"""Mode-weight least squares on synthetic deformations with known weights."""

import numpy as np
import pytest

from cisreg.boxtree import BoundingBoxTree
from cisreg.deformable import (
    DeformableOptions,
    deformable_registration,
    format_deformable_history,
    mode_basis,
    solve_combined_step,
    solve_mode_weights,
)
from cisreg.fileio import read_mesh, read_modes
from cisreg.frames import Frame, frame_difference, is_rotation, random_rotation, rotation_from_vector
from cisreg.search import MeshMatches
from cisreg.shape_model import ShapeModel
from conftest import DATA_DIR


@pytest.fixture(scope="module")
def model():
    mesh = read_mesh(DATA_DIR / "Problem5MeshFile.sur")
    return ShapeModel.from_atlas(read_modes(DATA_DIR / "Problem5Modes.txt"), mesh)


def random_surface_points(model, weights, rng, n):
    """Points on the deformed surface, with the triangle and barycentric weights used."""
    tri = rng.integers(0, len(model.triangles), size=n)
    bary = rng.dirichlet(np.ones(3), size=n)
    vertices = model.vertices(weights)
    points = np.einsum("kj,kjd->kd", bary, vertices[model.triangles[tri]])
    return points, MeshMatches(points, np.zeros(n), tri, bary)


def test_basis_reproduces_the_deformed_point(model):
    rng = np.random.default_rng(0)
    weights = rng.normal(scale=60, size=6)
    points, matches = random_surface_points(model, weights, rng, 50)
    q = mode_basis(model, matches)
    assert q.shape == (7, 50, 3)
    assert np.allclose(q[0] + np.tensordot(weights, q[1:], axes=1), points)


@pytest.mark.parametrize("seed", range(5))
def test_known_weights_are_recovered_with_known_matches(seed, model):
    rng = np.random.default_rng(seed)
    weights = rng.normal(scale=60, size=6)
    points, matches = random_surface_points(model, weights, rng, 150)
    assert np.allclose(solve_mode_weights(points, mode_basis(model, matches)), weights, atol=1e-8)


def test_known_weights_are_recovered_with_closest_point_matches(model):
    """Matches found by the search on the deformed mesh may land on a neighbouring triangle
    (for points on edges), but the point is the same, so the solution is still exact."""
    rng = np.random.default_rng(10)
    weights = np.array([110.0, 50.0, 60.0, 45.0, -60.0, -20.0])
    points, _ = random_surface_points(model, weights, rng, 200)
    tree = BoundingBoxTree(model.mesh(np.zeros(6)))
    tree.update_vertices(model.vertices(weights))
    matches = tree.closest_points(points)
    assert np.max(matches.distances) < 1e-9
    assert np.allclose(solve_mode_weights(points, mode_basis(model, matches)), weights, atol=1e-6)


def test_noise_gives_close_weights(model):
    rng = np.random.default_rng(11)
    weights = rng.normal(scale=60, size=6)
    points, matches = random_surface_points(model, weights, rng, 400)
    noisy = points + rng.normal(scale=0.1, size=points.shape)
    estimate = solve_mode_weights(noisy, mode_basis(model, matches))
    assert np.max(np.abs(estimate - weights)) < 1.0


def test_combined_step_recovers_a_small_correction(model):
    """One linearized solve recovers alpha, epsilon and the weight change to second order."""
    rng = np.random.default_rng(13)
    true_weights = rng.normal(scale=60, size=6)
    target, matches = random_surface_points(model, true_weights, rng, 300)
    q = mode_basis(model, matches)
    dweights = rng.normal(scale=0.5, size=6)
    current = q[0] + np.tensordot(true_weights - dweights, q[1:], axes=1)
    alpha = np.array([1e-3, -2e-3, 1.5e-3])
    epsilon = np.array([0.05, -0.02, 0.03])
    delta = Frame(rotation_from_vector(alpha), epsilon)
    s = delta.inverse().apply(target)  # so that delta s_k is exactly on the target shape
    a, e, dw = solve_combined_step(s, current, q)
    assert np.allclose(a, alpha, atol=2e-5)
    assert np.allclose(e, epsilon, atol=5e-3)
    assert np.allclose(dw, dweights, atol=0.05)


def synthetic_problem(model, rng, n=200, noise=0.0):
    weights = rng.normal(scale=60, size=model.n_modes)
    F_true = Frame(random_rotation(rng, np.radians(3.0)), rng.uniform(-2, 2, size=3))
    points, _ = random_surface_points(model, weights, rng, n)
    points = points + rng.normal(scale=noise, size=points.shape) if noise else points
    return weights, F_true, F_true.inverse().apply(points)


def test_full_registration_recovers_pose_and_weights(model):
    rng = np.random.default_rng(14)
    weights, F_true, d = synthetic_problem(model, rng)
    result = deformable_registration(d, model)
    assert result.converged
    assert is_rotation(result.F.R)
    angle, distance = frame_difference(F_true, result.F)
    assert np.degrees(angle) < 1e-3 and distance < 1e-3
    assert np.max(np.abs(result.weights - weights)) < 0.01
    assert result.history[-1].rms_residual < 1e-3
    assert result.count("rigid") > 0 and result.count("mode") > 0 and result.count("combined") > 0
    log = format_deformable_history(result.history).splitlines()
    assert len(log) == len(result.history) + 1 and "lambda_6" in log[0]
    assert log[-1].split()[1] == "combined"


def test_combined_steps_alone_reach_the_same_answer(model):
    rng = np.random.default_rng(15)
    _, _, d = synthetic_problem(model, rng, noise=0.1)
    default = deformable_registration(d, model)
    combined_only = deformable_registration(d, model, DeformableOptions(use_alternation=False))
    assert combined_only.count("mode") == 0
    assert np.max(np.abs(default.weights - combined_only.weights)) < 0.01
    angle, distance = frame_difference(default.F, combined_only.F)
    assert np.degrees(angle) < 1e-3 and distance < 1e-3


def test_fewer_modes(model):
    rng = np.random.default_rng(12)
    small = ShapeModel(model.mean, model.modes[:3], model.triangles)
    weights = np.array([30.0, -20.0, 10.0])
    points, matches = random_surface_points(small, weights, rng, 100)
    assert np.allclose(solve_mode_weights(points, mode_basis(small, matches)), weights, atol=1e-8)
