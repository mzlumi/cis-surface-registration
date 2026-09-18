"""Mode-weight least squares on synthetic deformations with known weights."""

import numpy as np
import pytest

from cisreg.boxtree import BoundingBoxTree
from cisreg.deformable import mode_basis, solve_mode_weights
from cisreg.fileio import read_mesh, read_modes
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


def test_fewer_modes(model):
    rng = np.random.default_rng(12)
    small = ShapeModel(model.mean, model.modes[:3], model.triangles)
    weights = np.array([30.0, -20.0, 10.0])
    points, matches = random_surface_points(small, weights, rng, 100)
    assert np.allclose(solve_mode_weights(points, mode_basis(small, matches)), weights, atol=1e-8)
