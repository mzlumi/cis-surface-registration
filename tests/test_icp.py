"""Rigid ICP on synthetic data with known F_reg."""

import numpy as np
import pytest

from cisreg.boxtree import BoundingBoxTree
from cisreg.fileio import read_mesh
from cisreg.frames import Frame, frame_difference, random_rotation
from cisreg.icp import IcpOptions, format_history, icp, next_threshold, select_inliers
from conftest import DATA_DIR


@pytest.fixture(scope="module")
def bone():
    return read_mesh(DATA_DIR / "Problem4MeshFile.sur")


@pytest.fixture(scope="module")
def tree(bone):
    return BoundingBoxTree(bone)


def surface_points(mesh, rng, n):
    tri = rng.integers(0, mesh.n_triangles, size=n)
    w = rng.dirichlet(np.ones(3), size=n)
    return np.einsum("nk,nkd->nd", w, mesh.vertices[mesh.triangles[tri]])


def synthetic(mesh, rng, n=150, angle_deg=5.0, shift_mm=5.0, noise=0.0):
    """Points c on the surface, a known F_reg, and d = F_reg^-1 (c + noise)."""
    F_true = Frame(random_rotation(rng, np.radians(angle_deg)), rng.uniform(-1, 1, 3) * shift_mm / np.sqrt(3))
    c = surface_points(mesh, rng, n)
    c_noisy = c + rng.normal(scale=noise, size=c.shape) if noise else c
    return F_true, F_true.inverse().apply(c_noisy)


@pytest.mark.parametrize("seed", range(6))
def test_recovers_a_known_registration(seed, bone, tree):
    rng = np.random.default_rng(seed)
    F_true, d = synthetic(bone, rng)
    result = icp(d, tree)
    angle, distance = frame_difference(F_true, result.F)
    assert result.converged
    assert np.degrees(angle) < 1e-3 and distance < 1e-3
    assert result.history[-1].rms_residual < 1e-3


@pytest.mark.parametrize("seed", range(4))
def test_noisy_points_give_a_close_registration(seed, bone, tree):
    rng = np.random.default_rng(10 + seed)
    F_true, d = synthetic(bone, rng, n=200, noise=0.2)
    result = icp(d, tree)
    angle, distance = frame_difference(F_true, result.F)
    assert np.degrees(angle) < 0.3 and distance < 0.3


def test_without_rejection_the_rms_never_increases(bone, tree):
    """Matching and registration each lower the sum of squares, so the RMS is monotone."""
    rng = np.random.default_rng(20)
    _, d = synthetic(bone, rng, angle_deg=10.0, shift_mm=10.0, noise=0.1)
    result = icp(d, tree, options=IcpOptions(reject_outliers=False))
    rms = [h.rms_residual for h in result.history]
    assert all(b <= a + 1e-12 for a, b in zip(rms, rms[1:]))


def test_outlier_rejection_helps_with_gross_outliers(bone, tree):
    rng = np.random.default_rng(30)
    F_true, d = synthetic(bone, rng, n=200, noise=0.1)
    c = F_true.apply(d)
    bad = rng.choice(len(d), size=20, replace=False)
    direction = rng.normal(size=(20, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    c[bad] += direction * rng.uniform(8, 15, size=(20, 1))
    d = F_true.inverse().apply(c)
    with_rejection = icp(d, tree, options=IcpOptions(reject_outliers=True))
    without = icp(d, tree, options=IcpOptions(reject_outliers=False))
    error_with = frame_difference(F_true, with_rejection.F)
    error_without = frame_difference(F_true, without.F)
    assert error_with[1] < error_without[1] and error_with[0] < error_without[0]
    assert error_with[1] < 0.2
    assert with_rejection.history[-1].n_inliers <= 185


def test_threshold_shrinks_and_keeps_enough_pairs():
    options = IcpOptions(warmup_iterations=2, threshold_scale=3.0, eta_floor=0.5)
    residuals = np.array([0.1, 0.2, 0.3, 4.0])
    assert next_threshold(np.inf, 1, residuals, options) == np.inf
    eta = next_threshold(np.inf, 3, residuals, options)
    assert np.isclose(eta, 3.0 * residuals.mean())
    assert next_threshold(eta, 4, residuals * 2, options) == eta  # never grows
    assert next_threshold(eta, 5, residuals * 0.01, options) == 0.5  # floor
    tight = select_inliers(np.arange(10.0), 0.5, IcpOptions(min_inlier_fraction=0.5))
    assert tight.sum() == 5


def test_max_iterations_and_history_log(bone, tree):
    rng = np.random.default_rng(40)
    _, d = synthetic(bone, rng, angle_deg=10.0, shift_mm=10.0)
    result = icp(d, tree, options=IcpOptions(max_iterations=3))
    assert result.iterations == 3 and not result.converged
    text = format_history(result.history)
    assert len(text.splitlines()) == 4 and "inf" in text
