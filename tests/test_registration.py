"""Property tests for the SVD point-set registration on random transforms and noise."""

import numpy as np
import pytest

from cisreg.frames import Frame, frame_difference, is_rotation, random_rotation
from cisreg.registration import register_points, registration_residuals

SEEDS = range(50)


def random_problem(rng: np.random.Generator, n: int):
    F = Frame(random_rotation(rng), rng.uniform(-200, 200, size=3))
    a = rng.uniform(-50, 50, size=(n, 3))
    return F, a, F.apply(a)


@pytest.mark.parametrize("seed", SEEDS)
def test_exact_recovery(seed):
    rng = np.random.default_rng(seed)
    F, a, b = random_problem(rng, n=int(rng.integers(3, 40)))
    G = register_points(a, b)
    assert is_rotation(G.R)
    angle, distance = frame_difference(F, G)
    assert angle < 1e-9 and distance < 1e-8


@pytest.mark.parametrize("seed", SEEDS)
def test_noisy_fit_is_least_squares(seed):
    """With noise, the estimate fits at least as well as the true frame, and is close to it."""
    rng = np.random.default_rng(seed)
    F, a, b = random_problem(rng, n=30)
    sigma = 0.5
    b_noisy = b + rng.normal(scale=sigma, size=b.shape)
    G = register_points(a, b_noisy)
    assert is_rotation(G.R)
    fit_estimate = np.sum(registration_residuals(G, a, b_noisy) ** 2)
    fit_true = np.sum(registration_residuals(F, a, b_noisy) ** 2)
    assert fit_estimate <= fit_true + 1e-9
    angle, distance = frame_difference(F, G)
    assert np.degrees(angle) < 2.0 and distance < 2.0


@pytest.mark.parametrize("seed", SEEDS)
def test_estimate_is_a_local_minimum(seed):
    """Small rigid perturbations of the estimate never reduce the squared error."""
    rng = np.random.default_rng(seed)
    _, a, b = random_problem(rng, n=12)
    b = b + rng.normal(scale=2.0, size=b.shape)
    G = register_points(a, b)
    best = np.sum(registration_residuals(G, a, b) ** 2)
    for _ in range(20):
        delta = Frame(random_rotation(rng, max_angle=1e-3), rng.normal(scale=1e-3, size=3))
        assert np.sum(registration_residuals(delta @ G, a, b) ** 2) >= best - 1e-9


def test_coplanar_points_give_a_rotation_not_a_reflection():
    """Coplanar points are where V U^T can be a reflection; the fix must recover F exactly."""
    reflections = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        a = np.column_stack([rng.uniform(-50, 50, size=(8, 2)), np.zeros(8)])
        F = Frame(random_rotation(rng), rng.uniform(-100, 100, size=3))
        b = F.apply(a)
        U, _, Vt = np.linalg.svd((a - a.mean(axis=0)).T @ (b - b.mean(axis=0)))
        reflections += np.linalg.det(Vt.T @ U.T) < 0
        G = register_points(a, b)
        assert is_rotation(G.R)
        angle, distance = frame_difference(F, G)
        assert angle < 1e-9 and distance < 1e-8
    assert reflections > 0, "no seed exercised the reflection case"


def test_mirrored_points_still_give_a_proper_rotation():
    """A mirror image cannot be fit by a rotation, but the result must still be one."""
    rng = np.random.default_rng(7)
    a = rng.normal(size=(10, 3))
    b = a * np.array([-1.0, 1.0, 1.0])
    G = register_points(a, b)
    assert is_rotation(G.R)


def test_minimum_three_points_and_shape_checks():
    with pytest.raises(ValueError):
        register_points(np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        register_points(np.zeros((4, 3)), np.zeros((5, 3)))
