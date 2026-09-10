import numpy as np
import pytest

from cisreg.frames import (
    Frame,
    frame_difference,
    is_rotation,
    random_rotation,
    rotation_angle,
    rotation_from_vector,
    rotation_vector,
    skew,
)

SEEDS = range(20)


def random_frame(rng: np.random.Generator) -> Frame:
    return Frame(random_rotation(rng), rng.uniform(-100, 100, size=3))


@pytest.mark.parametrize("seed", SEEDS)
def test_random_rotation_is_proper(seed):
    assert is_rotation(random_rotation(np.random.default_rng(seed)))


@pytest.mark.parametrize("seed", SEEDS)
def test_inverse_and_composition(seed):
    rng = np.random.default_rng(seed)
    F, G = random_frame(rng), random_frame(rng)
    x = rng.normal(size=(10, 3))
    assert np.allclose((F @ F.inverse()).as_matrix(), np.eye(4))
    assert np.allclose((F @ G).apply(x), F.apply(G.apply(x)))
    assert np.allclose(F.inverse().apply(F.apply(x)), x)
    assert np.allclose((F @ G).as_matrix(), F.as_matrix() @ G.as_matrix())


def test_apply_single_point_and_matrix_round_trip():
    rng = np.random.default_rng(1)
    F = random_frame(rng)
    x = rng.normal(size=3)
    assert np.allclose(F.apply(x), F.R @ x + F.p)
    G = Frame.from_matrix(F.as_matrix())
    assert np.allclose(G.R, F.R) and np.allclose(G.p, F.p)


def test_skew_is_cross_product():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=3), rng.normal(size=3)
    assert np.allclose(skew(a) @ b, np.cross(a, b))
    assert np.allclose(skew(a), -skew(a).T)


@pytest.mark.parametrize("seed", SEEDS)
def test_rotation_vector_round_trip(seed):
    rng = np.random.default_rng(seed)
    alpha = rng.normal(size=3)
    alpha *= rng.uniform(0.0, 3.0) / np.linalg.norm(alpha)
    R = rotation_from_vector(alpha)
    assert is_rotation(R)
    assert np.isclose(rotation_angle(R), np.linalg.norm(alpha))
    assert np.allclose(R @ alpha, alpha)  # the axis is fixed
    assert np.allclose(rotation_vector(R), alpha)


def test_rotation_about_z():
    R = rotation_from_vector([0.0, 0.0, np.pi / 2])
    assert np.allclose(R @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])


def test_rotation_vector_near_pi():
    alpha = np.array([0.0, 1.0, 0.0]) * (np.pi - 1e-9)
    assert np.allclose(rotation_vector(rotation_from_vector(alpha)), alpha, atol=1e-6)


def test_small_angle_matches_first_order():
    alpha = np.array([1e-5, -2e-5, 3e-5])
    assert np.allclose(rotation_from_vector(alpha), np.eye(3) + skew(alpha), atol=1e-9)


def test_frame_difference():
    F = Frame(rotation_from_vector([0.0, 0.0, 0.1]), np.array([1.0, 2.0, 3.0]))
    angle, distance = frame_difference(Frame.identity(), F)
    assert np.isclose(angle, 0.1) and np.isclose(distance, np.sqrt(14.0))
    assert frame_difference(F, F) == pytest.approx((0.0, 0.0), abs=1e-7)
