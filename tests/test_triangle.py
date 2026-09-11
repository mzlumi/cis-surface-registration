"""Closest point on a triangle: interior, edges, vertices, in-plane points and degenerate cases."""

import numpy as np
import pytest

from cisreg.triangle import closest_point_on_triangle, project_on_segment

# A right triangle in the z = 0 plane: p at the origin, q on x, r on y.
P = np.array([0.0, 0.0, 0.0])
Q = np.array([4.0, 0.0, 0.0])
R = np.array([0.0, 3.0, 0.0])


def closest(a, p=P, q=Q, r=R):
    return closest_point_on_triangle(np.asarray(a, dtype=float), p, q, r)


def reference_distance(a, p, q, r, n=400):
    """Smallest distance from a to a dense grid of points covering the triangle."""
    s, t = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    keep = s + t <= 1.0
    s, t = s[keep], t[keep]
    samples = p + s[:, None] * (q - p) + t[:, None] * (r - p)
    return np.min(np.linalg.norm(samples - a, axis=1))


@pytest.mark.parametrize("height", [0.0, 2.5, -7.0])
def test_interior_point_projects_straight_down(height):
    c, w = closest([1.0, 1.0, height])
    assert np.allclose(c, [1.0, 1.0, 0.0])
    assert np.allclose(w, [1 - 0.25 - 1 / 3, 0.25, 1 / 3])


@pytest.mark.parametrize(
    "a, expected",
    [
        ([2.0, -1.0, 0.0], [2.0, 0.0, 0.0]),  # beyond edge pq
        ([-1.0, 1.5, 0.0], [0.0, 1.5, 0.0]),  # beyond edge rp
        # Beyond qr: t = (a - q).(r - q) / |r - q|^2 = 17 / 25, so c = q + 0.68 (r - q).
        ([2.0, 3.0, 0.0], [1.28, 2.04, 0.0]),
        ([2.0, -1.0, 5.0], [2.0, 0.0, 0.0]),  # beyond pq, above the plane
    ],
)
def test_edge_regions(a, expected):
    c, w = closest(a)
    assert np.allclose(c, expected)
    assert np.isclose(w.sum(), 1.0) and np.all(w >= -1e-12)
    assert np.allclose(w @ np.array([P, Q, R]), c)


@pytest.mark.parametrize(
    "a, vertex",
    [
        ([-1.0, -1.0, 0.0], P),
        ([-2.0, -0.5, 3.0], P),
        ([6.0, -1.0, 0.0], Q),
        ([5.0, 0.5, -2.0], Q),
        ([-0.5, 5.0, 0.0], R),
        ([0.5, 6.0, 1.0], R),
    ],
)
def test_vertex_regions(a, vertex):
    c, w = closest(a)
    assert np.allclose(c, vertex)
    assert np.isclose(w.max(), 1.0)


@pytest.mark.parametrize("a", [P, Q, R, (P + Q) / 2, (Q + R) / 2, (R + P) / 2, (P + Q + R) / 3])
def test_points_on_the_triangle_are_their_own_closest_point(a):
    c, _ = closest(a)
    assert np.allclose(c, a)


def test_points_in_the_plane_outside_the_triangle():
    rng = np.random.default_rng(3)
    for _ in range(200):
        a = np.append(rng.uniform(-5, 9, size=2), 0.0)
        c, _ = closest(a)
        assert np.isclose(c[2], 0.0)
        assert np.linalg.norm(c - a) <= reference_distance(a, P, Q, R, n=300) + 1e-9


def test_obtuse_triangle_with_two_negative_coordinates():
    """The case where a fixed edge table fails.

    For a = (12, -0.1, 0), lambda (weight of q) < 0 and 1 - lambda - mu (weight of p) < 0.
    Checking lambda < 0 first would pick edge rp, whose nearest point is r at distance 2.0.
    The true closest point is on edge qr, at distance 2.1 / sqrt(2) (about 1.485).
    """
    p, q, r = np.array([0.0, 0, 0]), np.array([11.0, 1, 0]), np.array([10.0, 0, 0])
    a = np.array([12.0, -0.1, 0.0])
    c, w = closest_point_on_triangle(a, p, q, r)
    assert np.allclose(c, [10.95, 0.95, 0.0])
    assert np.isclose(np.linalg.norm(a - c), 2.1 / np.sqrt(2))
    assert np.isclose(w[0], 0.0)  # on edge qr
    assert np.linalg.norm(a - c) <= reference_distance(a, p, q, r) + 1e-9
    table_choice, _ = project_on_segment(a, r, p)
    assert np.isclose(np.linalg.norm(a - table_choice), np.hypot(2.0, 0.1))


@pytest.mark.parametrize("seed", range(30))
def test_random_triangles_against_dense_sampling(seed):
    """c is on the triangle, and no sampled triangle point is closer to a than c."""
    rng = np.random.default_rng(seed)
    p, q, r = rng.normal(scale=5.0, size=(3, 3))
    for a in rng.normal(scale=8.0, size=(20, 3)):
        c, w = closest_point_on_triangle(a, p, q, r)
        assert np.isclose(w.sum(), 1.0) and np.all(w >= -1e-12)
        assert np.allclose(w @ np.array([p, q, r]), c)
        assert np.linalg.norm(a - c) <= reference_distance(a, p, q, r, n=200) + 1e-9


@pytest.mark.parametrize("seed", range(30))
def test_closest_point_is_a_local_minimum(seed):
    """Moving c anywhere else on the triangle never brings it closer to a."""
    rng = np.random.default_rng(100 + seed)
    p, q, r = rng.normal(scale=5.0, size=(3, 3))
    a = rng.normal(scale=8.0, size=3)
    c, _ = closest_point_on_triangle(a, p, q, r)
    weights = rng.dirichlet(np.ones(3), size=2000)
    others = weights @ np.array([p, q, r])
    assert np.linalg.norm(a - c) <= np.min(np.linalg.norm(others - a, axis=1)) + 1e-12


def test_degenerate_collinear_triangle():
    p, q, r = np.array([0.0, 0, 0]), np.array([2.0, 0, 0]), np.array([5.0, 0, 0])
    c, w = closest_point_on_triangle(np.array([3.0, 4.0, 0.0]), p, q, r)
    assert np.allclose(c, [3.0, 0.0, 0.0])
    assert np.allclose(w @ np.array([p, q, r]), c)
    c, _ = closest_point_on_triangle(np.array([7.0, 1.0, 0.0]), p, q, r)
    assert np.allclose(c, [5.0, 0.0, 0.0])


def test_degenerate_repeated_corners():
    p = np.array([1.0, 2.0, 3.0])
    c, w = closest_point_on_triangle(np.zeros(3), p, p, p)
    assert np.allclose(c, p) and np.isclose(w.sum(), 1.0)
    q = np.array([3.0, 2.0, 3.0])
    c, _ = closest_point_on_triangle(np.array([2.0, 5.0, 3.0]), p, p, q)
    assert np.allclose(c, [2.0, 2.0, 3.0])


def test_broadcasting_many_points_and_many_triangles():
    rng = np.random.default_rng(5)
    points = rng.normal(size=(7, 1, 3))
    corners = rng.normal(size=(3, 11, 3))
    c, w = closest_point_on_triangle(points, corners[0], corners[1], corners[2])
    assert c.shape == w.shape == (7, 11, 3)
    for i in range(7):
        for j in range(11):
            cij, _ = closest_point_on_triangle(points[i, 0], *corners[:, j])
            assert np.allclose(c[i, j], cij)


def test_project_on_segment():
    p, q = np.array([0.0, 0, 0]), np.array([2.0, 0, 0])
    c, t = project_on_segment(np.array([1.0, 1.0, 0.0]), p, q)
    assert np.allclose(c, [1.0, 0, 0]) and np.isclose(t, 0.5)
    c, t = project_on_segment(np.array([-3.0, 1.0, 0.0]), p, q)
    assert np.allclose(c, p) and t == 0.0
    c, t = project_on_segment(np.array([9.0, 1.0, 0.0]), p, q)
    assert np.allclose(c, q) and t == 1.0
    c, t = project_on_segment(np.array([9.0, 1.0, 0.0]), p, p)
    assert np.allclose(c, p) and t == 0.0
