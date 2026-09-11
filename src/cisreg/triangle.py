"""Closest point on a triangle.

Method (CIS I lecture "Finding point-pairs", R. H. Taylor, slides 11 to 15):

1. Write the point as a ~ p + lambda (q - p) + mu (r - p) and solve this 3 x 2
   system in the least-squares sense. The normal equations are 2 x 2,
       [e1.e1  e1.e2] [lambda]   [w.e1]
       [e1.e2  e2.e2] [mu    ] = [w.e2],   e1 = q - p, e2 = r - p, w = a - p,
   and are solved explicitly with Cramer's rule, as the lecture suggests. The
   point c = p + lambda e1 + mu e2 is the projection of a onto the plane of the
   triangle, and (1 - lambda - mu, lambda, mu) are its barycentric coordinates
   with respect to (p, q, r).
2. If lambda >= 0, mu >= 0 and lambda + mu <= 1, c is inside the triangle, and it
   is the closest point: the offset a - c is normal to the plane, so moving
   within the triangle can only make the distance larger.
3. Otherwise the closest point is on the boundary. The lecture's table picks one
   edge from the sign pattern (lambda < 0: edge rp, mu < 0: edge pq,
   lambda + mu > 1: edge qr). When two coordinates are negative, the point lies
   near the shared vertex and the answer can be on either of the two edges that
   meet there, so a fixed choice can be wrong for obtuse triangles (see
   tests/test_triangle.py). Here a is projected onto all three edges with
   ProjectOnSegment (projection clamped to the segment) and the nearest result
   is kept. This is always correct, and it also covers degenerate triangles
   (collinear or repeated corners), where the 2 x 2 system is singular.

All functions broadcast over leading dimensions, so one call can handle many
points against one triangle, one point against many triangles, or pairs.

Author: Parmida Mazloomi
"""

from __future__ import annotations

import numpy as np

# Triangles with |e1 x e2|^2 below this fraction of |e1|^2 |e2|^2 are treated as
# degenerate (sine of the corner angle below about 1e-6).
_DEGENERATE = 1e-12


def _dot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.einsum("...i,...i->...", x, y)


def project_on_segment(a: np.ndarray, p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closest point to a on the segment from p to q, and its parameter t in [0, 1].

    t = (a - p).(q - p) / |q - p|^2, clamped to [0, 1]; the point is p + t (q - p).
    A zero-length segment returns p with t = 0.
    """
    d = q - p
    length2 = _dot(d, d)
    safe = np.where(length2 > 0.0, length2, 1.0)
    t = np.clip(np.where(length2 > 0.0, _dot(a - p, d) / safe, 0.0), 0.0, 1.0)
    return p + t[..., None] * d, t


def closest_point_on_triangle(
    a: np.ndarray, p: np.ndarray, q: np.ndarray, r: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Closest point on triangle (p, q, r) to a.

    Arguments are (..., 3) arrays that broadcast against each other.
    Returns the closest points (..., 3) and their barycentric weights (..., 3)
    with respect to (p, q, r), so that point = w0 p + w1 q + w2 r.
    """
    a, p, q, r = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (a, p, q, r)))
    e1, e2, w = q - p, r - p, a - p
    d11, d12, d22 = _dot(e1, e1), _dot(e1, e2), _dot(e2, e2)
    w1, w2 = _dot(w, e1), _dot(w, e2)
    det = d11 * d22 - d12 * d12
    degenerate = det <= _DEGENERATE * d11 * d22
    safe_det = np.where(degenerate, 1.0, det)
    lam = (d22 * w1 - d12 * w2) / safe_det
    mu = (d11 * w2 - d12 * w1) / safe_det
    inside = ~degenerate & (lam >= 0.0) & (mu >= 0.0) & (lam + mu <= 1.0)

    point = p + lam[..., None] * e1 + mu[..., None] * e2
    weights = np.stack([1.0 - lam - mu, lam, mu], axis=-1)
    if np.all(inside):
        return point, weights

    # Boundary: nearest of the three clamped edge projections.
    c_pq, t_pq = project_on_segment(a, p, q)
    c_qr, t_qr = project_on_segment(a, q, r)
    c_rp, t_rp = project_on_segment(a, r, p)
    zero = np.zeros_like(t_pq)
    edge_points = np.stack([c_pq, c_qr, c_rp], axis=-2)
    edge_weights = np.stack(
        [
            np.stack([1.0 - t_pq, t_pq, zero], axis=-1),
            np.stack([zero, 1.0 - t_qr, t_qr], axis=-1),
            np.stack([t_rp, zero, 1.0 - t_rp], axis=-1),
        ],
        axis=-2,
    )
    offsets = edge_points - a[..., None, :]
    best = np.argmin(_dot(offsets, offsets), axis=-1)[..., None, None]
    boundary_point = np.take_along_axis(edge_points, best, axis=-2)[..., 0, :]
    boundary_weights = np.take_along_axis(edge_weights, best, axis=-2)[..., 0, :]

    point = np.where(inside[..., None], point, boundary_point)
    weights = np.where(inside[..., None], weights, boundary_weights)
    return point, weights
