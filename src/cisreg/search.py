"""Closest point on a triangle mesh.

BruteForceSearch is the simple linear search suggested by the PA3/PA4 handout:
for each query point it computes the closest point on every triangle and keeps
the nearest one. It is the reference that any faster method is checked against.

Author: Parmida Mazloomi
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cisreg.mesh import Mesh
from cisreg.triangle import closest_point_on_triangle


@dataclass(frozen=True)
class MeshMatches:
    """Closest points on a mesh for N query points.

    points: (N, 3) closest points c_k.
    distances: (N,) |query_k - c_k|.
    triangles: (N,) index of the triangle that contains c_k.
    weights: (N, 3) barycentric weights of c_k on that triangle's corners.
    """

    points: np.ndarray
    distances: np.ndarray
    triangles: np.ndarray
    weights: np.ndarray


class BruteForceSearch:
    """Linear search over all triangles of a mesh."""

    def __init__(self, mesh: Mesh, chunk_size: int = 16):
        self.triangles = mesh.triangles
        self.chunk_size = chunk_size
        self.update_vertices(mesh.vertices)

    def update_vertices(self, vertices: np.ndarray) -> None:
        """Use new vertex positions with the same triangles (for a deforming mesh)."""
        corners = np.asarray(vertices, dtype=float)[self.triangles]
        self.p, self.q, self.r = corners[:, 0], corners[:, 1], corners[:, 2]

    def closest_points(self, queries: np.ndarray, hint: np.ndarray | None = None) -> MeshMatches:
        """Closest mesh points to (N, 3) queries. hint is accepted for a common interface and ignored."""
        queries = np.atleast_2d(np.asarray(queries, dtype=float))
        n = len(queries)
        points = np.empty((n, 3))
        weights = np.empty((n, 3))
        distances = np.empty(n)
        triangles = np.empty(n, dtype=int)
        rows = np.arange(min(self.chunk_size, n))
        for start in range(0, n, self.chunk_size):
            block = queries[start : start + self.chunk_size]
            # (block, T, 3): every query in the block against every triangle.
            c, w = closest_point_on_triangle(block[:, None, :], self.p, self.q, self.r)
            d2 = np.sum((c - block[:, None, :]) ** 2, axis=-1)
            best = np.argmin(d2, axis=1)
            k = rows[: len(block)]
            sl = slice(start, start + len(block))
            points[sl] = c[k, best]
            weights[sl] = w[k, best]
            distances[sl] = np.sqrt(d2[k, best])
            triangles[sl] = best
        return MeshMatches(points, distances, triangles, weights)
