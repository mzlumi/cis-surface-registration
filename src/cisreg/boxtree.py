"""Bounding-box tree over the triangles of a mesh, for fast closest-point search.

Idea (CIS I lecture "Finding point-pairs", R. H. Taylor, slides 46 to 61 and
95 to 98): if we already know a point on the surface at distance `bound` from
the query a, a group of triangles can only contain something closer if the box
around that group is closer than `bound`. Organizing the groups as a tree lets
the search skip most of the mesh.

Construction. Every node holds a contiguous run of triangles and the
axis-aligned box around all of their corners (corners, not just the sort
points, so every triangle lies fully inside its node's box). A node with more
than `leaf_size` triangles is split in two: the triangles are sorted by their
centroid (the "sort point") along the longest axis of the node's centroid
spread, and divided at the median. This is the KD-tree variant of the lecture's
octree; the median split keeps the tree balanced, with depth about
log2(T / leaf_size).

Lower bound. The distance from a to a box [lo, hi] is |max(lo - a, 0, a - hi)|
(zero inside). It is never larger than the distance from a to anything inside
the box, so a node whose box is farther than `bound` cannot hold the closest
point and can be skipped. (The lecture tests each axis against the box enlarged
by `bound`; that test is looser but equally safe.)

Search, for a batch of query points:
1. Seed an upper bound for each query: the exact distance to a hint triangle
   (for example the triangle matched in the previous ICP iteration), or else to
   the triangles of the leaf reached by walking down the tree, always toward the
   child whose box is nearer.
2. Walk the tree one level at a time, keeping the (query, node) pairs whose box
   is within that query's bound. Children of surviving internal nodes form the
   next level; surviving leaves contribute their triangles.
3. Drop (query, triangle) pairs whose triangle box is farther than the bound,
   then compute the exact closest point for the rest (closest_point_on_triangle)
   and keep the nearest triangle for each query.
The true closest triangle is at most `bound` away, so every box on its path from
the root is too, and it always survives; the result is exactly the brute-force
result. The bound is not lowered during the walk (unlike the depth-first version
in the lecture), which costs some extra triangle tests but lets every step run
as one vectorized NumPy operation over all queries at once.

Deforming meshes (PA5). The tree topology only depends on which triangles are
grouped together, so after the vertices move it is enough to recompute the
boxes from the new corners, children before parents (`update_vertices`). Boxes
may then overlap more, which costs some speed but not correctness, as noted in
the PA5 notes.

Author: Parmida Mazloomi
"""

from __future__ import annotations

import numpy as np

from cisreg.mesh import Mesh
from cisreg.search import MeshMatches
from cisreg.triangle import closest_point_on_triangle


def box_distance2(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Squared distance from points (N, 3) to boxes [lo, hi] (broadcast), 0 inside."""
    gap = np.maximum(np.maximum(lo - points, points - hi), 0.0)
    return np.sum(gap * gap, axis=-1)


class BoundingBoxTree:
    """Binary tree of axis-aligned bounding boxes over mesh triangles."""

    def __init__(self, mesh: Mesh, leaf_size: int = 8):
        self.triangles = mesh.triangles
        self.leaf_size = leaf_size
        corners = np.asarray(mesh.vertices, dtype=float)[self.triangles]
        self._build(corners.mean(axis=1))
        self.update_vertices(mesh.vertices)
        self.triangle_checks = 0  # exact triangle tests in the last query, for reports

    def _build(self, centroids: np.ndarray) -> None:
        order = np.arange(len(centroids))
        start, end, left, right = [], [], [], []

        def build(first: int, last: int) -> int:
            node = len(start)
            start.append(first)
            end.append(last)
            left.append(-1)
            right.append(-1)
            if last - first > self.leaf_size:
                ids = order[first:last]
                spread = centroids[ids].max(axis=0) - centroids[ids].min(axis=0)
                axis = int(np.argmax(spread))
                half = (last - first) // 2
                order[first:last] = ids[np.argpartition(centroids[ids, axis], half)]
                left[node] = build(first, first + half)
                right[node] = build(first + half, last)
            return node

        build(0, len(centroids))
        self.order = order
        self.start = np.array(start)
        self.end = np.array(end)
        self.left = np.array(left)
        self.right = np.array(right)
        self.is_leaf = self.left < 0
        self.n_nodes = len(start)

    def update_vertices(self, vertices: np.ndarray) -> None:
        """Move the vertices (same triangles) and refit every box, children first."""
        corners = np.asarray(vertices, dtype=float)[self.triangles]
        self.p, self.q, self.r = corners[:, 0], corners[:, 1], corners[:, 2]
        self.tri_lo = corners.min(axis=1)
        self.tri_hi = corners.max(axis=1)
        self.lo = np.empty((self.n_nodes, 3))
        self.hi = np.empty((self.n_nodes, 3))
        # Nodes were numbered parent before children, so reverse order visits children first.
        for node in range(self.n_nodes - 1, -1, -1):
            if self.is_leaf[node]:
                tris = self.order[self.start[node] : self.end[node]]
                self.lo[node] = self.tri_lo[tris].min(axis=0)
                self.hi[node] = self.tri_hi[tris].max(axis=0)
            else:
                a, b = self.left[node], self.right[node]
                self.lo[node] = np.minimum(self.lo[a], self.lo[b])
                self.hi[node] = np.maximum(self.hi[a], self.hi[b])

    def depth(self) -> int:
        def depth_of(node: int) -> int:
            if self.is_leaf[node]:
                return 1
            return 1 + max(depth_of(self.left[node]), depth_of(self.right[node]))

        return depth_of(0)

    def closest_points(self, queries: np.ndarray, hint: np.ndarray | None = None) -> MeshMatches:
        """Closest mesh points to (N, 3) queries.

        hint: optional (N,) triangle indices used only to seed the search bound,
        for example the triangles matched in the previous ICP iteration.
        """
        queries = np.atleast_2d(np.asarray(queries, dtype=float))
        n = len(queries)
        if hint is None:
            k, t = self._leaf_pairs(np.arange(n), self._descend(queries))
        else:
            k, t = np.arange(n), np.asarray(hint, dtype=int)
        seed = self._best_of_pairs(queries, k, t)
        bound2 = np.full(n, np.inf)
        bound2[seed[0]] = seed[1]

        k, node = np.arange(n), np.zeros(n, dtype=int)
        leaf_k, leaf_node = [], []
        while k.size:
            keep = box_distance2(queries[k], self.lo[node], self.hi[node]) <= bound2[k]
            k, node = k[keep], node[keep]
            leaf = self.is_leaf[node]
            leaf_k.append(k[leaf])
            leaf_node.append(node[leaf])
            k, node = k[~leaf], node[~leaf]
            k = np.concatenate([k, k])
            node = np.concatenate([self.left[node], self.right[node]])

        k, t = self._leaf_pairs(np.concatenate(leaf_k), np.concatenate(leaf_node))
        keep = box_distance2(queries[k], self.tri_lo[t], self.tri_hi[t]) <= bound2[k]
        k, t = k[keep], t[keep]
        self.triangle_checks = len(k)
        _, d2, points, weights, triangles = self._best_of_pairs(queries, k, t)
        return MeshMatches(points, np.sqrt(d2), triangles, weights)

    def _descend(self, queries: np.ndarray) -> np.ndarray:
        """Leaf reached by each query when always stepping into the nearer child box."""
        node = np.zeros(len(queries), dtype=int)
        inner = np.flatnonzero(~self.is_leaf[node])
        while inner.size:
            a, b = self.left[node[inner]], self.right[node[inner]]
            pts = queries[inner]
            nearer_a = box_distance2(pts, self.lo[a], self.hi[a]) <= box_distance2(pts, self.lo[b], self.hi[b])
            node[inner] = np.where(nearer_a, a, b)
            inner = inner[~self.is_leaf[node[inner]]]
        return node

    def _leaf_pairs(self, k: np.ndarray, leaves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Expand (query, leaf) pairs into (query, triangle) pairs."""
        counts = self.end[leaves] - self.start[leaves]
        first = np.repeat(self.start[leaves], counts)
        offset = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        return np.repeat(k, counts), self.order[first + offset]

    def _best_of_pairs(self, queries: np.ndarray, k: np.ndarray, t: np.ndarray):
        """Exact test of (query k, triangle t) pairs; the nearest triangle for each query.

        Returns (queries, squared distances, points, weights, triangles), one entry per
        distinct query in k, sorted by query index.
        """
        a = queries[k]
        c, w = closest_point_on_triangle(a, self.p[t], self.q[t], self.r[t])
        d2 = np.sum((c - a) ** 2, axis=1)
        order = np.lexsort((d2, k))
        ks = k[order]
        first = order[np.r_[True, ks[1:] != ks[:-1]]]
        return k[first], d2[first], c[first], w[first], t[first]
