"""Triangle mesh representation.

A mesh is two arrays, as the PA3/PA4 handout suggests: the vertex coordinates
and, for each triangle, the indices of its three vertices.

Author: Parmida Mazloomi
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Mesh:
    """Triangle mesh in CT coordinates.

    vertices: (V, 3) vertex coordinates.
    triangles: (T, 3) vertex indices of each triangle.
    neighbors: (T, 3) index of the neighbour triangle opposite each vertex
        (-1 if none). Read from the file but not used by the algorithms.
    """

    vertices: np.ndarray
    triangles: np.ndarray
    neighbors: np.ndarray

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_triangles(self) -> int:
        return len(self.triangles)

    def corners(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the (T, 3) arrays of first, second and third triangle corners."""
        v = self.vertices[self.triangles]
        return v[:, 0], v[:, 1], v[:, 2]
