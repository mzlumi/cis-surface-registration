"""Statistical shape model (atlas) of the bone for PA5.

The atlas gives a mean shape m_0 (one point per mesh vertex) and modes of
variation m_1 ... m_M (one displacement per vertex). For mode weights
lambda = (lambda_1, ..., lambda_M), vertex i of the deformed bone is
    m_i(lambda) = m_0,i + sum_m lambda_m m_m,i
(PA5 handout). The triangles are those of the mean mesh. In the course data the
modes are orthonormal when each is flattened to a 3V vector (they are principal
components), so lambda_m is in mm of displacement along a unit-norm mode.

Author: Parmida Mazloomi
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cisreg.mesh import Mesh


@dataclass(frozen=True)
class ShapeModel:
    """mean: (V, 3), modes: (M, V, 3), triangles: (T, 3) vertex indices."""

    mean: np.ndarray
    modes: np.ndarray
    triangles: np.ndarray

    @classmethod
    def from_atlas(cls, atlas: np.ndarray, mesh: Mesh, n_modes: int | None = None, tol: float = 1e-5):
        """Build from read_modes() output; mode 0 must match the mesh vertices.

        n_modes selects the first modes to use (the PA5 sample header says how many).
        """
        check_mean_matches_mesh(atlas[0], mesh, tol)
        modes = atlas[1:] if n_modes is None else atlas[1 : 1 + n_modes]
        if n_modes is not None and len(modes) < n_modes:
            raise ValueError(f"the atlas has {len(atlas) - 1} modes, {n_modes} requested")
        return cls(mean=atlas[0].copy(), modes=modes.copy(), triangles=mesh.triangles)

    @property
    def n_modes(self) -> int:
        return len(self.modes)

    def vertices(self, weights: np.ndarray) -> np.ndarray:
        """Deformed vertices m_0 + sum_m lambda_m m_m."""
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (self.n_modes,):
            raise ValueError(f"expected {self.n_modes} weights, got shape {weights.shape}")
        return self.mean + np.tensordot(weights, self.modes, axes=1)

    def mesh(self, weights: np.ndarray) -> Mesh:
        return Mesh(self.vertices(weights), self.triangles, -np.ones_like(self.triangles))


def check_mean_matches_mesh(mean: np.ndarray, mesh: Mesh, tol: float = 1e-5) -> None:
    """Raise ValueError unless atlas mode 0 equals the mesh vertices to within tol (mm)."""
    if mean.shape != mesh.vertices.shape:
        raise ValueError(f"mode 0 has shape {mean.shape}, the mesh has {mesh.vertices.shape}")
    worst = float(np.max(np.abs(mean - mesh.vertices)))
    if worst > tol:
        raise ValueError(f"mode 0 differs from the mesh vertices by up to {worst:.3g} mm")
