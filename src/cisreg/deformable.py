"""Deformable registration to a statistical shape model (PA5).

Mode-weight step (PA5 handout, equations 1 to 5; PA5 notes, "Barycentric
coordinates of deforming triangle"). Suppose the closest point c_k to the sample
s_k lies on the triangle with vertices (s, t, u), with barycentric coordinates
(zeta_k, xi_k, psi_k). The deformed vertices are m_v = m_0,v + sum_m lambda_m m_m,v,
so the same barycentric point on the deformed triangle is
    c_k = q_0,k + sum_m lambda_m q_m,k,
    q_m,k = zeta_k m_m,s + xi_k m_m,t + psi_k m_m,u   (m = 0 ... M).
The point moves linearly with the weights. Keeping the triangle and the
barycentric coordinates of each match fixed, the weights that best move the
matched points onto the samples solve the linear least-squares problem
    s_k ~ q_0,k + sum_m lambda_m q_m,k   for all k,
i.e. A lambda ~ b with one 3-row block per sample: A_k = [q_1,k ... q_M,k] (3 x M)
and b_k = s_k - q_0,k. It is solved with numpy.linalg.lstsq (NumPy, which calls
LAPACK's SVD-based least-squares solver).

Author: Parmida Mazloomi
"""

from __future__ import annotations

import numpy as np

from cisreg.search import MeshMatches
from cisreg.shape_model import ShapeModel


def mode_basis(model: ShapeModel, matches: MeshMatches) -> np.ndarray:
    """The vectors q_m,k for m = 0 ... M, as an (M + 1, K, 3) array.

    q[0] interpolates the mean shape and q[m] interpolates mode m, using the
    triangle and barycentric weights of each match.
    """
    corners = model.triangles[matches.triangles]  # (K, 3) vertex indices
    w = matches.weights  # (K, 3)
    shapes = np.concatenate([model.mean[None], model.modes])  # (M + 1, V, 3)
    return np.einsum("kj,mkjd->mkd", w, shapes[:, corners])


def solve_mode_weights(s: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Least-squares lambda with s_k ~ q_0,k + sum_m lambda_m q_m,k, for (K, 3) samples s."""
    n_modes = len(q) - 1
    A = q[1:].reshape(n_modes, -1).T  # (3K, M): column m is q_m,k for all k, flattened
    b = (np.asarray(s, dtype=float) - q[0]).reshape(-1)
    weights, *_ = np.linalg.lstsq(A, b, rcond=None)
    return weights
