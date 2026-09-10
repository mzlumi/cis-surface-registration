"""Rigid 3D point-set to point-set registration.

Given paired points a_i and b_i, find the frame F = [R, p] that minimizes
    sum_i |R a_i + p - b_i|^2.

Method (Arun, Huang and Blostein, "Least-squares fitting of two 3-D point sets",
IEEE PAMI 9(5), 1987, as presented in the CIS I lecture "Point cloud to point
cloud rigid transformations", R. H. Taylor):

1. Subtract the centroids: a~_i = a_i - mean(a), b~_i = b_i - mean(b). The
   optimal translation is then p = mean(b) - R mean(a), so only R remains.
2. Form H = sum_i a~_i b~_i^T and take its SVD, H = U S V^T.
3. R = V D U^T with D = diag(1, 1, det(V U^T)).

The matrix D is the reflection fix. Without it, V U^T can have determinant -1
(a reflection), which happens when the points are coplanar (one singular value
is zero) or very noisy. Flipping the sign of the column of V that belongs to
the smallest singular value gives the best proper rotation; in the coplanar
case it is exact, because that column does not affect the fit (Arun et al.;
also Umeyama, IEEE PAMI 13(4), 1991).

The SVD is computed with numpy.linalg.svd (NumPy, Harris et al., Nature 585,
2020), which calls LAPACK. Everything else is written here.

Author: Parmida Mazloomi
"""

from __future__ import annotations

import numpy as np

from cisreg.frames import Frame


def register_points(a: np.ndarray, b: np.ndarray) -> Frame:
    """Least-squares rigid frame F with F a_i ~ b_i, for paired (N, 3) arrays, N >= 3."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"expected two (N, 3) arrays of the same shape, got {a.shape} and {b.shape}")
    if len(a) < 3:
        raise ValueError("at least 3 point pairs are needed to fix a rigid frame")
    a_mean = a.mean(axis=0)
    b_mean = b.mean(axis=0)
    H = (a - a_mean).T @ (b - b_mean)
    U, _, Vt = np.linalg.svd(H)
    V = Vt.T
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(V @ U.T)) or 1.0])
    R = V @ D @ U.T
    return Frame(R, b_mean - R @ a_mean)


def registration_residuals(F: Frame, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distances |F a_i - b_i| for each pair."""
    return np.linalg.norm(F.apply(a) - b, axis=1)
