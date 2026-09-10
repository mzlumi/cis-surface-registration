"""Rigid frames F = [R, p] and rotation helpers.

A frame maps a point x to F x = R x + p. Composition and inversion follow the
frames lecture of CIS I (R. H. Taylor):
    F1 F2 = [R1 R2, R1 p2 + p1],   F^-1 = [R^T, -R^T p].

Author: Parmida Mazloomi
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Frame:
    """Rigid transformation with rotation R (3, 3) and translation p (3,)."""

    R: np.ndarray
    p: np.ndarray

    @classmethod
    def identity(cls) -> Frame:
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> Frame:
        """Build a frame from a 4 x 4 homogeneous matrix."""
        matrix = np.asarray(matrix, dtype=float)
        return cls(matrix[:3, :3].copy(), matrix[:3, 3].copy())

    def as_matrix(self) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3, :3] = self.R
        matrix[:3, 3] = self.p
        return matrix

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Transform one point (3,) or an array of points (..., 3)."""
        return np.asarray(points) @ self.R.T + self.p

    def inverse(self) -> Frame:
        R_inv = self.R.T
        return Frame(R_inv, -R_inv @ self.p)

    def __matmul__(self, other: Frame) -> Frame:
        return Frame(self.R @ other.R, self.R @ other.p + self.p)


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix with skew(v) @ x == cross(v, x)."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotation_from_vector(alpha: np.ndarray) -> np.ndarray:
    """Rotation by angle |alpha| about the axis alpha / |alpha| (Rodrigues' formula).

    R = I + sin(t) K + (1 - cos(t)) K^2, with t = |alpha| and K = skew(alpha / t).
    This is an exact rotation for any alpha, so it is the right way to turn the
    small-angle vector of a linearized update into a proper rotation (rather than
    using I + skew(alpha), which is not orthogonal).
    """
    alpha = np.asarray(alpha, dtype=float)
    angle = float(np.linalg.norm(alpha))
    if angle < 1e-15:
        return np.eye(3) + skew(alpha)
    K = skew(alpha / angle)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def rotation_vector(R: np.ndarray) -> np.ndarray:
    """Inverse of rotation_from_vector: the axis times the angle (radians) of R."""
    angle = rotation_angle(R)
    if angle < 1e-12:
        return np.zeros(3)
    if np.pi - angle < 1e-6:
        # Near 180 degrees, the axis is the column of R + I with the largest norm.
        M = R + np.eye(3)
        axis = M[:, np.argmax(np.linalg.norm(M, axis=0))]
        return angle * axis / np.linalg.norm(axis)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return angle * w / (2.0 * np.sin(angle))


def rotation_angle(R: np.ndarray) -> float:
    """Rotation angle of R in radians.

    Uses trace(R) = 1 + 2 cos(angle) and |R - R^T| / 2 = sqrt(2) sin(angle), so
    atan2 stays accurate for small angles where arccos alone loses precision.
    """
    cos_angle = (np.trace(R) - 1.0) / 2.0
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    sin_angle = np.linalg.norm(w) / 2.0
    return float(np.arctan2(sin_angle, cos_angle))


def is_rotation(R: np.ndarray, tol: float = 1e-9) -> bool:
    """True if R is orthonormal with determinant +1."""
    return bool(np.allclose(R.T @ R, np.eye(3), atol=tol) and abs(np.linalg.det(R) - 1.0) < tol)


def random_rotation(rng: np.random.Generator, max_angle: float = np.pi) -> np.ndarray:
    """Rotation about a uniformly random axis by an angle uniform in [0, max_angle]."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return rotation_from_vector(axis * rng.uniform(0.0, max_angle))


def frame_difference(F1: Frame, F2: Frame) -> tuple[float, float]:
    """Rotation angle (radians) and translation norm of F1^-1 F2."""
    D = F1.inverse() @ F2
    return rotation_angle(D.R), float(np.linalg.norm(D.p))
