"""Rigid iterative closest point (ICP) registration of sample points to a mesh.

Problem (PA4): find F_reg such that the points s_k = F_reg d_k lie on the
surface. The correspondences are unknown, so ICP alternates two steps
(Besl and McKay, IEEE PAMI 14(2), 1992; CIS I lecture "Registration part 1",
R. H. Taylor, "Outline of a practical ICP code"):

1. Matching: with the current F_n, find the closest mesh point c_k to each
   s_k = F_n d_k.
2. Registration: treat (d_k, c_k) as known pairs and solve the rigid
   point-set registration F_{n+1} = argmin_F sum_k |F d_k - c_k|^2.

Each step can only lower the sum of squared distances (step 2 is optimal for
fixed pairs, and step 1 picks the nearest surface point for a fixed F), so the
error decreases monotonically to a local minimum. A good initial guess is
needed to reach the right minimum.

Outlier handling. A match threshold eta_n excludes pairs with |s_k - c_k| > eta_n
from step 2. Following the lecture, eta starts very large (all pairs used) and
then shrinks: after `warmup_iterations`, eta_n = max(eta_floor,
min(eta_{n-1}, threshold_scale * mean residual)). If fewer than
`min_inlier_fraction` of the pairs survive, eta is raised to keep that fraction,
since too tight a threshold can lock in a false minimum.

Stopping rule. Stop when the update moved every sample point by less than
`motion_tol` (max_k |F_{n+1} d_k - F_n d_k|), which bounds both the rotation and
the translation change of F_reg; or when the mean residual has changed by less
than the fraction `stall_tol` for `patience` iterations in a row (the
lecture's ratio test gamma <= eps_n / eps_{n-1} <= 1); or at `max_iterations`.

Author: Parmida Mazloomi
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cisreg.frames import Frame, frame_difference
from cisreg.registration import register_points
from cisreg.search import MeshMatches


@dataclass(frozen=True)
class IcpOptions:
    max_iterations: int = 500
    motion_tol: float = 1e-5  # mm
    stall_tol: float = 1e-9
    patience: int = 5
    reject_outliers: bool = True
    warmup_iterations: int = 3
    threshold_scale: float = 3.0
    eta_floor: float = 0.5  # mm, never reject pairs closer than this
    min_inlier_fraction: float = 0.5


@dataclass(frozen=True)
class IcpIteration:
    """Statistics after iteration n, computed on all N pairs.

    eta and n_inliers describe the pairs used for the registration of this
    iteration; motion is max_k |F_n d_k - F_{n-1} d_k|.
    """

    n: int
    mean_residual: float
    rms_residual: float
    max_residual: float
    eta: float
    n_inliers: int
    rotation_change: float  # radians
    translation_change: float  # mm
    motion: float  # mm


@dataclass(frozen=True)
class IcpResult:
    F: Frame
    s: np.ndarray  # (N, 3) final sample points F d_k
    matches: MeshMatches  # closest points to s
    history: list[IcpIteration] = field(default_factory=list)
    converged: bool = False
    reason: str = ""

    @property
    def iterations(self) -> int:
        return len(self.history)


def residual_stats(s: np.ndarray, matches: MeshMatches) -> tuple[float, float, float]:
    e = matches.distances
    return float(e.mean()), float(np.sqrt(np.mean(e**2))), float(e.max())


def next_threshold(eta: float, n: int, residuals: np.ndarray, options: IcpOptions) -> float:
    """Match threshold for iteration n (1-based), given the current residuals."""
    if not options.reject_outliers or n <= options.warmup_iterations:
        return np.inf
    return max(options.eta_floor, min(eta, options.threshold_scale * float(residuals.mean())))


def select_inliers(residuals: np.ndarray, eta: float, options: IcpOptions) -> np.ndarray:
    """Pairs within eta, widened if needed to keep min_inlier_fraction of them."""
    inliers = residuals <= eta
    needed = max(3, int(np.ceil(options.min_inlier_fraction * len(residuals))))
    if inliers.sum() < needed:
        inliers = residuals <= np.partition(residuals, needed - 1)[needed - 1]
    return inliers


def icp(
    d: np.ndarray,
    search,
    F0: Frame | None = None,
    options: IcpOptions | None = None,
) -> IcpResult:
    """Rigid ICP of the (N, 3) points d to the surface held by `search`.

    search must provide closest_points(queries, hint=None) -> MeshMatches, as
    BruteForceSearch and BoundingBoxTree do. Returns the final F_reg, the sample
    points s_k = F_reg d_k, their closest points and the per-iteration history.
    """
    options = options or IcpOptions()
    d = np.asarray(d, dtype=float)
    F = F0 or Frame.identity()
    s = F.apply(d)
    matches = search.closest_points(s)
    history: list[IcpIteration] = []
    eta = np.inf
    stalled = 0
    reason = "reached max_iterations"
    converged = False

    for n in range(1, options.max_iterations + 1):
        eta = next_threshold(eta, n, matches.distances, options)
        inliers = select_inliers(matches.distances, eta, options)
        F_new = register_points(d[inliers], matches.points[inliers])
        s_new = F_new.apply(d)
        motion = float(np.max(np.linalg.norm(s_new - s, axis=1)))
        rotation_change, translation_change = frame_difference(F, F_new)
        previous_mean = float(matches.distances.mean())
        F, s = F_new, s_new
        matches = search.closest_points(s, hint=matches.triangles)
        mean, rms, worst = residual_stats(s, matches)
        history.append(
            IcpIteration(n, mean, rms, worst, eta, int(inliers.sum()), rotation_change,
                         translation_change, motion)
        )
        if motion < options.motion_tol:
            converged, reason = True, "update moved the points less than motion_tol"
            break
        change = abs(previous_mean - mean) / max(previous_mean, 1e-12)
        stalled = stalled + 1 if change < options.stall_tol else 0
        if stalled >= options.patience:
            converged, reason = True, "mean residual stalled"
            break

    return IcpResult(F, s, matches, history, converged, reason)


def format_history(history: list[IcpIteration]) -> str:
    """Per-iteration residual log as a fixed-width text table."""
    lines = [
        "  iter   mean_mm    rms_mm    max_mm     eta_mm  inliers   d_rot_deg   d_trans_mm",
    ]
    for h in history:
        eta = "inf" if np.isinf(h.eta) else f"{h.eta:.4f}"
        lines.append(
            f"{h.n:6d} {h.mean_residual:9.5f} {h.rms_residual:9.5f} {h.max_residual:9.5f} "
            f"{eta:>10s} {h.n_inliers:8d} {np.degrees(h.rotation_change):11.6f} "
            f"{h.translation_change:12.6f}"
        )
    return "\n".join(lines)
