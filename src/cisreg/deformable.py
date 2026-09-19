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

Combined step (handout equations 6 and 7). Instead of holding F_reg fixed, also
look for a small rigid correction Delta F = [Delta R(alpha), epsilon] applied
to the current samples s_k = F_reg d_k. To first order Delta R(alpha) s ~ s + alpha x s,
so requiring the corrected sample to land on the re-weighted model point gives
    s_k + alpha x s_k + epsilon ~ c_k + sum_m Delta lambda_m q_m,k,
which rearranges (using alpha x s = -s x alpha) to the linear system
    s_k x alpha - epsilon + sum_m Delta lambda_m q_m,k ~ s_k - c_k.
Each sample gives three rows [skew(s_k), -I, q_1,k ... q_M,k] in the unknowns
(alpha, epsilon, Delta lambda). Because c_k = q_0,k + sum_m lambda_m q_m,k already
includes the current weights, the weights in this equation are the change
Delta lambda = lambda^(t+1) - lambda^(t) (the handout writes lambda^(t+1);
with c_k on the right-hand side the increment is what keeps the equation
consistent). After solving, Delta R is built from alpha with Rodrigues'
formula, which gives an exact rotation; I + skew(alpha) would not be orthogonal.
Then F_reg <- Delta F F_reg, lambda <- lambda + Delta lambda, the mesh and the
tree boxes are updated, and the matches are found again.

Overall procedure (deformable_registration). Following the handout, which
used "the first method, followed by the second method":
1. Rigid ICP to the mean shape (lambda = 0).
2. Alternate: with F_reg fixed, repeat mode-weight steps (match on the current
   deformed mesh, solve for lambda, deform) until lambda stops changing; then,
   with the shape fixed, run rigid ICP from the current F_reg. Repeat until
   neither step changes anything.
3. Refine with combined steps until the update moves the samples and the
   vertices by less than the tolerance. The alternating phase can be slow when
   a mode and a rigid motion pull the points in similar directions (on the
   debug sets, running it to convergence took about 250 mode steps and 280
   rigid ICP iterations); the combined step solves for both together. So by
   default the alternation is limited to a few rounds that bring the shape
   close, and the combined steps finish. Run to convergence, either phase alone
   reaches the same solution on the debug sets (weights within 0.001).

Author: Parmida Mazloomi
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cisreg.boxtree import BoundingBoxTree
from cisreg.frames import Frame, rotation_from_vector
from cisreg.icp import IcpOptions, icp
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


def solve_combined_step(s: np.ndarray, c: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve s_k x alpha - epsilon + sum_m dlambda_m q_m,k ~ s_k - c_k (handout eq. 7).

    Returns (alpha, epsilon, dlambda).
    """
    s = np.asarray(s, dtype=float)
    k = len(s)
    n_modes = len(q) - 1
    A = np.zeros((k, 3, 6 + n_modes))
    x, y, z = s[:, 0], s[:, 1], s[:, 2]
    # skew(s) alpha = s x alpha
    A[:, 0, 1], A[:, 0, 2] = -z, y
    A[:, 1, 0], A[:, 1, 2] = z, -x
    A[:, 2, 0], A[:, 2, 1] = -y, x
    A[:, :, 3:6] = -np.eye(3)
    A[:, :, 6:] = np.transpose(q[1:], (1, 2, 0))
    solution, *_ = np.linalg.lstsq(A.reshape(3 * k, -1), (s - c).reshape(-1), rcond=None)
    return solution[:3], solution[3:6], solution[6:]


@dataclass(frozen=True)
class DeformableOptions:
    max_outer: int = 3
    max_mode_iterations: int = 10
    max_combined_iterations: int = 300
    weight_tol: float = 1e-5  # stop the mode loop when max |change in lambda| is below this
    motion_tol: float = 1e-5  # mm, for the outer loop and the combined steps
    icp: IcpOptions = IcpOptions()
    use_alternation: bool = True
    use_combined: bool = True


@dataclass(frozen=True)
class DeformableStep:
    """One step of the deformable registration and the residuals after it (mm)."""

    kind: str  # "rigid", "mode" or "combined"
    mean_residual: float
    rms_residual: float
    max_residual: float
    weights: np.ndarray
    motion: float  # largest movement of a sample point or a model vertex caused by the step


@dataclass(frozen=True)
class DeformableResult:
    F: Frame
    weights: np.ndarray
    s: np.ndarray
    matches: MeshMatches
    history: list[DeformableStep] = field(default_factory=list)
    converged: bool = False

    def count(self, kind: str) -> int:
        return sum(step.kind == kind for step in self.history)


def deformable_registration(
    d: np.ndarray, model: ShapeModel, options: DeformableOptions | None = None
) -> DeformableResult:
    """Estimate F_reg and the mode weights so that F_reg d_k lie on the deformed model."""
    options = options or DeformableOptions()
    d = np.asarray(d, dtype=float)
    weights = np.zeros(model.n_modes)
    vertices = model.vertices(weights)
    tree = BoundingBoxTree(model.mesh(weights))
    history: list[DeformableStep] = []

    def record(kind: str, matches: MeshMatches, motion: float) -> None:
        e = matches.distances
        history.append(
            DeformableStep(kind, float(e.mean()), float(np.sqrt(np.mean(e**2))), float(e.max()),
                           weights.copy(), motion)
        )

    def run_rigid(F: Frame) -> tuple[Frame, MeshMatches, float]:
        result = icp(d, tree, F0=F, options=options.icp)
        for step in result.history:
            history.append(
                DeformableStep("rigid", step.mean_residual, step.rms_residual, step.max_residual,
                               weights.copy(), step.motion)
            )
        motion = float(np.max(np.linalg.norm(result.F.apply(d) - F.apply(d), axis=1)))
        return result.F, result.matches, motion

    F, matches, _ = run_rigid(Frame.identity())
    converged = False

    if options.use_alternation:
        for _ in range(options.max_outer):
            s = F.apply(d)
            mode_motion = 0.0
            for _ in range(options.max_mode_iterations):
                new_weights = solve_mode_weights(s, mode_basis(model, matches))
                change = float(np.max(np.abs(new_weights - weights)))
                weights = new_weights
                new_vertices = model.vertices(weights)
                step_motion = float(np.max(np.linalg.norm(new_vertices - vertices, axis=1)))
                mode_motion = max(mode_motion, step_motion)
                vertices = new_vertices
                tree.update_vertices(vertices)
                matches = tree.closest_points(s, hint=matches.triangles)
                record("mode", matches, step_motion)
                if change < options.weight_tol:
                    break
            F, matches, rigid_motion = run_rigid(F)
            if max(mode_motion, rigid_motion) < options.motion_tol:
                converged = True
                break

    if options.use_combined:
        converged = False
        for _ in range(options.max_combined_iterations):
            s = F.apply(d)
            alpha, epsilon, dweights = solve_combined_step(s, matches.points, mode_basis(model, matches))
            delta = Frame(rotation_from_vector(alpha), epsilon)
            F = delta @ F
            weights = weights + dweights
            new_vertices = model.vertices(weights)
            s_new = F.apply(d)
            motion = max(
                float(np.max(np.linalg.norm(s_new - s, axis=1))),
                float(np.max(np.linalg.norm(new_vertices - vertices, axis=1))),
            )
            vertices = new_vertices
            tree.update_vertices(vertices)
            matches = tree.closest_points(s_new, hint=matches.triangles)
            record("combined", matches, motion)
            if motion < options.motion_tol:
                converged = True
                break

    return DeformableResult(F, weights, F.apply(d), matches, history, converged)


def format_deformable_history(history: list[DeformableStep]) -> str:
    """Per-step log: kind, residuals, size of the step and the current weights."""
    n_modes = len(history[0].weights) if history else 0
    weight_names = "".join(f"{f'lambda_{m + 1}':>11s}" for m in range(n_modes))
    lines = [f"  step  kind       mean_mm    rms_mm    max_mm   motion_mm{weight_names}"]
    for i, h in enumerate(history, start=1):
        weights = "".join(f"{w:11.4f}" for w in h.weights)
        lines.append(
            f"{i:6d}  {h.kind:8s} {h.mean_residual:9.5f} {h.rms_residual:9.5f} "
            f"{h.max_residual:9.5f} {h.motion:11.2e}{weights}"
        )
    return "\n".join(lines)
