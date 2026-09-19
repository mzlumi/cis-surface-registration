"""Figures for the validation and the report (Matplotlib, Hunter, CiSE 9(3), 2007).

Author: Parmida Mazloomi
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cisreg.icp import IcpIteration  # noqa: E402


def plot_icp_convergence(histories: dict[str, list[IcpIteration]], path: Path, title: str) -> None:
    """RMS residual and size of the update (max point motion) against the iteration number."""
    fig, (ax_rms, ax_motion) = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = plt.get_cmap("tab20")
    for i, (label, history) in enumerate(histories.items()):
        n = [h.n for h in history]
        style = "--" if "Unknown" in label else "-"
        color = cmap(i % 20)
        ax_rms.semilogy(n, [h.rms_residual for h in history], style, color=color, label=label)
        ax_motion.semilogy(n, [max(h.motion, 1e-9) for h in history], style, color=color)
    ax_rms.set_xlabel("iteration")
    ax_rms.set_ylabel("RMS of |s_k - c_k| (mm)")
    ax_rms.set_title("Residual")
    ax_motion.set_xlabel("iteration")
    ax_motion.set_ylabel("max_k |F_n d_k - F_(n-1) d_k| (mm)")
    ax_motion.set_title("Size of the F_reg update")
    for ax in (ax_rms, ax_motion):
        ax.grid(True, which="both", alpha=0.3)
    fig.legend(loc="center right", fontsize=8, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_deformable_convergence(histories: dict, path: Path) -> None:
    """RMS residual against the step number for PA5, with the phase of each step marked.

    histories maps a set label to a list of cisreg.deformable.DeformableStep.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    cmap = plt.get_cmap("tab10")
    markers = {"rigid": ".", "mode": "^", "combined": "s"}
    for i, (label, history) in enumerate(histories.items()):
        ax = axes[1] if "Unknown" in label else axes[0]
        rms = np.array([h.rms_residual for h in history])
        steps = np.arange(1, len(history) + 1)
        color = cmap(i % 10)
        ax.semilogy(steps, rms, "-", color=color, lw=1, label=label)
        for kind, marker in markers.items():
            mask = np.array([h.kind == kind for h in history])
            ax.semilogy(steps[mask], rms[mask], marker, color=color, ms=3, ls="none")
    for ax, title in zip(axes, ("Debug sets", "Unknown sets")):
        ax.set_xlabel("step (rigid ICP iterations, mode steps and combined steps in order)")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("RMS of |s_k - c_k| (mm)")
    handles = [plt.Line2D([], [], marker=m, color="gray", ls="none", label=f"{k} step") for k, m in markers.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("PA5: deformable registration")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def residual_histogram(residuals: dict[str, np.ndarray], path: Path, title: str) -> None:
    """Histograms of the final residuals |s_k - c_k| for several sets."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for label, values in residuals.items():
        ax.hist(values, bins=30, histtype="step", label=label)
    ax.set_xlabel("|s_k - c_k| (mm)")
    ax.set_ylabel("samples")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
