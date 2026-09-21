"""Error analysis experiments and figures.

1. ICP convergence basin: start rigid ICP from a misaligned F_reg and record
   how often it reaches the right minimum, as a function of the initial
   rotation (about the centroid of the samples) and translation.
2. Marker noise: add Gaussian noise to every LED reading of a debug set and
   measure the resulting error in F_reg (PA4) and in the mode weights (PA5).
3. Number of modes: run PA5 with the first M atlas modes, M = 0 ... 6, and
   record the final residual.
4. Search timing: brute force against the bounding-box tree for growing
   numbers of query points.

Every experiment uses a fixed random seed. The reference answers are our own
noise-free solutions on the debug sets, so the experiments do not use the
answer key.

Usage:
    python -m cisreg.analysis            # all experiments
    python -m cisreg.analysis --quick    # fewer trials, for a fast check

Author: Parmida Mazloomi
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cisreg import datasets  # noqa: E402
from cisreg.boxtree import BoundingBoxTree  # noqa: E402
from cisreg.deformable import deformable_registration  # noqa: E402
from cisreg.fileio import SampleReadings, read_modes  # noqa: E402
from cisreg.frames import Frame, frame_difference, rotation_from_vector  # noqa: E402
from cisreg.icp import IcpOptions, icp  # noqa: E402
from cisreg.search import BruteForceSearch  # noqa: E402
from cisreg.shape_model import ShapeModel  # noqa: E402
from cisreg.tracking import pointer_tips  # noqa: E402

SUCCESS_ANGLE_DEG = 0.1
SUCCESS_DISTANCE_MM = 0.1


def misaligned(F: Frame, center: np.ndarray, angle: float, shift: float, rng) -> Frame:
    """F followed by a rotation of `angle` (radians, random axis) about `center` and a shift of `shift` mm."""
    axis, direction = rng.normal(size=(2, 3))
    R = rotation_from_vector(axis / np.linalg.norm(axis) * angle)
    direction /= np.linalg.norm(direction)
    about_center = Frame(R, center - R @ center + shift * direction)
    return about_center @ F


def convergence_basin(data_dir: Path, angles_deg, shifts_mm, trials: int, seed: int = 0):
    """Success rate of ICP started from misaligned frames, on PA4-B-Debug."""
    inputs = datasets.load_inputs(data_dir, "PA4", "B-Debug")
    tree = BoundingBoxTree(inputs.mesh)
    d = pointer_tips(inputs.body_a, inputs.body_b, inputs.samples)
    reference = icp(d, tree).F
    center = reference.apply(d).mean(axis=0)
    rng = np.random.default_rng(seed)
    options = IcpOptions(max_iterations=300)
    rates = np.zeros((len(shifts_mm), len(angles_deg)))
    iterations = np.zeros_like(rates)
    for i, shift in enumerate(shifts_mm):
        for j, angle in enumerate(angles_deg):
            ok, its = 0, []
            for _ in range(trials):
                F0 = misaligned(reference, center, np.radians(angle), shift, rng)
                result = icp(d, tree, F0=F0, options=options)
                rot, dist = frame_difference(reference, result.F)
                success = np.degrees(rot) < SUCCESS_ANGLE_DEG and dist < SUCCESS_DISTANCE_MM
                ok += success
                if success:
                    its.append(result.iterations)
            rates[i, j] = ok / trials
            iterations[i, j] = np.mean(its) if its else np.nan
    return rates, iterations


def _noisy(samples: SampleReadings, sigma: float, rng) -> SampleReadings:
    return replace(
        samples,
        a=samples.a + rng.normal(scale=sigma, size=samples.a.shape),
        b=samples.b + rng.normal(scale=sigma, size=samples.b.shape),
    )


def marker_noise_pa4(data_dir: Path, sigmas, trials: int, seed: int = 1):
    """Mean F_reg rotation (deg) and translation (mm) error and final RMS residual against marker noise."""
    inputs = datasets.load_inputs(data_dir, "PA4", "B-Debug")
    tree = BoundingBoxTree(inputs.mesh)
    reference = icp(pointer_tips(inputs.body_a, inputs.body_b, inputs.samples), tree).F
    rng = np.random.default_rng(seed)
    rows = []
    for sigma in sigmas:
        rot, dist, rms = [], [], []
        for _ in range(trials):
            d = pointer_tips(inputs.body_a, inputs.body_b, _noisy(inputs.samples, sigma, rng))
            result = icp(d, tree)
            a, t = frame_difference(reference, result.F)
            rot.append(np.degrees(a))
            dist.append(t)
            rms.append(result.history[-1].rms_residual)
        rows.append((sigma, np.mean(rot), np.mean(dist), np.mean(rms)))
    return np.array(rows)


def marker_noise_pa5(data_dir: Path, sigmas, trials: int, seed: int = 2):
    """RMS mode-weight error and mean F_reg rotation error against marker noise, on PA5-A-Debug."""
    inputs = datasets.load_inputs(data_dir, "PA5", "A-Debug")
    model = ShapeModel.from_atlas(read_modes(datasets.modes_path(data_dir)), inputs.mesh, inputs.samples.n_modes)
    reference = deformable_registration(pointer_tips(inputs.body_a, inputs.body_b, inputs.samples), model)
    rng = np.random.default_rng(seed)
    rows = []
    for sigma in sigmas:
        weight_rms, rot = [], []
        for _ in range(trials):
            d = pointer_tips(inputs.body_a, inputs.body_b, _noisy(inputs.samples, sigma, rng))
            result = deformable_registration(d, model)
            weight_rms.append(np.sqrt(np.mean((result.weights - reference.weights) ** 2)))
            rot.append(np.degrees(frame_difference(reference.F, result.F)[0]))
        rows.append((sigma, np.mean(weight_rms), np.mean(rot)))
    return np.array(rows)


def residual_vs_modes(data_dir: Path, labels):
    """Final RMS residual of PA5 using the first M modes, for M = 0 ... 6."""
    atlas = read_modes(datasets.modes_path(data_dir))
    table = {}
    for label in labels:
        inputs = datasets.load_inputs(data_dir, "PA5", label)
        d = pointer_tips(inputs.body_a, inputs.body_b, inputs.samples)
        residuals = []
        for m in range(0, len(atlas)):
            if m == 0:
                residuals.append(icp(d, BoundingBoxTree(inputs.mesh)).history[-1].rms_residual)
            else:
                model = ShapeModel.from_atlas(atlas, inputs.mesh, m)
                residuals.append(deformable_registration(d, model).history[-1].rms_residual)
        table[label] = residuals
    return table


def search_timing(data_dir: Path, sizes, seed: int = 3, repeats: int = 5):
    """Best-of-repeats time (s) for brute force and the tree, for random points near the bone."""
    mesh = datasets.load_inputs(data_dir, "PA4", "A-Debug").mesh
    brute, tree = BruteForceSearch(mesh), BoundingBoxTree(mesh)
    rng = np.random.default_rng(seed)
    lo, hi = mesh.vertices.min(axis=0) - 5, mesh.vertices.max(axis=0) + 5
    rows = []
    for n in sizes:
        queries = rng.uniform(lo, hi, size=(n, 3))
        times = []
        for search in (brute, tree):
            best = np.inf
            for _ in range(repeats):
                start = time.perf_counter()
                search.closest_points(queries)
                best = min(best, time.perf_counter() - start)
            times.append(best)
        rows.append((n, times[0], times[1]))
    return np.array(rows)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_basin(angles, shifts, rates, iterations, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for i, shift in enumerate(shifts):
        ax1.plot(angles, 100 * rates[i], "o-", label=f"shift {shift:g} mm")
        ax2.plot(angles, iterations[i], "o-", label=f"shift {shift:g} mm")
    ax1.set_xlabel("initial rotation error (deg)")
    ax1.set_ylabel("runs reaching the right minimum (%)")
    ax1.set_ylim(-5, 105)
    ax1.set_title("ICP convergence basin (PA4-B-Debug)")
    ax2.set_xlabel("initial rotation error (deg)")
    ax2.set_ylabel("iterations (successful runs)")
    ax2.set_title("Iterations to converge")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    _save(fig, path)


def plot_noise(pa4_rows, pa5_rows, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(pa4_rows[:, 0], pa4_rows[:, 3], "o-", label="final RMS residual (mm)")
    ax1.plot(pa4_rows[:, 0], pa4_rows[:, 2], "s-", label="F_reg translation error (mm)")
    ax1.plot(pa4_rows[:, 0], pa4_rows[:, 1], "^-", label="F_reg rotation error (deg)")
    ax1.set_xlabel("marker noise sigma (mm, per coordinate)")
    ax1.set_title("PA4 rigid ICP (B-Debug)")
    ax2.plot(pa5_rows[:, 0], pa5_rows[:, 1], "o-", color="C3", label="RMS mode-weight error")
    ax2.plot(pa5_rows[:, 0], pa5_rows[:, 2], "^-", color="C2", label="F_reg rotation error (deg)")
    ax2.set_xlabel("marker noise sigma (mm, per coordinate)")
    ax2.set_title("PA5 deformable registration (A-Debug)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    _save(fig, path)


def plot_modes(table, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for label, residuals in table.items():
        style = "--" if label in ("E-Debug", "F-Debug") else "-"
        ax.semilogy(range(len(residuals)), residuals, "o" + style, label=label)
    ax.set_xlabel("number of atlas modes used")
    ax.set_ylabel("final RMS of |s_k - c_k| (mm)")
    ax.set_title("PA5: residual against the number of modes")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_timing(rows, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.loglog(rows[:, 0], 1e3 * rows[:, 1], "o-", label="brute force (3135 triangles per point)")
    ax.loglog(rows[:, 0], 1e3 * rows[:, 2], "s-", label="bounding-box tree")
    ax.set_xlabel("number of query points")
    ax.set_ylabel("time (ms, best of 5)")
    ax.set_title("Closest-point search on the bone mesh")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the error-analysis experiments.")
    parser.add_argument("--data-dir", type=Path, default=datasets.DEFAULT_DATA_DIR)
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--out", type=Path, default=Path("results/error_analysis_data.md"))
    parser.add_argument("--quick", action="store_true", help="fewer trials")
    args = parser.parse_args(argv)
    basin_trials, pa4_trials, pa5_trials = (4, 3, 1) if args.quick else (12, 10, 4)

    angles = [0, 15, 30, 45, 60, 75, 90, 120, 150, 180]
    shifts = [0, 15, 30]
    rates, iterations = convergence_basin(args.data_dir, angles, shifts, basin_trials)
    plot_basin(angles, shifts, rates, iterations, args.figures / "icp_convergence_basin.png")

    sigmas = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    pa4_rows = marker_noise_pa4(args.data_dir, sigmas, pa4_trials)
    pa5_rows = marker_noise_pa5(args.data_dir, sigmas, pa5_trials)
    plot_noise(pa4_rows, pa5_rows, args.figures / "marker_noise.png")

    modes = residual_vs_modes(args.data_dir, ["A-Debug", "B-Debug", "C-Debug", "D-Debug", "E-Debug", "F-Debug"])
    plot_modes(modes, args.figures / "residual_vs_modes.png")

    timing = search_timing(args.data_dir, [1, 3, 10, 30, 100, 300, 1000])
    plot_timing(timing, args.figures / "search_timing.png")

    lines = [
        "# Error analysis data",
        "",
        "Generated by `python -m cisreg.analysis`" + (" --quick" if args.quick else "") + ".",
        "Discussion: [error_analysis.md](error_analysis.md).",
        "",
        f"## ICP convergence basin (PA4-B-Debug, {basin_trials} trials per cell)",
        "",
        "Percentage of runs ending within 0.1 deg and 0.1 mm of the reference F_reg.",
        "",
        "| Shift (mm) | " + " | ".join(f"{a} deg" for a in angles) + " |",
        "|---|" + "---|" * len(angles),
    ]
    for shift, row in zip(shifts, rates):
        lines.append(f"| {shift} | " + " | ".join(f"{100 * r:.0f}" for r in row) + " |")
    lines += [
        "",
        f"## Marker noise, PA4-B-Debug ({pa4_trials} trials per level)",
        "",
        "| sigma (mm) | rotation error (deg) | translation error (mm) | final RMS residual (mm) |",
        "|---|---|---|---|",
    ]
    lines += [f"| {r[0]:g} | {r[1]:.4f} | {r[2]:.4f} | {r[3]:.4f} |" for r in pa4_rows]
    lines += [
        "",
        f"## Marker noise, PA5-A-Debug ({pa5_trials} trials per level)",
        "",
        "| sigma (mm) | RMS weight error | rotation error (deg) |",
        "|---|---|---|",
    ]
    lines += [f"| {r[0]:g} | {r[1]:.4f} | {r[2]:.4f} |" for r in pa5_rows]
    lines += [
        "",
        "## Final RMS residual (mm) against the number of modes",
        "",
        "| Set | " + " | ".join(f"M = {m}" for m in range(7)) + " |",
        "|---|" + "---|" * 7,
    ]
    lines += [f"| {label} | " + " | ".join(f"{v:.4f}" for v in res) + " |" for label, res in modes.items()]
    lines += [
        "",
        "## Search time (ms, best of 5)",
        "",
        "| N | brute force | tree | speedup |",
        "|---|---|---|---|",
    ]
    lines += [f"| {int(r[0])} | {1e3 * r[1]:.1f} | {1e3 * r[2]:.1f} | {r[1] / r[2]:.1f} |" for r in timing]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
