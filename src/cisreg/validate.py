"""Compare program outputs with the instructor's debug Output and Answer files.

For each debug set, with our points s_k, c_k and the reference points s'_k, c'_k:
    e_s,k = |s_k - s'_k|,   e_c,k = |c_k - c'_k|,
    e_d,k = | |s_k - c_k| - |s'_k - c'_k| |,
and the table reports max_k and RMS_k = sqrt(mean_k e_k^2) of each (e_dist is
the difference of the distance magnitudes). For PA5 it also reports
max_m |lambda_m - lambda'_m| over the mode weights.

Usage:
    python -m cisreg.validate PA3

Author: Parmida Mazloomi
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cisreg import datasets
from cisreg.fileio import RegistrationOutput, read_output


@dataclass(frozen=True)
class SetComparison:
    label: str
    n_samples: int
    s_max: float
    s_rms: float
    c_max: float
    c_rms: float
    distance_max: float
    weights_max: float | None


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


def compare(label: str, ours: RegistrationOutput, reference: RegistrationOutput) -> SetComparison:
    if ours.n_samples != reference.n_samples:
        raise ValueError(f"{label}: {ours.n_samples} samples, reference has {reference.n_samples}")
    e_s = np.linalg.norm(ours.s - reference.s, axis=1)
    e_c = np.linalg.norm(ours.c - reference.c, axis=1)
    e_d = np.abs(ours.distance - reference.distance)
    weights = None
    if len(reference.weights):
        weights = float(np.max(np.abs(ours.weights - reference.weights)))
    return SetComparison(
        label, ours.n_samples, float(e_s.max()), _rms(e_s), float(e_c.max()), _rms(e_c),
        float(e_d.max()), weights,
    )


def compare_sets(
    assignment: str, data_dir: Path, output_dir: Path, kind: str = "Output"
) -> list[SetComparison]:
    """Compare every debug set that has an output in output_dir with the reference files."""
    reference_path = datasets.output_path if kind == "Output" else datasets.answer_path
    rows = []
    for label in datasets.list_sets(data_dir, assignment):
        ours_path = Path(output_dir) / f"{assignment}-{label}-Output.txt"
        if not datasets.is_debug(label) or not ours_path.exists():
            continue
        ours = read_output(ours_path)
        rows.append(compare(label, ours, read_output(reference_path(data_dir, assignment, label))))
    return rows


def matching_check(assignment: str, data_dir: Path, search_factory) -> str:
    """Closest-point step alone: match the reference s'_k and compare with the reference c'_k.

    This removes the pose and registration steps, so any difference comes from
    the closest-point search or from the rounding of the reference files. It also
    reports how far the reference c'_k are from the mesh that was read.
    """
    from cisreg.fileio import read_mesh

    search = search_factory(read_mesh(datasets.mesh_path(data_dir, assignment)))
    lines = [
        "| Set | max dist(c'_k, mesh) | max e_c | RMS e_c |",
        "|---|---|---|---|",
    ]
    for label in datasets.list_sets(data_dir, assignment):
        if not datasets.is_debug(label):
            continue
        ref = read_output(datasets.output_path(data_dir, assignment, label))
        on_mesh = search.closest_points(ref.c).distances
        e_c = np.linalg.norm(search.closest_points(ref.s).points - ref.c, axis=1)
        lines.append(f"| {label} | {on_mesh.max():.4f} | {e_c.max():.4f} | {_rms(e_c):.4f} |")
    return "\n".join(lines)


def frame_check(assignment: str, data_dir: Path, output_dir: Path, search_factory) -> str:
    """Compare our F_reg with the instructor's F_reg implied by the reference s'_k.

    The output files do not contain F_reg, but s_k = F_reg d_k with d_k known
    (our pointer tips), so F_reg can be recovered by registering d_k onto s_k.
    Doing this for our files and for the reference files gives the two frames.
    Both are then scored by the ICP objective, the sum over k of the squared
    distance from F d_k to the surface (for PA5, to the mean shape, so only the
    frame difference is meaningful there).
    """
    from cisreg.frames import frame_difference
    from cisreg.registration import register_points
    from cisreg.tracking import pointer_tips

    lines = [
        "| Set | angle(F^-1 F') (deg) | abs(p - p') (mm) | SSE with our F | SSE with reference F' |",
        "|---|---|---|---|---|",
    ]
    for label in datasets.list_sets(data_dir, assignment):
        ours_path = Path(output_dir) / f"{assignment}-{label}-Output.txt"
        if not datasets.is_debug(label) or not ours_path.exists():
            continue
        inputs = datasets.load_inputs(data_dir, assignment, label)
        search = search_factory(inputs.mesh)
        d = pointer_tips(inputs.body_a, inputs.body_b, inputs.samples)
        F_ours = register_points(d, read_output(ours_path).s)
        F_ref = register_points(d, read_output(datasets.output_path(data_dir, assignment, label)).s)
        angle, distance = frame_difference(F_ours, F_ref)
        sse_ours = float(np.sum(search.closest_points(F_ours.apply(d)).distances ** 2))
        sse_ref = float(np.sum(search.closest_points(F_ref.apply(d)).distances ** 2))
        lines.append(
            f"| {label} | {np.degrees(angle):.4f} | {distance:.4f} | {sse_ours:.4f} | {sse_ref:.4f} |"
        )
    return "\n".join(lines)


def markdown_table(rows: list[SetComparison], point: str = "s") -> str:
    with_weights = any(r.weights_max is not None for r in rows)
    header = (
        f"| Set | N | max e_{point} | RMS e_{point} | max e_c | RMS e_c | max e_dist |"
        + (" max weight diff |" if with_weights else "")
    )
    lines = [header, "|" + "---|" * (header.count("|") - 1)]
    for r in rows:
        line = (
            f"| {r.label} | {r.n_samples} | {r.s_max:.4f} | {r.s_rms:.4f} | {r.c_max:.4f} "
            f"| {r.c_rms:.4f} | {r.distance_max:.4f} |"
        )
        if with_weights:
            line += f" {r.weights_max:.4f} |"
        lines.append(line)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print validation tables for one assignment.")
    parser.add_argument("assignment", choices=["PA3", "PA4", "PA5"])
    parser.add_argument("--data-dir", type=Path, default=datasets.DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args(argv)
    point = "d" if args.assignment == "PA3" else "s"
    for kind in ("Output", "Answer"):
        print(f"Against the {kind} files:\n")
        rows = compare_sets(args.assignment, args.data_dir, args.output_dir, kind)
        print(markdown_table(rows, point) + "\n")
    if args.assignment == "PA3":
        from cisreg.search import BruteForceSearch

        print("Closest-point step alone, on the reference d'_k:\n")
        print(matching_check("PA3", args.data_dir, BruteForceSearch) + "\n")
    else:
        from cisreg.boxtree import BoundingBoxTree

        print("Our F_reg against the reference F_reg implied by s'_k:\n")
        print(frame_check(args.assignment, args.data_dir, args.output_dir, BoundingBoxTree) + "\n")


if __name__ == "__main__":
    main()
