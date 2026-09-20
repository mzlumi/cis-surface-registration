"""Compare the committed outputs with the instructor's log files (the answer key).

The log files give, for every set including the unknown ones, the instructor's
computed F_reg, the actual F_reg used to generate the data, RMS residuals,
iteration counts and (PA5) the solved and actual mode weights. This module is
only run after the outputs are final; it reads our output files and does not
change them.

Our F_reg is recovered from each output file by registering our pointer tips
d_k onto the written s_k (for PA3, F_reg = I by definition). In the logs the
compact frame form Fr([r],[p]] has r = rotation vector (axis times angle, in
radians); this was checked against the matrix blocks of the debug entries. The
matrix block is used when present, since it has more digits.

Usage:
    python -m cisreg.answer_key [--out results/answer_key_comparison.md]

Author: Parmida Mazloomi
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cisreg import datasets
from cisreg.fileio import LogEntry, read_logfile, read_output
from cisreg.frames import Frame, frame_difference, rotation_from_vector
from cisreg.registration import register_points
from cisreg.tracking import pointer_tips


def log_frames(entry: LogEntry) -> tuple[Frame, Frame]:
    """(computed, actual) F_reg of a log entry."""
    computed_R = entry.computed_R if entry.computed_R is not None else rotation_from_vector(entry.computed_rot_vec)
    actual_R = entry.actual_R if entry.actual_R is not None else rotation_from_vector(entry.actual_rot_vec)
    return Frame(computed_R, entry.computed_p), Frame(actual_R, entry.actual_p)


def our_frame(data_dir: Path, output_dir: Path, assignment: str, label: str) -> Frame:
    if assignment == "PA3":
        return Frame.identity()
    inputs = datasets.load_inputs(data_dir, assignment, label)
    d = pointer_tips(inputs.body_a, inputs.body_b, inputs.samples)
    return register_points(d, read_output(Path(output_dir) / f"{assignment}-{label}-Output.txt").s)


def _entries(data_dir: Path, assignment: str):
    """Log entries with their set label; repeated names get an entry number."""
    entries = read_logfile(datasets.logfile_path(data_dir, assignment))
    names = [e.name for e in entries]
    for i, entry in enumerate(entries):
        label = entry.name.split("-", 1)[1]
        repeat = names.count(entry.name) > 1
        shown = f"{label} (log entry {names[: i + 1].count(entry.name)})" if repeat else label
        yield label, shown, entry


def frame_table(data_dir: Path, output_dir: Path, assignment: str) -> str:
    lines = [
        "| Set | Noise (mm) | Their steps (rigid/mode/combined) | Our mean | Our RMS | Log \"RMS\" | "
        "Log \"RMS\" with actual F | Ours vs actual (deg, mm) | Theirs vs actual (deg, mm) | Ours vs theirs (deg, mm) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, shown, entry in _entries(data_dir, assignment):
        ours = our_frame(data_dir, output_dir, assignment, label)
        theirs, actual = log_frames(entry)
        output = read_output(Path(output_dir) / f"{assignment}-{label}-Output.txt")
        our_mean = float(np.mean(output.distance))
        our_rms = float(np.sqrt(np.mean(output.distance**2)))
        cells = []
        for a, b in ((ours, actual), (theirs, actual), (ours, theirs)):
            angle, distance = frame_difference(a, b)
            cells.append(f"{np.degrees(angle):.4f}, {distance:.4f}")
        lines.append(
            f"| {shown} | {entry.noise:.2f} | {entry.rigid_steps}/{entry.mode_steps}/{entry.combined_steps} "
            f"| {our_mean:.4f} | {our_rms:.4f} | {entry.rms_computed:.4f} | {entry.rms_actual:.4f} | "
            + " | ".join(cells) + " |"
        )
    return "\n".join(lines)


def weights_table(data_dir: Path, output_dir: Path) -> str:
    lines = [
        "| Set | max abs(ours - actual) | max abs(theirs - actual) | max abs(ours - theirs) | RMS(ours - actual) |",
        "|---|---|---|---|---|",
    ]
    for label, shown, entry in _entries(data_dir, "PA5"):
        ours = read_output(Path(output_dir) / f"PA5-{label}-Output.txt").weights
        actual, theirs = entry.weights_actual, entry.weights_solved
        lines.append(
            f"| {shown} | {np.abs(ours - actual).max():.4f} | {np.abs(theirs - actual).max():.4f} "
            f"| {np.abs(ours - theirs).max():.4f} | {np.sqrt(np.mean((ours - actual) ** 2)):.4f} |"
        )
    return "\n".join(lines)


def unknown_weights_listing(data_dir: Path, output_dir: Path) -> str:
    lines = ["| Set | Source | " + " | ".join(f"lambda_{m}" for m in range(1, 7)) + " |", "|---|---|" + "---|" * 6]
    for label, shown, entry in _entries(data_dir, "PA5"):
        if datasets.is_debug(label):
            continue
        ours = read_output(Path(output_dir) / f"PA5-{label}-Output.txt").weights
        for source, w in (("ours", ours), ("theirs", entry.weights_solved), ("actual", entry.weights_actual)):
            lines.append(f"| {shown} | {source} | " + " | ".join(f"{x:.4f}" for x in w) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare committed outputs with the instructor's log files.")
    parser.add_argument("--data-dir", type=Path, default=datasets.DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--out", type=Path, default=Path("results/answer_key_tables.md"))
    args = parser.parse_args(argv)
    parts = [
        "# Answer-key tables",
        "",
        "Generated by `python -m cisreg.answer_key` from the committed files in `output/` and the",
        "instructor's log files. Frame differences are the rotation angle (degrees) and translation",
        "(mm) of F1^-1 F2. \"Log RMS\" is the residual the log calls RMS; it equals the mean of",
        "|s_k - c_k| (see answer_key_comparison.md). Discussion: [answer_key_comparison.md](answer_key_comparison.md).",
        "",
    ]
    for assignment in ("PA3", "PA4", "PA5"):
        parts += [f"## {assignment} frames and residuals", "", frame_table(args.data_dir, args.output_dir, assignment), ""]
    parts += ["## PA5 mode weights", "", weights_table(args.data_dir, args.output_dir), ""]
    parts += ["## PA5 unknown-set weights", "", unknown_weights_listing(args.data_dir, args.output_dir), ""]
    text = "\n".join(parts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
