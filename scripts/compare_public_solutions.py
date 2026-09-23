"""Score public CIS I solutions for the same Fall 2025 data against the answer key.

Each repository is cloned at a fixed commit into a cache folder (default
/tmp/cis-public-solutions), its committed unknown-set output files are read,
and they are scored exactly like ours: F_reg is recovered by registering our
pointer tips d_k onto the written s_k and compared with the actual F_reg in the
instructor's log file; for PA5 the weights are compared with the actual ones;
and the sum of squared distances (SSE) from F_reg d_k to the surface measures
how well each F_reg fits the data. No code from these repositories is run or
reused; only their output files are read.

Usage:
    python scripts/compare_public_solutions.py [--cache /tmp/cis-public-solutions]

Author: Parmida Mazloomi
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

from cisreg import datasets
from cisreg.answer_key import log_frames
from cisreg.boxtree import BoundingBoxTree
from cisreg.fileio import read_logfile, read_output
from cisreg.frames import frame_difference
from cisreg.registration import register_points
from cisreg.tracking import pointer_tips

REPOS = {
    "mayasharma604/CIS-PA3": "93acba536e9593f9c6c3292c20c41d470504a4c9",
    "mayasharma604/CIS-PA4": "c6c1537e571a8475966f7a05879f0ffeab2792c4",
    "vibhakamath23/CIS-PA3": "f153df48700dfa2a7de5495a2eb67445e248fc8e",
    "pranhav16/CIS_PA4": "4ac772e763ee453f20ea1767df58e6a44813ac45",
    "SIDR73/cis2025": "335eb890ed71ed7c61f5af0a8df5c1f7a8e508f4",
}

# Output file of each solution for a set letter x (lower-case letter as l).
SOURCES = {
    "PA3": {
        "mayasharma604/CIS-PA3": "OUTPUT/PA3-{x}-Unknown-Output.txt",
        "vibhakamath23/CIS-PA3": "output/pa3-{l}-unknown-Output.txt",
        "SIDR73/cis2025": "CIS_PA_3/PA3-{x}-Unknown-myOutput.txt",
    },
    "PA4": {
        "mayasharma604/CIS-PA4": "output/Output-{x}.txt",
        "pranhav16/CIS_PA4": "output/PA4-{x}-Unknown-Output.txt",
        "SIDR73/cis2025": "CIS_PA_4/final_outputs/PA4-{x}-Unknown-Output.txt",
    },
    "PA5": {
        "SIDR73/cis2025": "CIS_PA_5/Outputs/PA5-{x}-Unknown-Output.txt",
    },
}


def fetch(cache: Path) -> dict[str, Path]:
    folders = {}
    for repo, commit in REPOS.items():
        folder = cache / repo.replace("/", "_")
        if not folder.exists():
            subprocess.run(["git", "clone", "-q", f"https://github.com/{repo}", str(folder)], check=True,
                           env={"GIT_LFS_SKIP_SMUDGE": "1", "PATH": "/usr/bin:/bin:/opt/homebrew/bin"})
        subprocess.run(["git", "-C", str(folder), "checkout", "-q", commit], check=True)
        folders[repo] = folder
    return folders


def score(data_dir: Path, output_dir: Path, folders: dict[str, Path]) -> str:
    lines = []
    for assignment, sources in SOURCES.items():
        logs = {e.name: e for e in read_logfile(datasets.logfile_path(data_dir, assignment))}
        header = "| Set | Solution | F_reg error (deg, mm) | SSE (mm^2) | Mean residual (mm) |"
        if assignment == "PA5":
            header += " Max weight error |"
        lines += [f"## {assignment} unknown sets", "", header, "|" + "---|" * (header.count("|") - 1)]
        for label in [l for l in datasets.list_sets(data_dir, assignment) if not datasets.is_debug(l)]:
            x = label[0]
            inputs = datasets.load_inputs(data_dir, assignment, label)
            d = pointer_tips(inputs.body_a, inputs.body_b, inputs.samples)
            tree = BoundingBoxTree(inputs.mesh)
            _, actual = log_frames(logs[f"{assignment}-{label}"])
            entry = logs[f"{assignment}-{label}"]
            candidates = {"ours": Path(output_dir) / f"{assignment}-{label}-Output.txt"}
            candidates.update({repo: folders[repo] / path.format(x=x, l=x.lower()) for repo, path in sources.items()})
            for name, path in candidates.items():
                result = read_output(path)
                F = register_points(d, result.s)
                angle, distance = frame_difference(F, actual)
                if assignment == "PA5":
                    from cisreg.fileio import read_modes
                    from cisreg.shape_model import ShapeModel

                    model = ShapeModel.from_atlas(read_modes(datasets.modes_path(data_dir)), inputs.mesh, 6)
                    surface = BoundingBoxTree(model.mesh(result.weights))
                else:
                    surface = tree
                sse = float(np.sum(surface.closest_points(F.apply(d)).distances ** 2))
                row = (
                    f"| {label} | {name} | {np.degrees(angle):.4f}, {distance:.4f} | {sse:.4f} "
                    f"| {np.mean(result.distance):.4f} |"
                )
                if assignment == "PA5":
                    row += f" {np.abs(result.weights - entry.weights_actual).max():.3f} |"
                lines.append(row)
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, default=Path("/tmp/cis-public-solutions"))
    parser.add_argument("--data-dir", type=Path, default=datasets.DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--out", type=Path, default=Path("results/public_solutions_tables.md"))
    args = parser.parse_args()
    folders = fetch(args.cache)
    text = "\n".join(
        [
            "# Public solutions scored against the answer key",
            "",
            "Generated by `python scripts/compare_public_solutions.py`. Discussion and repository",
            "details: [comparison_with_public_solutions.md](comparison_with_public_solutions.md).",
            "F_reg error is the rotation angle and translation of F^-1 F_actual. For PA3, F_reg = I",
            "for every solution, so the F_reg and SSE columns only confirm that.",
            "",
            score(args.data_dir, args.output_dir, folders),
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
