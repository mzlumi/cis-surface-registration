"""PA3: closest points on the bone surface with F_reg = I.

For each sample k the pointer tip d_k is found in body B (bone) coordinates,
the sample point is s_k = F_reg d_k with F_reg = I, and c_k is the closest
point to s_k on the CT surface mesh. The output lists d_k, c_k and |d_k - c_k|.

Usage:
    python -m cisreg.pa3 --set A-Debug
    python -m cisreg.pa3 --all

Author: Parmida Mazloomi
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cisreg import datasets
from cisreg.cli import build_parser, selected_labels
from cisreg.datasets import ProblemInputs
from cisreg.fileio import write_output
from cisreg.search import BruteForceSearch, MeshMatches
from cisreg.tracking import pointer_tips


@dataclass(frozen=True)
class Pa3Result:
    d: np.ndarray  # (K, 3) pointer tips in body B coordinates (= s_k, since F_reg = I)
    matches: MeshMatches


def solve_pa3(inputs: ProblemInputs, search=None) -> Pa3Result:
    """Pointer tips d_k and their closest points on the mesh."""
    search = search or BruteForceSearch(inputs.mesh)
    d = pointer_tips(inputs.body_a, inputs.body_b, inputs.samples)
    return Pa3Result(d=d, matches=search.closest_points(d))


def run_set(data_dir: Path, output_dir: Path, label: str) -> Path:
    inputs = datasets.load_inputs(data_dir, "PA3", label)
    result = solve_pa3(inputs)
    path = Path(output_dir) / f"PA3-{label}-Output.txt"
    write_output(path, result.d, result.matches.points)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = build_parser("PA3", "PA3: closest points on the bone surface with F_reg = I.")
    args = parser.parse_args(argv)
    for label in selected_labels(args, "PA3"):
        print(f"wrote {run_set(args.data_dir, args.output_dir, label)}")


if __name__ == "__main__":
    main()
