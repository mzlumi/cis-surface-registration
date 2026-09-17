"""Command-line options shared by the PA3, PA4 and PA5 programs.

Author: Parmida Mazloomi
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cisreg import datasets


def build_parser(assignment: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"python -m cisreg.{assignment.lower()}", description=description)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--set", dest="labels", action="append", help="set label, e.g. A-Debug (repeatable)")
    which.add_argument("--all", action="store_true", help="run every set in the data directory")
    parser.add_argument("--data-dir", type=Path, default=datasets.DEFAULT_DATA_DIR, help="input data folder")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="where output files go")
    return parser


def selected_labels(args: argparse.Namespace, assignment: str) -> list[str]:
    return datasets.list_sets(args.data_dir, assignment) if args.all else args.labels
