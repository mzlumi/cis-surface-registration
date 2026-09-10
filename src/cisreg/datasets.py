"""File names of the course data sets.

A set is named by its assignment ("PA3", "PA4" or "PA5") and its label as it
appears in the file names, for example "A-Debug", "A-Demo-Fast" or "G-Unknown".

Author: Parmida Mazloomi
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cisreg.fileio import RigidBody, SampleReadings, read_body, read_mesh, read_samples
from cisreg.mesh import Mesh

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROBLEM_NUMBER = {"PA3": 3, "PA4": 4, "PA5": 5}
_SAMPLE_SUFFIX = "-SampleReadingsTest.txt"


def _problem(assignment: str) -> int:
    try:
        return PROBLEM_NUMBER[assignment.upper()]
    except KeyError:
        raise ValueError(f"unknown assignment {assignment!r}, expected PA3, PA4 or PA5") from None


def sample_path(data_dir: Path, assignment: str, label: str) -> Path:
    return Path(data_dir) / f"{assignment.upper()}-{label}{_SAMPLE_SUFFIX}"


def output_path(data_dir: Path, assignment: str, label: str) -> Path:
    return Path(data_dir) / f"{assignment.upper()}-{label}-Output.txt"


def answer_path(data_dir: Path, assignment: str, label: str) -> Path:
    return Path(data_dir) / f"{assignment.upper()}-{label}-Answer.txt"


def mesh_path(data_dir: Path, assignment: str) -> Path:
    return Path(data_dir) / f"Problem{_problem(assignment)}MeshFile.sur"


def body_path(data_dir: Path, assignment: str, body: str) -> Path:
    return Path(data_dir) / f"Problem{_problem(assignment)}-Body{body.upper()}.txt"


def modes_path(data_dir: Path) -> Path:
    return Path(data_dir) / "Problem5Modes.txt"


def logfile_path(data_dir: Path, assignment: str) -> Path:
    return Path(data_dir) / f"{assignment.upper()}-Logfile.txt"


def list_sets(data_dir: Path, assignment: str) -> list[str]:
    """Labels of all sample sets of an assignment, debug sets first, sorted by letter."""
    prefix = f"{assignment.upper()}-"
    labels = [
        p.name[len(prefix) : -len(_SAMPLE_SUFFIX)]
        for p in Path(data_dir).glob(f"{prefix}*{_SAMPLE_SUFFIX}")
    ]
    return sorted(labels, key=lambda label: ("Unknown" in label, label))


def is_debug(label: str) -> bool:
    """Debug and demo sets come with instructor outputs; unknown sets do not."""
    return "Unknown" not in label


@dataclass(frozen=True)
class ProblemInputs:
    """Everything needed to run one sample set."""

    mesh: Mesh
    body_a: RigidBody
    body_b: RigidBody
    samples: SampleReadings


def load_inputs(data_dir: Path, assignment: str, label: str) -> ProblemInputs:
    body_a = read_body(body_path(data_dir, assignment, "A"))
    body_b = read_body(body_path(data_dir, assignment, "B"))
    samples = read_samples(
        sample_path(data_dir, assignment, label), len(body_a.markers), len(body_b.markers)
    )
    return ProblemInputs(
        mesh=read_mesh(mesh_path(data_dir, assignment)),
        body_a=body_a,
        body_b=body_b,
        samples=samples,
    )
