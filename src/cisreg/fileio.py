"""Readers for the PA3, PA4 and PA5 data files.

The files mix separators: sample readings use commas, while meshes, outputs and
answers use spaces. Every reader splits on commas and whitespace together, so
either style is accepted. Formats are described in the PA3/PA4 and PA5 handouts
(R. H. Taylor, JHU EN.601.455/655, Fall 2025) and in data/README.md.

Author: Parmida Mazloomi
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cisreg.mesh import Mesh

_SEPARATORS = re.compile(r"[,\s]+")


def _tokens(line: str) -> list[str]:
    return [t for t in _SEPARATORS.split(line.strip()) if t]


def _floats(line: str) -> list[float]:
    return [float(t) for t in _tokens(line)]


def _data_lines(path: str | Path) -> list[str]:
    """Return the lines of a file, without trailing blank lines."""
    lines = Path(path).read_text().splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _xyz_block(lines: list[str], start: int, count: int) -> np.ndarray:
    block = np.array([_floats(lines[i])[:3] for i in range(start, start + count)], dtype=float)
    return block.reshape(count, 3)


@dataclass(frozen=True)
class RigidBody:
    """Rigid body definition: LED marker positions and the tip, in body coordinates."""

    markers: np.ndarray  # (N, 3)
    tip: np.ndarray  # (3,)


@dataclass(frozen=True)
class SampleReadings:
    """LED positions in tracker coordinates for each sample frame k.

    a: (K, N_A, 3) markers of body A (the pointer).
    b: (K, N_B, 3) markers of body B (fixed to the bone).
    dummy: (K, N_D, 3) other markers, not used.
    n_modes: number of atlas modes to use (PA5); 0 for PA3 and PA4.
    """

    a: np.ndarray
    b: np.ndarray
    dummy: np.ndarray
    n_modes: int

    @property
    def n_samples(self) -> int:
        return len(self.a)


@dataclass(frozen=True)
class RegistrationOutput:
    """Contents of an Output or Answer file.

    s: (K, 3) sample points (d_k in PA3, s_k = F_reg d_k in PA4 and PA5).
    c: (K, 3) closest points on the (possibly deformed) surface.
    distance: (K,) the magnitudes |s_k - c_k| as written in the file.
    weights: (n_modes,) mode weights for PA5, an empty array otherwise.
    """

    name: str
    s: np.ndarray
    c: np.ndarray
    distance: np.ndarray
    weights: np.ndarray

    @property
    def n_samples(self) -> int:
        return len(self.s)


def read_mesh(path: str | Path) -> Mesh:
    """Read a ``.sur`` surface file: vertex count, vertices, triangle count, triangles."""
    lines = _data_lines(path)
    n_vertices = int(_tokens(lines[0])[0])
    vertices = _xyz_block(lines, 1, n_vertices)
    tri_header = 1 + n_vertices
    n_triangles = int(_tokens(lines[tri_header])[0])
    rows = [_tokens(lines[i]) for i in range(tri_header + 1, tri_header + 1 + n_triangles)]
    table = np.array(rows, dtype=int).reshape(n_triangles, 6)
    return Mesh(vertices=vertices, triangles=table[:, :3].copy(), neighbors=table[:, 3:].copy())


def read_body(path: str | Path) -> RigidBody:
    """Read a rigid body file: marker count and name, marker positions, then the tip."""
    lines = _data_lines(path)
    n_markers = int(_tokens(lines[0])[0])
    markers = _xyz_block(lines, 1, n_markers)
    tip = np.array(_floats(lines[1 + n_markers])[:3], dtype=float)
    return RigidBody(markers=markers, tip=tip)


def read_samples(path: str | Path, n_a: int, n_b: int) -> SampleReadings:
    """Read a SampleReadings file.

    The header is ``N_S, N_samps, filename [N_modes]``. Each sample frame then has
    N_S lines: the N_A markers of body A, the N_B markers of body B, and the
    remaining N_D = N_S - N_A - N_B dummy markers.
    """
    lines = _data_lines(path)
    header = _tokens(lines[0])
    n_leds, n_samples = int(header[0]), int(header[1])
    n_modes = int(header[3]) if len(header) > 3 else 0
    if n_a + n_b > n_leds:
        raise ValueError(f"{path}: {n_a} + {n_b} markers exceed the {n_leds} LEDs per frame")
    expected = 1 + n_leds * n_samples
    if len(lines) < expected:
        raise ValueError(f"{path}: expected {expected} lines, found {len(lines)}")
    frames = _xyz_block(lines, 1, n_leds * n_samples).reshape(n_samples, n_leds, 3)
    return SampleReadings(
        a=frames[:, :n_a].copy(),
        b=frames[:, n_a : n_a + n_b].copy(),
        dummy=frames[:, n_a + n_b :].copy(),
        n_modes=n_modes,
    )


def read_modes(path: str | Path) -> np.ndarray:
    """Read an atlas modes file.

    Returns an array of shape (N_modes + 1, N_vertices, 3): entry 0 is the mean
    shape, entries 1 to N_modes are the per-vertex mode displacements.
    """
    lines = _data_lines(path)
    header = lines[0]
    n_vertices = int(re.search(r"Nvertices\s*=\s*(\d+)", header).group(1))
    n_modes = int(re.search(r"Nmodes\s*=\s*(\d+)", header).group(1))
    modes = np.empty((n_modes + 1, n_vertices, 3))
    line = 1
    for m in range(n_modes + 1):
        if not lines[line].lstrip().startswith("Mode"):
            raise ValueError(f"{path}: expected a 'Mode {m}' line at line {line + 1}")
        modes[m] = _xyz_block(lines, line + 1, n_vertices)
        line += 1 + n_vertices
    return modes


def read_output(path: str | Path) -> RegistrationOutput:
    """Read an Output or Answer file (PA3, PA4 or PA5 format).

    The header is ``N_samps filename N_modes``. If N_modes > 0 (PA5), the next line
    holds the mode weights. Each remaining line is ``s_x s_y s_z c_x c_y c_z |s-c|``.
    """
    lines = _data_lines(path)
    header = _tokens(lines[0])
    n_samples, name = int(header[0]), header[1]
    n_modes = int(header[2]) if len(header) > 2 else 0
    first = 1
    weights = np.zeros(0)
    if n_modes > 0:
        weights = np.array(_floats(lines[1])[:n_modes])
        first = 2
    rows = np.array([_floats(lines[i])[:7] for i in range(first, first + n_samples)])
    rows = rows.reshape(n_samples, 7)
    return RegistrationOutput(
        name=name, s=rows[:, :3], c=rows[:, 3:6], distance=rows[:, 6], weights=weights
    )


@dataclass(frozen=True)
class LogEntry:
    """One ``<set>: summary`` block of an instructor log file (the answer key).

    The compact frame form ``Fr([r],[p]]`` is kept as written (rot_vec, p). When
    the block also has "Matrix forms of frames", R and p are taken from it; each
    ``R*x``, ``R*y``, ``R*z`` line is a column of R. Mode weights are listed in
    the log as "Mode 0" to "Mode N-1", which are atlas modes 1 to N.
    """

    name: str
    noise: float
    rigid_steps: int
    mode_steps: int
    combined_steps: int
    rms_computed: float
    rms_actual: float
    computed_rot_vec: np.ndarray
    computed_p: np.ndarray
    actual_rot_vec: np.ndarray
    actual_p: np.ndarray
    computed_R: np.ndarray | None
    actual_R: np.ndarray | None
    weights_solved: np.ndarray
    weights_actual: np.ndarray


_FR = re.compile(r"Fr\(\[([^\]]*)\],\[([^\]]*)\]\]")
_NUMBER = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"


def _matrix_block(lines: list[str], title: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Find ``title`` alone on a line, followed by P and the three columns of R."""
    for i, line in enumerate(lines):
        if line.strip() == title and i + 4 < len(lines):
            p = np.array(_floats(lines[i + 1].split("=", 1)[1]))
            columns = [_floats(lines[i + j].split("=", 1)[1]) for j in (2, 3, 4)]
            return np.array(columns).T, p
    return None


def _parse_log_block(lines: list[str]) -> LogEntry:
    name = lines[0].split(":")[0].strip()
    text = "\n".join(lines)
    noise = float(re.search(r"noise level\s*=\s*" + _NUMBER, text).group(1))
    steps = re.search(r"rigid steps (\d+)\s+mode steps (\d+)\s+combined steps (\d+)", text)
    rms = re.search(r"Computed\s*=\s*" + _NUMBER + r"\s+Actual\s*=\s*" + _NUMBER, text)
    fr = {}
    for line in lines:
        match = _FR.search(line)
        if match:
            key = line.split("Fr(")[0].strip()
            fr[key] = (np.array(_floats(match.group(1))), np.array(_floats(match.group(2))))
    solved, actual = [], []
    for match in re.finditer(r"Mode\s+\d+:\s+Solved\s+" + _NUMBER + r"\s+Actual\s+" + _NUMBER, text):
        solved.append(float(match.group(1)))
        actual.append(float(match.group(2)))
    computed_matrix = _matrix_block(lines, "Computed Freg")
    actual_matrix = _matrix_block(lines, "Actual Freg")
    computed_rot, computed_p = fr["Computed Freg"]
    actual_rot, actual_p = fr["Actual Freg"]
    if computed_matrix is not None:
        computed_p = computed_matrix[1]
    if actual_matrix is not None:
        actual_p = actual_matrix[1]
    return LogEntry(
        name=name,
        noise=noise,
        rigid_steps=int(steps.group(1)),
        mode_steps=int(steps.group(2)),
        combined_steps=int(steps.group(3)),
        rms_computed=float(rms.group(1)),
        rms_actual=float(rms.group(2)),
        computed_rot_vec=computed_rot,
        computed_p=computed_p,
        actual_rot_vec=actual_rot,
        actual_p=actual_p,
        computed_R=None if computed_matrix is None else computed_matrix[0],
        actual_R=None if actual_matrix is None else actual_matrix[0],
        weights_solved=np.array(solved),
        weights_actual=np.array(actual),
    )


def read_logfile(path: str | Path) -> list[LogEntry]:
    """Read every summary block of an instructor log file, in file order.

    A set name can appear more than once (PA4-B-Demo-Fast does), so a list is
    returned rather than a dictionary.
    """
    lines = _data_lines(path)
    starts = [i for i, line in enumerate(lines) if line.strip().endswith(": summary")]
    ends = starts[1:] + [len(lines)]
    return [_parse_log_block(lines[s:e]) for s, e in zip(starts, ends)]
