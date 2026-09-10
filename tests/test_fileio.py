"""Every file in data/ is read, and the counts match the headers and the handouts."""

import re

import numpy as np
import pytest

from cisreg import datasets
from cisreg.fileio import (
    _tokens,
    read_body,
    read_logfile,
    read_mesh,
    read_modes,
    read_output,
    read_samples,
)
from conftest import DATA_DIR

N_VERTICES = 1568
N_TRIANGLES = 3135
N_LEDS = 16
N_MARKERS = 6

ALL_FILES = sorted(p.name for p in DATA_DIR.iterdir() if p.name != "README.md")
SURFACES = [f for f in ALL_FILES if f.endswith(".sur")]
BODIES = [f for f in ALL_FILES if re.fullmatch(r"Problem\d-Body[AB]\.txt", f)]
SAMPLES = [f for f in ALL_FILES if f.endswith("-SampleReadingsTest.txt")]
RESULTS = [f for f in ALL_FILES if f.endswith(("-Output.txt", "-Answer.txt"))]
LOGS = [f for f in ALL_FILES if f.endswith("-Logfile.txt")]
MODES = ["Problem5Modes.txt"]


def test_every_data_file_has_a_reader():
    covered = set(SURFACES + BODIES + SAMPLES + RESULTS + LOGS + MODES)
    assert covered == set(ALL_FILES)


def test_separators_are_commas_or_whitespace():
    assert _tokens("16, 150, PA5-A-Debug-SampleReadingsTest.txt 6") == [
        "16",
        "150",
        "PA5-A-Debug-SampleReadingsTest.txt",
        "6",
    ]
    assert _tokens("  -21.49,   -22.52,   143.26") == ["-21.49", "-22.52", "143.26"]
    assert _tokens("15 PA3-A-Debug-Answer.txt 0") == ["15", "PA3-A-Debug-Answer.txt", "0"]


@pytest.mark.parametrize("name", SURFACES)
def test_read_mesh(name):
    mesh = read_mesh(DATA_DIR / name)
    assert mesh.vertices.shape == (N_VERTICES, 3)
    assert mesh.triangles.shape == (N_TRIANGLES, 3)
    assert mesh.neighbors.shape == (N_TRIANGLES, 3)
    assert mesh.triangles.min() >= 0 and mesh.triangles.max() < N_VERTICES
    assert mesh.neighbors.min() >= -1 and mesh.neighbors.max() < N_TRIANGLES
    assert np.allclose(mesh.vertices[0], [-23.786148, -16.420282, -48.229988])


@pytest.mark.parametrize("name", BODIES)
def test_read_body(name):
    body = read_body(DATA_DIR / name)
    assert body.markers.shape == (N_MARKERS, 3)
    assert body.tip.shape == (3,)
    assert np.all(np.isfinite(body.markers))


def test_body_a_tip_and_first_marker():
    body = read_body(DATA_DIR / "Problem3-BodyA.txt")
    assert np.allclose(body.markers[0], [-49.999, -36.846, 25.561])
    assert np.allclose(body.tip, [0.0, 0.0, -100.0])


@pytest.mark.parametrize("name", SAMPLES)
def test_read_samples(name):
    header = _tokens((DATA_DIR / name).read_text().splitlines()[0])
    n_samples = int(header[1])
    samples = read_samples(DATA_DIR / name, N_MARKERS, N_MARKERS)
    assert samples.n_samples == n_samples
    assert samples.a.shape == (n_samples, N_MARKERS, 3)
    assert samples.b.shape == (n_samples, N_MARKERS, 3)
    assert samples.dummy.shape == (n_samples, N_LEDS - 2 * N_MARKERS, 3)
    assert samples.n_modes == (6 if name.startswith("PA5") else 0)


def test_samples_first_values():
    samples = read_samples(DATA_DIR / "PA3-A-Debug-SampleReadingsTest.txt", 6, 6)
    assert np.allclose(samples.a[0, 0], [-21.49, -22.52, 143.26])
    assert samples.n_samples == 15


@pytest.mark.parametrize("name", RESULTS)
def test_read_output_and_answer(name):
    result = read_output(DATA_DIR / name)
    sample_name = re.sub(r"-(Output|Answer)\.txt$", "-SampleReadingsTest.txt", name)
    expected = read_samples(DATA_DIR / sample_name, N_MARKERS, N_MARKERS).n_samples
    assert result.n_samples == expected
    assert result.s.shape == result.c.shape == (expected, 3)
    assert result.weights.shape == ((6,) if name.startswith("PA5") else (0,))
    # Each coordinate is rounded to 0.01 mm, so each component of s - c can be off by up
    # to 0.01 and the recomputed norm by up to 0.01 * sqrt(3), plus 0.0005 for the
    # rounding of the stored distance.
    recomputed = np.linalg.norm(result.s - result.c, axis=1)
    assert np.allclose(recomputed, result.distance, atol=0.01 * np.sqrt(3) + 0.0005)


def test_read_modes_and_mean_shape_is_the_mesh():
    modes = read_modes(DATA_DIR / "Problem5Modes.txt")
    assert modes.shape == (7, N_VERTICES, 3)
    mesh = read_mesh(DATA_DIR / "Problem5MeshFile.sur")
    assert np.allclose(modes[0], mesh.vertices, atol=1e-6)


@pytest.mark.parametrize("name", LOGS)
def test_read_logfile(name):
    entries = read_logfile(DATA_DIR / name)
    assignment = name.split("-")[0]
    sets = {f"{assignment}-{label}" for label in datasets.list_sets(DATA_DIR, assignment)}
    assert {e.name for e in entries} <= sets
    expected_count = {"PA3": 9, "PA4": 15, "PA5": 10}[assignment]
    assert len(entries) == expected_count
    for entry in entries:
        assert entry.computed_p.shape == entry.actual_p.shape == (3,)
        assert entry.computed_rot_vec.shape == (3,)
        for R in (entry.computed_R, entry.actual_R):
            if R is not None:
                assert np.allclose(R.T @ R, np.eye(3), atol=1e-6)
        n_weights = 6 if assignment == "PA5" else 0
        assert len(entry.weights_solved) == len(entry.weights_actual) == n_weights


def test_list_sets(data_dir):
    pa4 = datasets.list_sets(data_dir, "PA4")
    assert pa4[:3] == ["A-Debug", "A-Demo-Fast", "A-Demo-Slow"]
    assert pa4[-4:] == ["G-Unknown", "H-Unknown", "J-Unknown", "K-Unknown"]
    assert len(datasets.list_sets(data_dir, "PA3")) == 9
    assert len(datasets.list_sets(data_dir, "PA5")) == 10


def test_load_inputs(data_dir):
    inputs = datasets.load_inputs(data_dir, "PA5", "A-Debug")
    assert inputs.samples.n_samples == 150
    assert inputs.samples.n_modes == 6
    assert inputs.mesh.n_triangles == N_TRIANGLES
