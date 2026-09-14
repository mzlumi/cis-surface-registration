import numpy as np

from cisreg import datasets, pa3
from cisreg.fileio import format_output, read_output, write_output
from conftest import DATA_DIR


def test_writer_reproduces_the_instructor_layout():
    """Rewriting an instructor file gives the same text, column for column.

    The distance column is recomputed from the rounded coordinates, so it is
    compared numerically; everything before it must match character for character.
    """
    for name in ["PA3-A-Debug-Output.txt", "PA4-B-Debug-Output.txt", "PA5-A-Debug-Output.txt"]:
        original = (DATA_DIR / name).read_text().splitlines()
        result = read_output(DATA_DIR / name)
        weights = result.weights if len(result.weights) else None
        rewritten = format_output(name, result.s, result.c, weights).splitlines()
        assert len(rewritten) == len(original)
        header_lines = 2 if weights is not None else 1
        assert rewritten[:header_lines] == original[:header_lines]
        for new, old in zip(rewritten[header_lines:], original[header_lines:]):
            assert len(new) == len(old) and new[:57] == old[:57]
            assert abs(float(new[57:]) - float(old[57:])) <= 0.01 * np.sqrt(3)


def test_write_and_read_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    s, c = rng.uniform(-90, 90, size=(5, 3)), rng.uniform(-90, 90, size=(5, 3))
    path = tmp_path / "PA5-Z-Test-Output.txt"
    write_output(path, s, c, weights=np.array([1.23456, -7.0]))
    back = read_output(path)
    assert back.name == path.name
    assert np.allclose(back.s, s, atol=0.005) and np.allclose(back.c, c, atol=0.005)
    assert np.allclose(back.distance, np.linalg.norm(s - c, axis=1), atol=0.0005)
    assert np.allclose(back.weights, [1.2346, -7.0])


def test_noise_free_debug_set_points_lie_on_the_mesh():
    """In PA3-A-Debug the samples are on the surface, so every distance is close to 0."""
    result = pa3.solve_pa3(datasets.load_inputs(DATA_DIR, "PA3", "A-Debug"))
    assert result.d.shape == (15, 3)
    assert np.max(result.matches.distances) < 0.05


def test_cli_writes_the_output_file(tmp_path):
    pa3.main(["--set", "A-Debug", "--data-dir", str(DATA_DIR), "--output-dir", str(tmp_path)])
    written = read_output(tmp_path / "PA3-A-Debug-Output.txt")
    assert written.n_samples == 15
    assert written.name == "PA3-A-Debug-Output.txt"
