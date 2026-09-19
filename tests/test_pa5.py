import numpy as np

from cisreg import datasets, pa5
from cisreg.fileio import read_modes, read_output
from conftest import DATA_DIR


def test_cli_writes_weights_then_points(tmp_path):
    pa5.main(
        [
            "--set", "B-Debug", "--data-dir", str(DATA_DIR), "--output-dir", str(tmp_path),
            "--log-dir", str(tmp_path / "logs"),
        ]
    )
    ours = read_output(tmp_path / "PA5-B-Debug-Output.txt")
    reference = read_output(datasets.output_path(DATA_DIR, "PA5", "B-Debug"))
    assert ours.n_samples == 150 and ours.weights.shape == (6,)
    # Weight differences at this level come from the 0.01 mm rounding of the readings
    # (see results/pa5_validation.md).
    assert np.max(np.abs(ours.weights - reference.weights)) < 0.1
    assert np.max(np.linalg.norm(ours.s - reference.s, axis=1)) < 0.03
    log = (tmp_path / "logs" / "PA5-B-Debug-deformable.txt").read_text()
    assert "combined" in log and "mode" in log


def test_summary_lists_every_run():
    inputs = datasets.load_inputs(DATA_DIR, "PA5", "A-Debug")
    d, result = pa5.solve_pa5(inputs, read_modes(datasets.modes_path(DATA_DIR)))
    text = pa5.summary_markdown([pa5.Pa5Run("A-Debug", d, result)])
    assert text.count("| A-Debug |") == 3
