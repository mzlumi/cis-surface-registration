import numpy as np

from cisreg import datasets, pa4
from cisreg.fileio import read_output
from conftest import DATA_DIR


def test_debug_set_converges_onto_the_surface():
    d, result = pa4.solve_pa4(datasets.load_inputs(DATA_DIR, "PA4", "C-Debug"))
    assert result.converged
    assert result.history[-1].rms_residual < 0.01
    reference = read_output(datasets.output_path(DATA_DIR, "PA4", "C-Debug"))
    assert np.max(np.linalg.norm(result.s - reference.s, axis=1)) < 0.03


def test_brute_force_and_tree_give_the_same_registration():
    inputs = datasets.load_inputs(DATA_DIR, "PA4", "A-Demo-Fast")
    _, fast = pa4.solve_pa4(inputs)
    _, slow = pa4.solve_pa4(inputs, brute_force=True)
    assert fast.iterations == slow.iterations
    assert np.allclose(fast.F.as_matrix(), slow.F.as_matrix(), atol=1e-12)


def test_cli_writes_output_log_summary_and_figure(tmp_path):
    pa4.main(
        [
            "--set", "A-Debug", "--data-dir", str(DATA_DIR), "--output-dir", str(tmp_path),
            "--log-dir", str(tmp_path / "logs"),
        ]
    )
    assert read_output(tmp_path / "PA4-A-Debug-Output.txt").n_samples == 75
    log = (tmp_path / "logs" / "PA4-A-Debug-icp.txt").read_text().splitlines()
    assert log[1].split()[:2] == ["iter", "mean_mm"] and len(log) > 2


def test_summary_table_has_one_row_per_run():
    inputs = datasets.load_inputs(DATA_DIR, "PA4", "A-Debug")
    d, result = pa4.solve_pa4(inputs)
    text = pa4.summary_markdown([pa4.Pa4Run("A-Debug", d, result)])
    assert text.count("| A-Debug | 75 |") == 1
