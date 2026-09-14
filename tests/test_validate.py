import numpy as np

from cisreg.fileio import RegistrationOutput, read_output
from cisreg.validate import compare, markdown_table
from conftest import DATA_DIR


def test_a_file_compared_with_itself_has_no_error():
    ref = read_output(DATA_DIR / "PA5-E-Debug-Output.txt")
    row = compare("E-Debug", ref, ref)
    assert row.s_max == row.c_rms == row.distance_max == row.weights_max == 0.0


def test_known_offsets():
    ref = RegistrationOutput("x", np.zeros((4, 3)), np.zeros((4, 3)), np.zeros(4), np.zeros(0))
    s = np.zeros((4, 3))
    s[0] = [0.0, 3.0, 4.0]
    ours = RegistrationOutput("x", s, np.zeros((4, 3)), np.linalg.norm(s, axis=1), np.zeros(0))
    row = compare("Z", ours, ref)
    assert row.s_max == 5.0 and np.isclose(row.s_rms, 2.5)
    assert row.c_max == 0.0 and row.distance_max == 5.0 and row.weights_max is None
    assert "| Z | 4 | 5.0000 | 2.5000 |" in markdown_table([row])
