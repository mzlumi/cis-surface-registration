from dataclasses import replace

import numpy as np

from cisreg import datasets
from cisreg.answer_key import log_frames
from cisreg.fileio import read_logfile
from cisreg.frames import rotation_from_vector
from conftest import DATA_DIR


def test_log_rotation_vectors_match_the_matrix_blocks():
    """The compact Fr([r],[p]] form uses r = axis times angle in radians."""
    checked = 0
    for assignment in ("PA4", "PA5"):
        for entry in read_logfile(datasets.logfile_path(DATA_DIR, assignment)):
            if entry.actual_R is not None:
                assert np.allclose(rotation_from_vector(entry.actual_rot_vec), entry.actual_R, atol=1e-6)
                checked += 1
    assert checked > 10


def test_every_entry_has_a_matrix_block_and_frames_are_rotations():
    for assignment in ("PA3", "PA4", "PA5"):
        for entry in read_logfile(datasets.logfile_path(DATA_DIR, assignment)):
            assert entry.computed_R is not None and entry.actual_R is not None
            for frame in log_frames(entry):
                assert np.allclose(frame.R.T @ frame.R, np.eye(3), atol=1e-6)


def test_frames_fall_back_to_the_rotation_vector():
    entry = read_logfile(datasets.logfile_path(DATA_DIR, "PA4"))[6]
    without_matrices = replace(entry, computed_R=None, actual_R=None)
    # The compact form is printed to about 1e-6 rad, so the two agree to that level.
    for with_m, without in zip(log_frames(entry), log_frames(without_matrices)):
        assert np.allclose(with_m.R, without.R, atol=5e-6)
