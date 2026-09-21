import numpy as np

from cisreg.analysis import misaligned, plot_basin, plot_modes, plot_noise, plot_timing
from cisreg.frames import Frame, frame_difference, random_rotation


def test_misalignment_has_the_requested_size():
    rng = np.random.default_rng(0)
    F = Frame(random_rotation(rng), rng.normal(size=3))
    center = np.array([10.0, -5.0, 20.0])
    G = misaligned(F, center, np.radians(30.0), 7.0, rng)
    angle, _ = frame_difference(F, G)
    assert np.isclose(np.degrees(angle), 30.0)
    # The rotation is about the centre, so the centre only moves by the shift.
    moved = (G @ F.inverse()).apply(center)
    assert np.isclose(np.linalg.norm(moved - center), 7.0)


def test_plots_are_written(tmp_path):
    plot_basin([0, 30], [0, 10], np.array([[1.0, 0.5], [1.0, 0.0]]), np.array([[5, 9], [6, np.nan]]),
               tmp_path / "basin.png")
    rows4 = np.array([[0.0, 0.0, 0.0, 0.002], [0.1, 0.1, 0.03, 0.15]])
    rows5 = np.array([[0.0, 0.0, 0.0], [0.1, 1.0, 0.08]])
    plot_noise(rows4, rows5, tmp_path / "noise.png")
    plot_modes({"A-Debug": [2.0, 1.0, 0.003]}, tmp_path / "modes.png")
    plot_timing(np.array([[1, 0.002, 0.001], [100, 0.2, 0.006]]), tmp_path / "timing.png")
    for name in ("basin", "noise", "modes", "timing"):
        assert (tmp_path / f"{name}.png").stat().st_size > 1000
