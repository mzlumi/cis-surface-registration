import numpy as np

import cisreg


def test_package_imports():
    assert cisreg.__version__


def test_numpy_available():
    assert np.allclose(np.eye(3) @ np.ones(3), np.ones(3))
