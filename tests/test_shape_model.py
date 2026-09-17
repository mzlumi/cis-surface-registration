import numpy as np
import pytest

from cisreg.fileio import read_mesh, read_modes
from cisreg.mesh import Mesh
from cisreg.shape_model import ShapeModel, check_mean_matches_mesh
from conftest import DATA_DIR


@pytest.fixture(scope="module")
def mesh():
    return read_mesh(DATA_DIR / "Problem5MeshFile.sur")


@pytest.fixture(scope="module")
def atlas():
    return read_modes(DATA_DIR / "Problem5Modes.txt")


def test_mode_zero_is_the_mesh(atlas, mesh):
    # The mesh file has 6 decimals and the modes file 7, so they agree to rounding.
    assert np.max(np.abs(atlas[0] - mesh.vertices)) < 5e-6
    check_mean_matches_mesh(atlas[0], mesh)


def test_mismatched_mean_is_rejected(atlas, mesh):
    moved = Mesh(mesh.vertices + 0.01, mesh.triangles, mesh.neighbors)
    with pytest.raises(ValueError):
        check_mean_matches_mesh(atlas[0], moved)
    with pytest.raises(ValueError):
        ShapeModel.from_atlas(atlas, moved)


def test_zero_weights_give_the_mean_shape(atlas, mesh):
    model = ShapeModel.from_atlas(atlas, mesh)
    assert model.n_modes == 6
    assert np.array_equal(model.vertices(np.zeros(6)), atlas[0])


def test_deformation_is_linear_in_the_weights(atlas, mesh):
    model = ShapeModel.from_atlas(atlas, mesh)
    rng = np.random.default_rng(0)
    a, b = rng.normal(scale=50, size=6), rng.normal(scale=50, size=6)
    lhs = model.vertices(a + b) - model.mean
    rhs = (model.vertices(a) - model.mean) + (model.vertices(b) - model.mean)
    assert np.allclose(lhs, rhs)
    single = np.zeros(6)
    single[2] = 10.0
    assert np.allclose(model.vertices(single), atlas[0] + 10.0 * atlas[3])


def test_modes_are_orthonormal(atlas):
    flat = atlas[1:].reshape(6, -1)
    assert np.allclose(flat @ flat.T, np.eye(6), atol=1e-5)


def test_selecting_fewer_modes(atlas, mesh):
    model = ShapeModel.from_atlas(atlas, mesh, n_modes=3)
    assert model.n_modes == 3 and np.array_equal(model.modes, atlas[1:4])
    with pytest.raises(ValueError):
        model.vertices(np.zeros(6))
    with pytest.raises(ValueError):
        ShapeModel.from_atlas(atlas, mesh, n_modes=7)


def test_deformed_mesh_keeps_the_triangles(atlas, mesh):
    model = ShapeModel.from_atlas(atlas, mesh)
    deformed = model.mesh(np.full(6, 20.0))
    assert np.array_equal(deformed.triangles, mesh.triangles)
    assert not np.allclose(deformed.vertices, mesh.vertices)
