"""Brute-force closest point on a mesh."""

import numpy as np
import pytest

from cisreg.fileio import read_mesh
from cisreg.mesh import Mesh
from cisreg.search import BruteForceSearch
from cisreg.triangle import closest_point_on_triangle
from conftest import DATA_DIR


def unit_cube() -> Mesh:
    """Closed unit cube, 12 triangles."""
    v = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = []
    for a, b, c, d in quads:
        tris += [(a, b, c), (a, c, d)]
    tris = np.array(tris)
    return Mesh(v, tris, -np.ones_like(tris))


@pytest.fixture(scope="module")
def bone() -> Mesh:
    return read_mesh(DATA_DIR / "Problem3MeshFile.sur")


def loop_reference(mesh: Mesh, a: np.ndarray) -> float:
    """Distance to the mesh by looping over triangles one at a time."""
    p, q, r = mesh.corners()
    return min(
        np.linalg.norm(closest_point_on_triangle(a, p[i], q[i], r[i])[0] - a)
        for i in range(mesh.n_triangles)
    )


def test_cube_known_answers():
    search = BruteForceSearch(unit_cube())
    queries = np.array([[0.5, 0.5, 2.0], [2.0, 2.0, 2.0], [0.5, 0.5, 0.4], [-1.0, 0.5, 0.5]])
    m = search.closest_points(queries)
    assert np.allclose(m.points, [[0.5, 0.5, 1.0], [1, 1, 1], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])
    assert np.allclose(m.distances, [1.0, np.sqrt(3), 0.4, 1.0])


def test_cube_against_loop_reference():
    mesh = unit_cube()
    rng = np.random.default_rng(0)
    queries = rng.uniform(-1, 2, size=(50, 3))
    m = BruteForceSearch(mesh).closest_points(queries)
    expected = [loop_reference(mesh, a) for a in queries]
    assert np.allclose(m.distances, expected)


def test_matches_are_consistent(bone):
    rng = np.random.default_rng(1)
    queries = rng.uniform(bone.vertices.min(0) - 10, bone.vertices.max(0) + 10, size=(40, 3))
    m = BruteForceSearch(bone, chunk_size=7).closest_points(queries)
    assert np.allclose(np.linalg.norm(queries - m.points, axis=1), m.distances)
    corners = bone.vertices[bone.triangles[m.triangles]]
    assert np.allclose(np.einsum("nk,nkd->nd", m.weights, corners), m.points)
    assert np.all(m.weights >= -1e-12) and np.allclose(m.weights.sum(axis=1), 1.0)


def test_bone_against_loop_reference(bone):
    rng = np.random.default_rng(2)
    queries = rng.uniform(bone.vertices.min(0), bone.vertices.max(0), size=(5, 3))
    m = BruteForceSearch(bone).closest_points(queries)
    assert np.allclose(m.distances, [loop_reference(bone, a) for a in queries])


def test_points_on_the_surface_have_zero_distance(bone):
    rng = np.random.default_rng(3)
    tri = rng.integers(0, bone.n_triangles, size=30)
    w = rng.dirichlet(np.ones(3), size=30)
    on_surface = np.einsum("nk,nkd->nd", w, bone.vertices[bone.triangles[tri]])
    m = BruteForceSearch(bone).closest_points(on_surface)
    assert np.allclose(m.distances, 0.0, atol=1e-9)
    assert np.allclose(m.points, on_surface)
    vertices = BruteForceSearch(bone).closest_points(bone.vertices[:50])
    assert np.allclose(vertices.distances, 0.0)


def test_update_vertices_moves_the_surface():
    mesh = unit_cube()
    search = BruteForceSearch(mesh)
    search.update_vertices(mesh.vertices + [0.0, 0.0, 1.0])
    m = search.closest_points(np.array([[0.5, 0.5, 3.0]]))
    assert np.allclose(m.points, [[0.5, 0.5, 2.0]])
