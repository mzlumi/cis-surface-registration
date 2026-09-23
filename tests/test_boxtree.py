"""The bounding-box tree must give exactly the brute-force answer."""

import numpy as np
import pytest

from cisreg import datasets
from cisreg.boxtree import BoundingBoxTree, box_distance2
from cisreg.fileio import read_mesh, read_modes
from cisreg.mesh import Mesh
from cisreg.search import BruteForceSearch, MeshMatches
from cisreg.tracking import pointer_tips
from cisreg.triangle import closest_point_on_triangle
from conftest import DATA_DIR

ALL_SETS = [(pa, label) for pa in ("PA3", "PA4", "PA5") for label in datasets.list_sets(DATA_DIR, pa)]


@pytest.fixture(scope="module")
def bone() -> Mesh:
    return read_mesh(DATA_DIR / "Problem3MeshFile.sur")


@pytest.fixture(scope="module", params=[False, True], ids=["axis-aligned", "oriented"])
def tree(bone, request) -> BoundingBoxTree:
    return BoundingBoxTree(bone, oriented=request.param)


@pytest.fixture(scope="module")
def brute(bone) -> BruteForceSearch:
    return BruteForceSearch(bone)


def assert_same_matches(mesh: Mesh, queries: np.ndarray, fast: MeshMatches, slow: MeshMatches):
    """Same distances and points; a different triangle is allowed only for an exact tie."""
    assert np.allclose(fast.distances, slow.distances, rtol=0, atol=1e-12)
    assert np.allclose(fast.points, slow.points, rtol=0, atol=1e-9)
    for i in np.flatnonzero(fast.triangles != slow.triangles):
        corners = mesh.vertices[mesh.triangles[fast.triangles[i]]]
        c, _ = closest_point_on_triangle(queries[i], *corners)
        assert np.isclose(np.linalg.norm(c - queries[i]), slow.distances[i], rtol=0, atol=1e-12)


@pytest.mark.parametrize("assignment, label", ALL_SETS)
def test_identical_to_brute_force_on_every_data_set(assignment, label, bone, tree, brute):
    inputs = datasets.load_inputs(DATA_DIR, assignment, label)
    d = pointer_tips(inputs.body_a, inputs.body_b, inputs.samples)
    assert_same_matches(bone, d, tree.closest_points(d), brute.closest_points(d))


@pytest.mark.parametrize("seed", range(5))
def test_identical_to_brute_force_on_random_points(seed, bone, tree, brute):
    rng = np.random.default_rng(seed)
    lo, hi = bone.vertices.min(axis=0), bone.vertices.max(axis=0)
    near = rng.uniform(lo - 5, hi + 5, size=(150, 3))
    far = rng.normal(scale=300.0, size=(50, 3))
    queries = np.vstack([near, far])
    assert_same_matches(bone, queries, tree.closest_points(queries), brute.closest_points(queries))


def test_single_query_and_points_on_the_surface(bone, tree):
    on_vertex = tree.closest_points(bone.vertices[17])
    assert np.isclose(on_vertex.distances[0], 0.0) and np.allclose(on_vertex.points[0], bone.vertices[17])
    centroids = bone.vertices[bone.triangles].mean(axis=1)
    m = tree.closest_points(centroids)
    assert np.allclose(m.distances, 0.0, atol=1e-12)


def test_any_hint_gives_the_same_answer(bone, tree, brute):
    """Hints only seed the bound, so even wrong hints must not change the result."""
    rng = np.random.default_rng(10)
    queries = rng.uniform(bone.vertices.min(axis=0), bone.vertices.max(axis=0), size=(100, 3))
    slow = brute.closest_points(queries)
    random_hint = rng.integers(0, bone.n_triangles, size=100)
    assert_same_matches(bone, queries, tree.closest_points(queries, hint=random_hint), slow)
    assert_same_matches(bone, queries, tree.closest_points(queries, hint=slow.triangles), slow)


@pytest.mark.parametrize("oriented", [False, True])
def test_refit_after_deformation(bone, oriented):
    """PA5 deforms the mesh; after update_vertices the tree must still be exact."""
    modes = read_modes(DATA_DIR / "Problem5Modes.txt")
    deformed = modes[0] + np.tensordot([80.0, -60.0, 40.0, 30.0, -50.0, 20.0], modes[1:], axes=1)
    tree = BoundingBoxTree(bone, oriented=oriented)
    tree.update_vertices(deformed)
    brute = BruteForceSearch(Mesh(deformed, bone.triangles, bone.neighbors))
    rng = np.random.default_rng(11)
    queries = rng.uniform(deformed.min(axis=0) - 5, deformed.max(axis=0) + 5, size=(200, 3))
    assert_same_matches(
        Mesh(deformed, bone.triangles, bone.neighbors),
        queries,
        tree.closest_points(queries),
        brute.closest_points(queries),
    )


def test_tree_structure(bone, tree):
    assert np.array_equal(np.sort(tree.order), np.arange(bone.n_triangles))
    corners = bone.vertices[bone.triangles]
    for node in range(tree.n_nodes):
        tris = tree.order[tree.start[node] : tree.end[node]]
        R = tree.frames[node]
        assert np.allclose(R.T @ R, np.eye(3))
        inside = (corners[tris] - tree.centers[node]) @ R
        assert np.all(inside >= tree.lo[node] - 1e-9) and np.all(inside <= tree.hi[node] + 1e-9)
        if tree.is_leaf[node]:
            assert len(tris) <= tree.leaf_size
        else:
            a, b = tree.left[node], tree.right[node]
            assert tree.start[a] == tree.start[node] and tree.end[b] == tree.end[node]
            assert tree.end[a] == tree.start[b]
    assert tree.depth() <= int(np.ceil(np.log2(bone.n_triangles / tree.leaf_size))) + 2


def test_tiny_mesh_is_a_single_leaf():
    v = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    t = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    tree = BoundingBoxTree(Mesh(v, t, -np.ones_like(t)))
    assert tree.n_nodes == 1
    m = tree.closest_points(np.array([[0.2, 0.2, -1.0]]))
    assert np.allclose(m.points, [[0.2, 0.2, 0.0]])


def test_oriented_boxes_are_tighter(bone):
    """Principal-axis boxes enclose less volume than axis-aligned ones."""
    aligned, oriented = BoundingBoxTree(bone), BoundingBoxTree(bone, oriented=True)
    volume = lambda t: np.prod(t.hi - t.lo, axis=1)[t.is_leaf].mean()  # noqa: E731
    assert volume(oriented) < 0.6 * volume(aligned)


def test_box_distance():
    lo, hi = np.zeros(3), np.ones(3)
    points = np.array([[0.5, 0.5, 0.5], [2.0, 0.5, 0.5], [2.0, 2.0, -1.0]])
    assert np.allclose(box_distance2(points, lo, hi), [0.0, 1.0, 3.0])
