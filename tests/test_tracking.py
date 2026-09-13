import numpy as np

from cisreg.fileio import RigidBody, SampleReadings
from cisreg.frames import Frame, random_rotation
from cisreg.tracking import body_poses, pointer_tips


def random_frame(rng):
    return Frame(random_rotation(rng), rng.uniform(-300, 300, size=3))


def synthetic_samples(rng, n_samples=10, noise=0.0):
    """Pointer A touches a point within 60 mm of body B, as when probing the bone."""
    body_a = RigidBody(rng.uniform(-50, 50, size=(6, 3)), np.array([0.0, 0.0, -100.0]))
    body_b = RigidBody(rng.uniform(-50, 50, size=(5, 3)), np.zeros(3))
    poses_b = [random_frame(rng) for _ in range(n_samples)]
    poses_a = []
    for F_b in poses_b:
        touched = F_b.apply(rng.uniform(-60, 60, size=3))
        R_a = random_rotation(rng)
        poses_a.append(Frame(R_a, touched - R_a @ body_a.tip))
    a = np.array([F.apply(body_a.markers) for F in poses_a])
    b = np.array([F.apply(body_b.markers) for F in poses_b])
    a += rng.normal(scale=noise, size=a.shape) if noise else 0.0
    b += rng.normal(scale=noise, size=b.shape) if noise else 0.0
    samples = SampleReadings(a=a, b=b, dummy=np.zeros((n_samples, 0, 3)), n_modes=0)
    expected = np.array([(Fb.inverse() @ Fa).apply(body_a.tip) for Fa, Fb in zip(poses_a, poses_b)])
    return body_a, body_b, samples, poses_a, expected


def test_body_poses_recover_the_true_frames():
    rng = np.random.default_rng(0)
    body_a, _, samples, poses_a, _ = synthetic_samples(rng)
    for F_true, F in zip(poses_a, body_poses(body_a.markers, samples.a)):
        assert np.allclose(F.as_matrix(), F_true.as_matrix())


def test_pointer_tips_noise_free():
    rng = np.random.default_rng(1)
    body_a, body_b, samples, _, expected = synthetic_samples(rng)
    assert np.allclose(pointer_tips(body_a, body_b, samples), expected)


def test_pointer_tips_with_marker_noise_stay_close():
    rng = np.random.default_rng(2)
    body_a, body_b, samples, _, expected = synthetic_samples(rng, n_samples=50, noise=0.1)
    error = np.linalg.norm(pointer_tips(body_a, body_b, samples) - expected, axis=1)
    # The tip is 100 mm from the A markers, so small rotation errors are amplified,
    # but 0.1 mm marker noise should give errors of a few tenths of a millimetre.
    assert np.mean(error) < 0.5


def test_tip_is_independent_of_a_common_tracker_motion():
    """Moving the whole scene (both bodies) relative to the tracker does not change d_k."""
    rng = np.random.default_rng(3)
    body_a, body_b, samples, _, expected = synthetic_samples(rng)
    G = random_frame(rng)
    moved = SampleReadings(G.apply(samples.a), G.apply(samples.b), samples.dummy, 0)
    assert np.allclose(pointer_tips(body_a, body_b, moved), expected)
