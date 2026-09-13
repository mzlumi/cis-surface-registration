"""Rigid body poses from tracker readings and the pointer tip in bone coordinates.

For each sample frame k, the pose of a rigid body with respect to the tracker is
the frame F_k that best maps its marker positions in body coordinates onto the
measured marker positions (point-set registration). The tip of pointer A in the
coordinates of body B, which is screwed into the bone, is then
    d_k = F_B,k^-1 F_A,k A_tip
(PA3/PA4 handout, "Hints on suggested procedure").

Author: Parmida Mazloomi
"""

from __future__ import annotations

import numpy as np

from cisreg.fileio import RigidBody, SampleReadings
from cisreg.frames import Frame
from cisreg.registration import register_points


def body_poses(body_markers: np.ndarray, readings: np.ndarray) -> list[Frame]:
    """Pose F_k of a body for each sample, from (N, 3) body markers and (K, N, 3) readings."""
    return [register_points(body_markers, frame_readings) for frame_readings in readings]


def pointer_tips(body_a: RigidBody, body_b: RigidBody, samples: SampleReadings) -> np.ndarray:
    """Tip of pointer A in body B coordinates for every sample, as a (K, 3) array."""
    poses_a = body_poses(body_a.markers, samples.a)
    poses_b = body_poses(body_b.markers, samples.b)
    return np.array([(F_b.inverse() @ F_a).apply(body_a.tip) for F_a, F_b in zip(poses_a, poses_b)])
