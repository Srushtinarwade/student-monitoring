"""
Head pose estimation using OpenCV solvePnP.

Computes pitch, yaw, and roll from 6 key facial landmarks mapped to a
generic 3-D face model.
"""

from __future__ import annotations

import numpy as np
import cv2

import config


# MediaPipe landmark indices used by solvePnP
_NOSE_TIP = 1
_CHIN = 152
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_OUTER = 263
_LEFT_MOUTH = 61
_RIGHT_MOUTH = 291

_POSE_LANDMARK_IDS: list[int] = [
    _NOSE_TIP,
    _CHIN,
    _LEFT_EYE_OUTER,
    _RIGHT_EYE_OUTER,
    _LEFT_MOUTH,
    _RIGHT_MOUTH,
]

# Generic 3-D model points (mm) corresponding to the landmarks above
_MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float64,
)


class HeadPoseEstimator:
    """Estimates head orientation (pitch, yaw, roll) from face landmarks."""

    def __init__(self, frame_w: int | None = None, frame_h: int | None = None) -> None:
        w = frame_w or config.FRAME_WIDTH
        h = frame_h or config.FRAME_HEIGHT
        focal_length = float(w)
        cx, cy = w / 2.0, h / 2.0
        self._camera_matrix = np.array(
            [[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]],
            dtype=np.float64,
        )
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    def estimate(
        self, landmarks: list, frame_w: int, frame_h: int
    ) -> tuple[float, float, float] | None:
        """Return (pitch, yaw, roll) in degrees, or *None* on failure."""
        try:
            image_points = np.array(
                [
                    (landmarks[i].x * frame_w, landmarks[i].y * frame_h)
                    for i in _POSE_LANDMARK_IDS
                ],
                dtype=np.float64,
            )
            success, rot_vec, trans_vec = cv2.solvePnP(
                _MODEL_POINTS_3D,
                image_points,
                self._camera_matrix,
                self._dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success:
                return None

            rot_mat, _ = cv2.Rodrigues(rot_vec)
            pose_mat = cv2.hstack((rot_mat, trans_vec))
            _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mat)
            return float(euler[0]), float(euler[1]), float(euler[2])
        except Exception:
            return None

    @staticmethod
    def is_head_ok(pitch: float, yaw: float) -> bool:
        """Check whether pitch & yaw are within acceptable thresholds."""
        return abs(yaw) <= config.YAW_THRESHOLD and abs(pitch) <= config.PITCH_THRESHOLD
