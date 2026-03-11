"""
Face detection, eye aspect ratio, gaze tracking, and blink handling.

Uses MediaPipe FaceMesh with refined landmarks for iris tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import mediapipe as mp

import config


# MediaPipe landmark indices for 6-point EAR computation
# Order: p1, p2, p3, p4, p5, p6  →  EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2·‖p1−p4‖)
_LEFT_EAR_IDX: list[int] = [33, 160, 158, 133, 153, 144]
_RIGHT_EAR_IDX: list[int] = [263, 387, 385, 362, 380, 373]

# Iris landmark ranges (MediaPipe refined landmarks)
_LEFT_IRIS_IDX: list[int] = list(range(468, 473))
_RIGHT_IRIS_IDX: list[int] = list(range(473, 478))

# Eye contour indices used for gaze bounding box
_LEFT_EYE_CONTOUR: list[int] = [33, 133, 160, 159, 158, 144, 153]
_RIGHT_EYE_CONTOUR: list[int] = [362, 263, 387, 386, 385, 373, 380]


@dataclass
class FaceAnalysis:
    """Result of face analysis for a single frame."""

    detected: bool = False
    eyes_open: bool = True
    gaze_centered: bool = True
    short_blink: bool = False
    long_eye_closure: bool = False
    avg_ear: float = 0.0
    closed_frames: int = 0

    # Iris centres for overlay drawing (pixel coords)
    left_iris_center: tuple[int, int] | None = None
    right_iris_center: tuple[int, int] | None = None

    # Eye bounding boxes (x_min, y_min, x_max, y_max) for overlay
    left_eye_box: tuple[int, int, int, int] | None = None
    right_eye_box: tuple[int, int, int, int] | None = None

    # Raw MediaPipe result for overlay drawing
    face_landmarks: object | None = field(default=None, repr=False)


class FaceDetector:
    """Detects face landmarks, computes EAR, gaze direction, and blink state."""

    def __init__(self) -> None:
        self._mp_face_mesh = mp.solutions.face_mesh
        self._mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._closed_frames: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, rgb_frame: np.ndarray) -> FaceAnalysis:
        """Analyse a single RGB frame and return a `FaceAnalysis`."""
        h, w = rgb_frame.shape[:2]
        result = self._mesh.process(rgb_frame)

        if not result.multi_face_landmarks:
            return FaceAnalysis(detected=False)

        lm = result.multi_face_landmarks[0].landmark

        # EAR ----------------------------------------------------------
        left_ear = self._ear(lm, _LEFT_EAR_IDX, w, h)
        right_ear = self._ear(lm, _RIGHT_EAR_IDX, w, h)
        avg_ear = (left_ear + right_ear) / 2.0
        eyes_open = avg_ear > config.EAR_CLOSED_THRESHOLD

        # Blink tracking -----------------------------------------------
        if not eyes_open:
            self._closed_frames += 1
        else:
            self._closed_frames = 0

        short_blink = 0 < self._closed_frames <= config.BLINK_MAX_FRAMES
        long_closure = self._closed_frames > config.BLINK_MAX_FRAMES

        # Iris / gaze --------------------------------------------------
        left_iris, _ = self._iris_center(lm, _LEFT_IRIS_IDX, w, h)
        right_iris, _ = self._iris_center(lm, _RIGHT_IRIS_IDX, w, h)

        left_box = self._eye_box(lm, _LEFT_EYE_CONTOUR, w, h)
        right_box = self._eye_box(lm, _RIGHT_EYE_CONTOUR, w, h)

        left_gaze_ok = self._gaze_ok(left_iris, left_box)
        right_gaze_ok = self._gaze_ok(right_iris, right_box)
        gaze_centered = left_gaze_ok and right_gaze_ok

        return FaceAnalysis(
            detected=True,
            eyes_open=eyes_open,
            gaze_centered=gaze_centered,
            short_blink=short_blink,
            long_eye_closure=long_closure,
            avg_ear=avg_ear,
            closed_frames=self._closed_frames,
            left_iris_center=left_iris,
            right_iris_center=right_iris,
            left_eye_box=left_box,
            right_eye_box=right_box,
            face_landmarks=result.multi_face_landmarks[0],
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._mesh.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ear(
        landmarks: list, indices: list[int], w: int, h: int
    ) -> float:
        """Compute Eye Aspect Ratio from 6 landmark indices."""
        pts = np.array(
            [(landmarks[i].x * w, landmarks[i].y * h) for i in indices],
            dtype=np.float32,
        )
        if pts.shape[0] != 6:
            vert = pts[:, 1].max() - pts[:, 1].min()
            horiz = (pts[:, 0].max() - pts[:, 0].min()) + 1e-6
            return float(vert / horiz)

        p1, p2, p3, p4, p5, p6 = pts
        vert1 = np.linalg.norm(p2 - p6)
        vert2 = np.linalg.norm(p3 - p5)
        horiz = np.linalg.norm(p1 - p4) + 1e-6
        return float((vert1 + vert2) / (2.0 * horiz))

    @staticmethod
    def _iris_center(
        landmarks: list, indices: list[int], w: int, h: int
    ) -> tuple[tuple[int, int], np.ndarray]:
        """Return (cx, cy) of iris centre and the raw points array."""
        pts = np.array(
            [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
        )
        cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
        return (cx, cy), pts

    @staticmethod
    def _eye_box(
        landmarks: list, indices: list[int], w: int, h: int
    ) -> tuple[int, int, int, int]:
        """Bounding box (x_min, y_min, x_max, y_max) for an eye contour."""
        pts = np.array(
            [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
        )
        return (
            int(pts[:, 0].min()),
            int(pts[:, 1].min()),
            int(pts[:, 0].max()),
            int(pts[:, 1].max()),
        )

    @staticmethod
    def _gaze_ok(
        iris_center: tuple[int, int],
        eye_box: tuple[int, int, int, int],
    ) -> bool:
        """Check whether the iris is roughly centred inside the eye box."""
        x_min, y_min, x_max, y_max = eye_box
        w = max(1, x_max - x_min)
        h = max(1, y_max - y_min)
        rel_x = (iris_center[0] - x_min) / w - 0.5
        rel_y = (iris_center[1] - y_min) / h - 0.5
        return abs(rel_x) <= config.GAZE_X_THRESHOLD and abs(rel_y) <= config.GAZE_Y_THRESHOLD
