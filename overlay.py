"""
OpenCV overlay rendering for the monitoring window.

Draws face contours, iris markers, eye boxes, and HUD text using pure
OpenCV — no dependency on ``mp.solutions.drawing_utils``.
"""

from __future__ import annotations

import cv2
import numpy as np

from detectors.face import FaceAnalysis, _FACE_OVAL


def draw_face(frame: np.ndarray, face: FaceAnalysis) -> None:
    """Draw face contour, iris centres, and eye bounding boxes."""
    if not face.detected or face.landmarks is None:
        return

    h, w = frame.shape[:2]
    lm = face.landmarks

    # Draw face oval contour
    for start_idx, end_idx in _FACE_OVAL:
        pt1 = (int(lm[start_idx].x * w), int(lm[start_idx].y * h))
        pt2 = (int(lm[end_idx].x * w), int(lm[end_idx].y * h))
        cv2.line(frame, pt1, pt2, (80, 256, 121), 1)

    # Iris centres
    if face.left_iris_center:
        cv2.circle(frame, face.left_iris_center, 3, (0, 255, 255), -1)
    if face.right_iris_center:
        cv2.circle(frame, face.right_iris_center, 3, (0, 255, 255), -1)

    # Eye bounding boxes
    for box in (face.left_eye_box, face.right_eye_box):
        if box:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)

    # EAR & blink debug info
    cv2.putText(
        frame,
        f"EAR:{face.avg_ear:.2f}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2,
    )
    cv2.putText(
        frame,
        f"ClosedFrames:{face.closed_frames}",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
    )


def draw_status(
    frame: np.ndarray,
    *,
    focused: bool,
    phone_detected: bool,
    phone_seconds: float,
    distraction_count: int,
) -> None:
    """Draw the status HUD: focus state, phone time, distraction counter."""
    h, w = frame.shape[:2]

    # Focus / distraction banner
    if focused:
        text, color = "FOCUSED", (0, 255, 0)
    else:
        text, color = "DISTRACTED", (0, 0, 255)
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    # Counters
    cv2.putText(
        frame,
        f"Phone secs: {phone_seconds:.1f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2,
    )
    cv2.putText(
        frame,
        f"Distractions: {distraction_count}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2,
    )

    # Phone warning
    if phone_detected:
        cv2.putText(
            frame,
            "PHONE DETECTED",
            (w - 360, 60),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
