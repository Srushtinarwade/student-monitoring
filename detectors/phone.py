"""
Phone detection using YOLOv8.

Runs YOLOv8-nano on a resized frame, applies temporal smoothing, and
tracks cumulative phone-visible duration.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

import config

logger = logging.getLogger(__name__)


@dataclass
class PhoneStatus:
    """Snapshot of phone-detection state after processing one frame."""

    detected: bool = False
    visible_seconds: float = 0.0


class PhoneDetector:
    """Detects phones via YOLOv8 with frame-skipping and smoothing."""

    def __init__(self) -> None:
        logger.info("Loading YOLOv8 model (%s)…", config.YOLO_MODEL)
        self._model = YOLO(config.YOLO_MODEL)

        self._skip_counter: int = 0
        self._consec_frames: int = 0
        self._last_seen_ts: float | None = None
        self._visible_seconds: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> PhoneStatus:
        """Run detection on *frame* and return the current `PhoneStatus`."""
        phone_in_frame = False

        self._skip_counter += 1
        if self._skip_counter >= config.YOLO_SKIP_FRAMES:
            self._skip_counter = 0
            phone_in_frame = self._run_yolo(frame)

        self._update_smoothing(phone_in_frame)

        return PhoneStatus(
            detected=self._consec_frames >= config.PHONE_CONFIRM_FRAMES,
            visible_seconds=self._visible_seconds,
        )

    def finalize(self) -> float:
        """Flush any remaining phone-visible time and return total seconds."""
        if (
            self._consec_frames >= config.PHONE_CONFIRM_FRAMES
            and self._last_seen_ts is not None
        ):
            self._visible_seconds += time.time() - self._last_seen_ts
            self._last_seen_ts = None
        return self._visible_seconds

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_yolo(self, frame: np.ndarray) -> bool:
        """Run YOLO inference on a resized copy of *frame*."""
        h, w = frame.shape[:2]
        new_h = int(config.YOLO_IMGSZ * h / w)
        small = cv2.resize(frame, (config.YOLO_IMGSZ, new_h))

        results = self._model(small, imgsz=config.YOLO_IMGSZ, verbose=False)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = self._model.names.get(cls_id, str(cls_id)).lower()
                if any(kw in label for kw in ("phone", "cell", "mobile")):
                    return True
        return False

    def _update_smoothing(self, phone_in_frame: bool) -> None:
        """Apply temporal smoothing to avoid single-frame false positives."""
        if phone_in_frame:
            self._consec_frames += 1
            if self._last_seen_ts is None:
                self._last_seen_ts = time.time()
        else:
            if self._consec_frames >= config.PHONE_CONFIRM_FRAMES:
                if self._last_seen_ts is not None:
                    self._visible_seconds += time.time() - self._last_seen_ts
                self._last_seen_ts = None
            self._consec_frames = 0
