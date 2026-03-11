"""
Configuration for the Student Monitoring System.

All tuneable parameters are defined here. Override via environment variables
where noted (prefixed with SM_).
"""

from __future__ import annotations

import os
import getpass
from pathlib import Path

# ---------------------------------------------------------------------------
# User / Session
# ---------------------------------------------------------------------------
USERNAME: str = os.environ.get("SM_USERNAME", getpass.getuser())

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_default_log_dir = Path.home() / ".student_monitoring" / "logs"
LOG_DIR: Path = Path(os.environ.get("SM_LOG_DIR", str(_default_log_dir)))

# ---------------------------------------------------------------------------
# Webcam
# ---------------------------------------------------------------------------
WEBCAM_INDEX: int = int(os.environ.get("SM_WEBCAM_INDEX", "0"))
FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720

# ---------------------------------------------------------------------------
# YOLO Phone Detection
# ---------------------------------------------------------------------------
YOLO_MODEL: str = "yolov8n.pt"
YOLO_IMGSZ: int = 640
YOLO_SKIP_FRAMES: int = 2

# ---------------------------------------------------------------------------
# FPS & Timing
# ---------------------------------------------------------------------------
FPS_ESTIMATE: float = 22.5

# Distraction must be continuous for this many seconds before triggering alert
DISTRACTION_SECONDS: float = 1.3
DISTRACTION_FRAMES: int = max(1, int(round(DISTRACTION_SECONDS * FPS_ESTIMATE)))

# Focus must be sustained for this many seconds before resetting alert state
FOCUS_RESET_SECONDS: float = 0.35
FOCUS_RESET_FRAMES: int = max(1, int(round(FOCUS_RESET_SECONDS * FPS_ESTIMATE)))

# Phone must be visible for this many seconds to count
PHONE_CONFIRM_SECONDS: float = 0.5
PHONE_CONFIRM_FRAMES: int = max(1, int(round(PHONE_CONFIRM_SECONDS * FPS_ESTIMATE)))

# ---------------------------------------------------------------------------
# Blink Handling
# ---------------------------------------------------------------------------
BLINK_MAX_SECONDS: float = 0.25
BLINK_MAX_FRAMES: int = max(1, int(round(BLINK_MAX_SECONDS * FPS_ESTIMATE)))

# ---------------------------------------------------------------------------
# Eye Aspect Ratio (EAR)
# ---------------------------------------------------------------------------
EAR_CLOSED_THRESHOLD: float = 0.18

# ---------------------------------------------------------------------------
# Gaze & Head Pose Thresholds
# ---------------------------------------------------------------------------
GAZE_X_THRESHOLD: float = 0.30
GAZE_Y_THRESHOLD: float = 0.40
YAW_THRESHOLD: float = 25.0
PITCH_THRESHOLD: float = 25.0

# ---------------------------------------------------------------------------
# Beep Alert Parameters
# ---------------------------------------------------------------------------
BEEP_FREQUENCY_HZ: int = 1500
BEEP_DURATION_MS: int = 600
