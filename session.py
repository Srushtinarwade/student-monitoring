"""
Session tracking and log persistence.

Records focus frames, distraction events, phone usage, and produces a
human-readable summary that is both printed and saved to a text file.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import config

logger = logging.getLogger(__name__)


class SessionTracker:
    """Accumulates focus metrics over a monitoring session."""

    def __init__(self) -> None:
        self._start_time: float = time.time()
        self._start_dt: datetime = datetime.now()
        self._total_frames: int = 0
        self._focused_frames: int = 0
        self._distraction_count: int = 0

    # ------------------------------------------------------------------
    # Frame-level updates
    # ------------------------------------------------------------------

    def tick(self, *, focused: bool) -> None:
        """Call once per frame to update counters."""
        self._total_frames += 1
        if focused:
            self._focused_frames += 1

    def record_distraction(self) -> None:
        """Increment the distraction counter (call once per event, not per frame)."""
        self._distraction_count += 1

    # ------------------------------------------------------------------
    # Summary / Logging
    # ------------------------------------------------------------------

    @property
    def duration_seconds(self) -> float:
        return time.time() - self._start_time

    @property
    def focus_percent(self) -> float:
        if self._total_frames == 0:
            return 0.0
        return self._focused_frames / self._total_frames * 100.0

    def get_summary_lines(self, phone_seconds: float) -> list[str]:
        """Return the session summary as a list of strings."""
        dur = self.duration_seconds
        dur_min = int(dur // 60)
        dur_sec = int(dur % 60)
        return [
            f"Session Start: {self._start_dt:%Y-%m-%d %H:%M:%S}",
            f"User: {config.USERNAME}",
            f"Total Time: {dur_min} min {dur_sec} sec",
            f"Total Distractions: {self._distraction_count}",
            f"Phone Usage (seconds): {int(phone_seconds)}",
            f"Focus Score: {self.focus_percent:.1f}%",
        ]

    def save_log(self, phone_seconds: float) -> Path:
        """Write the session log to disk and return the file path."""
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = self._start_dt.strftime("%Y%m%d_%H%M%S")
        # Use first word of username for filename readability
        first_name = config.USERNAME.split()[0] if config.USERNAME else "user"
        log_file = config.LOG_DIR / f"session_{first_name}_{timestamp}.txt"

        lines = self.get_summary_lines(phone_seconds)
        log_file.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Session log saved to %s", log_file)
        return log_file
