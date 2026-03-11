"""
Cross-platform audio alert system.

- **Windows** — uses ``winsound.Beep``
- **macOS** — uses ``afplay`` with a system sound, falling back to terminal bell
- **Linux / other** — terminal bell (``\\a``)
"""

from __future__ import annotations

import logging
import platform
import subprocess
import threading

import config

logger = logging.getLogger(__name__)

_SYSTEM = platform.system()


def _beep_once() -> None:
    """Play a single short beep appropriate for the current OS."""
    try:
        if _SYSTEM == "Windows":
            import winsound  # type: ignore[import-not-found]

            winsound.Beep(config.BEEP_FREQUENCY_HZ, config.BEEP_DURATION_MS)
        elif _SYSTEM == "Darwin":
            # macOS: use afplay with a built-in system sound
            subprocess.run(
                ["afplay", "/System/Library/Sounds/Tink.aiff"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            # Linux / fallback — terminal bell
            print("\a", end="", flush=True)
    except Exception:
        print("\a", end="", flush=True)


class AlertManager:
    """Non-blocking, debounced alert system."""

    def __init__(self) -> None:
        self._active: bool = False

    def trigger(self) -> None:
        """Play a beep in a daemon thread (non-blocking)."""
        if self._active:
            return  # already triggered, waiting for reset
        self._active = True
        t = threading.Thread(target=_beep_once, daemon=True)
        t.start()
        logger.debug("Distraction alert triggered")

    def reset(self) -> None:
        """Reset the alert so it can fire again on the next distraction."""
        if self._active:
            self._active = False
            logger.debug("Alert reset — student refocused")

    @property
    def is_active(self) -> bool:
        return self._active
