"""
Student Monitoring System — main entry point.

Run with:
    python main.py

Press 'q' in the OpenCV window to stop the session.
"""

from __future__ import annotations

import logging
import sys

import cv2

import config
from alerts import AlertManager
from detectors.face import FaceDetector
from detectors.head_pose import HeadPoseEstimator
from detectors.phone import PhoneDetector
from overlay import draw_face, draw_status
from session import SessionTracker

logger = logging.getLogger(__name__)

WINDOW_TITLE = "Student Monitoring — Press q to quit"


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    """Run the monitoring loop."""
    _configure_logging()

    logger.info("Initialising detectors…")
    face_det = FaceDetector()
    head_pose = HeadPoseEstimator()
    phone_det = PhoneDetector()
    alert = AlertManager()
    session = SessionTracker()

    # Smoothing counters
    consec_not_focused: int = 0
    consec_focused: int = 0
    distraction_count: int = 0

    # Open webcam
    cap = cv2.VideoCapture(config.WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        logger.error("Cannot open webcam at index %d", config.WEBCAM_INDEX)
        sys.exit(1)

    logger.info("Monitoring started for user '%s'. Press 'q' to quit.", config.USERNAME)
    logger.info(
        "Distraction threshold: %d frames (~%.1f s)",
        config.DISTRACTION_FRAMES,
        config.DISTRACTION_SECONDS,
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read webcam frame — stopping.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ----- Detection -------------------------------------------
            face = face_det.process(rgb)
            phone = phone_det.detect(frame)

            # ----- Focus decision --------------------------------------
            focused = False
            if face.detected:
                hp = head_pose.estimate(
                    face.face_landmarks.landmark,  # type: ignore[union-attr]
                    config.FRAME_WIDTH,
                    config.FRAME_HEIGHT,
                )
                head_ok = True
                if hp is not None:
                    pitch, yaw, _ = hp
                    head_ok = HeadPoseEstimator.is_head_ok(pitch, yaw)

                focused = (
                    not face.long_eye_closure
                    and face.eyes_open
                    and face.gaze_centered
                    and head_ok
                    and not phone.detected
                )

            # ----- Smoothing -------------------------------------------
            if focused:
                consec_not_focused = 0
                consec_focused += 1
            else:
                consec_focused = 0
                # Don't penalise short blinks
                if not face.short_blink:
                    consec_not_focused += 1

            # Trigger alert on sustained distraction
            if (
                consec_not_focused >= config.DISTRACTION_FRAMES
                and not alert.is_active
            ):
                alert.trigger()
                distraction_count += 1
                session.record_distraction()

            # Reset alert after sustained re-focus
            if alert.is_active and consec_focused >= config.FOCUS_RESET_FRAMES:
                alert.reset()

            session.tick(focused=focused)

            # ----- Rendering ------------------------------------------
            draw_face(frame, face)
            draw_status(
                frame,
                focused=focused,
                phone_detected=phone.detected,
                phone_seconds=phone.visible_seconds,
                distraction_count=distraction_count,
            )

            cv2.imshow(WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        face_det.close()
        phone_seconds = phone_det.finalize()

        # Session summary
        log_path = session.save_log(phone_seconds)
        lines = session.get_summary_lines(phone_seconds)

        print("\n===== SESSION SUMMARY =====")
        print("\n".join(lines))
        print(f"Log saved to: {log_path}")
        print("===========================")


if __name__ == "__main__":
    main()
