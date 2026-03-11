"""
Camera module for the Flask web dashboard.

Wraps the original main.py OpenCV loop into a background generator
that yields JPEG frames and updates a global stats dictionary for SSE.
"""

import cv2
import time
import logging

import config
from utils.alerts import AlertManager
from detectors.face import FaceDetector
from detectors.head_pose import HeadPoseEstimator
from detectors.phone import PhoneDetector
from utils.overlay import draw_face, draw_status
from core.session import SessionTracker

logger = logging.getLogger(__name__)

# Global state dictionary for the web dashboard to poll via SSE
live_stats = {
    "focused": True,
    "phone_seconds": 0.0,
    "distractions": 0,
    "focus_score": 100.0,
    "active": False,
}

class CameraStream:
    def __init__(self):
        self.cap = None
        self.face_det = None
        self.head_pose = None
        self.phone_det = None
        self.alert = None
        self.session = None
        
        self.consec_not_focused = 0
        self.consec_focused = 0
        self.distraction_count = 0
        self.running = False

    def start(self):
        self.running = True
        
    def stop(self):
        self.running = False

    def _init_tracking(self):
        logger.info("Starting camera stream...")
        self.cap = cv2.VideoCapture(config.WEBCAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        
        self.face_det = FaceDetector()
        self.head_pose = HeadPoseEstimator()
        self.phone_det = PhoneDetector()
        self.alert = AlertManager()
        self.session = SessionTracker()
        
        self.consec_not_focused = 0
        self.consec_focused = 0
        self.distraction_count = 0
        
        live_stats["active"] = True
        live_stats["focused"] = True
        live_stats["phone_seconds"] = 0.0
        live_stats["distractions"] = 0
        live_stats["focus_score"] = 100.0

    def _teardown_tracking(self):
        live_stats["active"] = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.face_det:
            self.face_det.close()
            self.face_det = None
        
        if self.session and self.phone_det:
            phone_secs = self.phone_det.finalize()
            self.session.save_log(phone_secs)
            self.session = None
            self.phone_det = None

    def generate_frames(self):
        # We start inactive. Yield black frames until explicitly started.
        import numpy as np
        blank_frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
        cv2.putText(blank_frame, "SESSION INACTIVE", (config.FRAME_WIDTH//2 - 150, config.FRAME_HEIGHT//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        _, blank_buffer = cv2.imencode('.jpg', blank_frame)
        blank_bytes = blank_buffer.tobytes()
        blank_yield = (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + blank_bytes + b'\r\n')

        is_tracking = False

        try:
            while True:
                if self.running and not is_tracking:
                    self._init_tracking()
                    is_tracking = True
                    
                elif not self.running and is_tracking:
                    self._teardown_tracking()
                    is_tracking = False

                if not is_tracking:
                    yield blank_yield
                    time.sleep(0.1)
                    continue
                    
                if not self.cap or not self.cap.isOpened():
                    yield blank_yield
                    time.sleep(0.1)
                    continue
                    
                ret, frame = self.cap.read()
                if not ret:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # ----- Detection -------------------------------------------
                face = self.face_det.process(rgb)
                phone = self.phone_det.detect(frame)

                # ----- Focus decision --------------------------------------
                focused = False
                if face.detected:
                    hp = self.head_pose.estimate(
                        face.landmarks,
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
                    self.consec_not_focused = 0
                    self.consec_focused += 1
                else:
                    self.consec_focused = 0
                    if not face.short_blink:
                        self.consec_not_focused += 1

                if self.consec_not_focused >= config.DISTRACTION_FRAMES and not self.alert.is_active:
                    self.alert.trigger()
                    self.distraction_count += 1
                    self.session.record_distraction()

                if self.alert.is_active and self.consec_focused >= config.FOCUS_RESET_FRAMES:
                    self.alert.reset()

                self.session.tick(focused=focused)

                # Update global stats for SSE
                live_stats["focused"] = focused
                live_stats["phone_seconds"] = round(phone.visible_seconds, 1)
                live_stats["distractions"] = self.distraction_count
                live_stats["focus_score"] = round(self.session.focus_percent, 1)

                # ----- Rendering ------------------------------------------
                draw_face(frame, face)
                draw_status(
                    frame,
                    focused=focused,
                    phone_detected=phone.detected,
                    phone_seconds=phone.visible_seconds,
                    distraction_count=self.distraction_count,
                )

                # Encode to JPEG for web streaming
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                    
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        finally:
            if is_tracking:
                self._teardown_tracking()
