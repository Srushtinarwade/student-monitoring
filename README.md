# 👀 Student Monitoring System

A real-time AI-based student monitoring system that tracks **eye gaze**, **head pose**, **phone usage**, and **focus level** using a webcam. When distraction is detected, the system plays a **beep alert**. Designed for student self-monitoring during study sessions.

---

## ✨ Features

| Feature | How it works |
|---------|-------------|
| **Eye tracking** | EAR (Eye Aspect Ratio) detects blinks vs drowsiness |
| **Gaze detection** | Iris tracking checks if eyes point at the screen |
| **Head pose** | OpenCV `solvePnP` detects if the student looks away |
| **Phone detection** | YOLOv8-nano with temporal smoothing avoids false positives |
| **Beep alerts** | Cross-platform (Windows / macOS / Linux), fires once per distraction event |
| **Session logging** | Text logs and SQLite database (`sessions.db`) tracking distractions, focus score & duration |

---

## 📁 Repository Structure

The project has been refactored into a highly modular, layered architecture:

```
student-monitoring/
├── app.py                # Flask server, API & web routes
├── config.py             # Global constants & tunable thresholds
├── core/                 # Core Business Logic Layer
│   ├── camera.py         # Main loop yielding webcam frames & state
│   └── session.py        # Focus tracking & metric accumulation
├── utils/                # Utility & Infrastructure Layer
│   ├── alerts.py         # Cross-platform audio beeps
│   ├── database.py       # SQLite connection & logging
│   └── overlay.py        # OpenCV drawing / HUD rendering
├── detectors/            # Machine Learning Models Layer
│   ├── face.py           # MediaPipe Tasks API (Landmarks, Blinks)
│   ├── head_pose.py      # OpenCV solvePnP (Pitch/Yaw)
│   └── phone.py          # YOLOv8-nano (Object Detection)
├── frontend/             # User Interface Layer
│   ├── templates/        # HTML Dashboard structure
│   └── static/           # Premium CSS and JS
```├── pyproject.toml      # Project metadata & dependencies
├── .gitignore
└── README.md

### File & Folder Descriptions

| File / Folder | What it does |
|---------------|-------------|
| `main.py` | The app's entry point. Opens the webcam, runs the detection loop (capture → detect → evaluate focus → alert → render), and prints a session summary on exit. |
| `config.py` | Stores every tuneable setting (EAR threshold, gaze limits, head pose limits, beep duration, timing windows). Override user-specific values via environment variables (`SM_USERNAME`, `SM_LOG_DIR`, `SM_WEBCAM_INDEX`). |
| `alerts.py` | Plays a non-blocking beep when distraction is confirmed (using `winsound`, `afplay`, or terminal bell). |
| `session.py` | Tracks per-session metrics and writes summaries to both a text log file and the SQLite database. |
| `database.py` | SQLite database logic for persistent session history used by the web dashboard. |
| `overlay.py` | Handles all OpenCV drawing: face mesh visualization, iris centre dots, eye bounding boxes, and the on-screen HUD (focus status, phone time, distraction count). |
| `detectors/` | A package containing the three detection modules: |
| `detectors/face.py` | Uses MediaPipe FaceMesh to compute the Eye Aspect Ratio (EAR), track blinks vs. drowsiness, detect iris position, and check gaze direction. Returns a `FaceAnalysis` dataclass. |
| `detectors/head_pose.py` | Estimates head orientation (pitch, yaw, roll) using OpenCV's `solvePnP` with 6 facial landmark points mapped to a 3D face model. |
| `detectors/phone.py` | Runs YOLOv8-nano for phone detection with frame-skipping (for performance) and temporal smoothing (to avoid false positives). Tracks cumulative phone-visible time. |
| `pyproject.toml` | Project metadata and dependency list — used by `pip install -e .` to install everything. |

---

## 🔄 How It Works

1. **Webcam capture** — `main.py` opens the webcam and enters a frame-by-frame loop
2. **Face analysis** — Each frame is passed to `FaceDetector` which computes EAR (are eyes open?), iris gaze (looking at screen?), and blink status (short blink or prolonged closure?)
3. **Head pose check** — `HeadPoseEstimator` calculates pitch & yaw to see if the student is looking away
4. **Phone detection** — `PhoneDetector` runs YOLOv8 every few frames to check for a phone, with smoothing to avoid false positives
5. **Focus decision** — The student is considered **focused** only if: eyes are open (not a long closure), gaze is centred, head is facing forward, and no phone is detected
6. **Alert** — If distraction is sustained for ~1.3 seconds, `AlertManager` plays a beep (once, not repeated). Alert resets after ~0.35 seconds of re-focusing
7. **Session log** — On stopping the session, a summary (duration, distractions, phone time, focus score) is saved to the text log and the SQLite database.

---

## 🚀 Getting Started

### Requirements

- Python 3.9+
- Webcam
- Works on **Windows**, **macOS**, and **Linux**
- GPU optional (CPU works fine)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Srushtinarwade/student-monitoring.git
cd student-monitoring

# 2. Create a virtual environment
python -m venv .venv

# On macOS / Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# 3. Download the required MediaPipe AI Model
# (This file is required for face and eye tracking to work!)
curl -sSL -o face_landmarker.task "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

# 4. Install the project
pip install -e .
```

### Usage

```bash
# Start the web dashboard server
python app.py

# Open http://127.0.0.1:5000 in your browser to monitor focus
```

*(Alternatively, run the native OpenCV headless script with `python main.py` and press 'q' to stop.)*

### Configuration

Override settings via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SM_USERNAME` | OS username | Name shown in session logs |
| `SM_LOG_DIR` | `~/.student_monitoring/logs` | Where session logs are saved |
| `SM_WEBCAM_INDEX` | `0` | Webcam device index |

All other thresholds (EAR, gaze, head pose, timing) are in [`config.py`](config.py).

---

## 🎯 Detection Logic

| Parameter | Rule |
|-----------|------|
| Eye Open State | EAR > threshold (default 0.18) |
| Blink Handling | Short blinks (< 0.25s) are ignored |
| Gaze Direction | Iris must be centred within eye bounding box |
| Head Pose | Pitch & yaw must be within ±25° |
| Phone Detection | YOLOv8-nano, confirmed after ~0.5s of visibility |
| Distraction Alert | Triggered after ~1.3s of continuous distraction |

---

## 📊 Sample Output

```
===== SESSION SUMMARY =====
Session Start: 2025-11-04 15:34:03
User: Srushti Narwade
Total Time: 1 min 55 sec
Total Distractions: 3
Phone Usage (seconds): 0
Focus Score: 63.8%
===========================
```

---

## 📌 Future Improvements

- Voice alerts using TTS via the browser
- Advanced historical graphs for focus analytics
- Multi-student support for classroom mode

---

## 🛠️ Tech Stack

| Component | Library |
|-----------|---------|
| Face mesh & eye tracking | [MediaPipe](https://google.github.io/mediapipe/) |
| Object detection (phone) | [YOLOv8](https://docs.ultralytics.com/) (Ultralytics) |
| Head pose estimation | OpenCV `solvePnP` |
| Audio alerts | `winsound` / `afplay` / terminal bell |
| Session logging | Python `pathlib` + SQLite |
