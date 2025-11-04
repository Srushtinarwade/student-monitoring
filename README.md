# student-monitoring

# 👀 Student Monitoring System (Beep-Only Alerts)

A real-time AI-based student monitoring system that tracks eye gaze, head pose, phone usage, and focus level using a webcam. When distraction is detected, the system plays a **beep alert** (no MP3 required). Designed for student self-monitoring during study sessions.

---

## 📌 Features

### 🎯 Focus Detection
- Tracks **eye openness (EAR)** to detect blinking vs. drowsiness
- Monitors if eyes are pointed at the screen using **iris tracking**
- **Head pose estimation** to detect if the student looks away
- Short blinks are allowed – no false distraction counts

### 📱 Phone Detection (YOLOv8)
- Detects if a phone is present using YOLOv8
- Uses smoothing to avoid false positives
- Tracks total phone usage time during the session

### 🔊 Alerts & Logging
- **BEEP sound alert** only when distraction is confirmed
- No repetitive sound spam
- Creates a text **session log** with:
  - Total distractions
  - Phone usage time
  - Focus score
  - Session duration

---

## 🖥️ Tech Stack

| Component | Library |
|----------|----------|
| Face Mesh & Eye Tracking | MediaPipe |
| Object Detection (Phone) | YOLOv8 (Ultralytics) |
| Head Pose Estimation | OpenCV + SolvePnP |
| Beep Alerts | winsound / terminal bell |
| Logging | Python text log |

---

## 🚀 Getting Started

### ✅ Requirements

- Python 3.8+
- Webcam
- Windows recommended (for beep sound)
- GPU optional (CPU works fine)

### 📦 Installation

```bash
# 1. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate  # For Windows

# 2. Install dependencies
pip install -r requirements.txt
📂 Student-Monitoring-System/
│── student_monitoring_beep_only.py
│── requirements.txt
│── README.md
│── .gitignore
└── Monitoring_Logs/ (auto-generated)

| Parameter       | Rule                             |
| --------------- | -------------------------------- |
| Eye Open State  | EAR > threshold                  |
| Blink Handling  | Short blinks ignored             |
| Gaze Direction  | Eyes must face screen            |
| Head Pose       | Pitch & yaw within allowed range |
| Phone Detection | If phone seen, not focused       |

OUTPUT
===== SESSION SUMMARY =====
User: Srushti Narwade
Total Time: 21 min 15 sec
Total Distractions: 4
Phone Usage: 13 sec
Focus Score: 92.8%
===========================

📌 Future Improvements

Add voice alerts using TTS

Add dashboard / graphs for focus analytics

Store logs in CSV format for progress tracking

Multi-student support for classroom mode
