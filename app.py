"""
Flask Web Dashboard for Student Monitoring System.
"""

from flask import Flask, render_template, Response, jsonify
import config
from utils import database
from core.camera import CameraStream, live_stats
import os

app = Flask(
    __name__,
    template_folder="frontend/templates",
    static_folder="frontend/static"
)
camera_stream = CameraStream()

@app.route("/")
def index():
    """Render the main dashboard."""
    # Ensure DB exists
    database.init_db()
    return render_template("index.html", user=config.USERNAME)

@app.route("/video_feed")
def video_feed():
    """MJPEG streaming route."""
    return Response(
        camera_stream.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/live_stats")
def stream_stats():
    """Server-Sent Events (SSE) for live tracking stats."""
    def generate():
        import json
        import time
        while True:
            # Send current global stats
            yield f"data: {json.dumps(live_stats)}\n\n"
            time.sleep(0.5)

    return Response(generate(), mimetype="text/event-stream")

@app.route("/api/sessions")
def get_sessions():
    """API to fetch history of past sessions."""
    sessions = database.get_recent_sessions(limit=10)
    return jsonify(sessions)

@app.route("/api/start", methods=["POST"])
def start_session():
    """API to start the camera session."""
    if not camera_stream.running:
        camera_stream.start()
    return jsonify({"status": "started"})

@app.route("/api/stop", methods=["POST"])
def stop_session():
    """API to stop the camera session and save logs."""
    if camera_stream.running:
        camera_stream.stop()
    return jsonify({"status": "stopped"})

if __name__ == "__main__":
    # Ensure logs dir / db exist
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    database.init_db()
    
    print(f"Starting web dashboard for {config.USERNAME}...")
    print("Open http://127.0.0.1:5000 in your browser.")
    
    # Run server (threaded for SSE + Video)
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)
