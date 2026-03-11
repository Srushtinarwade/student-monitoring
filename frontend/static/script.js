// Connect to Server-Sent Events for live stats
const evtSource = new EventSource("/live_stats");

const badge = document.getElementById("status-badge");
const valState = document.getElementById("val-state");
const valDistractions = document.getElementById("val-distractions");
const valPhone = document.getElementById("val-phone");
const valScore = document.getElementById("val-score");
const progressBar = document.getElementById("focus-progress");
const cameraStatus = document.getElementById("camera-status");

evtSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    // Update Camera Status
    if (data.active) {
        cameraStatus.innerHTML = '<span class="pulse-dot"></span> Camera Active';
        cameraStatus.style.color = 'var(--success)';
    } else {
        cameraStatus.innerHTML = '<span class="pulse-dot" style="background:var(--danger); animation:none"></span> Camera Inactive';
        cameraStatus.style.color = 'var(--danger)';
    }

    // Update Focus Status
    if (!data.active) {
        badge.textContent = "STANDBY";
        badge.className = "status-badge";
        valState.textContent = "Idle";
        valState.style.color = "var(--text-muted)";
    } else if (data.focused) {
        badge.textContent = "FOCUSED";
        badge.className = "status-badge focused";
        valState.textContent = "Focused";
        valState.style.color = "var(--success)";
    } else {
        badge.textContent = "DISTRACTED";
        badge.className = "status-badge distracted";
        valState.textContent = "Distracted";
        valState.style.color = "var(--danger)";
    }

    // Update Metrics
    valDistractions.textContent = data.distractions;
    valPhone.textContent = data.phone_seconds.toFixed(1);
    
    // Update Score Bar
    valScore.textContent = data.focus_score.toFixed(1) + "%";
    progressBar.style.width = data.focus_score + "%";
    
    // Change color based on score
    if (data.focus_score >= 80) {
        progressBar.style.backgroundColor = "var(--success)";
    } else if (data.focus_score >= 50) {
        progressBar.style.backgroundColor = "var(--accent)";
    } else {
        progressBar.style.backgroundColor = "var(--danger)";
    }
};

// Fetch Session History
async function fetchHistory() {
    try {
        const response = await fetch("/api/sessions");
        const sessions = await response.json();
        
        const tbody = document.getElementById("history-body");
        tbody.innerHTML = "";
        
        if (sessions.length === 0) {
            tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; color:var(--text-muted)">No sessions recorded yet. Start monitoring to see history.</td></tr>`;
            return;
        }

        sessions.forEach(session => {
            const tr = document.createElement("tr");
            
            // Format Duration
            const mins = Math.floor(session.duration_seconds / 60);
            const secs = session.duration_seconds % 60;
            const durationArr = [];
            if (mins > 0) durationArr.push(`${mins}m`);
            if (secs > 0 || mins === 0) durationArr.push(`${secs}s`);
            
            // Score Color
            let scoreClass = 'bg-success';
            let scoreColor = 'var(--success)';
            let scoreBg = 'rgba(16, 185, 129, 0.1)';
            if (session.focus_score < 50) {
                scoreColor = 'var(--danger)';
                scoreBg = 'rgba(244, 63, 94, 0.1)';
            } else if (session.focus_score < 80) {
                scoreColor = 'var(--accent)';
                scoreBg = 'rgba(56, 189, 248, 0.1)';
            }
            
            tr.innerHTML = `
                <td style="color:var(--text-muted)">${session.start_time}</td>
                <td>${durationArr.join(' ')}</td>
                <td>${session.distraction_count}</td>
                <td>${session.phone_usage_seconds}s</td>
                <td>
                    <span class="score-badge" style="color:${scoreColor}; background:${scoreBg}">
                        ${session.focus_score.toFixed(1)}%
                    </span>
                </td>
            `;
            tbody.appendChild(tr);
        });
    } catch (error) {
        console.error("Error fetching history:", error);
    }
}

// Initial fetch and bind refresh button
document.addEventListener("DOMContentLoaded", () => {
    fetchHistory();
    
    document.getElementById("refresh-btn").addEventListener("click", () => {
        fetchHistory();
    });

    // Control Buttons
    const btnStart = document.getElementById("btn-start");
    const btnStop = document.getElementById("btn-stop");

    btnStart.addEventListener("click", async () => {
        btnStart.disabled = true;
        try {
            await fetch("/api/start", { method: "POST" });
            btnStop.disabled = false;
        } catch (e) {
            console.error(e);
            btnStart.disabled = false;
        }
    });

    btnStop.addEventListener("click", async () => {
        btnStop.disabled = true;
        try {
            await fetch("/api/stop", { method: "POST" });
            btnStart.disabled = false;
            // Refresh history after a short delay to allow DB save
            setTimeout(fetchHistory, 1000);
        } catch (e) {
            console.error(e);
            btnStop.disabled = false;
        }
    });
});
