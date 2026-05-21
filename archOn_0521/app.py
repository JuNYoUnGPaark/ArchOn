import os
import json
import time
from datetime import datetime

from flask import Flask, render_template
from flask_socketio import SocketIO

from sensor_reader import SensorReader
from decision import DecisionEngine


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

reader = SensorReader()
engine = DecisionEngine(window_seconds=5, fsr_count=18)

running = False
session_data = []


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("start")
def start_monitoring():
    global running, session_data

    if running:
        return

    running = True
    session_data = []

    socketio.start_background_task(sensor_loop)


@socketio.on("stop")
def stop_monitoring():
    global running
    running = False


def sensor_loop():
    global running, session_data

    while running:
        ts = datetime.now().strftime("%H:%M:%S")

        sensor_data = reader.read()
        decision = engine.update(sensor_data)

        item = {
            "time": ts,
            **sensor_data,
            **decision,
        }

        session_data.append(item)

        print("=" * 60)
        print(f"[{ts}]")

        fsr_logs = []
        for i in range(1, 19):
            key = f"fsr{i}_voltage"
            value = item.get(key)

            if value is not None:
                fsr_logs.append(f"FSR{i}={value:.3f}V")

        print(" | ".join(fsr_logs))

        emg_v = item.get("emg_voltage")

        if emg_v is not None:
            print(f"EMG={emg_v:.3f}V")
        else:
            print("EMG=None")

        print(f"Pressure State: {item['pressure_state']}")
        print(f"EMG State: {item['emg_state']}")
        print(f"Feedback: {item['feedback']}")

        socketio.emit("sensor_data", item)

        time.sleep(1)

    save_session()


def save_session():
    if not session_data:
        return

    os.makedirs("data", exist_ok=True)

    filename = "data/session_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=4, ensure_ascii=False)

    print(f"Saved: {filename}")
    socketio.emit("saved", {"filename": filename})


if __name__ == "__main__":
    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=False
    )