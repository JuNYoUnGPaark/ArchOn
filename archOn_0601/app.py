import os
import json
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO

from sensor_reader import SensorReader
from decision import DecisionEngine


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

reader = SensorReader()
engine = DecisionEngine(config_path=str(BASE_DIR / "sensor_config.json"))

phase = "idle"
busy = False
current_session_id = None
current_jsonl_path = None

all_samples = []
baseline_samples = []
max_samples = []
exercise_results = []
current_exercise_samples = []


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/data")
def list_data():
    files = sorted(DATA_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    html = "<h2>Arch-On Data Files</h2><ul>"
    for f in files:
        html += f'<li><a href="/data/{f.name}" download>{f.name}</a></li>'
    html += "</ul>"
    return html


@app.route("/data/<path:filename>")
def download_data(filename):
    return send_from_directory(DATA_DIR, filename, as_attachment=True)


def new_session_if_needed():
    global current_session_id, current_jsonl_path
    if current_session_id is None:
        current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_jsonl_path = DATA_DIR / f"session_{current_session_id}_raw.jsonl"


def read_sample(stage="live"):
    sample = reader.read()
    sample["timestamp"] = datetime.now().isoformat(timespec="seconds")
    sample["time"] = datetime.now().strftime("%H:%M:%S")
    sample["stage"] = stage
    return sample


def append_sample(sample):
    new_session_if_needed()
    all_samples.append(sample)
    with open(current_jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def emit_status(message, level="info"):
    socketio.emit("status", {"message": message, "level": level})


def emit_phase():
    socketio.emit("phase", {"phase": phase, "label": phase_label(phase)})


def phase_label(p):
    return {
        "idle": "시작",
        "checked": "기준값 측정",
        "baseline_done": "최대값 측정",
        "max_done": "운동 시작",
        "exercising": "운동 종료",
    }.get(p, "시작")


def collect_samples(seconds, stage, message):
    samples = []
    interval = engine.config.get("timing", {}).get("sample_interval", 1.0)
    steps = int(seconds / interval)

    for i in range(steps):
        remain = int(seconds - i * interval)
        emit_status(f"{message} ({remain}초 남음)", "info")

        sample = read_sample(stage=stage)
        append_sample(sample)
        samples.append(sample)
        socketio.emit("sensor_data", sample)

        time.sleep(interval)

    return samples


@socketio.on("primary_action")
def primary_action():
    global phase, busy

    if busy:
        emit_status("현재 작업이 진행 중입니다.", "warning")
        return

    if phase == "idle":
        socketio.start_background_task(system_check_task)
    elif phase == "checked":
        socketio.start_background_task(baseline_task)
    elif phase == "baseline_done":
        socketio.start_background_task(max_task)
    elif phase == "max_done":
        start_exercise()
    elif phase == "exercising":
        stop_exercise()


@socketio.on("end_summary")
def end_summary():
    save_summary()


def system_check_task():
    global phase, busy, current_session_id, current_jsonl_path
    busy = True

    if current_session_id is None:
        new_session_if_needed()

    try:
        seconds = engine.config.get("timing", {}).get("system_check_seconds", 3)
        samples = collect_samples(seconds, "system_check", "시스템 검사 중")

        result = engine.system_check(samples)
        socketio.emit("system_check_result", result)

        if result["ok"]:
            phase = "checked"
            emit_status("시스템 정상", "success")
        else:
            phase = "idle"
            emit_status("센서 불량 감지: " + ", ".join(result["disconnected"]), "error")

        emit_phase()

    except Exception as e:
        emit_status(f"시스템 검사 오류: {e}", "error")
    finally:
        busy = False


def baseline_task():
    global phase, busy, baseline_samples
    busy = True

    try:
        seconds = engine.config.get("timing", {}).get("baseline_seconds", 10)
        baseline_samples = collect_samples(seconds, "baseline", "10초 평균 기준값 측정 중입니다. 가만히 유지하세요.")

        baseline = engine.compute_baseline(baseline_samples)
        save_json(f"session_{current_session_id}_baseline.json", baseline)

        phase = "baseline_done"
        socketio.emit("baseline_result", baseline)
        emit_status("기준값 측정 완료", "success")
        emit_phase()

    except Exception as e:
        emit_status(f"기준값 측정 오류: {e}", "error")
    finally:
        busy = False


def max_task():
    global phase, busy, max_samples
    busy = True

    try:
        timing = engine.config.get("timing", {})
        emg_seconds = timing.get("max_emg_seconds", 10)
        toe_seconds = timing.get("max_toe_seconds", 10)

        samples1 = collect_samples(emg_seconds, "max_emg", "무지외전근에 최대로 힘을 주세요.")
        samples2 = collect_samples(toe_seconds, "max_toe", "엄지발가락을 최대로 들어보세요.")

        max_samples = samples1 + samples2
        max_values = engine.compute_max_values(max_samples)
        save_json(f"session_{current_session_id}_max.json", max_values)

        phase = "max_done"
        socketio.emit("max_result", max_values)
        emit_status("최대값 측정 완료. 운동을 시작할 수 있습니다.", "success")
        emit_phase()

    except Exception as e:
        emit_status(f"최대값 측정 오류: {e}", "error")
    finally:
        busy = False


def start_exercise():
    global phase, current_exercise_samples

    if engine.baseline is None or engine.max_values is None:
        emit_status("기준값과 최대값을 먼저 측정하세요.", "warning")
        return

    phase = "exercising"
    current_exercise_samples = []
    engine.start_exercise()

    emit_status("운동 측정을 시작합니다.", "success")
    emit_phase()
    socketio.start_background_task(exercise_loop)


def exercise_loop():
    global phase, current_exercise_samples

    while phase == "exercising":
        sample = read_sample(stage="exercise")
        append_sample(sample)
        current_exercise_samples.append(sample)
        engine.update_exercise(sample)

        live = {
            **sample,
            "feedback": "운동 중입니다. 운동 종료를 누르면 이 구간을 분석합니다."
        }
        socketio.emit("sensor_data", live)

        time.sleep(engine.config.get("timing", {}).get("sample_interval", 1.0))


def stop_exercise():
    global phase, exercise_results

    result = engine.analyze_exercise()
    result["exercise_index"] = len(exercise_results) + 1
    result["sample_count"] = len(current_exercise_samples)
    result["ended_at"] = datetime.now().isoformat(timespec="seconds")

    exercise_results.append(result)

    save_json(f"session_{current_session_id}_exercise_{result['exercise_index']:02d}.json", {
        "result": result,
        "samples": current_exercise_samples
    })

    phase = "max_done"
    socketio.emit("exercise_result", result)
    emit_status(f"{result['exercise_index']}회차 운동 분석 완료: {result['grade']} / {result['score']}점", "success")
    emit_phase()


def save_summary():
    summary = engine.summarize_exercises(exercise_results)

    save_json(f"session_{current_session_id}_summary.json", {
        "session_id": current_session_id,
        "baseline": engine.baseline,
        "max_values": engine.max_values,
        "exercise_results": exercise_results,
        "summary": summary,
        "all_samples_file": current_jsonl_path.name if current_jsonl_path else None,
    })

    socketio.emit("summary_result", summary)
    emit_status("전체 종료 및 요약 저장 완료", "success")


def save_json(filename, obj):
    path = DATA_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)
    socketio.emit("saved", {"filename": f"data/{filename}"})


if __name__ == "__main__":
    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=False,
        allow_unsafe_werkzeug=True
    )