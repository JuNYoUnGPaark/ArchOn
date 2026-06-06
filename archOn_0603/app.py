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

phase = "ready"  # ready -> baseline_done -> max_lift_done -> mvic_1_done -> mvic_2_done -> calibration_done -> exercising
busy = False
current_session_id = None
current_jsonl_path = None
all_samples = []
exercise_results = []
current_exercise_samples = []
exercise_started_at = None
abort_requested = False


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


def reset_runtime_for_new_session():
    global current_session_id, current_jsonl_path, all_samples, exercise_results, current_exercise_samples, exercise_started_at, abort_requested
    current_session_id = None
    current_jsonl_path = None
    all_samples = []
    exercise_results = []
    current_exercise_samples = []
    exercise_started_at = None
    abort_requested = False
    engine.reset_calibration()


def clear_exercise_logs_after_recalibration():
    global exercise_results, current_exercise_samples, exercise_started_at
    exercise_results = []
    current_exercise_samples = []
    exercise_started_at = None
    engine.exercise_data = []
    socketio.emit("clear_exercise_logs", {})


def new_session_if_needed():
    global current_session_id, current_jsonl_path
    if current_session_id is None:
        current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_jsonl_path = DATA_DIR / f"session_{current_session_id}_raw.jsonl"


def read_sample(stage="live"):
    sample = reader.read()
    now = datetime.now()

    emg_window = reader.read_emg_window(duration=0.5, fs=None)
    sample.update(emg_window)

    sample["timestamp"] = now.isoformat(timespec="seconds")
    sample["time"] = now.strftime("%H:%M:%S")
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


def emit_action_enabled(enabled):
    socketio.emit("action_state", {"enabled": bool(enabled)})


def phase_label(p):
    return {
        "ready": "기준값 측정",
        "baseline_done": "최대 아치 형성값 측정",
        "max_lift_done": "MVIC 측정",
        "mvic_1_done": "2번째 MVIC 측정",
        "mvic_2_done": "3번째 MVIC 측정",
        "calibration_done": "운동 시작",
        "exercising": "운동 종료",
        "saving": "저장 중"
    }.get(p, "기준값 측정")


def collect_samples(seconds, stage, message, realtime_lift=False):
    global abort_requested
    samples = []
    interval = engine.config.get("timing", {}).get("sample_interval", 1.0)
    steps = max(1, int(round(seconds / interval)))
    for i in range(steps):
        if abort_requested:
            raise InterruptedError("SESSION_RESET_REQUESTED")
        remain = max(0, int(round(seconds - i * interval)))
        loop_start = time.time()
        sample = read_sample(stage=stage)
        append_sample(sample)
        samples.append(sample)

        payload = dict(sample)
        payload["monitoring"] = engine.calibration_monitoring(sample)

        if realtime_lift:
            metrics = engine.realtime_max_lift_metrics(sample)
            payload["max_lift_realtime"] = metrics
            if metrics.get("toe_decrease_percent") is not None:
                emit_status(f"{message} / 현재 엄지 압력 감소율 {metrics['toe_decrease_percent']:.1f}% ({remain}s left)", "info")
            else:
                emit_status(f"{message} ({remain}s left)", "info")
        else:
            emit_status(f"{message} ({remain}s left)", "info")
        socketio.emit("sensor_data", payload)


        elapsed = time.time() - loop_start
        time.sleep(max(0, interval - elapsed))
        if abort_requested:
            raise InterruptedError("SESSION_RESET_REQUESTED")
    return samples


@socketio.on("connect")
def on_connect():
    emit_phase()
    emit_action_enabled(True)
    emit_status("기준값 측정을 누르면 센서 점검 후 바로 기준값을 측정합니다.", "info")


@socketio.on("primary_action")
def primary_action():
    global busy
    if busy:
        emit_status("현재 작업이 진행 중입니다.", "warning")
        return
    if phase == "ready":
        socketio.start_background_task(system_check_then_baseline_task)
    elif phase == "baseline_done":
        socketio.start_background_task(max_lift_task)
    elif phase in ["max_lift_done", "mvic_1_done", "mvic_2_done"]:
        socketio.start_background_task(mvic_task)
    elif phase == "calibration_done":
        start_exercise()
    elif phase == "exercising":
        stop_exercise()


@socketio.on("end_summary")
def end_summary():
    if busy:
        emit_status("측정 중에는 먼저 세션 초기화를 선택하세요.", "warning")
        return
    save_summary_and_reset()




@socketio.on("reset_session")
def reset_session(data=None):
    global phase, busy, abort_requested
    reason = (data or {}).get("reason", "manual")

    if busy:
        abort_requested = True
        emit_status("진행 중인 측정을 중단하고 세션을 초기화합니다.", "warning")
        socketio.emit("ui_resetting", {"reason": reason})
        return

    phase = "ready"
    reset_runtime_for_new_session()
    socketio.emit("clear_exercise_logs", {})
    emit_phase()
    emit_action_enabled(True)
    emit_status("세션이 초기화되었습니다. 기준값 측정부터 다시 시작하세요.", "success")


@socketio.on("remeasure")
def remeasure(data):
    global phase, busy
    if busy:
        emit_status("현재 작업이 진행 중입니다.", "warning")
        return

    target = (data or {}).get("target")
    if target == "baseline":
        clear_exercise_logs_after_recalibration()
        engine.reset_calibration()
        phase = "ready"
        emit_phase()
        socketio.start_background_task(system_check_then_baseline_task)
    elif target == "max_lift":
        if engine.baseline is None:
            emit_status("기준값이 먼저 필요합니다.", "warning")
            return
        clear_exercise_logs_after_recalibration()
        engine.max_lift = None
        engine.mvic_trials = []
        engine.mvic = None
        phase = "baseline_done"
        emit_phase()
        socketio.start_background_task(max_lift_task)
    elif target == "mvic":
        if engine.baseline is None or engine.max_lift is None:
            emit_status("기준값과 최대 아치 형성값이 먼저 필요합니다.", "warning")
            return
        clear_exercise_logs_after_recalibration()
        engine.mvic_trials = []
        engine.mvic = None
        phase = "max_lift_done"
        emit_phase()
        socketio.start_background_task(mvic_task)
    else:
        emit_status("알 수 없는 재측정 요청입니다.", "warning")


def system_check_then_baseline_task():
    global phase, busy
    busy = True
    emit_action_enabled(False)
    new_session_if_needed()
    try:
        check_seconds = engine.config.get("timing", {}).get("system_check_seconds", 3)
        check_samples = collect_samples(check_seconds, "system_check", "기기 점검 중입니다")
        check = engine.system_check(check_samples)
        socketio.emit("system_check_result", check)
        if check.get("need_reconnect"):
            socketio.emit("reconnect_required", check)
            emit_status("재연결 필요: " + ", ".join(check.get("disconnected", [])), "error")
            phase = "ready"
            emit_phase()
            return
        warning_text = ""
        if check.get("disconnected") or check.get("warnings"):
            warning_text = f" 일부 센서 확인 필요: {', '.join(check.get('disconnected', []) + check.get('warnings', []))}."
        seconds = engine.config.get("timing", {}).get("baseline_seconds", 5)
        baseline_samples = collect_samples(seconds, "baseline", "발을 올리고 가만히 자세를 유지하세요")
        baseline = engine.compute_baseline(baseline_samples)
        socketio.emit("baseline_result", baseline)
        save_json(f"session_{current_session_id}_baseline.json", baseline)
        phase = "baseline_done"
        emit_status("기준값 저장 완료." + warning_text + " 이제 최대 아치 형성값을 측정하세요.", "success")
        emit_phase()
    except InterruptedError:
        emit_status("작업이 중단되었습니다. 기준값 측정부터 다시 시작하세요.", "warning")
    except Exception as e:
        emit_status(f"기준값 측정 오류: {e}", "error")
    finally:
        if abort_requested:
            reset_runtime_for_new_session()
            phase = "ready"
            socketio.emit("clear_exercise_logs", {})
            emit_phase()
        busy = False
        emit_action_enabled(True)


def max_lift_task():
    global phase, busy
    busy = True
    emit_action_enabled(False)
    try:
        seconds = engine.config.get("timing", {}).get("max_lift_seconds", 5)
        samples = collect_samples(seconds, "max_lift", "엄지발가락을 최대한 들어올리세요", realtime_lift=True)
        result = engine.compute_max_lift(samples)
        socketio.emit("max_lift_result", result)
        save_json(f"session_{current_session_id}_max_lift.json", result)
        if result.get("_ok"):
            phase = "max_lift_done"
            emit_status("최대 아치 형성값 저장 완료. MVIC 1회차를 측정하세요.", "success")
        else:
            phase = "baseline_done"
            emit_status("최대 아치 형성값 재측정 필요: " + " ".join(result.get("_feedback", [])), "warning")
        emit_phase()
    except InterruptedError:
        emit_status("작업이 중단되었습니다. 기준값 측정부터 다시 시작하세요.", "warning")
    except Exception as e:
        emit_status(f"최대 아치 형성값 측정 오류: {e}", "error")
    finally:
        if abort_requested:
            reset_runtime_for_new_session()
            phase = "ready"
            socketio.emit("clear_exercise_logs", {})
            emit_phase()
        busy = False
        emit_action_enabled(True)


def mvic_task():
    global phase, busy
    busy = True
    emit_action_enabled(False)
    try:
        current_trial = len(engine.mvic_trials) + 1
        seconds = engine.config.get("timing", {}).get("mvic_seconds", 3)
        samples = collect_samples(seconds, f"mvic_{current_trial}", f"앉아서 발바닥을 바닥에 붙이고 엄지발가락을 최대한 벌린 상태에서 동시에 엄지발가락을 좌에서 우 방향으로 밀어주세요. MVIC {current_trial}회차")
        result = engine.add_mvic_trial(samples)
        socketio.emit("mvic_result", result)
        save_json(f"session_{current_session_id}_mvic_trial_{current_trial:02d}.json", {"result": result, "samples": samples})
        if result.get("ok"):
            save_json(f"session_{current_session_id}_mvic_summary.json", result)
            phase = "calibration_done"
            emit_status("MVIC 3회 측정 완료. 운동을 시작할 수 있습니다.", "success")
        else:
            count = len(engine.mvic_trials)
            if result.get("reset"):
                phase = "max_lift_done"
                emit_status("MVIC 재측정 필요: " + " ".join(result.get("feedback", [])), "warning")
            elif count == 1:
                phase = "mvic_1_done"
                emit_status("MVIC 1회차 완료. 2회차를 측정하세요.", "info")
            elif count == 2:
                phase = "mvic_2_done"
                emit_status("MVIC 2회차 완료. 3회차를 측정하세요.", "info")
        emit_phase()
    except InterruptedError:
        emit_status("작업이 중단되었습니다. 기준값 측정부터 다시 시작하세요.", "warning")
    except Exception as e:
        emit_status(f"MVIC 측정 오류: {e}", "error")
    finally:
        if abort_requested:
            reset_runtime_for_new_session()
            phase = "ready"
            socketio.emit("clear_exercise_logs", {})
            emit_phase()
        busy = False
        emit_action_enabled(True)


def start_exercise():
    global phase, current_exercise_samples, exercise_started_at
    if engine.baseline is None or engine.max_lift is None or engine.mvic is None:
        emit_status("기준값, 최대 아치 형성값, MVIC를 먼저 측정하세요.", "warning")
        return
    phase = "exercising"
    current_exercise_samples = []
    exercise_started_at = time.time()
    engine.start_exercise()
    emit_status("운동 측정을 시작합니다. 3초 후 운동 종료 버튼을 누를 수 있습니다.", "success")
    emit_phase()
    socketio.emit("exercise_lock", {"locked_seconds": engine.config.get("timing", {}).get("exercise_min_seconds", 3)})
    socketio.start_background_task(exercise_loop)


def exercise_loop():
    global current_exercise_samples
    while phase == "exercising":
        loop_start = time.time()

        if abort_requested:
            break

        sample = read_sample(stage="exercise")
        append_sample(sample)
        current_exercise_samples.append(sample)
        engine.update_exercise(sample)

        live = engine.live_exercise_feedback()
        socketio.emit(
            "sensor_data",
            {
                **sample,
                "feedback": live.get("coaching_message", "운동 중입니다."),
                "live_exercise": live,
                "monitoring": engine.calibration_monitoring(sample),
            }
        )
        socketio.emit("live_exercise", live)

        interval = engine.config.get("timing", {}).get("sample_interval", 1.0)
        elapsed = time.time() - loop_start
        time.sleep(max(0, interval - elapsed))


def stop_exercise():
    global phase, exercise_results
    min_seconds = engine.config.get("timing", {}).get("exercise_min_seconds", 3)
    if exercise_started_at is not None and time.time() - exercise_started_at < min_seconds:
        emit_status(f"운동 시작 후 최소 {min_seconds}초가 지나야 종료할 수 있습니다.", "warning")
        return
    previous_phase = phase
    phase = "saving"
    emit_phase()
    emit_action_enabled(False)
    try:
        result = engine.analyze_exercise()
        result["exercise_index"] = len(exercise_results) + 1
        result["sample_count"] = len(current_exercise_samples)
        result["ended_at"] = datetime.now().isoformat(timespec="seconds")
        exercise_results.append(result)
        save_json(f"session_{current_session_id}_exercise_{result['exercise_index']:02d}.json", {"result": result, "samples": current_exercise_samples})
        phase = "calibration_done"
        socketio.emit("exercise_result", result)
        emit_status(f"{result['exercise_index']}회차 결과: {result['grade']} / {result['score']}점. {result.get('main_comment', '')}", "success")
    except Exception as e:
        phase = previous_phase
        emit_status(f"운동 분석 오류: {e}", "error")
    finally:
        emit_phase()
        emit_action_enabled(True)


def save_summary_and_reset():
    global phase
    if current_session_id is None:
        emit_status("저장할 세션이 없습니다.", "warning")
        return
    summary = engine.summarize_exercises(exercise_results)
    save_json(f"session_{current_session_id}_summary.json", {
        "session_id": current_session_id,
        "baseline": engine.baseline,
        "max_lift": engine.max_lift,
        "mvic": engine.mvic,
        "mvic_trials": engine.mvic_trials,
        "exercise_count": len(exercise_results),
        "exercise_results": exercise_results,
        "summary": summary,
        "all_samples_file": current_jsonl_path.name if current_jsonl_path else None,
        "saved_at": datetime.now().isoformat(timespec="seconds")
    })
    socketio.emit("summary_result", summary)
    emit_status("전체 운동 종료 및 요약 저장 완료. 새 세션을 바로 시작할 수 있습니다.", "success")
    reset_runtime_for_new_session()
    phase = "ready"
    emit_phase()
    emit_action_enabled(True)


def save_json(filename, obj):
    path = DATA_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)
    socketio.emit("saved", {"filename": f"data/{filename}"})


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
