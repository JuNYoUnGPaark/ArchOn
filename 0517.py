"""
EMG N개 + 압력센서 N개 실시간 모니터링 기본 코드

현재 코드의 목적
1. EMG 센서 N개, Pressure 센서 N개 값을 받아온다.
2. 최근 데이터를 버퍼에 저장한다.
3. window 단위로 feature를 계산한다.
4. baseline 대비 ratio를 계산한다.
5. 임시 heuristic으로 상태를 판단한다.
6. 원본 데이터(raw log)와 분석 결과(feature log)를 CSV로 저장한다.

나중에 바꿀 부분
- read_sensor_dummy() 부분만 Arduino Serial 입력으로 교체하면 됨.
- 판단 threshold는 회의 후 수정하면 됨.
"""

import os
import time
import csv
import numpy as np
import pandas as pd
from dataclasses import dataclass
from collections import deque
from datetime import datetime


# =========================================================
# 1. Config
# =========================================================
@dataclass
class SensorConfig:
    # -----------------------------
    # 센서 개수
    # -----------------------------
    num_emg: int = 1
    num_pressure: int = 4

    # -----------------------------
    # 샘플링 설정
    # 예: 100Hz면 1초에 100번 읽음
    # -----------------------------
    fs: int = 100

    # -----------------------------
    # 버퍼 / window 설정
    # buffer_seconds: 최근 몇 초 데이터를 들고 있을지
    # window_ms: 한 번 분석할 구간 길이
    # step_ms: 몇 ms마다 분석할지
    # -----------------------------
    buffer_seconds: float = 2.0
    window_ms: int = 500
    step_ms: int = 100

    # -----------------------------
    # smoothing 설정
    # 압력센서/EMG 값이 튀는 것 완화
    # -----------------------------
    smooth_window: int = 5

    # -----------------------------
    # 임시 판단 threshold
    # baseline 대비 ratio 기준
    # -----------------------------
    emg_low_thresh: float = 0.7
    emg_high_thresh: float = 1.5

    pressure_low_thresh: float = 0.7
    pressure_high_thresh: float = 1.5

    # -----------------------------
    # feedback 조건
    # 같은 상태가 몇 초 이상 지속될 때 경고할지
    # -----------------------------
    feedback_duration_sec: float = 2.0

    # -----------------------------
    # 로그 저장 폴더
    # -----------------------------
    log_dir: str = "logs"


# =========================================================
# 2. Utility functions
# =========================================================
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def moving_average_2d(x, window_size=5):
    """
    x shape: (T, C)

    T = 시간축
    C = 센서 채널 수

    각 센서 채널마다 moving average 적용
    """
    if window_size <= 1:
        return x

    x = np.asarray(x, dtype=np.float32)
    y = np.zeros_like(x)

    kernel = np.ones(window_size) / window_size

    for c in range(x.shape[1]):
        y[:, c] = np.convolve(x[:, c], kernel, mode="same")

    return y


def make_windows(signal, window_size, step_size):
    """
    전체 신호를 sliding window로 자르는 함수

    signal shape: (T, C)
    return: list of windows, each shape = (window_size, C)
    """
    windows = []

    for s in range(0, len(signal) - window_size + 1, step_size):
        e = s + window_size
        windows.append(signal[s:e])

    return windows


def safe_ratio(current, baseline, eps=1e-8):
    """
    baseline이 0에 가까울 때 나누기 오류 방지
    """
    return current / (baseline + eps)


# =========================================================
# 3. Feature extraction
# =========================================================
def extract_emg_features(emg_window, smooth_window=5):
    """
    EMG window feature 계산

    주의:
    - 현재 사용하는 EMG 모듈은 이미 filtering/rectification된 값을 줄 가능성이 큼.
    - 그래서 여기서는 복잡한 bandpass/notch 대신,
      smoothing + RMS/MAV 중심으로 처리.

    emg_window shape: (T, num_emg)
    """
    emg_processed = moving_average_2d(emg_window, smooth_window)

    rms = np.sqrt(np.mean(emg_processed ** 2, axis=0))
    mav = np.mean(np.abs(emg_processed), axis=0)
    mean_val = np.mean(emg_processed, axis=0)
    max_val = np.max(emg_processed, axis=0)

    total_rms = float(np.mean(rms))

    return {
        "emg_processed": emg_processed,
        "emg_rms": rms,
        "emg_mav": mav,
        "emg_mean": mean_val,
        "emg_max": max_val,
        "emg_total_rms": total_rms,
    }


def extract_pressure_features(pressure_window, smooth_window=5):
    """
    압력센서 window feature 계산

    pressure_window shape: (T, num_pressure)
    """
    pressure_processed = moving_average_2d(pressure_window, smooth_window)

    mean_pressure = np.mean(pressure_processed, axis=0)
    max_pressure = np.max(pressure_processed, axis=0)
    min_pressure = np.min(pressure_processed, axis=0)
    std_pressure = np.std(pressure_processed, axis=0)

    total_pressure = float(np.sum(mean_pressure))

    return {
        "pressure_processed": pressure_processed,
        "pressure_mean": mean_pressure,
        "pressure_max": max_pressure,
        "pressure_min": min_pressure,
        "pressure_std": std_pressure,
        "pressure_total": total_pressure,
    }


# =========================================================
# 4. Baseline
# =========================================================
def compute_baseline_stats(baseline_data, config: SensorConfig):
    """
    baseline_data는 dict 형태로 받음.

    baseline_data = {
        "emg": np.array,       shape = (T, num_emg)
        "pressure": np.array,  shape = (T, num_pressure)
    }

    baseline은 예를 들어:
    - 힘 안 준 상태
    - 중립 자세
    - 치료 시작 전 standing 자세
    에서 5~10초 정도 측정한 값
    """

    window_size = int(config.window_ms * config.fs / 1000)
    step_size = int(config.step_ms * config.fs / 1000)

    emg_windows = make_windows(baseline_data["emg"], window_size, step_size)
    pressure_windows = make_windows(baseline_data["pressure"], window_size, step_size)

    emg_rms_list = []
    pressure_mean_list = []
    pressure_total_list = []

    for w in emg_windows:
        feat = extract_emg_features(w, config.smooth_window)
        emg_rms_list.append(feat["emg_rms"])

    for w in pressure_windows:
        feat = extract_pressure_features(w, config.smooth_window)
        pressure_mean_list.append(feat["pressure_mean"])
        pressure_total_list.append(feat["pressure_total"])

    emg_rms_arr = np.array(emg_rms_list)
    pressure_mean_arr = np.array(pressure_mean_list)
    pressure_total_arr = np.array(pressure_total_list)

    baseline_stats = {
        "emg_rms_mean": np.mean(emg_rms_arr, axis=0),
        "emg_rms_std": np.std(emg_rms_arr, axis=0),
        "emg_total_rms_mean": float(np.mean(emg_rms_arr)),

        "pressure_mean": np.mean(pressure_mean_arr, axis=0),
        "pressure_std": np.std(pressure_mean_arr, axis=0),
        "pressure_total_mean": float(np.mean(pressure_total_arr)),
    }

    return baseline_stats


# =========================================================
# 5. Temporary decision logic
# =========================================================
def judge_by_ratio(ratio, low_thresh, high_thresh, low_name, high_name):
    """
    임시 판단 함수

    ratio < low_thresh      -> 부족
    low <= ratio <= high    -> 적절
    ratio > high_thresh     -> 과함

    이 로직은 회의 후 바꾸면 됨.
    """
    if ratio < low_thresh:
        return low_name
    elif ratio <= high_thresh:
        return "good"
    else:
        return high_name


def make_feedback(emg_state, pressure_state):
    """
    EMG 상태와 압력 상태를 조합해서 피드백 생성

    현재는 임시 문구.
    나중에 의학적/재활적 기준 정해지면 수정.
    """

    if emg_state == "emg_underactive" and pressure_state == "pressure_underload":
        return "근활성과 압력이 모두 부족합니다. 목표 부위에 조금 더 힘을 실어보세요."

    if emg_state == "emg_overactive" and pressure_state == "pressure_overload":
        return "근활성과 압력이 모두 과합니다. 힘을 조금 빼고 자연스럽게 유지해보세요."

    if emg_state == "emg_underactive":
        return "근활성이 부족합니다. 해당 근육을 조금 더 사용해보세요."

    if emg_state == "emg_overactive":
        return "근활성이 과합니다. 과도한 긴장을 줄여보세요."

    if pressure_state == "pressure_underload":
        return "압력이 부족합니다. 압력판에 조금 더 체중을 실어보세요."

    if pressure_state == "pressure_overload":
        return "압력이 과합니다. 압력을 조금 줄여보세요."

    return "좋습니다. 현재 상태가 적절하게 유지되고 있습니다."


# =========================================================
# 6. Logger
# =========================================================
class CSVLogger:
    """
    raw sensor data와 feature 결과를 각각 CSV로 저장

    raw_log:
    - 매 샘플마다 저장
    - timestamp, emg_1, emg_2, ..., pressure_1, ...

    feature_log:
    - window 분석 결과마다 저장
    - timestamp, ratios, states, feedback 등
    """

    def __init__(self, config: SensorConfig):
        os.makedirs(config.log_dir, exist_ok=True)

        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.raw_path = os.path.join(config.log_dir, f"raw_log_{session_id}.csv")
        self.feature_path = os.path.join(config.log_dir, f"feature_log_{session_id}.csv")

        self.config = config

        self._init_raw_file()
        self._init_feature_file()

    def _init_raw_file(self):
        header = ["timestamp"]

        for i in range(self.config.num_emg):
            header.append(f"emg_{i+1}")

        for i in range(self.config.num_pressure):
            header.append(f"pressure_{i+1}")

        with open(self.raw_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def _init_feature_file(self):
        header = [
            "timestamp",
            "emg_total_rms",
            "emg_total_ratio",
            "pressure_total",
            "pressure_total_ratio",
            "emg_state",
            "pressure_state",
            "feedback",
        ]

        for i in range(self.config.num_emg):
            header.append(f"emg_{i+1}_rms")
            header.append(f"emg_{i+1}_ratio")

        for i in range(self.config.num_pressure):
            header.append(f"pressure_{i+1}_mean")
            header.append(f"pressure_{i+1}_ratio")

        with open(self.feature_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def log_raw(self, timestamp, emg_values, pressure_values):
        row = [timestamp]
        row += list(emg_values)
        row += list(pressure_values)

        with open(self.raw_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def log_feature(self, result):
        row = [
            result["timestamp"],
            result["emg_total_rms"],
            result["emg_total_ratio"],
            result["pressure_total"],
            result["pressure_total_ratio"],
            result["emg_state"],
            result["pressure_state"],
            result["feedback"],
        ]

        for rms, ratio in zip(result["emg_rms"], result["emg_ratio"]):
            row.append(rms)
            row.append(ratio)

        for mean_val, ratio in zip(result["pressure_mean"], result["pressure_ratio"]):
            row.append(mean_val)
            row.append(ratio)

        with open(self.feature_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)


# =========================================================
# 7. Main Monitor
# =========================================================
class RehabSensorMonitor:
    def __init__(self, config: SensorConfig):
        self.config = config

        self.buffer_size = int(config.buffer_seconds * config.fs)
        self.window_size = int(config.window_ms * config.fs / 1000)
        self.step_size = int(config.step_ms * config.fs / 1000)

        # 각각 최근 N초 데이터를 저장
        self.emg_buffer = deque(maxlen=self.buffer_size)
        self.pressure_buffer = deque(maxlen=self.buffer_size)

        self.baseline_stats = None

        self.feature_history = []
        self.state_history = []

        self.logger = CSVLogger(config)

    def set_baseline(self, baseline_data):
        """
        baseline_data:
        {
            "emg": shape (T, num_emg),
            "pressure": shape (T, num_pressure)
        }
        """
        self.baseline_stats = compute_baseline_stats(baseline_data, self.config)

        print("\n[Baseline 설정 완료]")
        print("EMG RMS mean:", self.baseline_stats["emg_rms_mean"])
        print("Pressure mean:", self.baseline_stats["pressure_mean"])
        print("Pressure total mean:", self.baseline_stats["pressure_total_mean"])

    def update(self, emg_values, pressure_values):
        """
        센서값 한 줄이 들어올 때마다 호출

        emg_values shape:
        - (num_emg,)

        pressure_values shape:
        - (num_pressure,)
        """

        emg_values = np.asarray(emg_values, dtype=np.float32)
        pressure_values = np.asarray(pressure_values, dtype=np.float32)

        if len(emg_values) != self.config.num_emg:
            raise ValueError("EMG 센서 개수가 config와 다름")

        if len(pressure_values) != self.config.num_pressure:
            raise ValueError("압력센서 개수가 config와 다름")

        timestamp = now_str()

        # 1) raw data 저장
        self.logger.log_raw(timestamp, emg_values, pressure_values)

        # 2) buffer 업데이트
        self.emg_buffer.append(emg_values)
        self.pressure_buffer.append(pressure_values)

    def analyze_current_window(self):
        """
        현재 buffer에서 최신 window만 가져와 분석
        """
        if self.baseline_stats is None:
            raise ValueError("먼저 set_baseline()을 호출해야 함")

        if len(self.emg_buffer) < self.window_size:
            return None

        if len(self.pressure_buffer) < self.window_size:
            return None

        emg_window = np.array(list(self.emg_buffer)[-self.window_size:])
        pressure_window = np.array(list(self.pressure_buffer)[-self.window_size:])

        emg_feat = extract_emg_features(emg_window, self.config.smooth_window)
        pressure_feat = extract_pressure_features(pressure_window, self.config.smooth_window)

        emg_rms = emg_feat["emg_rms"]
        pressure_mean = pressure_feat["pressure_mean"]

        emg_ratio = safe_ratio(emg_rms, self.baseline_stats["emg_rms_mean"])
        pressure_ratio = safe_ratio(pressure_mean, self.baseline_stats["pressure_mean"])

        emg_total_rms = emg_feat["emg_total_rms"]
        pressure_total = pressure_feat["pressure_total"]

        emg_total_ratio = float(
            safe_ratio(emg_total_rms, self.baseline_stats["emg_total_rms_mean"])
        )

        pressure_total_ratio = float(
            safe_ratio(pressure_total, self.baseline_stats["pressure_total_mean"])
        )

        emg_state = judge_by_ratio(
            emg_total_ratio,
            self.config.emg_low_thresh,
            self.config.emg_high_thresh,
            low_name="emg_underactive",
            high_name="emg_overactive"
        )

        pressure_state = judge_by_ratio(
            pressure_total_ratio,
            self.config.pressure_low_thresh,
            self.config.pressure_high_thresh,
            low_name="pressure_underload",
            high_name="pressure_overload"
        )

        feedback = make_feedback(emg_state, pressure_state)

        result = {
            "timestamp": now_str(),

            "emg_rms": emg_rms,
            "emg_ratio": emg_ratio,
            "emg_total_rms": emg_total_rms,
            "emg_total_ratio": emg_total_ratio,

            "pressure_mean": pressure_mean,
            "pressure_ratio": pressure_ratio,
            "pressure_total": pressure_total,
            "pressure_total_ratio": pressure_total_ratio,

            "emg_state": emg_state,
            "pressure_state": pressure_state,
            "feedback": feedback,
        }

        self.feature_history.append(result)
        self.state_history.append((emg_state, pressure_state))

        # feature 결과 저장
        self.logger.log_feature(result)

        return result

    def check_persistent_warning(self):
        """
        같은 문제가 일정 시간 이상 지속되는지 확인
        """
        step_sec = self.step_size / self.config.fs
        required_steps = int(self.config.feedback_duration_sec / step_sec)

        if len(self.state_history) < required_steps:
            return None

        recent = self.state_history[-required_steps:]

        emg_states = [x[0] for x in recent]
        pressure_states = [x[1] for x in recent]

        if all(s == "emg_underactive" for s in emg_states):
            return "EMG 부족 상태가 지속되고 있습니다."

        if all(s == "emg_overactive" for s in emg_states):
            return "EMG 과활성 상태가 지속되고 있습니다."

        if all(s == "pressure_underload" for s in pressure_states):
            return "압력 부족 상태가 지속되고 있습니다."

        if all(s == "pressure_overload" for s in pressure_states):
            return "압력 과다 상태가 지속되고 있습니다."

        return None

    def summarize_session(self):
        if len(self.feature_history) == 0:
            return None

        emg_ratios = [x["emg_total_ratio"] for x in self.feature_history]
        pressure_ratios = [x["pressure_total_ratio"] for x in self.feature_history]

        emg_states = [x["emg_state"] for x in self.feature_history]
        pressure_states = [x["pressure_state"] for x in self.feature_history]

        summary = {
            "avg_emg_ratio": float(np.mean(emg_ratios)),
            "avg_pressure_ratio": float(np.mean(pressure_ratios)),

            "emg_underactive_ratio": emg_states.count("emg_underactive") / len(emg_states),
            "emg_good_ratio": emg_states.count("good") / len(emg_states),
            "emg_overactive_ratio": emg_states.count("emg_overactive") / len(emg_states),

            "pressure_underload_ratio": pressure_states.count("pressure_underload") / len(pressure_states),
            "pressure_good_ratio": pressure_states.count("good") / len(pressure_states),
            "pressure_overload_ratio": pressure_states.count("pressure_overload") / len(pressure_states),

            "num_windows": len(self.feature_history),
        }

        return summary


# =========================================================
# 8. Dummy sensor input
# =========================================================
def read_sensor_dummy(config: SensorConfig, mode="good"):
    """
    실제 센서 연결 전 테스트용 함수

    나중에 이 함수만 Arduino Serial read로 바꾸면 됨.

    return:
    emg_values: shape (num_emg,)
    pressure_values: shape (num_pressure,)
    """

    if mode == "under":
        emg_base = 150
        pressure_base = 150
    elif mode == "over":
        emg_base = 800
        pressure_base = 800
    else:
        emg_base = 400
        pressure_base = 400

    emg_values = emg_base + np.random.randn(config.num_emg) * 20
    pressure_values = pressure_base + np.random.randn(config.num_pressure) * 20

    emg_values = np.clip(emg_values, 0, 1023)
    pressure_values = np.clip(pressure_values, 0, 1023)

    return emg_values, pressure_values


def collect_dummy_baseline(config: SensorConfig, duration_sec=10):
    """
    baseline 측정 흉내

    실제 구현에서는:
    - 사용자에게 "중립 자세를 유지하세요" 출력
    - duration_sec 동안 센서값 수집
    - 그걸 baseline으로 저장
    """

    n_samples = int(duration_sec * config.fs)

    emg_list = []
    pressure_list = []

    for _ in range(n_samples):
        emg_values, pressure_values = read_sensor_dummy(config, mode="good")
        emg_list.append(emg_values)
        pressure_list.append(pressure_values)

    baseline_data = {
        "emg": np.array(emg_list),
        "pressure": np.array(pressure_list),
    }

    return baseline_data


# =========================================================
# 9. Run example
# =========================================================
if __name__ == "__main__":
    config = SensorConfig(
        num_emg=1,
        num_pressure=4,
        fs=100,
        buffer_seconds=2.0,
        window_ms=500,
        step_ms=100,
        log_dir="logs"
    )

    monitor = RehabSensorMonitor(config)

    # 1) baseline 설정
    baseline_data = collect_dummy_baseline(config, duration_sec=10)
    monitor.set_baseline(baseline_data)

    # 2) 실시간 모니터링 시작
    print("\n[실시간 모니터링 시작]")
    print("Ctrl+C를 누르면 종료됩니다.")

    sample_interval = 1.0 / config.fs
    analyze_every_samples = monitor.step_size

    sample_count = 0

    try:
        while True:
            # 현재는 dummy 값
            # 나중에는 여기만 실제 센서 read 함수로 교체
            emg_values, pressure_values = read_sensor_dummy(config, mode="good")

            # raw 저장 + buffer 업데이트
            monitor.update(emg_values, pressure_values)

            sample_count += 1

            # step_size마다 window 분석
            if sample_count % analyze_every_samples == 0:
                result = monitor.analyze_current_window()

                if result is not None:
                    print(
                        f"[{result['timestamp']}] "
                        f"EMG ratio={result['emg_total_ratio']:.2f}, "
                        f"Pressure ratio={result['pressure_total_ratio']:.2f}, "
                        f"EMG state={result['emg_state']}, "
                        f"Pressure state={result['pressure_state']}, "
                        f"Feedback={result['feedback']}"
                    )

                    warning = monitor.check_persistent_warning()
                    if warning is not None:
                        print("경고:", warning)

            time.sleep(sample_interval)

    except KeyboardInterrupt:
        print("\n[모니터링 종료]")

        summary = monitor.summarize_session()

        if summary is not None:
            print("\n[세션 요약]")
            for k, v in summary.items():
                if isinstance(v, float):
                    print(f"{k}: {v:.4f}")
                else:
                    print(f"{k}: {v}")

        print("\n저장된 로그:")
        print("Raw log    :", monitor.logger.raw_path)
        print("Feature log:", monitor.logger.feature_path)
