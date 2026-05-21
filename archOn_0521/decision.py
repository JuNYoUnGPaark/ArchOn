from collections import deque
import numpy as np


class DecisionEngine:
    def __init__(self, window_seconds=5, fsr_count=18):
        self.window_size = window_seconds
        self.fsr_count = fsr_count
        self.buffer = deque(maxlen=window_seconds)

        self.fsr_low = 0.3
        self.fsr_high = 2.0

        self.emg_low = 2.70
        self.emg_high = 2.90

    def update(self, sample):
        self.buffer.append(sample)
        recent = list(self.buffer)

        fsr_means = {}

        for i in range(1, self.fsr_count + 1):
            key = f"fsr{i}_voltage"

            values = [
                x[key]
                for x in recent
                if key in x and x[key] is not None
            ]

            if values:
                fsr_means[f"fsr{i}_avg_5s"] = round(float(np.mean(values)), 4)

        fsr_avg = float(np.mean(list(fsr_means.values()))) if fsr_means else 0.0

        emg_values = [
            x["emg_voltage"]
            for x in recent
            if "emg_voltage" in x and x["emg_voltage"] is not None
        ]

        emg_avg = float(np.mean(emg_values)) if emg_values else 0.0

        pressure_state = self.judge_pressure(fsr_avg)
        emg_state = self.judge_emg(emg_avg)
        feedback = self.make_feedback(pressure_state, emg_state)

        return {
            **fsr_means,
            "fsr_total_avg_5s": round(fsr_avg, 4),
            "emg_avg_5s": round(emg_avg, 4),
            "pressure_state": pressure_state,
            "emg_state": emg_state,
            "feedback": feedback,
        }

    def judge_pressure(self, v):
        if v < self.fsr_low:
            return "underload"
        elif v <= self.fsr_high:
            return "good"
        else:
            return "overload"

    def judge_emg(self, v):
        if v < self.emg_low:
            return "underactive"
        elif v <= self.emg_high:
            return "good"
        else:
            return "overactive"

    def make_feedback(self, pressure_state, emg_state):
        if pressure_state == "underload":
            return "전체 압력이 부족합니다."
        if pressure_state == "overload":
            return "전체 압력이 과합니다."
        if emg_state == "underactive":
            return "근육 활성도가 낮습니다."
        if emg_state == "overactive":
            return "근육 활성도가 높습니다."
        return "좋습니다. 압력과 근활성이 적절합니다."