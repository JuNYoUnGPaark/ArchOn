import json
import numpy as np


class DecisionEngine:
    def __init__(self, config_path="sensor_config.json"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.fsr_count = self.config["fsr_count"]
        self.groups = self.config["groups"]
        self.th = self.config["thresholds"]
        self.emg_key = self.config["emg_key"]

        self.baseline = None
        self.max_values = None
        self.exercise_data = []

    def fsr_key(self, i):
        return f"fsr{i}_voltage"

    def get_value(self, sample, key):
        v = sample.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def group_mean(self, sample_or_stats, group_name):
        keys = self.groups.get(group_name, [])
        values = []

        for fsr_name in keys:
            key = f"{fsr_name}_voltage"
            v = sample_or_stats.get(key) if sample_or_stats else None
            if v is not None:
                values.append(float(v))

        if not values:
            return None

        return float(np.mean(values))

    def system_check(self, samples):
        disconnected = []
        warnings = []

        for i in range(1, self.fsr_count + 1):
            key = self.fsr_key(i)
            values = [
                self.get_value(s, key)
                for s in samples
                if self.get_value(s, key) is not None
            ]

            if not values:
                disconnected.append(f"FSR{i}")
                continue

            avg = float(np.mean(values))
            std = float(np.std(values))

            if avg <= self.th["sensor_dead_low"] or avg >= self.th["sensor_dead_high"]:
                disconnected.append(f"FSR{i}")
            elif std <= self.th.get("sensor_flat_std", 0.0):
                warnings.append(f"FSR{i}: 값 변화가 거의 없습니다.")

        emg_values = [
            self.get_value(s, self.emg_key)
            for s in samples
            if self.get_value(s, self.emg_key) is not None
        ]

        if not emg_values:
            disconnected.append("EMG")
        else:
            emg_avg = float(np.mean(emg_values))
            if emg_avg <= self.th["sensor_dead_low"] or emg_avg >= self.th["sensor_dead_high"]:
                disconnected.append("EMG")

        return {
            "ok": len(disconnected) == 0,
            "disconnected": disconnected,
            "warnings": warnings,
        }

    def compute_baseline(self, samples):
        self.baseline = self._compute_mean_stats(samples)
        return self.baseline

def compute_max_values(self, samples):
    self.max_values = {}

    top_percent = self.config.get("max_rule", {}).get("top_percent", 5)

    for i in range(1, self.fsr_count + 1):
        key = self.fsr_key(i)
        values = [
            self.get_value(s, key)
            for s in samples
            if self.get_value(s, key) is not None
        ]

        self.max_values[key] = self._top_percent_mean(values, top_percent)

    emg_values = [
        self.get_value(s, self.emg_key)
        for s in samples
        if self.get_value(s, self.emg_key) is not None
    ]

    self.max_values[self.emg_key] = self._top_percent_mean(emg_values, top_percent)

    return self.max_values


def _top_percent_mean(self, values, top_percent=5):
    if not values:
        return None

    values = np.array(values, dtype=float)
    values = np.sort(values)

    n = max(1, int(np.ceil(len(values) * top_percent / 100.0)))

    top_values = values[-n:]

    return float(np.mean(top_values))

    def _compute_mean_stats(self, samples):
        stats = {}

        for i in range(1, self.fsr_count + 1):
            key = self.fsr_key(i)
            values = [
                self.get_value(s, key)
                for s in samples
                if self.get_value(s, key) is not None
            ]
            stats[key] = float(np.mean(values)) if values else None

        emg_values = [
            self.get_value(s, self.emg_key)
            for s in samples
            if self.get_value(s, self.emg_key) is not None
        ]
        stats[self.emg_key] = float(np.mean(emg_values)) if emg_values else None

        return stats

    def normalize(self, key, value):
        if self.baseline is None or self.max_values is None:
            return None

        base = self.baseline.get(key)
        max_v = self.max_values.get(key)

        if base is None or max_v is None or value is None:
            return None

        denom = max_v - base

        if abs(denom) < 1e-8:
            return None

        ratio = (float(value) - base) / denom
        return max(0.0, min(1.0, float(ratio)))

    def start_exercise(self):
        self.exercise_data = []

    def update_exercise(self, sample):
        self.exercise_data.append(sample)

    def analyze_exercise(self):
        if not self.exercise_data:
            return {
                "ok": False,
                "grade": "bad",
                "score": 0.0,
                "feedback": ["운동 데이터가 없습니다."]
            }

        avg = self._compute_mean_stats(self.exercise_data)

        emg_result = self.judge_emg(avg)
        toe_result = self.judge_toe(self.exercise_data)
        heel_result = self.judge_heel(avg)
        shortfoot_result = self.judge_shortfoot(avg)
        balance_result = self.judge_balance(avg)

        results = [emg_result, toe_result, heel_result, shortfoot_result, balance_result]

        feedback = []
        for r in results:
            feedback.extend(r.get("feedback", []))

        good_count = sum(1 for r in results if r.get("grade") == "good")
        normal_count = sum(1 for r in results if r.get("grade") == "normal")
        bad_count = sum(1 for r in results if r.get("grade") == "bad")

        score = (good_count + 0.5 * normal_count) / len(results) * 100.0

        if score >= 80:
            grade = "good"
        elif score >= 50:
            grade = "normal"
        else:
            grade = "bad"

        return {
            "ok": True,
            "grade": grade,
            "score": round(float(score), 1),
            "emg_result": emg_result,
            "toe_result": toe_result,
            "heel_result": heel_result,
            "shortfoot_result": shortfoot_result,
            "balance_result": balance_result,
            "feedback": feedback
        }

    def judge_emg(self, avg):
        emg = avg.get(self.emg_key)
        ratio = self.normalize(self.emg_key, emg)

        if ratio is None:
            return {
                "state": "unknown",
                "grade": "bad",
                "ratio": None,
                "feedback": ["EMG 기준값/최대값이 부족합니다."]
            }

        if ratio >= self.th["emg_best_high"]:
            return {
                "state": "bad_compensation",
                "grade": "bad",
                "ratio": round(float(ratio), 4),
                "feedback": ["EMG가 45% 이상입니다. 보상작용 가능성이 있습니다."]
            }
        if ratio >= self.th["emg_normal_low"]:
            return {
                "state": "best",
                "grade": "good",
                "ratio": round(float(ratio), 4),
                "feedback": ["EMG가 35~45% 구간입니다. 무지외전근 활성도가 가장 적절합니다."]
            }
        if ratio >= self.th["emg_bad_low"]:
            return {
                "state": "normal",
                "grade": "normal",
                "ratio": round(float(ratio), 4),
                "feedback": ["EMG가 15~35% 구간입니다. 보통 수준입니다."]
            }
        return {
            "state": "bad_underactive",
            "grade": "bad",
            "ratio": round(float(ratio), 4),
            "feedback": ["EMG가 15% 이하입니다. 무지외전근의 힘을 더 이용하세요."]
        }

    def judge_toe(self, exercise_samples):
        toe_base = self.group_mean(self.baseline, "toe") if self.baseline else None
        if toe_base is None:
            return {
                "state": "unknown",
                "grade": "bad",
                "reps": 0,
                "feedback": ["엄지발가락 센서 매핑이 부족합니다."]
            }

        threshold = toe_base * self.th["toe_lift_ratio"]
        below_flags = []

        for s in exercise_samples:
            toe_now = self.group_mean(s, "toe")
            below_flags.append(toe_now is not None and toe_now < threshold)

        reps = self._count_rising_edges(below_flags)

        if reps >= 1:
            return {
                "state": "toe_lift_detected",
                "grade": "good",
                "reps": reps,
                "feedback": [f"엄지발가락 압력 감소가 {reps}회 감지되어 운동 반복으로 인정됩니다."]
            }

        return {
            "state": "no_toe_lift",
            "grade": "bad",
            "reps": 0,
            "feedback": ["엄지발가락 압력 감소가 감지되지 않았습니다. 엄지발가락을 들어올리는 동작을 명확히 해보세요."]
        }

    def _count_rising_edges(self, flags):
        count = 0
        prev = False
        for flag in flags:
            if flag and not prev:
                count += 1
            prev = flag
        return count

    def judge_heel(self, avg):
        heel_now = self.group_mean(avg, "heel")
        heel_base = self.group_mean(self.baseline, "heel") if self.baseline else None

        if heel_now is None or heel_base is None:
            return {
                "state": "unknown",
                "grade": "bad",
                "feedback": ["뒤꿈치 센서 매핑이 부족합니다."]
            }

        if heel_now < heel_base * self.th["heel_lift_ratio"]:
            return {
                "state": "heel_lift",
                "grade": "bad",
                "feedback": ["뒤꿈치 압력이 기준값 이하입니다. 뒤꿈치가 떨어지지 않도록 하세요."]
            }

        return {
            "state": "good",
            "grade": "good",
            "feedback": ["뒤꿈치 압력은 안정적으로 유지되고 있습니다."]
        }

    def judge_shortfoot(self, avg):
        arch_now = self.group_mean(avg, "arch")
        arch_base = self.group_mean(self.baseline, "arch") if self.baseline else None

        emg = avg.get(self.emg_key)
        emg_ratio = self.normalize(self.emg_key, emg) if emg is not None else None

        if arch_now is None or arch_base is None:
            return {
                "state": "unknown",
                "grade": "bad",
                "feedback": ["아치 센서 매핑이 부족합니다."]
            }

        arch_decreased = arch_now < arch_base * self.th["arch_decrease_ratio"]
        emg_active = emg_ratio is not None and emg_ratio >= self.th["emg_bad_low"]

        if arch_decreased and emg_active:
            return {
                "state": "good",
                "grade": "good",
                "feedback": ["아치 압력 감소와 EMG 상승이 함께 확인됩니다. 숏풋 운동을 잘 수행하고 있습니다."]
            }

        if arch_decreased and not emg_active:
            return {
                "state": "emg_low",
                "grade": "normal",
                "feedback": ["아치 압력은 감소했지만 EMG 상승이 부족합니다. 무지외전근의 힘을 이용하세요."]
            }

        return {
            "state": "arch_not_decreased",
            "grade": "bad",
            "feedback": ["아치 쪽 압력 감소가 부족합니다. 아치를 유지시켜보려고 하세요."]
        }

    def judge_balance(self, avg):
        left = self.group_mean(avg, "left")
        right = self.group_mean(avg, "right")

        if left is None or right is None:
            return {
                "state": "unknown",
                "grade": "bad",
                "balance_index": None,
                "feedback": ["좌우 밸런스 센서 매핑이 부족합니다."]
            }

        denom = left + right

        if denom == 0:
            return {
                "state": "unknown",
                "grade": "bad",
                "balance_index": None,
                "feedback": ["좌우 압력 합이 0입니다."]
            }

        balance_index = (right - left) / denom

        if balance_index > self.th["balance_threshold"]:
            return {
                "state": "lean_right",
                "grade": "normal",
                "balance_index": round(float(balance_index), 4),
                "feedback": ["오른쪽으로 체중이 기울어졌습니다."]
            }
        if balance_index < -self.th["balance_threshold"]:
            return {
                "state": "lean_left",
                "grade": "normal",
                "balance_index": round(float(balance_index), 4),
                "feedback": ["왼쪽으로 체중이 기울어졌습니다."]
            }

        return {
            "state": "balanced",
            "grade": "good",
            "balance_index": round(float(balance_index), 4),
            "feedback": ["좌우 밸런스가 안정적입니다."]
        }

    def summarize_exercises(self, exercise_results):
        if not exercise_results:
            return {
                "total_exercises": 0,
                "success_rate": 0.0,
                "most_common_issue": "없음",
                "comments": ["운동 기록이 없습니다."]
            }

        grades = [r.get("grade", "bad") for r in exercise_results]
        scores = [r.get("score", 0.0) for r in exercise_results]

        success_rate = sum(1 for g in grades if g == "good") / len(grades) * 100.0

        issue_counter = {}
        comments = []

        for idx, r in enumerate(exercise_results, start=1):
            fb = r.get("feedback", [])
            comments.append(f"{idx}회차 [{r.get('grade')} / {r.get('score')}점]: " + " ".join(fb))

            for item_key in ["emg_result", "toe_result", "heel_result", "shortfoot_result", "balance_result"]:
                item = r.get(item_key, {})
                if item.get("grade") != "good":
                    state = item.get("state", "unknown")
                    issue_counter[state] = issue_counter.get(state, 0) + 1

        most_common_issue = "없음"
        if issue_counter:
            most_common_issue = max(issue_counter, key=issue_counter.get)

        return {
            "total_exercises": len(exercise_results),
            "success_rate": round(float(success_rate), 1),
            "avg_score": round(float(np.mean(scores)), 1),
            "most_common_issue": most_common_issue,
            "comments": comments
        }