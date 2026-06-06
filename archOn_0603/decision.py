import json
import numpy as np
from collections import Counter


class DecisionEngine:
    def __init__(self, config_path="sensor_config.json"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.fsr_count = self.config["fsr_count"]
        self.groups = self.config["groups"]
        self.th = self.config["thresholds"]
        self.rules = self.config.get("rules", {})
        self.emg_key = self.config["emg_key"]
        self.mvic_instruction = "앉아서 발바닥을 바닥에 붙이고 엄지발가락을 최대한 벌린 상태에서 동시에 엄지발가락을 좌에서 우 방향으로 밀어주세요."

        self.reset_calibration()

    def reset_calibration(self):
        self.baseline = None
        self.max_lift = None
        self.mvic_trials = []
        self.mvic = None
        self.exercise_data = []

    def fsr_key(self, i):
        return f"fsr{i}_voltage"

    def get_value(self, sample, key):
        try:
            v = sample.get(key)
            return None if v is None else float(v)
        except (TypeError, ValueError):
            return None

    def values_for_key(self, samples, key):
        values = []
        for s in samples:
            v = self.get_value(s, key)
            if v is not None:
                values.append(v)
        return values

    def group_values(self, stats, group_name):
        values = []
        for fsr_name in self.groups.get(group_name, []):
            v = stats.get(f"{fsr_name}_voltage") if stats else None
            if v is not None:
                values.append(float(v))
        return values

    def group_mean(self, stats, group_name):
        values = self.group_values(stats, group_name)
        return float(np.mean(values)) if values else None

    def _safe_ratio(self, now, base):
        if now is None or base is None or abs(base) < 1e-8:
            return None
        return float((now - base) / abs(base))

    def _fsr_value(self, stats, idx):
        return self.get_value(stats, self.fsr_key(idx)) if stats else None

    def _sum_fsr_indices(self, stats, indices):
        values = [self._fsr_value(stats, i) for i in indices]
        values = [v for v in values if v is not None]
        return float(np.sum(values)) if values else None

    def _arch_eval_indices(self):
        return self.rules.get("arch_eval_sensors", [7, 8, 10, 11])

    def _arch_eval_sum(self, stats):
        return self._sum_fsr_indices(stats, self._arch_eval_indices())

    def _sample_arch_eval_sum(self, sample):
        return self._sum_fsr_indices(sample, self._arch_eval_indices())

    def _baseline_arch_eval_sum(self):
        return self._arch_eval_sum(self.baseline) if self.baseline else None

    def _max_lift_arch_eval_sum(self):
        if not self.max_lift:
            return None
        v = self.max_lift.get("arch_eval_sum")
        return None if v is None else float(v)

    def _mvic_emg_value(self):
        if not self.mvic:
            return None
        return self.mvic.get("emg_voltage")

    def _classify_arch_from_sum(self, k):
        n = self._baseline_arch_eval_sum()
        m = self._max_lift_arch_eval_sum()
        if k is None or n is None or m is None or abs(n - m) < 1e-8:
            return {"grade": "Bad", "score": 0.0, "progress": None, "n": n, "m": m, "k": k}

        # 압력 감소가 아치 형성이다. N > K > M이 이상적이지만, 역전/초과도 0~1로 보정한다.
        progress = (n - k) / (n - m)
        progress = max(0.0, min(1.0, float(progress)))

        if progress >= 2.0 / 3.0:
            grade = "Good"
        elif progress >= 1.0 / 3.0:
            grade = "Normal"
        else:
            grade = "Bad"

        return {"grade": grade, "score": progress, "progress": progress, "n": n, "m": m, "k": k}

    def _emg_mvic_ratio(self, emg_now):
        mvic_emg = self._mvic_emg_value()
        if emg_now is None or mvic_emg is None or abs(mvic_emg) < 1e-8:
            return None
        return float(emg_now / abs(mvic_emg))

    def _compute_mean_stats(self, samples):
        stats = {}

        for i in range(1, self.fsr_count + 1):
            key = self.fsr_key(i)
            values = self.values_for_key(samples, key)
            stats[key] = float(np.mean(values)) if values else None

        emg_values = self.values_for_key(samples, self.emg_key)
        stats[self.emg_key] = float(np.mean(emg_values)) if emg_values else None

        stats["toe_mean"] = self.group_mean(stats, "toe")
        stats["arch_mean"] = self.group_mean(stats, "arch")
        stats["heel_mean"] = self.group_mean(stats, "heel")
        stats["arch_eval_sum"] = self._arch_eval_sum(stats)

        fsr_values = [
            stats.get(self.fsr_key(i))
            for i in range(1, self.fsr_count + 1)
            if stats.get(self.fsr_key(i)) is not None
        ]
        stats["fsr_total_mean"] = float(np.mean(fsr_values)) if fsr_values else None

        return stats

    def system_check(self, samples):
        disconnected = []
        warnings = []
        sensor_status = {}

        severe_low = self.th.get("severe_dead_low", -0.02)
        severe_high = self.th.get("severe_dead_high", 3.9)
        soft_low = self.th.get("sensor_dead_low", -0.05)
        soft_high = self.th.get("sensor_dead_high", 3.6)

        for i in range(1, self.fsr_count + 1):
            name = f"FSR{i}"
            key = self.fsr_key(i)
            values = self.values_for_key(samples, key)

            if not values:
                disconnected.append(name)
                sensor_status[name] = {"state": "reconnect", "label": "재연결", "avg": None}
                continue

            avg = float(np.mean(values))

            if avg <= severe_low or avg >= severe_high:
                disconnected.append(name)
                sensor_status[name] = {"state": "reconnect", "label": "재연결", "avg": round(avg, 4)}
            elif avg <= soft_low or avg >= soft_high:
                warnings.append(name)
                sensor_status[name] = {"state": "warning", "label": "경고", "avg": round(avg, 4)}
            else:
                sensor_status[name] = {"state": "ok", "label": "정상", "avg": round(avg, 4)}

        emg_values = self.values_for_key(samples, self.emg_key)

        if not emg_values:
            disconnected.append("EMG")
            sensor_status["EMG"] = {"state": "reconnect", "label": "재연결", "avg": None}
        else:
            emg_avg = float(np.mean(emg_values))

            if emg_avg <= severe_low or emg_avg >= severe_high:
                disconnected.append("EMG")
                sensor_status["EMG"] = {"state": "reconnect", "label": "재연결", "avg": round(emg_avg, 4)}
            elif emg_avg <= soft_low or emg_avg >= soft_high:
                warnings.append("EMG")
                sensor_status["EMG"] = {"state": "warning", "label": "경고", "avg": round(emg_avg, 4)}
            else:
                sensor_status["EMG"] = {"state": "ok", "label": "정상", "avg": round(emg_avg, 4)}

        total = self.fsr_count + 1
        connected_ratio = (total - len(disconnected)) / total
        min_connected_ratio = self.th.get("min_connected_ratio", 0.72)

        # 초기 실험/시연에서는 일부 센서 경고가 있어도 전체 연결 비율이 충분하면 통과시킨다.
        ok = connected_ratio >= min_connected_ratio and len(disconnected) <= 4
        need_reconnect = not ok

        return {
            "ok": ok,
            "need_reconnect": need_reconnect,
            "disconnected": disconnected,
            "warnings": warnings,
            "sensor_status": sensor_status,
            "connected_ratio": round(float(connected_ratio), 3),
            "message": "재연결 필요" if need_reconnect else "점검 통과",
        }

    def compute_baseline(self, samples):
        baseline = self._compute_mean_stats(samples)
        self.baseline = baseline
        return {
            **baseline,
            "_ok": True,
            "_feedback": ["기준값 저장 완료"],
            "_calibration_log": self.calibration_monitoring(None),
        }

    def realtime_max_lift_metrics(self, sample):
        if self.baseline is None:
            return {}

        toe_base = self.group_mean(self.baseline, "toe")
        toe_now = self.group_mean(sample, "toe")
        heel_base = self.group_mean(self.baseline, "heel")
        heel_now = self.group_mean(sample, "heel")
        emg_base = self.baseline.get(self.emg_key)
        emg_now = self.get_value(sample, self.emg_key)

        toe_decrease = None
        if toe_base is not None and toe_now is not None and abs(toe_base) > 1e-8:
            toe_decrease = (toe_base - toe_now) / abs(toe_base)

        heel_shift = None
        if heel_base is not None and heel_now is not None and abs(heel_base) > 1e-8:
            heel_shift = abs(heel_now - heel_base) / abs(heel_base)

        emg_change = None
        emg_ratio = None
        if emg_base is not None and emg_now is not None and abs(emg_base) > 1e-8:
            emg_change = (emg_now - emg_base) / abs(emg_base)
            emg_ratio = emg_now / abs(emg_base)

        return {
            "toe_decrease_ratio": None if toe_decrease is None else round(float(toe_decrease), 4),
            "toe_decrease_percent": None if toe_decrease is None else round(float(toe_decrease * 100), 1),
            "heel_shift_ratio": None if heel_shift is None else round(float(heel_shift), 4),
            "heel_shift_percent": None if heel_shift is None else round(float(heel_shift * 100), 1),
            "emg_change_ratio": None if emg_change is None else round(float(emg_change), 4),
            "emg_change_percent": None if emg_change is None else round(float(emg_change * 100), 1),
            "emg_ratio_percent": None if emg_ratio is None else round(float(emg_ratio * 100), 1),
        }

    def compute_max_lift(self, samples):
        """
        최대 아치 형성 측정 단계.
        - EMG 최대값은 기존처럼 저장한다.
        - 동시에 센서 7, 8, 10, 11의 압력 합을 M으로 저장한다.
          아치가 잘 형성될수록 중족부 압력이 낮아지므로, 측정 구간 중 가장 낮은 20% 평균을
          개인의 최대 아치 형성 압력 기준값으로 사용한다.
        """
        emg_values = self.values_for_key(samples, self.emg_key)
        emg_max = float(np.max(emg_values)) if emg_values else None
        emg_mean = float(np.mean(emg_values)) if emg_values else None

        arch_sums = []
        for sample in samples:
            v = self._sample_arch_eval_sum(sample)
            if v is not None:
                arch_sums.append(v)

        arch_eval_sum = None
        arch_eval_mean = None
        if arch_sums:
            arr = np.array(arch_sums, dtype=float)
            arch_eval_mean = float(np.mean(arr))
            take = max(1, int(np.ceil(len(arr) * self.rules.get("max_lift_lowest_fraction", 0.20))))
            arch_eval_sum = float(np.mean(np.sort(arr)[:take]))

        result = {
            self.emg_key: emg_max,
            "emg_max": emg_max,
            "emg_mean": emg_mean,
            "arch_eval_sum": arch_eval_sum,
            "arch_eval_mean": arch_eval_mean,
            "arch_eval_sensors": self._arch_eval_indices(),
        }

        ok = emg_max is not None and arch_eval_sum is not None

        feedback = []
        if ok:
            feedback.append(f"최대 아치 형성값 저장 완료: EMG {emg_max:.4f}, 아치 압력합 M {arch_eval_sum:.4f}")
            self.max_lift = result
        else:
            if emg_max is None:
                feedback.append("EMG 값이 수집되지 않았습니다. EMG 센서 연결을 확인하세요.")
            if arch_eval_sum is None:
                feedback.append("아치 영역 압력값이 수집되지 않았습니다. FSR 7, 8, 10, 11을 확인하세요.")

        return {
            **result,
            "_ok": ok,
            "_feedback": feedback,
            "_metrics": {
                "emg_max": None if emg_max is None else round(float(emg_max), 4),
                "emg_mean": None if emg_mean is None else round(float(emg_mean), 4),
                "arch_eval_sum": None if arch_eval_sum is None else round(float(arch_eval_sum), 4),
                "arch_eval_mean": None if arch_eval_mean is None else round(float(arch_eval_mean), 4),
            },
            "_calibration_log": self.calibration_monitoring(None),
        }

    def _max_lift_emg_value(self):
        if not self.max_lift:
            return None

        for key in ("emg_max", self.emg_key):
            v = self.max_lift.get(key)
            if v is not None:
                return float(v)

        return None

    def _emg_arch_ratio_from_max_lift(self, emg_now):
        emg_max = self._max_lift_emg_value()

        if emg_now is None or emg_max is None or abs(emg_max) < 1e-8:
            return None

        return float(emg_now / abs(emg_max))

    def add_mvic_trial(self, samples):
        stats = self._compute_mean_stats(samples)

        trial = {
            "trial_index": len(self.mvic_trials) + 1,
            "toe_mean": stats.get("toe_mean"),
            "arch_mean": stats.get("arch_mean"),
            "heel_mean": stats.get("heel_mean"),
            "emg_voltage": stats.get(self.emg_key),
            "stats": stats,
        }

        self.mvic_trials.append(trial)

        required = self.config["timing"].get("mvic_trials", 3)
        toe_values = np.array([t["toe_mean"] for t in self.mvic_trials if t.get("toe_mean") is not None], dtype=float)

        mean = float(np.mean(toe_values)) if len(toe_values) else None
        std = float(np.std(toe_values)) if len(toe_values) else None
        cv = float(std / abs(mean) * 100.0) if mean is not None and abs(mean) > 1e-8 else None

        if len(self.mvic_trials) < required:
            return {
                "ok": False,
                "trial_count": len(self.mvic_trials),
                "cv_percent": cv,
                "trials": self.mvic_trials,
                "instruction": self.mvic_instruction,
                "feedback": [f"MVIC {len(self.mvic_trials)}/{required}회 완료"],
            }

        limit = self.rules.get("mvic_cv_limit_percent", 60.0)

        if cv is None or cv > limit:
            old_cv = cv
            self.mvic_trials = []
            self.mvic = None
            return {
                "ok": False,
                "trial_count": required,
                "cv_percent": old_cv,
                "trials": [],
                "reset": True,
                "instruction": self.mvic_instruction,
                "feedback": [f"MVIC 변동계수 {old_cv:.1f}%로 {limit:.0f}% 초과. 1회차부터 다시 측정하세요."],
            }

        self.mvic = {
            "toe_mean": mean,
            "emg_voltage": (
                float(np.mean([t["emg_voltage"] for t in self.mvic_trials if t.get("emg_voltage") is not None]))
                if any(t.get("emg_voltage") is not None for t in self.mvic_trials)
                else None
            ),
            "cv_percent": cv,
            "trials": self.mvic_trials,
        }

        return {
            "ok": True,
            "trial_count": required,
            "cv_percent": cv,
            "mvic": self.mvic,
            "trials": self.mvic_trials,
            "instruction": self.mvic_instruction,
            "feedback": ["MVIC 기준값 저장 완료"],
            "_calibration_log": self.calibration_monitoring(None),
        }

    def start_exercise(self):
        self.exercise_data = []

    def update_exercise(self, sample):
        self.exercise_data.append(sample)


    def _round_or_none(self, v, ndigits=4):
        return None if v is None else round(float(v), ndigits)

    def _percent_change(self, now, ref):
        if now is None or ref is None or abs(ref) < 1e-8:
            return None
        return float((now - ref) / abs(ref) * 100.0)

    def _percent_of(self, now, ref):
        if now is None or ref is None or abs(ref) < 1e-8:
            return None
        return float(now / abs(ref) * 100.0)

    def calibration_monitoring(self, sample=None):
        """
        기준값 / 최대 아치 EMG / MVIC 기준값과 현재값을 한 번에 확인하기 위한 모니터링 정보.
        app.py에서 sensor_loop마다 이 값을 emit하면 프론트에서 실시간 로그/표시가 가능하다.
        """
        current = {}
        if sample is not None:
            current = {
                "toe_mean": self.group_mean(sample, "toe"),
                "arch_mean": self.group_mean(sample, "arch"),
                "heel_mean": self.group_mean(sample, "heel"),
                self.emg_key: self.get_value(sample, self.emg_key),
            }

        baseline = {
            "toe_mean": self.group_mean(self.baseline, "toe") if self.baseline else None,
            "arch_mean": self.group_mean(self.baseline, "arch") if self.baseline else None,
            "heel_mean": self.group_mean(self.baseline, "heel") if self.baseline else None,
            self.emg_key: self.baseline.get(self.emg_key) if self.baseline else None,
        }

        max_lift_emg = self._max_lift_emg_value()
        max_lift = {
            "emg_max": max_lift_emg,
            self.emg_key: max_lift_emg,
        }

        mvic = {
            "toe_mean": self.mvic.get("toe_mean") if self.mvic else None,
            self.emg_key: self.mvic.get("emg_voltage") if self.mvic else None,
            "cv_percent": self.mvic.get("cv_percent") if self.mvic else None,
        }

        comparisons = {}
        if sample is not None:
            for key in ["toe_mean", "arch_mean", "heel_mean", self.emg_key]:
                now = current.get(key)
                base = baseline.get(key)
                comparisons[f"{key}_vs_baseline_change_percent"] = self._percent_change(now, base)
                comparisons[f"{key}_of_baseline_percent"] = self._percent_of(now, base)

            emg_now = current.get(self.emg_key)
            comparisons["emg_of_max_lift_percent"] = self._percent_of(emg_now, max_lift_emg)
            comparisons["emg_vs_max_lift_change_percent"] = self._percent_change(emg_now, max_lift_emg)
            comparisons["emg_of_mvic_percent"] = self._percent_of(emg_now, mvic.get(self.emg_key))
            comparisons["emg_vs_mvic_change_percent"] = self._percent_change(emg_now, mvic.get(self.emg_key))

            comparisons["toe_of_mvic_percent"] = self._percent_of(current.get("toe_mean"), mvic.get("toe_mean"))
            comparisons["toe_vs_mvic_change_percent"] = self._percent_change(current.get("toe_mean"), mvic.get("toe_mean"))

        def rounded_dict(d):
            return {k: self._round_or_none(v, 4) for k, v in d.items()}

        return {
            "current": rounded_dict(current),
            "baseline": rounded_dict(baseline),
            "max_lift": rounded_dict(max_lift),
            "mvic": rounded_dict(mvic),
            "comparisons": {k: self._round_or_none(v, 1) for k, v in comparisons.items()},
        }

    def _current_feedback_checks(self, sample):
        rules = self.rules
        toe_base = self._fsr_value(self.baseline, 1) if self.baseline else None
        toe_now = self._fsr_value(sample, 1)
        heel_base = self.group_mean(self.baseline, "heel") if self.baseline else None
        heel_now = self.group_mean(sample, "heel")
        front_base = self._fsr_value(self.baseline, 4) if self.baseline else None
        front_now = self._fsr_value(sample, 4)
        emg_base = self.baseline.get(self.emg_key) if self.baseline else None
        emg_now = self.get_value(sample, self.emg_key)
        mvic_toe = self.mvic.get("toe_mean") if self.mvic else None

        left_sum = self._sum_fsr_indices(sample, [1, 4, 7, 10, 13, 16])
        right_sum = self._sum_fsr_indices(sample, [3, 6, 9, 12, 15, 18])
        total_sum = self._sum_fsr_indices(sample, list(range(1, self.fsr_count + 1)))

        toe_increase = self._safe_ratio(toe_now, toe_base)
        toe_of_base = None if toe_now is None or toe_base is None or abs(toe_base) < 1e-8 else toe_now / abs(toe_base)
        heel_of_base = None if heel_now is None or heel_base is None or abs(heel_base) < 1e-8 else heel_now / abs(heel_base)
        front_of_base = None if front_now is None or front_base is None or abs(front_base) < 1e-8 else front_now / abs(front_base)
        emg_increase = self._safe_ratio(emg_now, emg_base)
        toe_of_mvic = None if toe_now is None or mvic_toe is None or abs(mvic_toe) < 1e-8 else toe_now / abs(mvic_toe)
        emg_of_mvic = self._emg_mvic_ratio(emg_now)
        left_share = None if left_sum is None or total_sum is None or abs(total_sum) < 1e-8 else left_sum / abs(total_sum)
        right_share = None if right_sum is None or total_sum is None or abs(total_sum) < 1e-8 else right_sum / abs(total_sum)

        emg_base_limit = rules.get("live_emg_over_baseline_increase_ratio", None)
        checks = {
            "toe_flexion_compensation": toe_increase is not None and toe_increase >= rules.get("live_toe_flexion_increase_ratio", 0.30),
            "toe_no_contact": toe_of_base is not None and toe_of_base < rules.get("live_toe_contact_min_of_baseline", 0.10),
            "heel_lift": heel_of_base is not None and heel_of_base < rules.get("live_heel_contact_min_of_baseline", 0.10),
            "forefoot_lift": front_of_base is not None and front_of_base < rules.get("live_front_contact_min_of_baseline", 0.10),
            "excessive_activation": (
                (emg_base_limit is not None and emg_increase is not None and emg_increase >= emg_base_limit) or
                (emg_of_mvic is not None and emg_of_mvic >= rules.get("live_emg_excessive_mvic_ratio", 0.90)) or
                (toe_of_mvic is not None and toe_of_mvic >= rules.get("live_toe_excessive_mvic_ratio", 0.90))
            ),
            "weight_shift": (
                (left_share is not None and left_share >= rules.get("live_side_weight_share_limit", 0.90)) or
                (right_share is not None and right_share >= rules.get("live_side_weight_share_limit", 0.90))
            ),
        }

        metrics = {
            "toe_increase_ratio": toe_increase,
            "toe_of_baseline_ratio": toe_of_base,
            "heel_of_baseline_ratio": heel_of_base,
            "front4_of_baseline_ratio": front_of_base,
            "emg_increase_ratio": emg_increase,
            "emg_of_mvic_ratio": emg_of_mvic,
            "toe_of_mvic_ratio": toe_of_mvic,
            "left_share": left_share,
            "right_share": right_share,
        }
        return checks, metrics

    def _arch_hold_feedback_needed(self):
        min_seconds = self.rules.get("live_arch_hold_min_seconds", 3.0)
        interval = self.config.get("timing", {}).get("sample_interval", 1.0)
        min_samples = max(2, int(np.ceil(min_seconds / max(interval, 1e-8))))
        if len(self.exercise_data) < 2:
            return False

        states = []
        for sample in self.exercise_data:
            arch = self._classify_arch_from_sum(self._sample_arch_eval_sum(sample))
            states.append(arch["grade"] in ["Normal", "Good"])

        runs = []
        cur = 0
        for ok in states:
            if ok:
                cur += 1
            elif cur:
                runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)

        # 아치가 한 번 형성되었지만 모든 연속 유지 구간이 3초 미만이면 유지시간 부족 피드백.
        return bool(runs) and max(runs) < min_samples

    def live_exercise_feedback(self):
        if not self.exercise_data:
            return {
                "rings": {"arch": 0, "toe": 0, "heel": 0, "emg": 0},
                "feedback_flags": {"arch_good": False, "toe_over": False, "heel_unstable": False, "emg_low": False},
                "coaching_message": "운동 데이터를 수집 중입니다.",
                "monitoring": self.calibration_monitoring(None),
            }

        rates = self._exercise_success_rates(self.exercise_data)
        rings = {k: round(float(v * 100), 1) for k, v in rates.items()}
        sample = self.exercise_data[-1]
        checks, metrics = self._current_feedback_checks(sample)

        flags = {
            "arch_good": rates["arch"] >= 0.70,
            "toe_over": checks.get("toe_flexion_compensation", False),
            "heel_unstable": checks.get("heel_lift", False),
            "emg_low": rates["emg"] < 0.70,
        }

        if checks["toe_flexion_compensation"]:
            msg = "엄지발가락에 힘이 과도하게 들어갔습니다. 발가락을 말지 말고 발 안쪽 아치를 끌어올리는 느낌으로 수행해 주세요."
        elif checks["toe_no_contact"]:
            msg = "엄지발가락이 바닥에 닿지 않았습니다. 아치를 유지한 상태에서 엄지발가락을 천천히 내려놓아 주세요."
        elif checks["heel_lift"]:
            msg = "뒤꿈치가 바닥에서 떨어졌습니다. 뒤꿈치를 고정한 상태에서 아치만 형성해 주세요."
        elif checks["forefoot_lift"]:
            msg = "발 앞부분이 바닥에서 떨어졌습니다. 발 전체를 바닥에 둔 상태에서 아치를 형성해 주세요."
        elif checks["excessive_activation"]:
            msg = "근육에 힘이 과도하게 들어갔습니다. 강하게 힘을 주기보다 발 안쪽 아치를 부드럽게 유지하는 데 집중해 주세요."
        elif checks["weight_shift"]:
            msg = "발 전체가 한쪽으로 기울어진 상태입니다. 체중을 발 전체에 고르게 두고 아치만 형성해 주세요."
        elif self._arch_hold_feedback_needed():
            msg = "아치는 형성되었지만 유지 시간이 부족합니다. 짧게 힘을 주기보다 아치를 일정 시간 유지하는 연습이 필요합니다."
        else:
            msg = "좋습니다. 발 전체를 안정적으로 둔 상태에서 아치를 유지하고 있습니다."

        return {
            "rings": rings,
            "feedback_flags": flags,
            "coaching_message": msg,
            "rates": {k: round(float(v), 4) for k, v in rates.items()},
            "live_checks": checks,
            "live_metrics": {k: self._round_or_none(v, 4) for k, v in metrics.items()},
            "monitoring": self.calibration_monitoring(sample),
        }

    def analyze_exercise(self):
        if not self.exercise_data:
            return {
                "ok": False,
                "grade": "Bad",
                "score": 0.0,
                "feedback": ["운동 데이터가 없습니다."],
                "main_comment": "운동 데이터가 없습니다.",
                "rings": {"arch": 0, "toe": 0, "heel": 0, "emg": 0},
                "score_parts": {"arch": 0, "toe": 0, "heel": 0, "emg": 0},
                "score_detail": "운동 데이터 없음",
                "good_points": [],
                "improve_points": ["운동 데이터가 수집되지 않았습니다."],
                "feedback_flags": {"arch_good": False, "toe_over": False, "heel_unstable": False, "emg_low": True},
            }

        weights = self.config.get("score_weights", {"arch": 50, "emg": 30, "compensation": 20})
        rates = self._exercise_success_rates(self.exercise_data)
        avg = self._compute_mean_stats(self.exercise_data)

        arch_result = self.judge_arch(avg)
        emg_result = self.judge_emg(avg)
        toe_result = self.judge_toe(avg)
        heel_result = self.judge_heel(avg)

        arch_score = rates["arch"] * weights.get("arch", 50)
        emg_score = rates["emg"] * weights.get("emg", 30)
        compensation_score = rates["compensation"] * weights.get("compensation", 20)
        total_score = arch_score + emg_score + compensation_score

        # 최종 등급은 사용자가 제시한 아치 3등분 기준을 우선 반영한다.
        grade = arch_result["grade"]
        if grade == "Good" and rates["emg"] < 0.70:
            grade = "Normal"
        elif grade == "Normal" and rates["emg"] < 0.40:
            grade = "Bad"

        item_results = {
            "arch": arch_result,
            "emg": emg_result,
            "toe": toe_result,
            "heel": heel_result,
        }

        good_points = []
        improve_points = []

        if arch_result["grade"] == "Good":
            good_points.append("아치 압력 감소가 최대 아치 형성 상태에 가까워 Good 구간입니다.")
        elif arch_result["grade"] == "Normal":
            improve_points.append("아치가 부분적으로 형성되었습니다. 최대 아치 형성 상태에 조금 더 가깝게 유지해 보세요.")
        else:
            improve_points.append("아치 압력 감소가 부족하여 Bad 구간입니다. 발을 들기보다 발 안쪽 아치를 천천히 끌어올리세요.")

        if emg_result["grade"] == "Good":
            good_points.append("EMG가 30~50% MVIC 범위에 있어 적절한 근활성도입니다.")
        elif emg_result["grade"] == "Normal":
            improve_points.append("EMG가 목표 범위에 가깝지만 조금 낮거나 높습니다. 30~50% MVIC를 목표로 조절하세요.")
        else:
            improve_points.append("EMG가 적절 범위에서 벗어났습니다. 너무 약하거나 강하게 힘을 주지 않도록 조절하세요.")

        if rates["compensation"] >= 0.70:
            good_points.append(f"보상작용이 비교적 잘 억제되었습니다. 안정률 {rates['compensation'] * 100:.0f}%.")
        else:
            improve_points.append(f"운동 중 보상작용이 자주 감지되었습니다. 안정률 {rates['compensation'] * 100:.0f}%.")

        if not good_points:
            good_points.append("아직 뚜렷하게 안정적인 항목이 없습니다.")
        if not improve_points:
            improve_points.append("큰 개선점 없이 안정적으로 수행했습니다.")

        feedback = [item_results[key]["comment"] for key in ["arch", "emg", "toe", "heel"]]
        main_comment = self.grade_comment(grade)

        rings = {
            "arch": round(float(rates["arch"] * 100), 1),
            "toe": round(float(rates["toe"] * 100), 1),
            "heel": round(float(rates["heel"] * 100), 1),
            "emg": round(float(rates["emg"] * 100), 1),
        }

        score_parts = {
            "arch": round(float(arch_score), 1),
            "emg": round(float(emg_score), 1),
            "compensation": round(float(compensation_score), 1),
        }

        score_detail = (
            f"아치 형성 {arch_score:.1f}/{weights.get('arch', 50)} + "
            f"EMG 30~50% MVIC {emg_score:.1f}/{weights.get('emg', 30)} + "
            f"보상작용 억제 {compensation_score:.1f}/{weights.get('compensation', 20)} "
            f"= {total_score:.1f}점"
        )

        return {
            "ok": True,
            "grade": grade,
            "score": round(float(total_score), 1),
            "main_comment": main_comment,
            "feedback": feedback,
            "items": item_results,
            "rings": rings,
            "score_parts": score_parts,
            "score_detail": score_detail,
            "good_points": good_points,
            "improve_points": improve_points,
            "feedback_flags": {
                "arch_good": arch_result["grade"] in ["Normal", "Good"],
                "toe_over": rates["toe"] < 0.70,
                "heel_unstable": rates["heel"] < 0.70,
                "emg_low": rates["emg"] < 0.70,
            },
            "rates": {k: round(float(v), 4) for k, v in rates.items()},
            "arch_reference": arch_result.get("reference"),
        }

    def _score_emg_ratio(self, ratio, req):
        """
        EMG 기반 아치 형성 점수화.
        - req(기본 0.30) 이상이면 '적절한 수행'으로 인정한다.
        - 하지만 req를 넘었다고 바로 100점이 되지는 않게 한다.
        - req에서 약 70점, 최대 아치 EMG의 60% 이상에서 100점으로 본다.
        """
        if ratio is None:
            return 0.0

        ratio = max(0.0, float(ratio))
        req = max(float(req), 1e-8)
        perfect = self.rules.get("exercise_emg_perfect_ratio", max(req * 2.0, 0.60))

        if ratio < req:
            return min(0.69, ratio / req * 0.69)

        if abs(perfect - req) < 1e-8:
            return 1.0

        return min(1.0, 0.70 + (ratio - req) / (perfect - req) * 0.30)

    def _score_upper_limit(self, value, limit, allow_negative_as_good=True):
        """
        toe/heel처럼 '너무 커지면 나쁜' 항목을 점수화한다.
        기존처럼 조건을 만족하면 무조건 100점이 아니라,
        제한값에 가까워질수록 점수가 내려가게 한다.
        """
        if value is None or limit is None or abs(limit) < 1e-8:
            return 0.0

        value = float(value)
        limit = float(limit)

        if allow_negative_as_good and value <= 0:
            return 1.0

        if value >= limit:
            return 0.0

        return max(0.0, min(1.0, 1.0 - value / limit))

    def _exercise_success_rates(self, samples):
        arch_scores = []
        toe_scores = []
        heel_scores = []
        emg_scores = []
        compensation_scores = []

        emg_min = self.rules.get("exercise_emg_mvic_min_ratio", 0.30)
        emg_max = self.rules.get("exercise_emg_mvic_max_ratio", 0.50)

        for sample in samples:
            checks, _ = self._current_feedback_checks(sample)

            arch = self._classify_arch_from_sum(self._sample_arch_eval_sum(sample))
            arch_scores.append(arch["score"] if arch["score"] is not None else 0.0)

            emg_ratio = self._emg_mvic_ratio(self.get_value(sample, self.emg_key))
            if emg_ratio is None:
                emg_scores.append(0.0)
            elif emg_min <= emg_ratio <= emg_max:
                emg_scores.append(1.0)
            elif emg_ratio < emg_min:
                emg_scores.append(max(0.0, min(0.69, emg_ratio / max(emg_min, 1e-8) * 0.69)))
            else:
                # 50% MVIC를 넘으면 과활성으로 보고, 90% MVIC에서는 0점에 가깝게 낮춘다.
                upper_zero = self.rules.get("live_emg_excessive_mvic_ratio", 0.90)
                emg_scores.append(max(0.0, 1.0 - (emg_ratio - emg_max) / max(upper_zero - emg_max, 1e-8)))

            toe_ok = not (checks.get("toe_flexion_compensation") or checks.get("toe_no_contact"))
            heel_ok = not checks.get("heel_lift")
            compensation_ok = not any(checks.values())
            toe_scores.append(1.0 if toe_ok else 0.0)
            heel_scores.append(1.0 if heel_ok else 0.0)
            compensation_scores.append(1.0 if compensation_ok else 0.0)

        def rate(scores):
            return float(np.mean(scores)) if scores else 0.0

        return {
            "arch": rate(arch_scores),
            "toe": rate(toe_scores),
            "heel": rate(heel_scores),
            "emg": rate(emg_scores),
            "compensation": rate(compensation_scores),
        }

    def judge_toe(self, avg):
        toe_base = self._fsr_value(self.baseline, 1) if self.baseline else None
        toe_now = self._fsr_value(avg, 1)
        inc = self._safe_ratio(toe_now, toe_base)

        if inc is None:
            return {"name": "엄지 압력", "grade": "Bad", "ratio": None, "comment": "엄지 압력 계산 불가"}

        if inc >= self.rules.get("live_toe_flexion_increase_ratio", 0.30):
            grade = "Bad"
            comment = f"엄지발가락 굴곡 보상작용: 기준값 대비 {inc * 100:.1f}% 증가"
        elif toe_now is not None and toe_base is not None and toe_now / abs(toe_base) < self.rules.get("live_toe_contact_min_of_baseline", 0.10):
            grade = "Bad"
            comment = "엄지발가락 미접촉: 기준값 대비 10% 미만"
        else:
            grade = "Good"
            comment = f"엄지발가락 접촉과 과보상 억제가 양호합니다: 기준값 대비 {inc * 100:.1f}% 변화"

        return {"name": "엄지 압력", "grade": grade, "ratio": round(float(inc), 4), "comment": comment}

    def judge_arch(self, avg):
        k = avg.get("arch_eval_sum")
        arch = self._classify_arch_from_sum(k)
        n, m = arch["n"], arch["m"]
        progress = arch["progress"]

        if progress is None:
            return {
                "name": "아치 형성",
                "grade": "Bad",
                "ratio": None,
                "comment": "아치 평가 기준값 N, M 또는 운동값 K 계산 불가",
                "reference": {"N": n, "M": m, "K": k},
            }

        if arch["grade"] == "Good":
            comment = f"Good: K가 최대 아치 압력합 M에 가까운 구간입니다. 아치 형성도 {progress * 100:.1f}%."
        elif arch["grade"] == "Normal":
            comment = f"Normal: K가 N과 M의 중간 구간입니다. 아치 형성도 {progress * 100:.1f}%."
        else:
            comment = f"Bad: K가 정적 기준값 N에 가까운 구간입니다. 아치 형성도 {progress * 100:.1f}%."

        return {
            "name": "아치 형성",
            "grade": arch["grade"],
            "ratio": round(float(progress), 4),
            "comment": comment,
            "reference": {
                "N_baseline_arch_sum": None if n is None else round(float(n), 4),
                "M_max_lift_arch_sum": None if m is None else round(float(m), 4),
                "K_exercise_arch_sum": None if k is None else round(float(k), 4),
            },
        }

    def judge_emg(self, avg):
        now = avg.get(self.emg_key)
        ratio = self._emg_mvic_ratio(now)
        low = self.rules.get("exercise_emg_mvic_min_ratio", 0.30)
        high = self.rules.get("exercise_emg_mvic_max_ratio", 0.50)

        if ratio is None:
            return {"name": "EMG 활성도", "grade": "Bad", "ratio": None, "comment": "MVIC EMG 기준값 또는 현재 EMG 계산 불가"}

        if low <= ratio <= high:
            grade = "Good"
            comment = f"EMG 활성도 적절: {ratio * 100:.1f}% MVIC로 30~50% MVIC 범위입니다."
        elif ratio < low:
            grade = "Normal" if ratio >= low * 0.7 else "Bad"
            comment = f"EMG 활성도 낮음: {ratio * 100:.1f}% MVIC입니다. 목표는 30~50% MVIC입니다."
        else:
            grade = "Normal" if ratio <= self.rules.get("live_emg_excessive_mvic_ratio", 0.90) else "Bad"
            comment = f"EMG 활성도 높음: {ratio * 100:.1f}% MVIC입니다. 강하게 힘을 주기보다 부드럽게 유지하세요."

        return {"name": "EMG 활성도", "grade": grade, "ratio": round(float(ratio), 4), "comment": comment}

    def judge_heel(self, avg):
        heel_base = self.group_mean(self.baseline, "heel") if self.baseline else None
        heel_now = self.group_mean(avg, "heel")
        of_base = None if heel_base is None or heel_now is None or abs(heel_base) < 1e-8 else heel_now / abs(heel_base)

        if of_base is None:
            return {"name": "뒤꿈치 안정성", "grade": "Bad", "ratio": None, "comment": "뒤꿈치 압력 계산 불가"}

        if of_base < self.rules.get("live_heel_contact_min_of_baseline", 0.10):
            grade = "Bad"
            comment = "뒤꿈치가 바닥에서 떨어졌을 가능성: 기준값 대비 10% 미만"
        else:
            grade = "Good"
            comment = f"뒤꿈치 접촉 유지: 기준값 대비 {of_base * 100:.1f}%"

        return {"name": "뒤꿈치 안정성", "grade": grade, "ratio": round(float(of_base), 4), "comment": comment}

    def grade_comment(self, grade):
        if grade == "Good":
            return "전반적으로 좋은 숏풋 패턴입니다. 현재 감각을 유지하세요."
        if grade == "Normal":
            return "기본 동작은 가능하지만 일부 보상 움직임이 있습니다. 압력을 조금 더 안정적으로 유지하세요."
        return "보상 움직임이 뚜렷합니다. 힘을 줄이거나 발 위치를 다시 잡고 천천히 반복하세요."

    def summarize_exercises(self, exercise_results):
        if not exercise_results:
            return {
                "total_exercises": 0,
                "avg_score": 0.0,
                "best_grade_count": {},
                "expected_effect": "운동 기록이 없어 기대 효과를 산출할 수 없습니다.",
                "most_common_comment": "없음",
                "comments": [],
            }

        scores = [r.get("score", 0.0) for r in exercise_results]
        comments = []
        all_item_comments = []
        grade_counter = Counter()

        for idx, r in enumerate(exercise_results, 1):
            grade_counter[r.get("grade", "Bad")] += 1
            item_comments = r.get("feedback", [])
            all_item_comments.extend(item_comments)
            comments.append(f"{idx}회차: {r.get('grade')} / {r.get('score')}점 - {r.get('main_comment', '')}")

        most_common_comment = Counter(all_item_comments).most_common(1)[0][0] if all_item_comments else "없음"
        avg_score = float(np.mean(scores))

        expected_effect = "반복 수행을 통해 내재근 활성, 아치 유지 감각, 엄지 보상 감소 훈련 효과를 기대할 수 있습니다."

        if avg_score < 50:
            expected_effect = "현재는 보상 움직임이 많아 정확한 자세 재학습이 우선입니다. 낮은 강도에서 천천히 반복하세요."

        return {
            "total_exercises": len(exercise_results),
            "avg_score": round(avg_score, 1),
            "best_grade_count": dict(grade_counter),
            "expected_effect": expected_effect,
            "most_common_comment": most_common_comment,
            "comments": comments,
        }
