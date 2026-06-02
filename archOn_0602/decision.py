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
        return [v for s in samples if (v := self.get_value(s, key)) is not None]

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
        stats["fsr_total_mean"] = float(np.mean([v for i in range(1, self.fsr_count + 1) if (v := stats.get(self.fsr_key(i))) is not None])) if any(stats.get(self.fsr_key(i)) is not None for i in range(1, self.fsr_count + 1)) else None
        return stats

    def system_check(self, samples):
        disconnected, warnings = [], []
        severe_low = self.th.get("severe_dead_low", -0.02)
        severe_high = self.th.get("severe_dead_high", 3.9)
        soft_low = self.th.get("sensor_dead_low", -0.05)
        soft_high = self.th.get("sensor_dead_high", 3.6)

        for i in range(1, self.fsr_count + 1):
            key = self.fsr_key(i)
            values = self.values_for_key(samples, key)
            if not values:
                disconnected.append(f"FSR{i}")
                continue
            avg = float(np.mean(values))
            if avg <= severe_low or avg >= severe_high:
                disconnected.append(f"FSR{i}")
            elif avg <= soft_low or avg >= soft_high:
                warnings.append(f"FSR{i}")

        emg_values = self.values_for_key(samples, self.emg_key)
        if not emg_values:
            disconnected.append("EMG")
        else:
            emg_avg = float(np.mean(emg_values))
            if emg_avg <= severe_low or emg_avg >= severe_high:
                disconnected.append("EMG")
            elif emg_avg <= soft_low or emg_avg >= soft_high:
                warnings.append("EMG")

        total = self.fsr_count + 1
        connected_ratio = (total - len(disconnected)) / total
        min_connected_ratio = self.th.get("min_connected_ratio", 0.72)
        # 웬만하면 통과: 일부 센서 경고/부분 오류는 warning으로만 표시하고 진행.
        ok = connected_ratio >= min_connected_ratio and len(disconnected) <= 4
        need_reconnect = not ok
        return {
            "ok": ok,
            "need_reconnect": need_reconnect,
            "disconnected": disconnected,
            "warnings": warnings,
            "connected_ratio": round(float(connected_ratio), 3),
            "message": "재연결 필요" if need_reconnect else "점검 통과"
        }

    def compute_baseline(self, samples):
        baseline = self._compute_mean_stats(samples)
        self.baseline = baseline
        return {**baseline, "_ok": True, "_feedback": ["기준값 저장 완료"]}

    def realtime_max_lift_metrics(self, sample):
        if self.baseline is None:
            return {}
        toe_base = self.group_mean(self.baseline, "toe")
        toe_now = self.group_mean(sample, "toe")
        heel_base = self.group_mean(self.baseline, "heel")
        heel_now = self.group_mean(sample, "heel")
        toe_decrease = None if toe_base is None or toe_now is None or abs(toe_base) < 1e-8 else (toe_base - toe_now) / abs(toe_base)
        heel_shift = None if heel_base is None or heel_now is None or abs(heel_base) < 1e-8 else abs(heel_now - heel_base) / abs(heel_base)
        return {
            "toe_decrease_ratio": None if toe_decrease is None else round(float(toe_decrease), 4),
            "toe_decrease_percent": None if toe_decrease is None else round(float(toe_decrease * 100), 1),
            "heel_shift_ratio": None if heel_shift is None else round(float(heel_shift), 4),
            "heel_shift_percent": None if heel_shift is None else round(float(heel_shift * 100), 1)
        }

    def compute_max_lift(self, samples):
        result = self._compute_mean_stats(samples)
        metrics = self._validate_max_lift_samples(samples, result)
        ok = metrics["toe_decrease_ratio"] is not None and metrics["toe_decrease_ratio"] >= self.rules.get("max_lift_toe_decrease_min", 0.15)
        ok = ok and metrics["toe_decrease_range"] is not None and metrics["toe_decrease_range"] <= self.rules.get("max_lift_toe_variation_limit", 0.70)
        ok = ok and metrics["heel_shift_ratio"] is not None and metrics["heel_shift_ratio"] <= self.rules.get("max_lift_heel_shift_limit", 0.50)
        feedback = []
        if metrics["toe_decrease_ratio"] is None or metrics["toe_decrease_ratio"] < self.rules.get("max_lift_toe_decrease_min", 0.15):
            feedback.append("엄지발가락 압력이 기준값 대비 15% 이상 감소하지 않았습니다.")
        if metrics["toe_decrease_range"] is None or metrics["toe_decrease_range"] > self.rules.get("max_lift_toe_variation_limit", 0.70):
            feedback.append("5초 동안 압력 변화가 너무 큽니다. 더 안정적으로 유지하세요.")
        if metrics["heel_shift_ratio"] is None or metrics["heel_shift_ratio"] > self.rules.get("max_lift_heel_shift_limit", 0.50):
            feedback.append("뒤꿈치 압력 변화가 큽니다. 뒤꿈치를 고정하세요.")
        if ok:
            feedback.append("최대 아치 형성값 저장 완료")
            self.max_lift = result
        return {**result, "_ok": ok, "_feedback": feedback, "_metrics": metrics}

    def _validate_max_lift_samples(self, samples, avg):
        toe_base = self.group_mean(self.baseline, "toe") if self.baseline else None
        heel_base = self.group_mean(self.baseline, "heel") if self.baseline else None
        toe_lift = self.group_mean(avg, "toe")
        heel_lift = self.group_mean(avg, "heel")
        toe_decrease = None if toe_base is None or toe_lift is None or abs(toe_base) < 1e-8 else (toe_base - toe_lift) / abs(toe_base)
        heel_shift = None if heel_base is None or heel_lift is None or abs(heel_base) < 1e-8 else abs(heel_lift - heel_base) / abs(heel_base)
        per_sample = []
        if toe_base is not None and abs(toe_base) > 1e-8:
            for s in samples:
                toe_now = self.group_mean(s, "toe")
                if toe_now is not None:
                    per_sample.append((toe_base - toe_now) / abs(toe_base))
        toe_range = float(np.max(per_sample) - np.min(per_sample)) if per_sample else None
        return {
            "toe_decrease_ratio": None if toe_decrease is None else round(float(toe_decrease), 4),
            "toe_decrease_percent": None if toe_decrease is None else round(float(toe_decrease * 100), 1),
            "toe_decrease_range": None if toe_range is None else round(float(toe_range), 4),
            "heel_shift_ratio": None if heel_shift is None else round(float(heel_shift), 4),
            "heel_shift_percent": None if heel_shift is None else round(float(heel_shift * 100), 1)
        }

    def add_mvic_trial(self, samples):
        stats = self._compute_mean_stats(samples)
        trial = {
            "trial_index": len(self.mvic_trials) + 1,
            "toe_mean": stats.get("toe_mean"),
            "arch_mean": stats.get("arch_mean"),
            "heel_mean": stats.get("heel_mean"),
            "emg_voltage": stats.get(self.emg_key),
            "stats": stats
        }
        self.mvic_trials.append(trial)
        required = self.config["timing"].get("mvic_trials", 3)
        toe_values = np.array([t["toe_mean"] for t in self.mvic_trials if t.get("toe_mean") is not None], dtype=float)
        mean = float(np.mean(toe_values)) if len(toe_values) else None
        std = float(np.std(toe_values)) if len(toe_values) else None
        cv = float(std / abs(mean) * 100.0) if mean is not None and abs(mean) > 1e-8 else None
        if len(self.mvic_trials) < required:
            return {"ok": False, "trial_count": len(self.mvic_trials), "cv_percent": cv, "trials": self.mvic_trials, "feedback": [f"MVIC {len(self.mvic_trials)}/{required}회 완료"]}
        limit = self.rules.get("mvic_cv_limit_percent", 60.0)
        if cv is None or cv > limit:
            old_cv = cv
            self.mvic_trials = []
            self.mvic = None
            return {"ok": False, "trial_count": required, "cv_percent": old_cv, "trials": [], "reset": True, "feedback": [f"MVIC 변동계수 {old_cv:.1f}%로 {limit:.0f}% 초과. 1회차부터 다시 측정하세요."]}
        self.mvic = {
            "toe_mean": mean,
            "emg_voltage": float(np.mean([t["emg_voltage"] for t in self.mvic_trials if t.get("emg_voltage") is not None])) if any(t.get("emg_voltage") is not None for t in self.mvic_trials) else None,
            "cv_percent": cv,
            "trials": self.mvic_trials
        }
        return {"ok": True, "trial_count": required, "cv_percent": cv, "mvic": self.mvic, "trials": self.mvic_trials, "feedback": ["MVIC 기준값 저장 완료"]}

    def start_exercise(self):
        self.exercise_data = []

    def update_exercise(self, sample):
        self.exercise_data.append(sample)

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
                "feedback_flags": {"arch_good": False, "toe_over": False, "heel_unstable": False, "emg_low": True}
            }

        weights = self.config.get("score_weights", {
            "arch": 35,
            "toe": 25,
            "heel": 20,
            "emg": 20
        })

        rates = self._exercise_success_rates(self.exercise_data)

        arch_score = rates["arch"] * weights.get("arch", 35)
        toe_score = rates["toe"] * weights.get("toe", 25)
        heel_score = rates["heel"] * weights.get("heel", 20)
        emg_score = rates["emg"] * weights.get("emg", 20)
        total_score = arch_score + toe_score + heel_score + emg_score
        grade = "Good" if total_score >= 80 else "Normal" if total_score >= 50 else "Bad"

        avg = self._compute_mean_stats(self.exercise_data)
        item_results = {
            "arch": self.judge_arch(avg),
            "toe": self.judge_toe(avg),
            "heel": self.judge_heel(avg),
            "emg": self.judge_emg(avg)
        }

        good_points = []
        improve_points = []

        if rates["arch"] >= 0.70:
            good_points.append(f"아치 유지가 안정적입니다. 조건 만족률 {rates['arch'] * 100:.0f}%.")
        else:
            improve_points.append(f"아치 유지가 부족합니다. 조건 만족률 {rates['arch'] * 100:.0f}%.")

        if rates["toe"] >= 0.70:
            good_points.append(f"엄지 과보상이 잘 억제되었습니다. 안정률 {rates['toe'] * 100:.0f}%.")
        else:
            improve_points.append(f"엄지 압력이 약간 과합니다. 과보상 억제율 {rates['toe'] * 100:.0f}%.")

        if rates["heel"] >= 0.70:
            good_points.append(f"뒤꿈치 압력이 안정적입니다. 안정률 {rates['heel'] * 100:.0f}%.")
        else:
            improve_points.append(f"뒤꿈치 압력이 흔들립니다. 안정률 {rates['heel'] * 100:.0f}%.")

        if rates["emg"] >= 0.70:
            good_points.append(f"EMG 활성도가 충분합니다. 활성 조건 만족률 {rates['emg'] * 100:.0f}%.")
        else:
            improve_points.append(f"EMG 활성도가 부족합니다. 활성 조건 만족률 {rates['emg'] * 100:.0f}%.")

        if not good_points:
            good_points.append("아직 뚜렷하게 안정적인 항목이 없습니다.")
        if not improve_points:
            improve_points.append("큰 개선점 없이 안정적으로 수행했습니다.")

        feedback = [item_results[key]["comment"] for key in ["arch", "toe", "heel", "emg"]]
        main_comment = self.grade_comment(grade)

        rings = {
            "arch": round(float(rates["arch"] * 100), 1),
            "toe": round(float(rates["toe"] * 100), 1),
            "heel": round(float(rates["heel"] * 100), 1),
            "emg": round(float(rates["emg"] * 100), 1)
        }

        score_parts = {
            "arch": round(float(arch_score), 1),
            "toe": round(float(toe_score), 1),
            "heel": round(float(heel_score), 1),
            "emg": round(float(emg_score), 1)
        }

        score_detail = (
            f"아치 유지 {arch_score:.1f}/{weights.get('arch', 35)} + "
            f"엄지 과보상 억제 {toe_score:.1f}/{weights.get('toe', 25)} + "
            f"뒤꿈치 안정성 {heel_score:.1f}/{weights.get('heel', 20)} + "
            f"EMG 활성도 {emg_score:.1f}/{weights.get('emg', 20)} "
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
                "arch_good": rates["arch"] >= 0.70,
                "toe_over": rates["toe"] < 0.70,
                "heel_unstable": rates["heel"] < 0.70,
                "emg_low": rates["emg"] < 0.70
            },
            "rates": {k: round(float(v), 4) for k, v in rates.items()}
        }

    def _exercise_success_rates(self, samples):
        arch_flags = []
        toe_flags = []
        heel_flags = []
        emg_flags = []

        arch_req = self.rules.get("exercise_arch_decrease_min_ratio", 0.15)
        toe_limit = self.rules.get("exercise_toe_increase_limit", 0.60)
        heel_limit = self.rules.get("exercise_heel_decrease_limit", 0.50)
        emg_req = self.rules.get("exercise_emg_increase_min_ratio", 0.05)

        toe_base = self.group_mean(self.baseline, "toe") if self.baseline else None
        arch_base = self.group_mean(self.baseline, "arch") if self.baseline else None
        heel_base = self.group_mean(self.baseline, "heel") if self.baseline else None
        emg_base = self.baseline.get(self.emg_key) if self.baseline else None

        for sample in samples:
            toe_now = self.group_mean(sample, "toe")
            arch_now = self.group_mean(sample, "arch")
            heel_now = self.group_mean(sample, "heel")
            emg_now = self.get_value(sample, self.emg_key)

            toe_inc = self._safe_ratio(toe_now, toe_base)
            arch_dec = None if arch_base is None or arch_now is None or abs(arch_base) < 1e-8 else (arch_base - arch_now) / abs(arch_base)
            heel_dec = None if heel_base is None or heel_now is None or abs(heel_base) < 1e-8 else (heel_base - heel_now) / abs(heel_base)
            emg_inc = self._safe_ratio(emg_now, emg_base)

            toe_flags.append(toe_inc is not None and toe_inc <= toe_limit)
            arch_flags.append(arch_dec is not None and arch_dec >= arch_req)
            heel_flags.append(heel_dec is not None and heel_dec <= heel_limit)
            emg_flags.append(emg_inc is not None and emg_inc >= emg_req)

        def rate(flags):
            return float(np.mean(flags)) if flags else 0.0

        return {
            "arch": rate(arch_flags),
            "toe": rate(toe_flags),
            "heel": rate(heel_flags),
            "emg": rate(emg_flags)
        }

    def judge_toe(self, avg):
        toe_base = self.group_mean(self.baseline, "toe") if self.baseline else None
        toe_now = self.group_mean(avg, "toe")
        inc = self._safe_ratio(toe_now, toe_base)
        limit = self.rules.get("exercise_toe_increase_limit", 0.60)
        if inc is None:
            return {"name": "엄지 압력", "grade": "Bad", "ratio": None, "comment": "엄지 압력 계산 불가"}
        if inc <= limit * 0.7:
            grade, comment = "Good", f"엄지 보상 양호: 기준값 대비 {inc*100:.1f}% 증가"
        elif inc <= limit:
            grade, comment = "Normal", f"엄지 압력이 약간 큼: 기준값 대비 {inc*100:.1f}% 증가"
        else:
            grade, comment = "Bad", f"엄지로 과하게 누름: 기준값 대비 {inc*100:.1f}% 증가"
        return {"name": "엄지 압력", "grade": grade, "ratio": round(float(inc), 4), "comment": comment}

    def judge_arch(self, avg):
        arch_base = self.group_mean(self.baseline, "arch") if self.baseline else None
        arch_now = self.group_mean(avg, "arch")
        dec = None if arch_base is None or arch_now is None or abs(arch_base) < 1e-8 else (arch_base - arch_now) / abs(arch_base)
        req = self.rules.get("exercise_arch_decrease_min_ratio", 0.15)
        if dec is None:
            return {"name": "아치 유지", "grade": "Bad", "ratio": None, "comment": "아치 압력 계산 불가"}
        if dec >= req * 1.5:
            grade, comment = "Good", f"아치 형성 좋음: 기준값 대비 {dec*100:.1f}% 감소 유지"
        elif dec >= req:
            grade, comment = "Normal", f"아치 형성 최소 기준 통과: 기준값 대비 {dec*100:.1f}% 감소"
        else:
            grade, comment = "Bad", f"아치 형성이 부족함: 기준값 대비 {dec*100:.1f}% 감소"
        return {"name": "아치 유지", "grade": grade, "ratio": round(float(dec), 4), "comment": comment}

    def judge_emg(self, avg):
        base = self.baseline.get(self.emg_key) if self.baseline else None
        now = avg.get(self.emg_key)
        inc = self._safe_ratio(now, base)
        req = self.rules.get("exercise_emg_increase_min_ratio", 0.05)
        if inc is None:
            return {"name": "EMG 활성도", "grade": "Bad", "ratio": None, "comment": "EMG 활성도 계산 불가"}
        if inc >= req * 2:
            grade, comment = "Good", f"EMG 활성도 좋음: 기준값 대비 {inc*100:.1f}% 증가"
        elif inc >= req:
            grade, comment = "Normal", f"EMG 활성도 최소 기준 통과: 기준값 대비 {inc*100:.1f}% 증가"
        else:
            grade, comment = "Bad", f"EMG 활성도가 부족함: 기준값 대비 {inc*100:.1f}% 증가"
        return {"name": "EMG 활성도", "grade": grade, "ratio": round(float(inc), 4), "comment": comment}

    def judge_heel(self, avg):
        heel_base = self.group_mean(self.baseline, "heel") if self.baseline else None
        heel_now = self.group_mean(avg, "heel")
        dec = None if heel_base is None or heel_now is None or abs(heel_base) < 1e-8 else (heel_base - heel_now) / abs(heel_base)
        limit = self.rules.get("exercise_heel_decrease_limit", 0.50)
        if dec is None:
            return {"name": "뒤꿈치 안정성", "grade": "Bad", "ratio": None, "comment": "뒤꿈치 압력 계산 불가"}
        if dec <= limit * 0.5:
            grade, comment = "Good", f"뒤꿈치 안정적: 기준값 대비 {dec*100:.1f}% 감소"
        elif dec <= limit:
            grade, comment = "Normal", f"뒤꿈치 압력 변화 주의: 기준값 대비 {dec*100:.1f}% 감소"
        else:
            grade, comment = "Bad", f"뒤꿈치가 들렸을 가능성: 기준값 대비 {dec*100:.1f}% 감소"
        return {"name": "뒤꿈치 안정성", "grade": grade, "ratio": round(float(dec), 4), "comment": comment}

    def grade_comment(self, grade):
        if grade == "Good":
            return "전반적으로 좋은 숏풋 패턴입니다. 현재 감각을 유지하세요."
        if grade == "Normal":
            return "기본 동작은 가능하지만 일부 보상 움직임이 있습니다. 압력을 조금 더 안정적으로 유지하세요."
        return "보상 움직임이 뚜렷합니다. 힘을 줄이거나 발 위치를 다시 잡고 천천히 반복하세요."

    def summarize_exercises(self, exercise_results):
        if not exercise_results:
            return {"total_exercises": 0, "avg_score": 0.0, "best_grade_count": {}, "expected_effect": "운동 기록이 없어 기대 효과를 산출할 수 없습니다.", "most_common_comment": "없음", "comments": []}
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
            "comments": comments
        }
