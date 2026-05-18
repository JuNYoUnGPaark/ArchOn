#!/usr/bin/env python3
"""
Arch-On 센서 데이터 수신 코드
- FSR 406 x3 (전족부/중족부/후족부) → ADS1115 A0~A2
- EMG SZH-HWS010                  → ADS1115 A3
- Raspberry Pi 4 기준
"""

import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from collections import deque
import numpy as np

# ══════════════════════════════════════
# 설정값
# ══════════════════════════════════════

# FSR 판정 기준
CSI_THRESHOLD   = 45.0   # CSI 유사값 45% 이상 = 편평발 의심
FSR_MIN_ACTIVE  = 100    # FSR 최소 활성 감지값 (노이즈 필터)

# EMG 판정 기준
EMG_BASELINE_V  = 1.5    # EMG 기준 전압 (안정 시 약 1.5V)
EMG_ACTIVE_THR  = 0.15   # 기준 전압 대비 ±0.15V 이상이면 활성
TA_ABH_RATIO_THR = 1.0   # TA/AbH ratio 1.0 초과 = 보상작용

# 샘플링
SAMPLE_RATE     = 0.1    # 0.1초 간격 (10Hz)
RMS_WINDOW      = 20     # RMS 계산 윈도우 (20샘플 = 2초)

# ══════════════════════════════════════
# I2C 및 ADS1115 초기화
# ══════════════════════════════════════

def init_sensors():
    """센서 초기화 및 채널 설정"""
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        ads = ADS.ADS1115(i2c, address=0x48)

        # 채널 설정
        fsr_a = AnalogIn(ads, ADS.P0)   # 전족부
        fsr_b = AnalogIn(ads, ADS.P1)   # 중족부
        fsr_c = AnalogIn(ads, ADS.P2)   # 후족부
        emg   = AnalogIn(ads, ADS.P3)   # EMG (AbH 기준)

        print("✅ ADS1115 초기화 완료 (I2C 주소: 0x48)")
        print(f"   채널 구성: A0=전족부FSR | A1=중족부FSR | A2=후족부FSR | A3=EMG")
        return fsr_a, fsr_b, fsr_c, emg

    except Exception as e:
        print(f"❌ 센서 초기화 실패: {e}")
        print("   확인사항: I2C 활성화 여부 (raspi-config), 배선 점검")
        raise

# ══════════════════════════════════════
# FSR 처리 함수
# ══════════════════════════════════════

def read_fsr(fsr_a, fsr_b, fsr_c):
    """
    FSR 3채널 읽기 및 CSI 유사값 계산
    반환: (a값, b값, c값, 전압a, 전압b, 전압c, CSI값, 판정)
    """
    a = fsr_a.value
    b = fsr_b.value
    c = fsr_c.value
    va = fsr_a.voltage
    vb = fsr_b.voltage
    vc = fsr_c.voltage

    # 노이즈 필터 (최소값 이하는 0으로)
    a = a if a > FSR_MIN_ACTIVE else 0
    b = b if b > FSR_MIN_ACTIVE else 0
    c = c if c > FSR_MIN_ACTIVE else 0

    # CSI 유사값 계산: 중족부 / (전족부 + 후족부) x 100
    denom = a + c
    csi = (b / denom * 100) if denom > 0 else 0.0

    # 판정
    if a + b + c < FSR_MIN_ACTIVE * 3:
        status = "대기중"
    elif csi > CSI_THRESHOLD:
        status = "⚠️ 편평발 의심"
    else:
        status = "✅ 정상"

    return a, b, c, va, vb, vc, csi, status


def calc_arch_index(a, b, c):
    """
    Cavanagh Arch Index 유사값 계산
    전체 압력 대비 중족부 비율
    """
    total = a + b + c
    if total == 0:
        return 0.0
    return (b / total) * 100


# ══════════════════════════════════════
# EMG 처리 함수
# ══════════════════════════════════════

class EMGProcessor:
    """EMG 신호 처리 클래스 (RMS 계산 + 판정)"""

    def __init__(self, window_size=RMS_WINDOW):
        self.window   = deque(maxlen=window_size)
        self.baseline = EMG_BASELINE_V
        self.calibrated = False

    def calibrate(self, emg_ch, duration=3):
        """
        EMG 기준값(Baseline) 캘리브레이션
        - 환자를 안정 상태로 3초간 측정
        - 평균값을 기준 전압으로 설정
        """
        print(f"\n📐 EMG 캘리브레이션 시작 ({duration}초간 안정 상태 유지)")
        samples = []
        for i in range(duration * 10):
            samples.append(emg_ch.voltage)
            time.sleep(0.1)
            print(f"   측정 중... {i+1}/{duration*10}", end="\r")

        self.baseline = np.mean(samples)
        self.calibrated = True
        print(f"\n✅ 캘리브레이션 완료 — 기준 전압: {self.baseline:.3f}V")

    def process(self, emg_ch):
        """
        EMG 신호 처리
        1. 원시 전압 읽기
        2. 기준 전압 기준 편차 계산
        3. RMS 계산
        4. 활성도 % 산출
        반환: (원시전압, 편차, RMS, 활성도%, 판정)
        """
        raw_v   = emg_ch.voltage
        dev     = raw_v - self.baseline        # 기준 전압 대비 편차
        self.window.append(abs(dev))

        # RMS 계산
        rms = np.sqrt(np.mean(np.array(self.window) ** 2)) if self.window else 0.0

        # 활성도 % (최대 편차 1.5V 기준 정규화)
        activity_pct = min((rms / 1.5) * 100, 100.0)

        # 보상작용 판정 (단일 센서 사용 시)
        # 참고: 두 채널 사용 시 TA/AbH ratio로 대체
        if not self.calibrated:
            verdict = "캘리브레이션 필요"
        elif rms < EMG_ACTIVE_THR:
            verdict = "😴 근육 비활성"
        elif activity_pct > 60:
            verdict = "🔴 과활성 (보상 의심)"
        else:
            verdict = "🟢 정상 활성"

        return raw_v, dev, rms, activity_pct, verdict


# ══════════════════════════════════════
# 통합 판정 함수
# ══════════════════════════════════════

def integrated_verdict(csi, emg_activity, pronation_angle=None):
    """
    FSR + EMG + IMU(선택) 통합 판정
    반환: 수행 등급 문자열
    """
    fsr_pass = csi <= CSI_THRESHOLD
    emg_pass = 10 < emg_activity < 60   # 너무 낮거나 너무 높으면 Fail
    imu_pass = (pronation_angle is None) or (pronation_angle <= 6.0)

    pass_count = sum([fsr_pass, emg_pass, imu_pass])

    if pass_count == 3:
        return "✅ 완전 수행 — 모든 조건 통과"
    elif pass_count == 2:
        fails = []
        if not fsr_pass: fails.append("압력 분포")
        if not emg_pass: fails.append("근활성")
        if not imu_pass: fails.append("발목 자세")
        return f"△ 부분 수행 — {', '.join(fails)} 교정 필요"
    elif pass_count == 1:
        return "⚠️ 잘못된 수행 — 자세 재교정 필요"
    else:
        return "❌ 운동 미수행 또는 전체 실패"


# ══════════════════════════════════════
# 메인 루프
# ══════════════════════════════════════

def main():
    print("=" * 55)
    print("   Arch-On 센서 모니터링 시작")
    print("   FSR(전족/중족/후족) + EMG")
    print("=" * 55)

    # 센서 초기화
    fsr_a, fsr_b, fsr_c, emg_ch = init_sensors()
    emg_proc = EMGProcessor()

    # EMG 캘리브레이션 선택
    ans = input("\nEMG 캘리브레이션 진행하시겠습니까? (y/n): ").strip().lower()
    if ans == 'y':
        emg_proc.calibrate(emg_ch, duration=3)
    else:
        print(f"⚠️  기본 기준 전압 {EMG_BASELINE_V}V 사용")

    print("\n📊 실시간 측정 시작 (Ctrl+C로 종료)")
    print("-" * 55)

    session_data = []   # 세션 데이터 저장용

    try:
        while True:
            # ── FSR 읽기
            a, b, c, va, vb, vc, csi, fsr_status = read_fsr(fsr_a, fsr_b, fsr_c)
            arch_idx = calc_arch_index(a, b, c)

            # ── EMG 읽기
            raw_v, dev, rms, activity, emg_status = emg_proc.process(emg_ch)

            # ── 통합 판정
            verdict = integrated_verdict(csi, activity)

            # ── 출력
            ts = time.strftime("%H:%M:%S")
            print(f"\n[{ts}]")
            print(f"  FSR   전족:{a:5d}({va:.2f}V)  중족:{b:5d}({vb:.2f}V)  후족:{c:5d}({vc:.2f}V)")
            print(f"        CSI:{csi:5.1f}%  ArchIdx:{arch_idx:5.1f}%  {fsr_status}")
            print(f"  EMG   원시:{raw_v:.3f}V  편차:{dev:+.3f}V  RMS:{rms:.4f}  활성:{activity:5.1f}%")
            print(f"        {emg_status}")
            print(f"  판정  {verdict}")
            print(f"  {'─'*51}")

            # 데이터 저장
            session_data.append({
                'time'    : ts,
                'fsr_a'   : a, 'fsr_b': b, 'fsr_c': c,
                'csi'     : round(csi, 2),
                'arch_idx': round(arch_idx, 2),
                'emg_raw' : round(raw_v, 4),
                'emg_rms' : round(rms, 4),
                'emg_act' : round(activity, 2),
                'verdict' : verdict
            })

            time.sleep(SAMPLE_RATE)

    except KeyboardInterrupt:
        print("\n\n⏹️  측정 종료")
        _print_session_summary(session_data)


# ══════════════════════════════════════
# 세션 요약 출력
# ══════════════════════════════════════

def _print_session_summary(data):
    """세션 종료 후 요약 통계 출력"""
    if not data:
        return

    print("\n" + "=" * 55)
    print("   세션 요약")
    print("=" * 55)

    csi_vals  = [d['csi']      for d in data]
    emg_vals  = [d['emg_act']  for d in data]
    verdicts  = [d['verdict']  for d in data]

    print(f"  총 측정 횟수  : {len(data)}회")
    print(f"  측정 시간     : {data[0]['time']} ~ {data[-1]['time']}")
    print(f"  CSI 유사값    : 평균 {np.mean(csi_vals):.1f}%  최대 {max(csi_vals):.1f}%")
    print(f"  EMG 활성도    : 평균 {np.mean(emg_vals):.1f}%  최대 {max(emg_vals):.1f}%")

    complete = sum(1 for v in verdicts if "완전" in v)
    partial  = sum(1 for v in verdicts if "부분" in v)
    fail     = sum(1 for v in verdicts if "잘못" in v or "미수행" in v)
    accuracy = (complete / len(verdicts) * 100) if verdicts else 0

    print(f"  수행 정확도   : {accuracy:.1f}%")
    print(f"    완전 수행   : {complete}회")
    print(f"    부분 수행   : {partial}회")
    print(f"    실패        : {fail}회")

    # CSV 저장 여부
    ans = input("\nCSV로 저장하시겠습니까? (y/n): ").strip().lower()
    if ans == 'y':
        import csv, os
        fname = f"arch_on_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        with open(fname, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        print(f"✅ 저장 완료: {fname}")


if __name__ == "__main__":
    main()
