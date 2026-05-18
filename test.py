#!/usr/bin/env python3

import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from collections import deque
import numpy as np

CSI_THRESHOLD   = 45.0
FSR_MIN_ACTIVE  = 100

EMG_BASELINE_V  = 1.5
EMG_ACTIVE_THR  = 0.15
TA_ABH_RATIO_THR = 1.0

SAMPLE_RATE     = 0.1
RMS_WINDOW      = 20


def init_sensors():
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        ads = ADS.ADS1115(i2c, address=0x48)

        fsr_a = AnalogIn(ads, 0)
        fsr_b = AnalogIn(ads, 1)
        fsr_c = AnalogIn(ads, 2)
        emg   = AnalogIn(ads, 3)

        print("ADS1115 initialized (I2C address: 0x48)")
        print(f"Channel mapping: A0=FSR | A1=FSR | A2=FSR | A3=EMG")
        return fsr_a, fsr_b, fsr_c, emg

    except Exception as e:
        print(f"Sensor initialization failed: {e}")
        raise


def read_fsr(fsr_a, fsr_b, fsr_c):
    a = fsr_a.value
    b = fsr_b.value
    c = fsr_c.value

    va = fsr_a.voltage
    vb = fsr_b.voltage
    vc = fsr_c.voltage

    a = a if a > FSR_MIN_ACTIVE else 0
    b = b if b > FSR_MIN_ACTIVE else 0
    c = c if c > FSR_MIN_ACTIVE else 0

    denom = a + c
    csi = (b / denom * 100) if denom > 0 else 0.0

    if a + b + c < FSR_MIN_ACTIVE * 3:
        status = "IDLE"
    elif csi > CSI_THRESHOLD:
        status = "FLAT"
    else:
        status = "NORMAL"

    return a, b, c, va, vb, vc, csi, status


def calc_arch_index(a, b, c):
    total = a + b + c
    if total == 0:
        return 0.0
    return (b / total) * 100


class EMGProcessor:

    def __init__(self, window_size=RMS_WINDOW):
        self.window = deque(maxlen=window_size)
        self.baseline = EMG_BASELINE_V
        self.calibrated = False

    def calibrate(self, emg_ch, duration=3):
        print(f"\nEMG calibration start ({duration} sec)")
        samples = []

        for i in range(duration * 10):
            samples.append(emg_ch.voltage)
            time.sleep(0.1)
            print(f"Measuring... {i+1}/{duration*10}", end="\r")

        self.baseline = np.mean(samples)
        self.calibrated = True

        print(f"\nCalibration done | Baseline: {self.baseline:.3f}V")

    def process(self, emg_ch):
        raw_v = emg_ch.voltage
        dev = raw_v - self.baseline

        self.window.append(abs(dev))

        rms = np.sqrt(np.mean(np.array(self.window) ** 2)) if self.window else 0.0

        activity_pct = min((rms / 1.5) * 100, 100.0)

        if not self.calibrated:
            verdict = "CALIBRATION REQUIRED"
        elif rms < EMG_ACTIVE_THR:
            verdict = "LOW ACTIVITY"
        elif activity_pct > 60:
            verdict = "HIGH ACTIVITY"
        else:
            verdict = "NORMAL ACTIVITY"

        return raw_v, dev, rms, activity_pct, verdict


def integrated_verdict(csi, emg_activity, pronation_angle=None):

    fsr_pass = csi <= CSI_THRESHOLD
    emg_pass = 10 < emg_activity < 60
    imu_pass = (pronation_angle is None) or (pronation_angle <= 6.0)

    pass_count = sum([fsr_pass, emg_pass, imu_pass])

    if pass_count == 3:
        return "GOOD"
    elif pass_count == 2:
        fails = []

        if not fsr_pass:
            fails.append("PRESSURE")

        if not emg_pass:
            fails.append("EMG")

        if not imu_pass:
            fails.append("POSTURE")

        return f"PARTIAL ({', '.join(fails)})"

    elif pass_count == 1:
        return "BAD"

    else:
        return "FAIL"


def main():

    print("=" * 55)
    print("Arch-On Sensor Monitor")
    print("=" * 55)

    fsr_a, fsr_b, fsr_c, emg_ch = init_sensors()

    emg_proc = EMGProcessor()

    ans = input("\nRun EMG calibration? (y/n): ").strip().lower()

    if ans == 'y':
        emg_proc.calibrate(emg_ch, duration=3)
    else:
        print(f"Using default baseline voltage: {EMG_BASELINE_V}V")

    print("\nRealtime monitoring started (Ctrl+C to stop)")
    print("-" * 55)

    session_data = []

    try:
        while True:

            a, b, c, va, vb, vc, csi, fsr_status = read_fsr(
                fsr_a,
                fsr_b,
                fsr_c
            )

            arch_idx = calc_arch_index(a, b, c)

            raw_v, dev, rms, activity, emg_status = emg_proc.process(emg_ch)

            verdict = integrated_verdict(csi, activity)

            ts = time.strftime("%H:%M:%S")

            print(f"\n[{ts}]")
            print(f"FSR  A:{a:5d}({va:.2f}V)  B:{b:5d}({vb:.2f}V)  C:{c:5d}({vc:.2f}V)")
            print(f"     CSI:{csi:5.1f}%  ArchIdx:{arch_idx:5.1f}%  {fsr_status}")

            print(f"EMG  Raw:{raw_v:.3f}V  Dev:{dev:+.3f}V  RMS:{rms:.4f}  Activity:{activity:5.1f}%")
            print(f"     {emg_status}")

            print(f"Verdict: {verdict}")
            print(f"{'-'*55}")

            session_data.append({
                'time'    : ts,
                'fsr_a'   : a,
                'fsr_b'   : b,
                'fsr_c'   : c,
                'csi'     : round(csi, 2),
                'arch_idx': round(arch_idx, 2),
                'emg_raw' : round(raw_v, 4),
                'emg_rms' : round(rms, 4),
                'emg_act' : round(activity, 2),
                'verdict' : verdict
            })

            time.sleep(SAMPLE_RATE)

    except KeyboardInterrupt:
        print("\n\nMeasurement stopped")
        _print_session_summary(session_data)


def _print_session_summary(data):

    if not data:
        return

    print("\n" + "=" * 55)
    print("Session Summary")
    print("=" * 55)

    csi_vals = [d['csi'] for d in data]
    emg_vals = [d['emg_act'] for d in data]
    verdicts = [d['verdict'] for d in data]

    print(f"Total samples : {len(data)}")
    print(f"Time range    : {data[0]['time']} ~ {data[-1]['time']}")

    print(f"CSI avg/max   : {np.mean(csi_vals):.1f}% / {max(csi_vals):.1f}%")
    print(f"EMG avg/max   : {np.mean(emg_vals):.1f}% / {max(emg_vals):.1f}%")

    complete = sum(1 for v in verdicts if "GOOD" in v)
    partial  = sum(1 for v in verdicts if "PARTIAL" in v)
    fail     = sum(1 for v in verdicts if "BAD" in v or "FAIL" in v)

    accuracy = (complete / len(verdicts) * 100) if verdicts else 0

    print(f"Success rate  : {accuracy:.1f}%")
    print(f"GOOD          : {complete}")
    print(f"PARTIAL       : {partial}")
    print(f"FAIL          : {fail}")

    ans = input("\nSave as CSV? (y/n): ").strip().lower()

    if ans == 'y':
        import csv

        fname = f"arch_on_{time.strftime('%Y%m%d_%H%M%S')}.csv"

        with open(fname, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

        print(f"Saved: {fname}")


if __name__ == "__main__":
    main()
