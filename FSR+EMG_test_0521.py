"""
(1) 실시간 수집 코드
    - 라즈베리파이 + ADS1115
    - 1초마다 FSR/EMG 읽기
    - 판단 로직 실행
    - 터미널 로그 출력
    - 종료 시 JSON 저장

(2) 분석/시각화 코드
    - 저장된 JSON 읽기
    - pandas DataFrame 변환
    - matplotlib 시각화
    - 상태 변화 분석

[
  {
    "time": "20:31:01",
    "fsr": 4821,
    "fsr_voltage": 0.61,
    "emg": 22310,
    "emg_voltage": 2.78,
    "pressure_state": "good",
    "emg_state": "underactive",
    "feedback": "압력은 적절하지만 근육 활성 부족"
  }
]
"""

#!/usr/bin/env python3

import time
import json
import board
import busio
import numpy as np
import adafruit_ads1x15.ads1115 as ADS

from adafruit_ads1x15.analog_in import AnalogIn


i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2c, address=0x48)

fsr = AnalogIn(ads, ADS.P0)
emg = AnalogIn(ads, ADS.P3)

session_data = []

print("Monitoring started")
print("type 'exit' to save and quit\n")


def judge_pressure(v):

    if v < 0.4:
        return "underload"

    elif v < 1.5:
        return "good"

    else:
        return "overload"


def judge_emg(v):

    if v < 2.72:
        return "underactive"

    elif v < 2.90:
        return "good"

    else:
        return "overactive"


def generate_feedback(p_state, e_state):

    if p_state == "underload":
        return "압력이 부족합니다"

    if p_state == "overload":
        return "압력이 과합니다"

    if e_state == "underactive":
        return "근육 활성 부족"

    if e_state == "overactive":
        return "근육 과활성"

    return "상태 양호"


while True:

    fsr_value = fsr.value
    fsr_voltage = fsr.voltage

    emg_value = emg.value
    emg_voltage = emg.voltage

    pressure_state = judge_pressure(fsr_voltage)
    emg_state = judge_emg(emg_voltage)

    feedback = generate_feedback(
        pressure_state,
        emg_state
    )

    ts = time.strftime("%H:%M:%S")

    item = {
        "time": ts,
        "fsr": int(fsr_value),
        "fsr_voltage": round(fsr_voltage, 3),
        "emg": int(emg_value),
        "emg_voltage": round(emg_voltage, 3),
        "pressure_state": pressure_state,
        "emg_state": emg_state,
        "feedback": feedback
    }

    session_data.append(item)

    print("\n" + "=" * 50)
    print(f"[{ts}]")

    print(
        f"FSR : {fsr_value:6d} "
        f"({fsr_voltage:.3f}V) "
        f"| {pressure_state}"
    )

    print(
        f"EMG : {emg_value:6d} "
        f"({emg_voltage:.3f}V) "
        f"| {emg_state}"
    )

    print(f"Feedback : {feedback}")

    cmd = input("\ncontinue ? (enter / exit): ")

    if cmd.lower() == "exit":

        filename = (
            "session_"
            + time.strftime("%Y%m%d_%H%M%S")
            + ".json"
        )

        with open(filename, "w") as f:
            json.dump(
                session_data,
                f,
                indent=4
            )

        print(f"\nsaved -> {filename}")
        break

    time.sleep(1)
