#!/usr/bin/env python3

import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2c, address=0x48)

fsr = AnalogIn(ads, 0)
emg = AnalogIn(ads, 3)

print("ADS1115 connected")

while True:

    fsr_value = fsr.value
    fsr_voltage = fsr.voltage

    emg_value = emg.value
    emg_voltage = emg.voltage

    ts = time.strftime("%H:%M:%S")

    print("\n" + "=" * 50)
    print(f"[{ts}]")

    print(
        f"FSR : {fsr_value:6d} ({fsr_voltage:.3f}V)"
    )

    print(
        f"EMG : {emg_value:6d} ({emg_voltage:.3f}V)"
    )

    time.sleep(3)
