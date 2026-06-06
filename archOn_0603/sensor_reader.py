import time
import numpy as np
import serial

import board
import busio
import adafruit_tca9548a
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn


class SensorReader:
    def __init__(self):
        # =========================
        # FSR: Raspberry Pi + ADS1115
        # =========================
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.tca = adafruit_tca9548a.TCA9548A(self.i2c, address=0x70)

        self.ads_list = {
            0: ADS.ADS1115(self.tca[0], address=0x48),
            1: ADS.ADS1115(self.tca[1], address=0x48),
            2: ADS.ADS1115(self.tca[2], address=0x48),

            3: ADS.ADS1115(self.tca[3], address=0x48),
            4: ADS.ADS1115(self.tca[4], address=0x48),
        }

        for ads in self.ads_list.values():
            ads.data_rate = 128

        self.fsr_map = {
            "fsr1": (0, 0), "fsr2": (0, 1), "fsr3": (0, 2), "fsr4": (0, 3),
            "fsr5": (1, 0), "fsr6": (1, 1), "fsr7": (1, 2), "fsr8": (1, 3),
            "fsr9": (2, 0), "fsr10": (2, 1), "fsr11": (2, 2), "fsr12": (2, 3),
            "fsr13": (3, 0), "fsr14": (3, 1), "fsr15": (3, 2), "fsr16": (3, 3),
            "fsr17": (4, 0), "fsr18": (4, 1),
        }

        self.fsr_channels = {}
        for name, (tca_channel, ads_channel) in self.fsr_map.items():
            try:
                ch = AnalogIn(self.ads_list[tca_channel], ads_channel)
                _ = ch.voltage
                self.fsr_channels[name] = ch
                print(f"[FSR] {name} OK: TCA {tca_channel}, ADS ch {ads_channel}")
            except Exception as e:
                self.fsr_channels[name] = None
                print(f"[FSR] {name} FAIL: TCA {tca_channel}, ADS ch {ads_channel}, error={e}")

        # =========================
        # EMG: Arduino Serial
        # =========================
        self.arduino_port = "/dev/ttyACM0"
        self.arduino_baudrate = 115200
        self.arduino = None

        try:
            self.arduino = serial.Serial(
                self.arduino_port,
                self.arduino_baudrate,
                timeout=0.02
            )
            time.sleep(2)
            self.arduino.reset_input_buffer()
            print(f"[EMG] Arduino connected: {self.arduino_port}")
        except Exception as e:
            print(f"[EMG] Arduino connection failed: {e}")
            self.arduino = None

    def read_emg_raw(self):
        if self.arduino is None:
            return None

        try:
            line = self.arduino.readline().decode("utf-8", errors="ignore").strip()

            if not line:
                return None

            # 숫자만 들어오는 경우
            try:
                return float(line)
            except ValueError:
                pass

            # 기존 형식: Zero_Line:0,Max_Limit:2000,Muscle_Power:2048.00
            if "Muscle_Power:" in line:
                value = line.split("Muscle_Power:")[-1].strip()
                return float(value)

            return None

        except Exception:
            return None

    def read(self):
        data = {}

        # =========================
        # FSR 읽기
        # =========================
        for i in range(1, 19):
            name = f"fsr{i}"

            ch = self.fsr_channels.get(name)

            if ch is not None:
                try:
                    data[name] = int(ch.value)
                    data[f"{name}_voltage"] = round(float(ch.voltage), 4)
                except Exception as e:
                    print(f"[FSR READ FAIL] {name}: {e}")
                    data[name] = None
                    data[f"{name}_voltage"] = None
            else:
                data[name] = None
                data[f"{name}_voltage"] = None

        # =========================
        # EMG는 Arduino에서 읽음
        # =========================
        emg_value = self.read_emg_raw()

        if emg_value is not None:
            data["emg"] = round(float(emg_value), 4)
            data["emg_voltage"] = round(float(emg_value), 4)
        else:
            data["emg"] = None
            data["emg_voltage"] = None

        return data

    def read_emg_window(self, duration=0.5, fs=None):
        """
        duration 동안 Arduino EMG 값을 여러 개 모아서
        mean, rms, peak를 계산한다.

        sensor_config.json에서 emg_key가 emg_rms이므로
        실제 판정에는 emg_rms가 사용된다.
        """
        values = []
        start = time.time()

        while time.time() - start < duration:
            v = self.read_emg_raw()

            if v is not None:
                values.append(v)

            if fs is not None:
                time.sleep(1.0 / fs)

        if not values:
            return {
                "emg_voltage": None,
                "emg_rms": None,
                "emg_peak": None,
                "emg_sample_count": 0,
            }

        arr = np.array(values, dtype=float)

        return {
            "emg_voltage": round(float(np.mean(arr)), 4),
            "emg_rms": round(float(np.sqrt(np.mean(np.square(arr)))), 4),
            "emg_peak": round(float(np.max(arr)), 4),
            "emg_sample_count": len(values),
        }
