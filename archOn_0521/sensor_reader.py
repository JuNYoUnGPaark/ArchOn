import board
import busio
import adafruit_tca9548a
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn


class SensorReader:
    def __init__(self):
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.tca = adafruit_tca9548a.TCA9548A(self.i2c, address=0x70)

        self.ads_list = {
            0: ADS.ADS1115(self.tca[0], address=0x48),
            1: ADS.ADS1115(self.tca[1], address=0x48),
        }

        self.fsr_map = {
            "fsr1": (0, 0),
            "fsr2": (0, 1),
            "fsr3": (0, 2),
            "fsr4": (0, 3),

            "fsr5": (1, 0),
            "fsr6": (1, 1),
            "fsr7": (1, 2),
            "fsr8": (1, 3),
        }

        self.fsr_channels = {}

        for name, (tca_channel, ads_channel) in self.fsr_map.items():
            self.fsr_channels[name] = AnalogIn(
                self.ads_list[tca_channel],
                ads_channel
            )

        self.emg_channel = None

    def read(self):
        data = {}

        for i in range(1, 19):
            name = f"fsr{i}"

            if name in self.fsr_channels:
                ch = self.fsr_channels[name]
                data[name] = int(ch.value)
                data[f"{name}_voltage"] = round(float(ch.voltage), 4)
            else:
                data[name] = None
                data[f"{name}_voltage"] = None

        if self.emg_channel is not None:
            data["emg"] = int(self.emg_channel.value)
            data["emg_voltage"] = round(float(self.emg_channel.voltage), 4)
        else:
            data["emg"] = None
            data["emg_voltage"] = None

        return data