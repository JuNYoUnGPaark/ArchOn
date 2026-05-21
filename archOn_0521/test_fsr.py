import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

i2c = busio.I2C(board.SCL, board.SDA)

ads = ADS.ADS1115(i2c)

fsr = AnalogIn(ads, 0)

while True:
    print(f"VALUE={fsr.value}  VOLT={fsr.voltage:.3f}V")
    time.sleep(1)