import board
import busio
import adafruit_tca9548a
import adafruit_ads1x15.ads1115 as ADS

i2c = busio.I2C(board.SCL, board.SDA)
tca = adafruit_tca9548a.TCA9548A(i2c, address=0x70)

for ch in range(8):
    try:
        ads = ADS.ADS1115(tca[ch], address=0x48)
        print(f"TCA channel {ch}: ADS1115 OK")
    except Exception:
        print(f"TCA channel {ch}: 없음")