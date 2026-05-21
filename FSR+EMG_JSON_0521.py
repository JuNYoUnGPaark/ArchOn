import json
import pandas as pd
import matplotlib.pyplot as plt


filename = "session_20260521_203000.json"

with open(filename, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

print(df.head())


plt.figure(figsize=(12, 5))

plt.plot(
    df["fsr_voltage"],
    label="FSR Voltage"
)

plt.plot(
    df["emg_voltage"],
    label="EMG Voltage"
)

plt.xlabel("Sample")
plt.ylabel("Voltage")
plt.title("FSR / EMG Signal")

plt.legend()
plt.grid()

plt.show()
