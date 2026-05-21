import json
import sys
import pandas as pd
import matplotlib.pyplot as plt


if len(sys.argv) < 2:
    print("사용법: python3 visualize.py data/session_xxxxx.json")
    sys.exit(1)

filename = sys.argv[1]

with open(filename, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

print(df.head())

print("\n총 샘플 수:", len(df))

plt.figure(figsize=(14, 6))

for i in range(1, 19):
    col = f"fsr{i}_voltage"

    if col in df.columns and df[col].notna().any():
        plt.plot(df[col], label=f"FSR{i}")

if "emg_voltage" in df.columns:
    plt.plot(df["emg_voltage"], label="EMG", linewidth=2)

plt.xlabel("Sample")
plt.ylabel("Voltage")
plt.title("FSR 1~18 / EMG Voltage")
plt.legend(ncol=4)
plt.grid()
plt.tight_layout()
plt.show()

if "pressure_state" in df.columns:
    state_counts = df["pressure_state"].value_counts()

    plt.figure(figsize=(7, 5))
    plt.bar(state_counts.index, state_counts.values)
    plt.xlabel("Pressure State")
    plt.ylabel("Count")
    plt.title("Pressure State Distribution")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

if "emg_state" in df.columns:
    emg_counts = df["emg_state"].value_counts()

    plt.figure(figsize=(7, 5))
    plt.bar(emg_counts.index, emg_counts.values)
    plt.xlabel("EMG State")
    plt.ylabel("Count")
    plt.title("EMG State Distribution")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()