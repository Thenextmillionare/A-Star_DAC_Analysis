import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load & sort your DAC data
df = pd.read_csv("DAC_data.csv")
df = df.sort_values("Decimal Value").reset_index(drop=True)

# Extract arrays
codes = df["Decimal Value"].astype(int).to_numpy()      # 0…65535
y_raw = df["Analog Output (V)"].to_numpy()

# Compute the ideal straight‐line transfer (end‐point method)
m = (y_raw[-1] - y_raw[0]) / (codes[-1] - codes[0])
b = y_raw[0] - m * codes[0]
y_ideal = m * codes + b

# Build LUT‐corrected output by snapping each ideal point to the nearest measured value
meas_sorted = np.sort(y_raw)
inds = np.searchsorted(meas_sorted, y_ideal)
inds_lo = np.clip(inds - 1, 0, len(meas_sorted) - 1)
inds_hi = np.clip(inds,     0, len(meas_sorted) - 1)
# distances to lower/upper neighbors
dl = np.abs(y_ideal - meas_sorted[inds_lo])
dh = np.abs(meas_sorted[inds_hi] - y_ideal)
# choose the closest neighbor
y_corr = np.where(dh < dl, meas_sorted[inds_hi], meas_sorted[inds_lo])

# Compute LSB size (ideal voltage step per code)
lsb = (y_raw[-1] - y_raw[0]) / (codes[-1] - codes[0])

# Compute residual error (in LSB) between corrected output and ideal
residual_lsb = (y_corr - y_ideal) / lsb

# % of codes within ±1 LSB
within_1 = np.sum(np.abs(residual_lsb) <= 1)
total    = len(residual_lsb)
pct1     = within_1/total * 100

# “True LSB range” of the raw DAC
raw_steps   = np.diff(y_raw)          # volts between successive codes
norm_steps  = raw_steps/lsb           # in LSB units
min_step, max_step = norm_steps.min(), norm_steps.max()

# Print key metrics
print(f"Max residual error: {np.max(np.abs(residual_lsb)):.3f} LSB")
print(f"RMS residual error: {np.sqrt(np.mean(residual_lsb**2)):.3f} LSB")
print(f"Stddev residual error: {np.std(residual_lsb):.3f} LSB")
print(f"Codes within ±1 LSB: {within_1}/{total} → {pct1:.2f}%")
print(f"Raw step size range: {min_step:.3f}–{max_step:.3f} LSB")

# Plot the residual error vs. code
plt.figure(figsize=(10, 5))
plt.plot(codes, residual_lsb, '.', markersize=2, alpha=0.6)
plt.title("Residual Error of LUT-Corrected Output vs Ideal")
plt.xlabel("Decimal Code")
plt.ylabel("Error (LSB)")
plt.grid(True)
plt.tight_layout()
plt.show()
