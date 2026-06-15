import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parameters 
thermo_bits = 5
binary_bits = 16 - thermo_bits

# Load & Sort
df = pd.read_csv("DAC_data.csv")
df = df.sort_values("Decimal Value").reset_index(drop=True)
decimal = df["Decimal Value"].astype(int).to_numpy()
analog  = df["Analog Output (V)"].to_numpy()

# Data-Driven Segmentation 
df["Thermo_LSB"] = decimal & ((1 << thermo_bits) - 1)  # lower 5 bits
df["Binary_MSB"] = decimal >> thermo_bits              # upper 11 bits

# Assertions to Verify Split 
n_thermo = df["Thermo_LSB"].nunique()
n_binary = df["Binary_MSB"].nunique()
assert n_thermo == 2**thermo_bits, f"Expected {2**thermo_bits} thermo codes, got {n_thermo}"
assert n_binary == 2**binary_bits, f"Expected {2**binary_bits} binary codes, got {n_binary}"

# Forward Sweep Plot 
plt.figure(figsize=(10, 6))
for t in sorted(df["Thermo_LSB"].unique()):
    grp = df[df["Thermo_LSB"] == t]
    plt.plot(grp["Binary_MSB"], grp["Analog Output (V)"],
             '.', alpha=0.2, markersize=3)

# Calculate Mean ±1σ AT Each Course Code 
msb_stats = (
    df.groupby("Binary_MSB")["Analog Output (V)"]
      .agg(mean="mean", std="std")
      .reset_index()
      .sort_values("Binary_MSB")   # ensure sorted before fitting
)

# Overlay Mean ±1σ
plt.errorbar(
    msb_stats["Binary_MSB"], msb_stats["mean"],
    yerr=msb_stats["std"], fmt='o-', capsize=3,
    color='k', label="Mean ±1σ"
)

plt.title(f"Forward Sweep (thermo={thermo_bits}b, binary={binary_bits}b)")
plt.xlabel(f"Binary MSB Code (0–{2**binary_bits - 1})")
plt.ylabel("Analog Output (V)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Linear Fit & Report 
x = msb_stats["Binary_MSB"].to_numpy()
y = msb_stats["mean"].to_numpy()

m, b = np.polyfit(x, y, 1)
y_pred = m * x + b
r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)

print(f"✔ Coarse step (slope)  = {m:.6e} V/code")
print(f"✔ Offset (intercept)   = {b:.6e} V")
print(f"✔ Linearity (R²)       = {r2:.6f}")

# Plot the Fit
plt.figure(figsize=(8, 4))
plt.plot(x, y, 'o', label="Mean outputs")
plt.plot(x, y_pred, '-', label=f"Fit: y={m:.2e}x + {b:.2e}")
plt.title("Linear Fit to Mean Analog vs Binary MSB")
plt.xlabel("Binary MSB Code")
plt.ylabel("Analog Output (V)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
