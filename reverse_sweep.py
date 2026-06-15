import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load & Prep
df = pd.read_csv("DAC_data.csv")
df = df.sort_values("Decimal Value").reset_index(drop=True)
decimal = df["Decimal Value"].astype(int).to_numpy()
analog = df["Analog Output (V)"].to_numpy()

# Hard-code segmentation: 10-bit thermo, 6-bit binary
thermo_bits = 10
binary_bits = 6
df["Thermo_LSB"] = decimal & ((1 << thermo_bits) - 1)   # mask lower 10 bits
df["Binary_MSB"] = decimal >> thermo_bits               # shift off 10 bits

# Sanity check
print("Thermo_LSB range:", df["Thermo_LSB"].min(), "to", df["Thermo_LSB"].max())
print("Binary_MSB range:", df["Binary_MSB"].min(), "to", df["Binary_MSB"].max())
print("Counts per MSB (should be 1024):", df.groupby("Binary_MSB").size().unique())

# Reverse Sweep Plot
plt.figure(figsize=(12, 6))
for w in range(2**binary_bits):  # 0…63
    grp = df[df["Binary_MSB"] == w]
    plt.plot(
        grp["Thermo_LSB"],
        grp["Analog Output (V)"],
        marker='.', linestyle='-', alpha=0.3
    )

plt.title(f"Reverse Sweep: Analog vs Thermo_LSB  (thermo={thermo_bits}b, binary={binary_bits}b)")
plt.xlabel("Thermometer LSB Code (0–1023)")
plt.ylabel("Analog Output (V)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Fit each trace to a line and collect stats
slopes = []
intercepts = []
for w in range(2**binary_bits):
    grp = df[df["Binary_MSB"] == w]
    x = grp["Thermo_LSB"].to_numpy()
    y = grp["Analog Output (V)"].to_numpy()
    m, b = np.polyfit(x, y, 1)
    slopes.append(m)
    intercepts.append(b)

slopes = np.array(slopes)
intercepts = np.array(intercepts)

# Print summary
print(f"LSB-weight (slope) stats:\n",
      f"  mean = {slopes.mean():.6e} V/LSB\n",
      f"   std = {slopes.std():.6e} V/LSB")
print(f"Binary-offset (intercept) stats:\n",
      f"  mean = {intercepts.mean():.6e} V\n",
      f"   std = {intercepts.std():.6e} V")

# Plot slopes vs MSB and histogram
plt.figure(figsize=(10,4))
plt.plot(range(2**binary_bits), slopes, 'o-')
plt.title("Thermo-LSB Weight (slope) vs Binary MSB Code")
plt.xlabel("Binary MSB Code")
plt.ylabel("Slope (V per Thermo LSB)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.hist(slopes, bins=15, edgecolor='k')
plt.title("Histogram of Thermo-LSB Slopes")
plt.xlabel("Slope (V per Thermo LSB)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
