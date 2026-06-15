import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Load & sort
df = pd.read_csv("DAC_data.csv").sort_values("Decimal Value").reset_index(drop=True)
y = df["Analog Output (V)"].to_numpy()

# Compute raw DNL
steps      = np.diff(y)
ideal_step = steps.mean()
dnl        = np.insert((steps/ideal_step) - 1, 0, 0)

# Pick top‐1% largest |DNL| spikes
thr        = np.percentile(np.abs(dnl), 99)
spike_idxs = np.where(np.abs(dnl) > thr)[0]

# Unique, sorted spike positions → code indices
uniq       = np.unique(spike_idxs)

# Gaps between successive spikes
gaps       = np.diff(uniq)

# Filter out “trivial” gaps from fine‐bit toggles (≤5 codes)
gaps_filt  = gaps[gaps > 5]

# Count exact gap frequencies
cnt        = Counter(gaps_filt)
gap, freq  = cnt.most_common(1)[0]

# Infer segment bits
thermo_bits = int(np.round(np.log2(gap)))
binary_bits = 16 - thermo_bits

print(f"Most common large gap = {gap:.0f} codes  (count = {freq})")
print(f"→ Inferred thermometer bits = {thermo_bits}")
print(f"→ Inferred binary bits     = {binary_bits}")

# Plot one-code-wide histogram
plt.figure(figsize=(6, 4))

# bins=range makes each integer gap get its own bar
plt.hist(gaps_filt, bins=range(0, int(gaps_filt.max())+2), align='left', edgecolor='k')
plt.axvline(gap, color='r', linestyle='--', label=f'Peak at {gap:.0f}')
plt.title("Histogram of Non-Trivial Gaps Between DNL Spikes")
plt.xlabel("Gap (codes)")
plt.ylabel("Count")
plt.xlim(0, 200)   # zoom in around the peak (optional)
plt.legend()
plt.tight_layout()
plt.show()
