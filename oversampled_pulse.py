import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load DAC mapping data from CSV 
dac_df = pd.read_csv("DAC_data.csv")  # Adjust path if needed
dac_df = dac_df[["Decimal Value", "Analog Output (V)"]]

# Convert to numpy arrays for fast lookup
codes = dac_df["Decimal Value"].values
voltages = dac_df["Analog Output (V)"].values

# Function to find 2 closest DAC output voltages around a target
def find_closest_two(target_v):
    diffs = voltages - target_v
    idx_below = np.where(diffs <= 0, diffs, -np.inf).argmax()
    idx_above = np.where(diffs > 0, diffs, np.inf).argmin()

    return idx_below, idx_above

# Function to generate oversampled code sequence 
def generate_oversampled_sequence(target_v, total_samples=100):
    idx_low, idx_high = find_closest_two(target_v)
    v_low = voltages[idx_low]
    v_high = voltages[idx_high]

    if np.isclose(v_low, v_high):  # exact match
        return [codes[idx_low]] * total_samples, [v_low] * total_samples

    # Solve weights
    n_high = int(round((target_v - v_low) / (v_high - v_low) * total_samples))
    n_low = total_samples - n_high

    sequence_codes = [codes[idx_low]] * n_low + [codes[idx_high]] * n_high
    sequence_voltages = [v_low] * n_low + [v_high] * n_high
    np.random.shuffle(sequence_voltages)

    return sequence_codes, sequence_voltages

# Low-pass filter (simple IIR)
def low_pass_filter(sequence, alpha=0.05):
    filtered = [sequence[0]]
    for v in sequence[1:]:
        filtered.append(alpha * v + (1 - alpha) * filtered[-1])
    return filtered

# Main test
target_voltage = 0.00347  # EDIT THIS
codes_seq, voltages_seq = generate_oversampled_sequence(target_voltage, total_samples=200)
filtered_output = low_pass_filter(voltages_seq, alpha=0.03)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(voltages_seq, label="Raw DAC Output", linestyle="--", marker='o', markersize=3, alpha=0.6)
plt.plot(filtered_output, label="Low-pass Filtered Output", linewidth=2)
plt.axhline(y=target_voltage, color="r", linestyle=":", label="Target Voltage")
plt.title(f"Oversampled DAC Simulation for Target {target_voltage:.6f} V")
plt.xlabel("Sample Index")
plt.ylabel("Voltage (V)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Min DAC V: {voltages.min()}, Max DAC V: {voltages.max()}")
