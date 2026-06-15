import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and sort DAC data
df = pd.read_csv('DAC_data.csv')
df = df.sort_values(by='Decimal Value').reset_index(drop=True)

# Create ideal linear reference line (from zoomed window, can change range)
zoom_start = 0
zoom_end = 100
Y_start = df.loc[df['Decimal Value'] == zoom_start, 'Analog Output (V)'].values[0]
Y_end = df.loc[df['Decimal Value'] == zoom_end, 'Analog Output (V)'].values[0]
m = (Y_end - Y_start) / (zoom_end - zoom_start)
b = Y_start - m * zoom_start
df['Y_ideal'] = m * df['Decimal Value'] + b

# Lookup table correction using real measured values
all_measured = df['Analog Output (V)'].values
df['Y_corrected_lookup'] = df['Analog Output (V)'].copy()
for idx in df.index:
    actual = df.at[idx, 'Analog Output (V)']
    ideal = df.at[idx, 'Y_ideal']
    closest_idx = np.argmin(np.abs(all_measured - ideal))
    df.at[idx, 'Y_corrected_lookup'] = all_measured[closest_idx]

# PWM-style oversampled output based on lookup-corrected values
def pwm_interpolated_output(code_frac, corrected_df):
    low = int(np.floor(code_frac))
    high = int(np.ceil(code_frac))
    if high > corrected_df['Decimal Value'].max():
        high = low  # Clamp at upper bound
    f = code_frac - low  # fractional distance
    v_low = corrected_df.loc[corrected_df['Decimal Value'] == low, 'Y_corrected_lookup'].values[0]
    v_high = corrected_df.loc[corrected_df['Decimal Value'] == high, 'Y_corrected_lookup'].values[0]
    return (1 - f) * v_low + f * v_high

# Generate oversampled input range
num_points = 1000
input_codes = np.linspace(0, 100, num_points)
outputs_pwm = np.array([pwm_interpolated_output(c, df) for c in input_codes])

# Compute ideal step size from endpoints
ideal_step = (outputs_pwm[-1] - outputs_pwm[0]) / (num_points - 1)

# DNL and INL
steps = np.diff(outputs_pwm)
dnl = steps / ideal_step - 1
dnl = np.insert(dnl, 0, 0)
inl = np.cumsum(dnl)
inl -= np.mean(inl)

# Save DNL, INL, and fractional input codes
np.save("pwm_dnl.npy", dnl)
np.save("pwm_inl.npy", inl)
np.save("pwm_codes.npy", input_codes)

# Plot DNL
plt.figure(figsize=(10, 4))
plt.stem(input_codes, dnl, linefmt='g-', markerfmt='go', basefmt='k-')
plt.title("DNL (PWM-Oversampled Output)")
plt.xlabel("Input Code (Fractional)")
plt.ylabel("DNL (LSB)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot INL
plt.figure(figsize=(10, 4))
plt.plot(input_codes, inl, color='maroon')
plt.axhline(0, linestyle='--', color='black')
plt.title("INL (PWM-Oversampled Output)")
plt.xlabel("Input Code (Fractional)")
plt.ylabel("INL (LSB)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Print error summary
print("DNL Check:")
print(f"  Max DNL: {np.max(dnl):.3f} LSB")
print(f"  Min DNL: {np.min(dnl):.3f} LSB")
print(f"  DNL Range (peak-to-peak): {np.ptp(dnl):.3f} LSB\n")

print("INL Check:")
print(f"  Max INL: {np.max(inl):.3f} LSB")
print(f"  Min INL: {np.min(inl):.3f} LSB")
print(f"  INL Range (peak-to-peak): {np.ptp(inl):.3f} LSB")

pwm_export = pd.DataFrame({
    'Input Code': input_codes,
    'DNL': dnl,
    'INL': inl
})
pwm_export.to_csv('pwm.csv', index=False)