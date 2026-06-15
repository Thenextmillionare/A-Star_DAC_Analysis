import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load both CSVs
df_lut = pd.read_csv('lookup.csv')
df_pwm = pd.read_csv('pwm.csv')

# Interpolate LUT data to PWM resolution
lut_interp_dnl = np.interp(df_pwm['Input Code'], df_lut['Decimal Value'], df_lut['DNL'])
lut_interp_inl = np.interp(df_pwm['Input Code'], df_lut['Decimal Value'], df_lut['INL'])

# Difference
dnl_diff = df_pwm['DNL'] - lut_interp_dnl
inl_diff = df_pwm['INL'] - lut_interp_inl

# Plot DNL difference
plt.figure(figsize=(12, 4))
plt.plot(df_pwm['Input Code'], dnl_diff, color='blue')
plt.axhline(0, linestyle='--', color='gray')
plt.title("DNL Difference (PWM - LUT)")
plt.xlabel("Input Code (Fractional)")
plt.ylabel("DNL Difference (LSB)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot INL difference
plt.figure(figsize=(12, 4))
plt.plot(df_pwm['Input Code'], inl_diff, color='purple')
plt.axhline(0, linestyle='--', color='gray')
plt.title("INL Difference (PWM - LUT)")
plt.xlabel("Input Code (Fractional)")
plt.ylabel("INL Difference (LSB)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Summary
print("DNL Diff Summary:")
print(f"Max: {np.max(dnl_diff):.2f} LSB, Min: {np.min(dnl_diff):.2f} LSB, Range: {np.ptp(dnl_diff):.2f} LSB")
print("INL Diff Summary:")
print(f"Max: {np.max(inl_diff):.2f} LSB, Min: {np.min(inl_diff):.2f} LSB, Range: {np.ptp(inl_diff):.2f} LSB")
