import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your DAC data
df = pd.read_csv("DAC_data.csv")  # Make sure the file is in the same directory
df = df.sort_values(by='Decimal Value').reset_index(drop=True)

# Generate Ideal Output Line
x_min, x_max = df['Decimal Value'].min(), df['Decimal Value'].max()
y_min = df.loc[df['Decimal Value'] == x_min, 'Analog Output (V)'].values[0]
y_max = df.loc[df['Decimal Value'] == x_max, 'Analog Output (V)'].values[0]
m = (y_max - y_min) / (x_max - x_min)
b = y_min - m * x_min
df['Y_ideal'] = m * df['Decimal Value'] + b

# Apply Lookup Table Correction
all_measured = df['Analog Output (V)'].values
df['Y_corrected_lookup'] = df['Analog Output (V)']
for idx in df.index:
    ideal_val = df.at[idx, 'Y_ideal']
    closest_idx = np.argmin(np.abs(all_measured - ideal_val))
    df.at[idx, 'Y_corrected_lookup'] = all_measured[closest_idx]

# DNL Calculation
df['step'] = df['Y_corrected_lookup'].diff().fillna(0)
n_codes = x_max - x_min
ideal_step = (df['Y_corrected_lookup'].iloc[-1] - df['Y_corrected_lookup'].iloc[0]) / n_codes
df['DNL'] = df['step'] / ideal_step - 1
df.loc[0, 'DNL'] = 0

# INL Calculation
df['INL'] = df['DNL'].cumsum()
df['INL'] -= df['INL'].mean()

# At the end of INL/DNL computation (before plotting)
np.save("lookup_dnl.npy", df['DNL'].values)
np.save("lookup_inl.npy", df['INL'].values)
np.save("lookup_codes.npy", df['Decimal Value'].values)  # Optional, for x-axis

# Plot DNL
plt.figure(figsize=(12, 4))
plt.stem(df['Decimal Value'], df['DNL'], linefmt='g-', markerfmt='go', basefmt='k-')
plt.title("DNL (Lookup-Corrected with Uniform Codes)")
plt.xlabel("Decimal Value")
plt.ylabel("DNL (LSB)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot INL
plt.figure(figsize=(12, 4))
plt.plot(df['Decimal Value'], df['INL'], color='maroon')
plt.axhline(0, linestyle='--', color='gray')
plt.title("INL (Lookup-Corrected with Uniform Codes)")
plt.xlabel("Decimal Value")
plt.ylabel("INL (LSB)")
plt.grid(True)
plt.tight_layout()
plt.show()

lookup_export = df[['Decimal Value', 'DNL', 'INL']]
lookup_export.to_csv('lookup.csv', index=False)

num_unique = len(np.unique(df['Y_corrected_lookup']))
total = len(df)
print(f"{num_unique} unique outputs used for {total} digital codes.")

