import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load & Sort 
df = pd.read_csv('DAC_data.csv')
df = df.sort_values(by='Decimal Value').reset_index(drop=True)

# Identify “Drops” 
df['Y_runmax'] = df['Analog Output (V)'].cummax()
df['is_drop'] = df['Analog Output (V)'] < df['Y_runmax']

# Replace Dropped Y's with NaN (in‐place)
df.loc[df['is_drop'], 'Analog Output (V)'] = np.nan

# Interpolate in X-space 
series = df.set_index('Decimal Value')['Analog Output (V)']
series_interp = series.interpolate(method='index').ffill()
df['Y_corrected'] = series_interp.values

# Compute ΔV and ideal LSB 
df['step'] = df['Y_corrected'].diff()
ideal_step = df['step'].mean()

# Compute DNL and INL 
df['DNL'] = df['step'] / ideal_step - 1
df['INL'] = df['DNL'].cumsum()

# Plot DNL vs. Decimal Value 
plt.figure(figsize=(8, 4))
markerline, stemlines, baseline = plt.stem(
    df['Decimal Value'],
    df['DNL'],
    linefmt='C1-',
    markerfmt='C1o',
    basefmt='gray'
)
plt.setp(baseline, color='gray', linewidth=1)
plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)

plt.xlabel('Decimal Value (code)')
plt.ylabel('DNL (LSB)')
plt.title('DNL vs. Decimal Value')
plt.grid(True, which='both', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()

# Plot INL vs. Decimal Value 
plt.figure(figsize=(8, 4))
plt.plot(
    df['Decimal Value'],
    df['INL'],
    marker='o',
    linestyle='-',
    color='darkgreen'
)
plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)

plt.xlabel('Decimal Value (code)')
plt.ylabel('INL (LSB)')
plt.title('INL vs. Decimal Value')
plt.grid(True, which='both', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()
