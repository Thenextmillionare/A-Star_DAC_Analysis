import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load & sort
df = pd.read_csv('DAC_data.csv')
df = df.sort_values(by='Decimal Value').reset_index(drop=True)

# Identify “drops”
# Running maximum of Y
df['Y_runmax'] = df['Analog Output (V)'].cummax()

# Mask all rows where Y is below that runmax
df['is_drop'] = df['Analog Output (V)'] < df['Y_runmax']

# Replace dropped Y's with NaN & interpolate
# Temporarily set index to Decimal Value so interpolation is in X-space
series = df.set_index('Decimal Value')['Analog Output (V)']
mask   = df['is_drop'].values

series_masked = series.copy()
series_masked[mask] = np.nan

# Linear interpolation in index (Decimal Value)
series_interp = series_masked.interpolate(method='index')

# Forward-fill any trailing NaNs (after last “valid” point)
series_interp = series_interp.ffill()

# Put the corrected values back into df
df['Y_corrected'] = series_interp.values

# Filter to only first 1000 Decimal Values
df_zoom = df[df['Decimal Value'] <= 8000]

# Plot
plt.figure(figsize=(9, 5))

plt.scatter(df_zoom['Decimal Value'], df_zoom['Analog Output (V)'],
            color='steelblue', alpha=0.3, label='Original Y')
plt.plot(df_zoom['Decimal Value'], df_zoom['Y_corrected'],
         color='firebrick', linewidth=2, marker='o', label='Y after interpolation')

plt.xlabel('Decimal Value')
plt.ylabel('Analog Output (V)')
plt.title('DAC Transfer (First 1000 Codes): Original vs. Interpolated')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()