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
df['Y_corrected_lookup'] = df['Y_corrected'].copy()

# Compute ideal linear output (based on Y_corrected range)
min_y = df['Y_corrected'].min()
max_y = df['Y_corrected'].max()

df['Y_ideal'] = np.linspace(min_y, max_y, len(df))

# Plot
plt.figure(figsize=(9, 5))
plt.scatter(df['Decimal Value'], df['Analog Output (V)'],
            color='steelblue', alpha=0.3, label='Original Y')
plt.plot(df['Decimal Value'], df['Y_corrected'],
         color='firebrick', linewidth=2, marker='o', label='Y after interpolation')
plt.xlabel('Decimal Value')
plt.ylabel('Analog Output (V)')
plt.title('DAC Transfer: Original vs. Interpolated (Drops Removed)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Zoomed plot (first 1000 codes) with final correction
df_zoom = df[df['Decimal Value'] <= 1000]

plt.figure(figsize=(10, 6))

# Ideal output: black dashed line, drawn last so it stands out
plt.plot(df_zoom['Decimal Value'], df_zoom['Y_corrected_lookup'],
         color='green', linewidth=4, alpha=0.5, label='Lookup-Corrected (Option A)')
plt.plot(df_zoom['Decimal Value'], df_zoom['Y_corrected'],
         color='orange', linewidth=2, label='Interpolated')
plt.scatter(df_zoom['Decimal Value'], df_zoom['Analog Output (V)'],
            alpha=0.3, color='steelblue', label='Original Output', s=10)
plt.plot(df_zoom['Decimal Value'], df_zoom['Y_ideal'],
         linestyle='--', color='black', linewidth=2, label='Ideal Output')

plt.title('DAC Output Correction Comparison (First 1000 Codes)')
plt.xlabel('Decimal Value')
plt.ylabel('Analog Output (V)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
