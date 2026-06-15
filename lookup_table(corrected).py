import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load & sort
df = pd.read_csv('DAC_data.csv')
df = df.sort_values(by='Decimal Value').reset_index(drop=True)

# Define zoom window (this affects ideal line computation too)
zoom_start = 0
zoom_end = 100
df_zoom = df[(df['Decimal Value'] >= zoom_start) & (df['Decimal Value'] <= zoom_end)].copy()

# Compute ideal output for zoom range
if not ((df['Decimal Value'] == zoom_start).any() and (df['Decimal Value'] == zoom_end).any()):
    raise ValueError(f"Decimal Value {zoom_start} or {zoom_end} not found in dataset.")

Y_start = df.loc[df['Decimal Value'] == zoom_start, 'Analog Output (V)'].values[0]
Y_end = df.loc[df['Decimal Value'] == zoom_end, 'Analog Output (V)'].values[0]

m = (Y_end - Y_start) / (zoom_end - zoom_start)
b = Y_start - m * zoom_start
df['Y_ideal'] = m * df['Decimal Value'] + b
df_zoom['Y_ideal'] = m * df_zoom['Decimal Value'] + b

# Apply corrected lookup (from original real values)
# Use all raw outputs as candidate replacements
all_measured = df['Analog Output (V)'].values
df['Y_corrected_lookup'] = df['Analog Output (V)'].copy()

for idx in df.index:
    actual = df.at[idx, 'Analog Output (V)']
    ideal = df.at[idx, 'Y_ideal']
    
    # If the value is far from ideal, replace it with closest real value
    if abs(actual - ideal) > 1e-6:
        closest_idx = np.argmin(np.abs(all_measured - ideal))
        df.at[idx, 'Y_corrected_lookup'] = all_measured[closest_idx]

df_zoom['Y_corrected_lookup'] = df['Y_corrected_lookup'][df_zoom.index]

# Full-range plot (original only)
plt.figure(figsize=(10, 5))
plt.scatter(df['Decimal Value'], df['Analog Output (V)'],
            alpha=0.3, color='steelblue', label='Original Output')
# Removed lookup-corrected from full view
plt.xlabel('Decimal Value')
plt.ylabel('Analog Output (V)')
plt.title('Full DAC Transfer (Raw Output Only)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Zoomed-in plot (dynamic ideal line)
plt.figure(figsize=(10, 6))
plt.plot(df_zoom['Decimal Value'], df_zoom['Y_corrected_lookup'],
         color='green', linewidth=2, label='Lookup-Corrected')
plt.scatter(df_zoom['Decimal Value'], df_zoom['Analog Output (V)'],
            alpha=0.3, color='steelblue', label='Original Output', s=10)
plt.plot(df_zoom['Decimal Value'], df_zoom['Y_ideal'],
         linestyle='--', color='black', linewidth=2, label='Ideal Output')
plt.title(f'DAC Correction Comparison (Decimal {zoom_start}–{zoom_end})')
plt.xlabel('Decimal Value')
plt.ylabel('Analog Output (V)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
