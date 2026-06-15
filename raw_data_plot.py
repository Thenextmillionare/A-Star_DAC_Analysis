import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV into a DataFrame
# Replace 'your_file.csv' with the path to your CSV file.
df = pd.read_csv("DAC_data.csv")

# Extract the two columns you want to plot.
# Replace 'ColumnX' and 'ColumnY' with your actual column names.
x = df["Decimal Value"]
y = df["Analog Output (V)"]

# 3. Make a simple line (or scatter) plot of Y vs. X
plt.figure(figsize=(8, 5))
plt.plot(x, y, marker='o', linestyle='-')  # For a line with markers
# plt.scatter(x, y)  # Or use this for a scatter-only plot

# 4. Label axes and add a title (optional)
plt.xlabel('Decimal Value')
plt.ylabel('Analog Output (V)')
plt.title('Analog Output (V) vs. Decimal Value')

# 5. Show the grid and display
plt.grid(True)
plt.tight_layout()
plt.show()
