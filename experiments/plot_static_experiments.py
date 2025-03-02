import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from scipy.stats import zscore

# Enable LaTeX formatting for plots
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# List of file paths
file_paths = [
    "experiment1/static_test.csv",
    "experiment3/static_test.csv",
    "experiment4/static_test.csv",
    "experiment5/static_test.csv",
    "experiment6/static_test.csv",
    "experiment7/static_test.csv"
]

# Initialize list to store summary statistics
summary_stats = []
delta_x_values = []
delta_z_values = []

# Loop through each CSV file
for idx, path in enumerate(file_paths, start=1):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(script_dir, path)
    df = pd.read_csv(file_path)

    # Extract position (x, y, z)
    positions = df[['X', 'Y', 'Z']].values * 1000  # Convert to millimeters

    # Outlier detection on positions
    position_z_scores = np.abs(zscore(positions))
    position_mask = (position_z_scores < 2.5).all(axis=1)
    filtered_positions = positions[position_mask]

    # Compute min and max for x and z (converted to mm)
    x_min, x_max = filtered_positions[:, 0].min(), filtered_positions[:, 0].max()
    z_min, z_max = filtered_positions[:, 2].min(), filtered_positions[:, 2].max()

    # Compute delta_x and delta_z
    delta_x = x_max - x_min
    delta_z = z_max - z_min
    delta_x_values.append(delta_x)
    delta_z_values.append(delta_z)

    # Append summary statistics for this CSV (no theta or standard deviation included)
    summary_stats.append([
        f"{x_min:.2f} - {x_max:.2f}",
        f"{z_min:.2f} - {z_max:.2f}"
    ])

# Prepare headers and row labels for the table
headers = [
    r"$\textbf{Test}$",
    r"${x_H}_{\text{min}} - {x_H}_{\text{max}}\ (\text{mm})$",
    r"${z_H}_{\text{min}} - {z_H}_{\text{max}}\ (\text{mm})$"
]

# Combine headers and statistics into a table format
table_data = [headers]
for i, stats in enumerate(summary_stats, start=1):
    table_data.append([f"{i}"] + stats)

# Dynamically adjust figure width based on longest content
max_content_length = max(len(str(item)) for row in table_data for item in row)
fig_width = max(12, max_content_length * 0.5)  # Adjust figure width based on content length

# Create the table plot
fig, ax = plt.subplots(figsize=(fig_width, 8))
ax.axis('off')

# Set cell height multiplier
cell_height = 1.5  # You can adjust this value

# Create the table
table = ax.table(
    cellText=table_data,
    cellLoc='center',
    loc='center'
)

# Adjust table aesthetics
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, cell_height)  # Set custom cell height

# Manually set custom widths for each column
column_widths = [0.05, 0.15, 0.15]  # Adjust widths as needed

for (i, j), cell in table.get_celld().items():
    if j < len(column_widths):
        cell.set_width(column_widths[j])

# Display the plot
plt.show()

# Save delta_x and delta_z to CSV file
delta_df = pd.DataFrame({"Delta X (mm)": delta_x_values, "Delta Z (mm)": delta_z_values})
delta_csv_path = os.path.join(script_dir, "Deltas.csv")
delta_df.to_csv(delta_csv_path, index=False)

# Display delta_x and delta_z vectors
print("Delta X values:", delta_x_values)
print("Delta Z values:", delta_z_values)
print(f"Deltas saved to: {delta_csv_path}")

