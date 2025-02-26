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

# Define function to extract values between headers
def extract_values(content, start_label, end_label):
    start_idx = content.find(start_label) + len(start_label)
    end_idx = content.find(end_label) if end_label else len(content)
    values_str = content[start_idx:end_idx].strip().replace("\n", "")
    values = [float(v) for v in values_str.split(',') if v.strip()]
    return np.array(values)

# List of file paths
file_paths = [
    "experiment1/iteration15.csv",
    "experiment3/iteration15.csv",
    "experiment4/iteration15.csv",
    "experiment5/iteration15.csv",
    "experiment6/iteration15.csv",
    "experiment7/iteration15.csv"
]

# Initialize list to store summary statistics
summary_stats = []

# Loop through each CSV file
for idx, path in enumerate(file_paths, start=1):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(script_dir, path)

    # Read the entire file content
    with open(file_path, 'r') as file:
        content = file.read()

    # Extract values based on the provided structure
    x_values = extract_values(content, 'X Values', 'Y Values') * 1000  # Convert to mm
    y_values = extract_values(content, 'Y Values', 'Z Values') * 1000  # Convert to mm
    z_values = extract_values(content, 'Z Values', 'QX Values') * 1000  # Convert to mm

    # Combine positions
    positions = np.column_stack((x_values, y_values, z_values))

    # Outlier detection on positions
    position_z_scores = np.abs(zscore(positions))
    position_mask = (position_z_scores < 3).all(axis=1)
    filtered_positions = positions[position_mask]

    # Compute min and max for x and z (converted to mm)
    x_min, x_max = filtered_positions[:, 0].min(), filtered_positions[:, 0].max()
    z_min, z_max = filtered_positions[:, 2].min(), filtered_positions[:, 2].max()

    # Append summary statistics for this CSV (no theta included)
    summary_stats.append([
        f"{x_min:.2f} - {x_max:.2f}",
        f"{z_min:.2f} - {z_max:.2f}"
    ])

# Prepare headers and summary statistics (removed the theta column)
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
