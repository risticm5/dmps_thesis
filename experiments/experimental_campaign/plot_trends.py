import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from matplotlib.ticker import MultipleLocator  # For custom grid spacing

# Enable LaTeX formatting for plots
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 12  # Ensure clarity
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# List of CSV file paths
file_paths = [
    "test1_christian/answers_test1.csv",
    "test3_xenia/answers_test3.csv",
    "test4_mihailo/answers_test4.csv",
    "test5_marko/answers_test5.csv",
    "test6_eleonora/answers_test6.csv",
    "test7_bozidar/answers_test7.csv"
]

# Set distinct, easily distinguishable colors for different participants
colors = plt.cm.tab10(np.linspace(0, 1, len(file_paths)))  # Use tab10 for clear, distinguishable colors

# Create a figure with 4 subplots arranged horizontally
fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # 1 row, 4 columns

# Custom titles for the subplots
title_labels = [r"$\bar{\tau}$", r"$k_s$", r"$k_m$", r"A\%"]

# Store lines and labels for the shared legend
lines = []
labels = []

# Loop through each file
for idx, file_path in enumerate(file_paths):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    full_path = os.path.join(script_dir, file_path)

    # Read the file line by line to handle variable-length rows
    with open(full_path, 'r') as file:
        lines_data = [line.strip().split(',') for line in file]

    # Convert to DataFrame, padding with NaN where needed
    max_length = max(len(line) for line in lines_data)
    padded_data = [line + [np.nan] * (max_length - len(line)) for line in lines_data]
    df = pd.DataFrame(padded_data).apply(pd.to_numeric, errors='coerce')

    # Extract the 4th row (index 3) and the last 3 rows
    row_4 = df.iloc[3].dropna()  # Remove NaNs
    last_3_rows = df.tail(3).apply(lambda x: x.dropna(), axis=1)  # Remove NaNs in each row

    # Create a 'values' vector for x-axis (integers from 1 to max_length)
    values = np.arange(1, 1 + max_length)

    # Vector for the y spacing
    y_spacing = [0.01, 0.05, 0.05]

    # Plot the last 3 rows in the first three subplots with integer x-ticks and all y-values
    for i in range(3):
        row_data = last_3_rows.iloc[i].values
        x_values = values[:len(row_data)]  # Match data length
        line, = axes[i].plot(x_values, row_data, marker='o', color=colors[idx], label=f"Participant {idx+1}")
        
        # Highlight x=8 with a red star for Participant 1
        if idx == 0 and 8 in x_values:
            y_value = row_data[np.where(x_values == 8)[0][0]]
            axes[i].plot(8, y_value, marker='*', color='red', markersize=15)
        
        axes[i].set_xlabel(r"Iteration", fontsize=20)
        axes[i].set_xticks(x_values)
        axes[i].set_yticks(sorted(np.unique(row_data)))  # Show all unique y-values
        axes[i].set_title(title_labels[i], fontsize=20)
        axes[i].xaxis.set_major_locator(MultipleLocator(2))  # X-axis grid every 2 units
        axes[i].yaxis.set_major_locator(MultipleLocator(y_spacing[i]))  # Y-axis grid
        axes[i].grid(True, which='both', linestyle='--', linewidth=0.7)
        axes[i].axvline(x=6, color='black', linestyle='--', linewidth=0.7, alpha=0.6)

    # Plot trend of values from the 4th row in the 4th subplot with grid
    x_indices = np.arange(2, len(row_4) + 2)
    line, = axes[3].plot(x_indices, row_4.values * 10, marker='o', linestyle='-', color=colors[idx], label=f"Participant {idx+1}")
    
    # Highlight x=8 with a red star for Participant 1
    if idx == 0 and 8 in x_indices:
        y_value = (row_4.values * 10)[np.where(x_indices == 8)[0][0]]
        axes[3].plot(8, y_value, marker='*', color='red', markersize=15)
    
    axes[3].set_xlabel(r"Iteration", fontsize=20)
    axes[3].set_title(title_labels[3], fontsize=20)
    axes[3].set_yticks(sorted(np.unique(row_4.values * 10)))
    axes[3].xaxis.set_major_locator(MultipleLocator(2))
    axes[3].yaxis.set_major_locator(MultipleLocator(10))
    axes[3].grid(True, which='both', linestyle='--', linewidth=0.7)
    axes[3].axvline(x=6, color='black', linestyle='--', linewidth=0.7, alpha=0.6)

    # Collect lines and labels for shared legend
    lines.append(line)
    labels.append(f"Participant {idx + 1}")

# Add a common legend below all plots
fig.legend(lines, labels, loc='lower center', ncol=len(file_paths), fontsize=12, frameon=False)

# Adjust layout for better spacing and legend
plt.tight_layout(rect=[0, 0.1, 1, 0.95])
plt.subplots_adjust(bottom=0.2)  # Adjust space for the common legend

# Display the plot
plt.show()
