import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib as mpl

# LaTeX settings
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# File paths and participant data
file_paths = [
    "test1_christian/answers_test1.csv",
    "test3_xenia/answers_test3.csv",
    "test4_mihailo/answers_test4.csv",
    "test5_marko/answers_test5.csv",
    "test6_eleonora/answers_test6.csv",
    "test7_bozidar/answers_test7.csv"
]

candidates = ['Christian', 'Xenia', 'Mihailo', 'Marko', 'Eleonora', 'Bozidar']
heights = [182, 168, 192, 192, 155, 195]

# Custom labels and heights
custom_labels = [
    ("1", r"$\text{expert}$"),
    ("2", r"$\text{non-expert}$"),
    ("3", r"$\text{non-expert}$"),
    ("4", r"$\text{expert}$"),
    ("5", r"$\text{expert}$"),
    ("6", r"$\text{non-expert}$")
]

# Custom headers for the table (adding Height column)
headers = [
    r"$\textbf{Participant}$", r"$\textbf{Height (cm)}$", r"$\textbf{Status}$",
    r"$\textbf{Q$_1$}$", r"$\textbf{Q$_{1 \setminus \bar{N}}$}$",
    r"$\textbf{Q$_2$}$", r"$\textbf{Q$_{2 \setminus \bar{N}}$}$",
    r"$\textbf{Q$_3$}$", r"$\textbf{Q$_{3 \setminus \bar{N}}$}$",
    r"$\textbf{Q$_4$}$"
]

# Process data
means_list, means_excl_first_5_list, q4_values = [], [], []

for path in file_paths:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(script_dir, path)

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Read all 15 values for each row
    rows = [list(map(float, line.strip().split(','))) for line in lines[:3]]

    # Mean using all 15 values for Q1, Q2, Q3
    means = [np.mean(row) for row in rows]
    means_list.append(means)

    # Mean excluding the first 5 values for Q11, Q22, Q33
    means_excl_first_5 = [np.mean(row[5:]) if len(row) > 5 else 0 for row in rows]
    means_excl_first_5_list.append(means_excl_first_5)

    # Q4 value from the 5th row
    q4_values.append(float(lines[4].strip()))

# Flattened list of all means for color normalization
all_means = [mean for means in means_list for mean in means] + [mean for means in means_excl_first_5_list for mean in means]
min_mean, max_mean = min(all_means), max(all_means)

# Normalize means for color mapping
mean_norm = Normalize(vmin=min_mean, vmax=max_mean)

# Normalize Q4 values for magenta coloring
min_q4, max_q4 = min(q4_values), max(q4_values)
q4_norm = Normalize(vmin=min_q4, vmax=max_q4)

# Plotting table with larger figure size
fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('off')

# Custom colors with consistent normalization
blue_cmap = mpl.colors.LinearSegmentedColormap.from_list("custom_blues", [(0.9, 0.95, 1), (0.1, 0.3, 0.9)])  # Light to Dark Blue
magenta_cmap = mpl.colors.LinearSegmentedColormap.from_list("custom_magentas", [(1, 0.9, 1), (0.5, 0, 0.5)])  # Light to Dark Magenta

# Prepare table data and cell colors
table_data = [headers]
cell_colors = [[(1, 1, 1)] * 10]  # Updated to 10 columns

for i, (label1, label2) in enumerate(custom_labels):
    means = means_list[i]
    excluded_means = means_excl_first_5_list[i]
    q4_value = q4_values[i]

    # Insert heights into the row data
    row_data = [
        f"${label1}$", f"${heights[i]}$", f"${label2}$",
        f"${means[0]:.2f}$", f"${excluded_means[0]:.2f}$",
        f"${means[1]:.2f}$", f"${excluded_means[1]:.2f}$",
        f"${means[2]:.2f}$", f"${excluded_means[2]:.2f}$",
        f"${q4_value:.2f}$"
    ]

    # Apply colors with proper normalization
    row_colors = [(1, 1, 1)] * 3  # No color for Participant, Height, and Status
    row_colors += [
        blue_cmap(mean_norm(means[0])), blue_cmap(mean_norm(excluded_means[0])),
        blue_cmap(mean_norm(means[1])), blue_cmap(mean_norm(excluded_means[1])),
        blue_cmap(mean_norm(means[2])), blue_cmap(mean_norm(excluded_means[2]))
    ]
    row_colors.append(magenta_cmap(q4_norm(q4_value)))  # Magenta color for Q4

    table_data.append(row_data)
    cell_colors.append(row_colors)

# Create table
thickness = 1  # Line thickness

custom_table = ax.table(
    cellText=table_data,
    cellColours=cell_colors,
    cellLoc='center',
    loc='center'
)
custom_table.scale(1.5, 1.5)

# Set line thickness and column widths
widths = [0.07, 0.07, 0.07, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]
for key, cell in custom_table.get_celld().items():
    cell.set_linewidth(thickness)
    cell.set_width(widths[key[1]])
    cell.set_height(0.04)

# Colorbars (Proper normalization shared with table)
sm_means = cm.ScalarMappable(cmap=blue_cmap, norm=mean_norm)
sm_means.set_array([])
cbar_means = plt.colorbar(sm_means, ax=ax, orientation='horizontal', pad=0.02, shrink=0.7, aspect=20)
cbar_means.ax.tick_params(width=2, labelsize=12)  # Thicker ticks and larger labels

sm_q4 = cm.ScalarMappable(cmap=magenta_cmap, norm=q4_norm)
sm_q4.set_array([])
cbar_q4 = plt.colorbar(sm_q4, ax=ax, orientation='horizontal', pad=0.1, shrink=0.7, aspect=20)
cbar_q4.ax.tick_params(width=2, labelsize=12)  # Thicker ticks and larger labels


# Display the plot
plt.show()
