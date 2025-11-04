import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from matplotlib.ticker import MultipleLocator  # For custom grid spacing

# LaTeX formatting
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 12
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# CSVs
file_paths = [
    "test1_christian/answers_test1.csv",
    "test3_xenia/answers_test3.csv",
    "test4_mihailo/answers_test4.csv",
    "test5_marko/answers_test5.csv",
    "test6_eleonora/answers_test6.csv",
    "test7_bozidar/answers_test7.csv",
    "test1_new_alessandro/answers_test1_new.csv",
    "test2_new_daniele/answers_test1.csv",
    "test3_new_eleonora/answers_test1.csv"
]

# >>> Made-up participant names (same order/length as file_paths)
participant_names = [
    "Chris", "Xenia", "Mihailo", "Marko", "Ele1",
    "Bozidar", "Ale", "Dani", "Ele2"
]

# Colors
colors = plt.cm.tab10(np.linspace(0, 1, len(file_paths)))

# Figure & axes
fig, axes = plt.subplots(1, 4, figsize=(20, 6))
title_labels = [r"$\boldsymbol{\bar{\tau}}$", r"$\boldsymbol{k_s}$", r"$\boldsymbol{k_m}$", r"$\boldsymbol{A\%}$"]

lines = []
legend_labels = []  # We'll fill with "P i – Name"

for idx, file_path in enumerate(file_paths):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    full_path = os.path.join(script_dir, file_path)

    with open(full_path, 'r') as file:
        lines_data = [line.strip().split(',') for line in file]

    max_length = max(len(line) for line in lines_data)
    padded_data = [line + [np.nan] * (max_length - len(line)) for line in lines_data]
    df = pd.DataFrame(padded_data).apply(pd.to_numeric, errors='coerce')

    row_4 = df.iloc[3].dropna()
    last_3_rows = df.tail(3).apply(lambda x: x.dropna(), axis=1)

    values = np.arange(1, 1 + max_length)
    y_spacing = [0.01, 0.05, 0.05]

    for i in range(3):
        row_data = last_3_rows.iloc[i].values
        x_values = values[:len(row_data)]
        line_i, = axes[i].plot(x_values, row_data, marker='o', color=colors[idx],
                               label=f"Participant {idx+1}")
        if idx == 0 and 8 in x_values:
            y_value = row_data[np.where(x_values == 8)[0][0]]
            axes[i].plot(8, y_value, marker='*', color='red', markersize=15)

        axes[i].set_xlabel(r"Iteration", fontsize=25)
        axes[i].set_xticks(x_values)
        axes[i].set_yticks(sorted(np.unique(row_data)))
        axes[i].set_title(title_labels[i], fontsize=25)
        axes[i].xaxis.set_major_locator(MultipleLocator(2))
        axes[i].yaxis.set_major_locator(MultipleLocator(y_spacing[i]))
        axes[i].grid(True, which='both', linestyle='--', linewidth=0.5)
        axes[i].axvline(x=6, color='black', linestyle='--', linewidth=1.0, alpha=0.6)
        axes[i].tick_params(axis='both', which='major', labelsize=18)

    x_indices = np.arange(2, len(row_4) + 2)
    line, = axes[3].plot(x_indices, row_4.values * 10, marker='o', linestyle='-',
                         color=colors[idx], label=f"Participant {idx+1}")

    if idx == 0 and 8 in x_indices:
        y_value = (row_4.values * 10)[np.where(x_indices == 8)[0][0]]
        axes[3].plot(8, y_value, marker='*', color='red', markersize=15)

    axes[3].set_xlabel(r"Iteration", fontsize=25)
    axes[3].set_title(title_labels[3], fontsize=25)
    axes[3].set_yticks(sorted(np.unique(row_4.values * 10)))
    axes[3].xaxis.set_major_locator(MultipleLocator(2))
    axes[3].yaxis.set_major_locator(MultipleLocator(10))
    axes[3].grid(True, which='both', linestyle='--', linewidth=0.5)
    axes[3].axvline(x=6, color='black', linestyle='--', linewidth=1.0, alpha=0.6)
    axes[3].tick_params(axis='both', which='major', labelsize=18)

    # Collect the *line* for this participant only once (from the 4th subplot)
    lines.append(line)
    # Label includes participant index and name
    legend_labels.append(rf"P {idx+1} -- {participant_names[idx]}")

# Shared legend
fig.legend(lines, legend_labels, loc='lower center',
           ncol=len(file_paths), fontsize=15, frameon=False)

plt.tight_layout(rect=[0, 0.1, 1, 0.95])
plt.subplots_adjust(bottom=0.2)
plt.show()
