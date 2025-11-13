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
    "test1_christian/answers.csv",
    "test2_xenia/answers.csv",
    "test3_mihailo/answers.csv",
    "test4_marko/answers.csv",
    "test5_eleonora/answers.csv",
    "test6_bozidar/answers.csv",
    "test7_alessandro/answers.csv",
    "test8_daniele/answers.csv",
    "test9_eleonora/answers.csv",
    "test10_paola/answers.csv",
    "test11_antonio/answers.csv",
    "test12_mario/answers.csv",
    "test13_marko/answers.csv",
    "test14_edoardo/answers.csv",
    "test15_matteo/answers.csv"
]

participant_names = [
    "Chr(1.82)", "Xe(1.68)", "Mih(1.92)", "Ma(1.92)", "El1(1.55)",
    "Boz(1.95)", "Ale(1.7)", "Dan(1.97)", "El2(1.82)", "Pao(1.76)", "Ant(1.75)",
    "Mar(1.77)", "Ma(1.85)", "Edo(1.91)", "Mat(1.75)"
]

# Colors
cmap = plt.get_cmap('tab20', len(file_paths))
colors = cmap(range(len(file_paths)))

# Figure & axes
fig, axes = plt.subplots(1, 4, figsize=(20, 6))
title_labels = [r"$\boldsymbol{\bar{\tau}}$", r"$\boldsymbol{k_s}$",
                r"$\boldsymbol{k_m}$", r"$\boldsymbol{A\%}$"]

lines = []
legend_labels = []

### NEW: list to store final A% (last value of row_4) for each participant
final_A_values = []

for idx, file_path in enumerate(file_paths):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    full_path = os.path.join(script_dir, file_path)

    with open(full_path, 'r') as file:
        lines_data = [line.strip().split(',') for line in file]

    max_length = max(len(line) for line in lines_data)
    padded_data = [line + [np.nan] * (max_length - len(line)) for line in lines_data]
    df = pd.DataFrame(padded_data).apply(pd.to_numeric, errors='coerce')

    # 4th row: A% over iterations
    row_4 = df.iloc[3].dropna()

    ### NEW: store final A% (last non-NaN) * 10
    final_A = row_4.values[-1] * 10
    final_A_values.append(final_A)

    last_3_rows = df.tail(3).apply(lambda x: x.dropna(), axis=1)

    values = np.arange(1, 1 + max_length)
    y_spacing = [0.01, 0.05, 0.05]

    # First 3 line plots (unchanged)
    for i in range(3):
        row_data = last_3_rows.iloc[i].values
        x_values = values[:len(row_data)]
        line_i, = axes[i].plot(x_values, row_data, marker='o', color=colors[idx],
                               label=f"Participant {idx+1}")

        # optional red star on iteration 8 for participant 1
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

        ### NEW: collect one line handle per participant (from first subplot)
        if i == 0:
            lines.append(line_i)
            legend_labels.append(rf"P {idx+1} -- {participant_names[idx]}")

# ===== NEW 4th subplot: horizontal bars with final A% =====
ax4 = axes[3]

y_pos = np.arange(len(file_paths))  # one bar per participant
bars = ax4.barh(y_pos, final_A_values, color=colors)

ax4.set_yticks(y_pos)
# you can use participant_names or "P i" style
ax4.set_yticklabels([rf"P {i+1}" for i in range(len(file_paths))], fontsize=18)
ax4.invert_yaxis()  # P1 at top

ax4.set_xlabel(r"$A\%$", fontsize=25)
ax4.set_title(title_labels[3], fontsize=25)

# nice x-axis scaling (0–100, step 10)
ax4.set_xlim(0, 100)
ax4.xaxis.set_major_locator(MultipleLocator(10))
ax4.grid(True, axis='x', which='both', linestyle='--', linewidth=0.5)
ax4.tick_params(axis='both', which='major', labelsize=18)

# Shared legend (still based on line colors)
fig.legend(lines, legend_labels, loc='lower center',
           ncol=len(file_paths), fontsize=7, frameon=False)

plt.tight_layout(rect=[0, 0.1, 1, 0.95])
plt.subplots_adjust(bottom=0.2)

# ====== SAVE FIGURE AS SVG ======
output_dir = os.path.join(script_dir, "../Plots")  # Folder where to save SVGs
os.makedirs(output_dir, exist_ok=True)  # Create folder if it doesn't exist

output_path = os.path.join(output_dir, f"tau_km\_ks_trends.svg")
plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')

print(f"✅ Figure saved successfully at: {output_path}")

plt.show()
