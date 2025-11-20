import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats import norm  # For Gaussian distributions
import matplotlib as mpl

# LaTeX settings for plots
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Helper to extract values from iteration file
def extract_values(content, start_label, end_label):
    start_idx = content.find(start_label) + len(start_label)
    end_idx = content.find(end_label) if end_label else len(content)
    values_str = content[start_idx:end_idx].strip().replace("\n", "")
    values = [float(v) for v in values_str.split(',') if v.strip()]
    return np.array(values)

# File base directory
script_dir = os.path.dirname(os.path.realpath(__file__))

# ===================== PARTICIPANT CONFIGURATION ===================== #
# These are taken from your if/elif block, with experiment1 using defaults.
participants = [
    # exp_name, lab_cand, hp, dh_static, dh_opt
    ("experiment1",  1, 1.82, 0.3,  0.3),
    ("experiment2",  2, 1.68, 0.3,  0.25),
    ("experiment3",  3, 1.92, 0.3,  0.3),
    ("experiment4",  4, 1.92, 0.3,  0.3),
    ("experiment5",  5, 1.55, 0.3,  0.3),
    ("experiment6",  6, 1.95, 0.3,  0.3),
    ("experiment7",  7, 1.70, 0.3,  0.25),
    ("experiment8",  8, 1.97, 0.3,  0.25),
    ("experiment9",  9, 1.82, 0.3,  0.25),
    ("experiment10", 10, 1.71, 0.3,  0.3),
    ("experiment11", 11, 1.74, 0.3,  0.3),
    ("experiment12", 12, 1.77, 0.35, 0.3),
    ("experiment13", 13, 1.85, 0.3,  0.3),
    ("experiment14", 14, 1.88, 0.3,  0.3),
    ("experiment15", 15, 1.75, 0.35, 0.26),
]

dT = 0.935  # Table height

# ===================== FIRST PASS: LOAD & COMPUTE ===================== #
all_data = []
global_min = np.inf
global_max = -np.inf

for exp_name, lab_cand, hp, dh_static, dh_opt in participants:
    file1_path = os.path.join(exp_name, "iteration15.csv")
    file2_path = os.path.join(exp_name, "static_test.csv")

    full_file1_path = os.path.join(script_dir, file1_path)
    full_file2_path = os.path.join(script_dir, file2_path)

    # ----- PROCESS static_test.csv -----
    data_static = pd.read_csv(full_file2_path)
    z_values_static = data_static['Z'].to_numpy()

    # ----- PROCESS iteration15.csv -----
    with open(full_file1_path, 'r') as file:
        content1 = file.read()

    z_values_iter = extract_values(content1, 'Z Values', 'QX Values')

    # Shoulder heights
    hS_static = hp - dh_static
    hS_iter = hp - dh_opt

    # Adjust Z values (normalized heights)
    hG_static = (z_values_static + dT) / hS_static
    hG_iter = (z_values_iter + dT) / hS_iter

    # Compute statistics
    mean_static = np.mean(hG_static)
    std_dev_static = np.std(hG_static, ddof=1)
    mean_iter = np.mean(hG_iter)
    std_dev_iter = np.std(hG_iter, ddof=1)

    # Update global vertical bounds
    global_min = min(global_min, np.min(hG_static), np.min(hG_iter))
    global_max = max(global_max, np.max(hG_static), np.max(hG_iter))

    # Lambda_z
    lambda_z = (std_dev_static - std_dev_iter) / std_dev_static if std_dev_static != 0 else np.nan

    all_data.append({
        "exp_name": exp_name,
        "lab_cand": lab_cand,
        "hG_static": hG_static,
        "hG_iter": hG_iter,
        "mean_static": mean_static,
        "std_static": std_dev_static,
        "mean_iter": mean_iter,
        "std_iter": std_dev_iter,
        "lambda_z": lambda_z,
    })

# Determine y-axis limits with offsets (global)
y_min = global_min - 0.02  # Offset below min
y_max = global_max + 0.1   # Offset above max

# Common z-range for all Gaussians
z_range = np.linspace(y_min, y_max, 300)

# ===================== PLOTTING: MEGA PLOT ===================== #
# Wider horizontally than vertically
fig, ax = plt.subplots(figsize=(15, 4))

cluster_spacing = 0.5  # Horizontal spacing between participants

for idx, pdata in enumerate(all_data):
    lab_cand = pdata["lab_cand"]
    hG_static = pdata["hG_static"]
    hG_iter = pdata["hG_iter"]
    mean_static = pdata["mean_static"]
    std_dev_static = pdata["std_static"]
    mean_iter = pdata["mean_iter"]
    std_dev_iter = pdata["std_iter"]
    lambda_z = pdata["lambda_z"]

    # Base x-positions for this participant
    base_x = idx * cluster_spacing
    x_static = base_x
    x_iter = base_x + 0.2

    # Gaussian curves for this participant
    gaussian_static = norm.pdf(z_range, mean_static, std_dev_static) if std_dev_static > 0 else np.zeros_like(z_range)
    gaussian_iter = norm.pdf(z_range, mean_iter, std_dev_iter) if std_dev_iter > 0 else np.zeros_like(z_range)

    # Scale Gaussians to a fixed width
    gaussian_static_scaled = gaussian_static / np.max(gaussian_static) * 0.1 if np.max(gaussian_static) > 0 else gaussian_static
    gaussian_iter_scaled = gaussian_iter / np.max(gaussian_iter) * 0.1 if np.max(gaussian_iter) > 0 else gaussian_iter

    # Scatter plot (static at x_static, iter at x_iter)
    ax.scatter(np.full_like(hG_static, x_static), hG_static, color='green', alpha=0.6, s=30)
    ax.scatter(np.full_like(hG_iter, x_iter), hG_iter, color='blue', alpha=0.6, s=30)

    # Gaussian curves
    ax.plot(x_static + gaussian_static_scaled, z_range, color='green', linestyle='solid')
    ax.plot(x_iter + gaussian_iter_scaled, z_range, color='blue', linestyle='solid')

    # Dashed lines for mean and std deviation
    for mean, std_dev, color, x_pos, gaussian in [
        (mean_static, std_dev_static, 'green', x_static, gaussian_static_scaled),
        (mean_iter, std_dev_iter, 'blue',  x_iter,  gaussian_iter_scaled)
    ]:
        if std_dev <= 0 or np.all(gaussian == 0):
            continue

        # Compute the Gaussian curve width at the respective height
        gaussian_end = x_pos + gaussian[np.argmin(np.abs(z_range - mean))]
        gaussian_end_low = x_pos + gaussian[np.argmin(np.abs(z_range - (mean - std_dev)))]
        gaussian_end_high = x_pos + gaussian[np.argmin(np.abs(z_range - (mean + std_dev)))]

        # Draw correct dashed lines from scatter points to the Gaussian
        ax.hlines(mean, x_pos, gaussian_end, color=color, linestyle='dashed', linewidth=2)
        ax.hlines(mean - std_dev, x_pos, gaussian_end_low, color=color, linestyle='dashed', linewidth=1, alpha=0.6)
        ax.hlines(mean + std_dev, x_pos, gaussian_end_high, color=color, linestyle='dashed', linewidth=1, alpha=0.6)

        # Fill area under Gaussian and above dataset, limited by std dev lines
        ax.fill_betweenx(
            z_range, x_pos, x_pos + gaussian,
            where=((z_range >= mean - std_dev) & (z_range <= mean + std_dev)),
            color=color, alpha=0.2
        )

    # Add λ_z label for this participant above its cluster
    extra_label = rf"$\lambda_{{z,{lab_cand}}} = {lambda_z:.3f}$" if not np.isnan(lambda_z) else rf"$\lambda_{{z,{lab_cand}}} = \text{{n/a}}$"
    ax.text(base_x + 0.1, y_max - 0.02, extra_label, ha='center', va='top', fontsize=8)

    # Participant label at the bottom (optional, like small x-axis label)
    ax.text(base_x + 0.1, y_min + 0.02, rf"$P_{{{lab_cand}}}$", ha='center', va='bottom', fontsize=8)

# Vertical dashed lines between participant clusters

num_participants = len(all_data)
'''
for i in range(num_participants - 1):
    xline = (i + 0.5) * cluster_spacing
    ax.axvline(xline, color='gray', linestyle='dashed', linewidth=1, alpha=0.7)
'''
# Shoulder height horizontal dashed line at y = 1
ax.axhline(1, color='red', linestyle='dashed', linewidth=2.0)

# Labels and formatting
x_min = -0.1
x_max = (num_participants - 1) * cluster_spacing + 0.4
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
#ax.set_ylabel(rf'$\xi_{{{lab_cand}}} = h_{{G,{lab_cand}}}/h_{{S,{lab_cand}}}$', fontsize=25)
ax.grid(True)

# Simple legend for colors (static vs iter)
static_proxy = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Static')
iter_proxy = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Iterative')
ax.legend(handles=[static_proxy, iter_proxy], fontsize=10, loc='upper right')

ax.set_title(r'$\textbf{Participants\ 1\text{–}15}$', fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=10)  # More reasonable tick size for mega plot

ax.set_xticklabels([])

plt.tight_layout()

# ====== SAVE FIGURE AS SVG ======
output_dir = os.path.join(script_dir, "Plots")  # Folder where to save SVGs
os.makedirs(output_dir, exist_ok=True)  # Create folder if it doesn't exist

output_path = os.path.join(output_dir, "p_all_xiz_height.svg")
plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')

print(f"✅ Mega figure saved successfully at: {output_path}")

plt.show()
