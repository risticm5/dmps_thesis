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

# File paths
file1_path = "experiment15/iteration15.csv"
file2_path = "experiment15/static_test.csv"
script_dir = os.path.dirname(os.path.realpath(__file__))
full_file1_path = os.path.join(script_dir, file1_path)
full_file2_path = os.path.join(script_dir, file2_path)

# ====== PROCESS static_test.csv ======
data_static = pd.read_csv(full_file2_path)
z_values_static = data_static['Z'].to_numpy()

# ====== PROCESS iteration15.csv ======
with open(full_file1_path, 'r') as file:
    content1 = file.read()

def extract_values(content, start_label, end_label):
    start_idx = content.find(start_label) + len(start_label)
    end_idx = content.find(end_label) if end_label else len(content)
    values_str = content[start_idx:end_idx].strip().replace("\n", "")
    values = [float(v) for v in values_str.split(',') if v.strip()]
    return np.array(values)

z_values_iter = extract_values(content1, 'Z Values', 'QX Values')

# Height of the participant
hp = 1.82  # Default height
dh_static = 0.3
dh_opt = 0.3
if file1_path == "experiment2/iteration15.csv":
    hp = 1.68
    dh_static = 0.3
    dh_opt = 0.25
    lab_cand = 2
elif file1_path == "experiment3/iteration15.csv":
    hp = 1.92
    dh_static = 0.3
    dh_opt = 0.3
    lab_cand = 3
elif file1_path == "experiment4/iteration15.csv":
    hp = 1.92
    dh_static = 0.3
    dh_opt = 0.3
    lab_cand = 4
elif file1_path == "experiment5/iteration15.csv":
    hp = 1.55
    dh_static = 0.3
    dh_opt = 0.3
    lab_cand = 5
elif file1_path == "experiment6/iteration15.csv":
    hp = 1.95
    dh_static = 0.3
    dh_opt = 0.3
    lab_cand = 6
elif file1_path == "experiment7/iteration15.csv":
    hp = 1.70
    dh_static = 0.3
    dh_opt = 0.25
    lab_cand = 7
elif file1_path == "experiment8/iteration15.csv":
    hp = 1.97
    dh_static = 0.3
    dh_opt = 0.25
    lab_cand = 8
elif file1_path == "experiment9/iteration15.csv":
    hp = 1.82
    dh_static = 0.3
    dh_opt = 0.25
    lab_cand = 9
elif file1_path == "experiment10/iteration15.csv":
    hp = 1.71
    dh_static = 0.3
    dh_opt = 0.3
    lab_cand = 10
elif file1_path == "experiment11/iteration15.csv":
    hp = 1.74
    dh_static = 0.3
    dh_opt = 0.3
    lab_cand = 11
elif file1_path == "experiment12/iteration15.csv":
    hp = 1.77
    dh_static = 0.35
    dh_opt = 0.3
    lab_cand = 12
elif file1_path == "experiment13/iteration15.csv":
    hp = 1.85
    dh_static = 0.3
    dh_opt = 0.3
    lab_cand = 13
elif file1_path == "experiment14/iteration15.csv":
    hp = 1.88
    dh_static = 0.3
    dh_opt = 0.3
    lab_cand = 14
elif file1_path == "experiment15/iteration15.csv":
    hp = 1.75
    dh_static = 0.35
    dh_opt = 0.26
    lab_cand = 15
else:
    lab_cand = 1

dT = 0.935  # Table height
hS_static = hp - dh_static  # Shoulder height
hS_iter = hp - dh_opt  # Shoulder height

# Adjust Z values
hG_static = (z_values_static + dT) / hS_static
hG_iter = (z_values_iter + dT) / hS_iter

# Compute statistics
mean_static = np.mean(hG_static)
std_dev_static = np.std(hG_static, ddof=1)
mean_iter = np.mean(hG_iter)
std_dev_iter = np.std(hG_iter, ddof=1)

# Determine y-axis limits with offsets
y_min = min(np.min(hG_static), np.min(hG_iter)) - 0.15  # Offset below min
y_max = max(np.max(hG_static), np.max(hG_iter)) + 0.1  # Offset above max

# Generate Gaussian curves
z_range = np.linspace(y_min, y_max, 300)

gaussian_static = norm.pdf(z_range, mean_static, std_dev_static)
gaussian_static_scaled = gaussian_static / np.max(gaussian_static) * 0.1

gaussian_iter = norm.pdf(z_range, mean_iter, std_dev_iter)
gaussian_iter_scaled = gaussian_iter / np.max(gaussian_iter) * 0.1

# ====== PLOTTING ======
fig, ax = plt.subplots(figsize=(3, 8))

# Scatter plot (static at x=0, iter at x=0.4)
ax.scatter(np.zeros_like(hG_static), hG_static, color='green', alpha=0.6, s=100)
ax.scatter(np.full_like(hG_iter, 0.2), hG_iter, color='blue', alpha=0.6, s=100)

# Gaussian curves
ax.plot(gaussian_static_scaled, z_range, color='green', linestyle='solid')
ax.plot(0.2 + gaussian_iter_scaled, z_range, color='blue', linestyle='solid')

# Dashed lines for mean and std deviation
# Dashed lines for mean and std deviation
for mean, std_dev, color, x_pos, gaussian in [(mean_static, std_dev_static, 'green', 0, gaussian_static_scaled),
                                              (mean_iter, std_dev_iter, 'blue', 0.2, gaussian_iter_scaled)]:
    # Compute where the Gaussian curve is at the respective height
        # Compute the Gaussian curve width at the respective height
    gaussian_end = x_pos + gaussian[np.argmin(np.abs(z_range - mean))]
    gaussian_end_low = x_pos + gaussian[np.argmin(np.abs(z_range - (mean - std_dev)))]
    gaussian_end_high = x_pos + gaussian[np.argmin(np.abs(z_range - (mean + std_dev)))]
    
    # Draw correct dashed lines from scatter points to the Gaussian
    ax.hlines(mean, x_pos, gaussian_end, color=color, linestyle='dashed', linewidth=2)
    ax.hlines(mean - std_dev, x_pos, gaussian_end_low, color=color, linestyle='dashed', linewidth=1, alpha=0.6)
    ax.hlines(mean + std_dev, x_pos, gaussian_end_high, color=color, linestyle='dashed', linewidth=1, alpha=0.6)

    
    # Fill area under Gaussian and above dataset, limited by std dev lines
    ax.fill_betweenx(z_range, x_pos, x_pos + gaussian,
                      where=((z_range >= mean - std_dev) & (z_range <= mean + std_dev)),
                      color=color, alpha=0.2)


# Shoulder height horizontal dashed line
ax.axhline(1, xmin=-0.1, xmax=2, color='red', linestyle='dashed', linewidth=2.0)

# Compute the ratios
lambda_z = (std_dev_static - std_dev_iter) / std_dev_static
extra_label = rf"$\lambda_{{z,{lab_cand}}} = {lambda_z:.3f}$"
ax.plot([], [], ' ', label=extra_label)  # Empty plot for legend entry

# Labels and formatting
ax.set_xlim(-0.05, 0.35)
ax.set_ylim(y_min, y_max)
#ax.set_ylabel(rf'$\xi_{{{lab_cand}}} = h_{{G,{lab_cand}}}/h_{{S,{lab_cand}}}$', fontsize=25)
ax.legend(fontsize=10)
ax.grid(True)
ax.set_title(rf'$\textbf{{Participant\ {lab_cand}}}$', fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=25)  # Increase size for major ticks

ax.set_xticklabels([])

plt.tight_layout()

# ====== SAVE FIGURE AS SVG ======
output_dir = os.path.join(script_dir, "Plots")  # Folder where to save SVGs
os.makedirs(output_dir, exist_ok=True)  # Create folder if it doesn't exist

output_path = os.path.join(output_dir, f"p{lab_cand}_xiz_height.svg")
plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')

print(f"✅ Figure saved successfully at: {output_path}")

plt.show()


