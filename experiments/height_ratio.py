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
file1_path = "experiment7/iteration15.csv"
file2_path = "experiment7/static_test.csv"
script_dir = os.path.dirname(os.path.realpath(__file__))
full_file1_path = os.path.join(script_dir, file1_path)
full_file2_path = os.path.join(script_dir, file2_path)

# ====== PROCESS static_test.csv ======
# Load the static test data
data_static = pd.read_csv(full_file2_path)

# Extract Z values from static_test.csv
z_values_static = data_static['Z'].to_numpy()

# ====== PROCESS iteration15.csv ======
# Read iteration15.csv content
with open(full_file1_path, 'r') as file:
    content1 = file.read()

# Function to extract values from iteration15.csv
def extract_values(content, start_label, end_label):
    start_idx = content.find(start_label) + len(start_label)
    end_idx = content.find(end_label) if end_label else len(content)
    values_str = content[start_idx:end_idx].strip().replace("\n", "")
    values = [float(v) for v in values_str.split(',') if v.strip()]
    return np.array(values)

# Extract Z values from iteration15.csv
z_values_iter = extract_values(content1, 'Z Values', 'QX Values')

# Define the reference heights
dT = 0.935  # Table height
hp = 1.950  # Person height
delta_H = 0.300  # Height difference between head and shoulder
hS = hp - delta_H  # Shoulder height

# Adjust all the z values to include the table height
hG_static = z_values_static + dT
hG_iter = z_values_iter + dT

# Compute mean and standard deviation
mean_static = np.mean(hG_static)
std_dev_static = np.std(hG_static, ddof=1)

mean_iter = np.mean(hG_iter)
std_dev_iter = np.std(hG_iter, ddof=1)

# Generate Gaussian curves (Normalized)
z_range_static = np.linspace(mean_static - 4 * std_dev_static, mean_static + 4 * std_dev_static, 300)
gaussian_static = norm.pdf(z_range_static, mean_static, std_dev_static)
gaussian_static_scaled = gaussian_static / np.max(gaussian_static) * 0.3  # Reduced height

z_range_iter = np.linspace(mean_iter - 4 * std_dev_iter, mean_iter + 4 * std_dev_iter, 300)
gaussian_iter = norm.pdf(z_range_iter, mean_iter, std_dev_iter)
gaussian_iter_scaled = gaussian_iter / np.max(gaussian_iter) * 0.3  # Reduced height

# ====== PLOTTING ======
fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# ---- Function to plot each subplot ----
def plot_gaussian(ax, hG_data, mean, std_dev, gaussian_scaled, z_range, color, label):
    # Scatter plot of the raw data (vertical dots)
    ax.scatter(np.zeros_like(hG_data), hG_data, label=label, alpha=0.6, color=color)

    # Gaussian curve
    ax.plot(gaussian_scaled, z_range, color=color, linestyle='solid', label=r"Gaussian Fit")

    # Compute intersection points (where dashed lines should stop)
    def find_intersection(z_values, gaussian, value):
        idx = np.argmin(np.abs(z_values - value))  # Find closest index
        return gaussian[idx]  # Return corresponding gaussian height

    # Vertical limits
    ymin, ymax = min(hG_data), max(hG_data)

    # Mean line
    mean_height = find_intersection(z_range, gaussian_scaled, mean)
    ax.axhline(mean, xmax=mean_height / 0.3, color=color, linestyle='dashed', linewidth=1, label=rf'Mean: {mean:.3f} m')

    # ±1 Standard deviation lines
    std_low = mean - std_dev
    std_high = mean + std_dev

    std_low_height = find_intersection(z_range, gaussian_scaled, std_low)
    std_high_height = find_intersection(z_range, gaussian_scaled, std_high)

    ax.axhline(std_low, xmax=std_low_height / 0.3, color=color, linestyle='dashed', linewidth=1, alpha=0.6)
    ax.axhline(std_high, xmax=std_high_height / 0.3, color=color, linestyle='dashed', linewidth=1, alpha=0.6)

    # Shoulder height point (Red Circle)
    ax.scatter(0, hS, color='red', s=60, edgecolors='black', label=rf'Shoulder Height: {hS:.3f} m', zorder=3)

    # Fill area inside dashed lines and Gaussian curve
    ax.fill_betweenx(z_range, 0, gaussian_scaled, where=((z_range >= std_low) & (z_range <= std_high)), color=color, alpha=0.2)

    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_title(label, fontsize=14)
    ax.legend()
    ax.grid(True)

# ---- First subplot: Static Test ----
plot_gaussian(axs[0], hG_static, mean_static, std_dev_static, gaussian_static_scaled, z_range_static, 'green', r"\textbf{Static Test}")

# ---- Second subplot: Iteration 15 ----
plot_gaussian(axs[1], hG_iter, mean_iter, std_dev_iter, gaussian_iter_scaled, z_range_iter, 'blue', r"\textbf{Iteration 15}")

# Adjust layout and show plot
plt.tight_layout()
plt.show()
