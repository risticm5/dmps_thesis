import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib as mpl
from scipy.stats import norm  # For Gaussian curves

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

# Extract X values from static_test.csv
x_values_static = data_static['X'].to_numpy()

# Compute mean and standard deviation for static test X values
mean_static = np.mean(x_values_static)
std_dev_static = np.std(x_values_static, ddof=1)  # Sample standard deviation

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

# Extract X values from iteration15.csv
x_values_iter = extract_values(content1, 'X Values', 'Y Values')

# Ensure x_values_iter is reshaped for clustering
x_values_iter_reshaped = x_values_iter.reshape(-1, 1)

# Apply K-Means clustering with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(x_values_iter_reshaped)

# Get cluster labels
labels = kmeans.labels_

# Separate data into two clusters
x_cluster_1 = x_values_iter[labels == 0]
x_cluster_2 = x_values_iter[labels == 1]

# Compute cluster means and standard deviations
mean_cluster_1 = np.mean(x_cluster_1)
mean_cluster_2 = np.mean(x_cluster_2)
std_dev_cluster_1 = np.std(x_cluster_1, ddof=1)  # Sample standard deviation
std_dev_cluster_2 = np.std(x_cluster_2, ddof=1)

# ====== PLOTTING ======
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# ---- First subplot: Static Test Data ----
axs[0].scatter(x_values_static, np.zeros_like(x_values_static), label=rf"$\mu$: {mean_static:.2f}, $\sigma$: {std_dev_static:.2f}", alpha=0.6, color='green')

# Generate Gaussian curve for static test data
x_range = np.linspace(mean_static - 4 * std_dev_static, mean_static + 4 * std_dev_static, 300)
gaussian_static = norm.pdf(x_range, mean_static, std_dev_static)  
gaussian_static_scaled = gaussian_static / np.max(gaussian_static) * 0.01  # Reduced height
axs[0].plot(x_range, gaussian_static_scaled, color='green', linestyle='solid')

# Shade only between ±1σ
x_shade = np.linspace(mean_static - std_dev_static, mean_static + std_dev_static, 200)
gaussian_shade = norm.pdf(x_shade, mean_static, std_dev_static)
gaussian_shade_scaled = gaussian_shade / np.max(gaussian_static) * 0.01
axs[0].fill_between(x_shade, 0, gaussian_shade_scaled, color='green', alpha=0.2)

# Vertical lines stopping at Gaussian height
for x_val in [mean_static, mean_static - std_dev_static, mean_static + std_dev_static]:
    y_val = gaussian_static_scaled[np.argmin(np.abs(x_range - x_val))]
    axs[0].axvline(x_val, ymax=y_val / 0.01, color='green', linestyle='dashed', linewidth=1, alpha=0.6)

# Updated legend with mean and std deviation values
axs[0].legend(fontsize = 15)
axs[0].set_ylabel(r"\textbf{Static case participant 6}", fontsize=15)
axs[0].grid(True)

# ---- Second subplot: K-Means Clustering ----
axs[1].scatter(x_cluster_1, np.zeros_like(x_cluster_1), label=rf"Cluster 1, $\mu$: {mean_cluster_1:.2f}, $\sigma$: {std_dev_cluster_1:.2f}", alpha=0.6, color='red')
axs[1].scatter(x_cluster_2, np.zeros_like(x_cluster_2), label=rf"Cluster 2, $\mu$: {mean_cluster_2:.2f}, $\sigma$: {std_dev_cluster_2:.2f}", alpha=0.6, color='blue')

# Generate Gaussian curves for clusters
for mean, std_dev, color in [(mean_cluster_1, std_dev_cluster_1, 'red'), (mean_cluster_2, std_dev_cluster_2, 'blue')]:
    x_range = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 300)
    gaussian_curve = norm.pdf(x_range, mean, std_dev)
    gaussian_scaled = gaussian_curve / np.max(gaussian_curve) * 0.01  # Scale height
    axs[1].plot(x_range, gaussian_scaled, color=color, linestyle='solid')

    # Shade only between ±1σ
    x_shade = np.linspace(mean - std_dev, mean + std_dev, 200)
    gaussian_shade = norm.pdf(x_shade, mean, std_dev)
    gaussian_shade_scaled = gaussian_shade / np.max(gaussian_curve) * 0.01
    axs[1].fill_between(x_shade, 0, gaussian_shade_scaled, color=color, alpha=0.2)

    # Vertical lines stopping at Gaussian height
    for x_val in [mean, mean - std_dev, mean + std_dev]:
        y_val = gaussian_scaled[np.argmin(np.abs(x_range - x_val))]
        axs[1].axvline(x_val, ymax=y_val / 0.01, color=color, linestyle='dashed', linewidth=1, alpha=0.6)

# Updated legend with mean and std deviation values for clusters
axs[1].legend(fontsize = 15)
axs[1].set_xlabel(r"$x_H$ (m)", fontsize=15)
axs[1].set_ylabel(r"\textbf{Optimal solution participant 6}", fontsize=15)
axs[1].grid(True)

plt.tight_layout()
plt.show()
