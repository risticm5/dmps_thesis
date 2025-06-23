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
file1_path = "experiment1/iteration15.csv"
file2_path = "experiment1/static_test.csv"
script_dir = os.path.dirname(os.path.realpath(__file__))
full_file1_path = os.path.join(script_dir, file1_path)
full_file2_path = os.path.join(script_dir, file2_path)

# Height of the partecipant
if file1_path == "experiment1/iteration15.csv":
    lab_cand = 1
elif file1_path == "experiment3/iteration15.csv":
    lab_cand = 2
elif file1_path == "experiment4/iteration15.csv":
    lab_cand = 3
elif file1_path == "experiment5/iteration15.csv":
    lab_cand = 4
elif file1_path == "experiment6/iteration15.csv":
    lab_cand = 5
elif file1_path == "experiment7/iteration15.csv":
    lab_cand = 6

scaling_gauss = 1

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

# Determine the global x-axis limits
x_min = min(np.min(x_values_static), np.min(x_values_iter)) - 0.05  # Slightly below min
x_max = max(np.max(x_values_static), np.max(x_values_iter)) + 0.05  # Slightly above max

# Adjust Gaussian curves to be within these limits
x_range = np.linspace(x_min, x_max, 300)


# ====== PLOTTING ======
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# ---- First subplot: Static Test Data ----
axs[0].scatter(x_values_static, np.zeros_like(x_values_static), alpha=0.6, color='green', s=100)

# Generate Gaussian curve for static test data
x_range = np.linspace(mean_static - 4 * std_dev_static, mean_static + 4 * std_dev_static, 300)
gaussian_static = norm.pdf(x_range, mean_static, std_dev_static)  
gaussian_static_scaled = gaussian_static / np.max(gaussian_static) * scaling_gauss  # Reduced height
axs[0].plot(x_range, gaussian_static_scaled, color='green', linestyle='solid')

# Shade only between ±1σ
x_shade = np.linspace(mean_static - std_dev_static, mean_static + std_dev_static, 200)
gaussian_shade = norm.pdf(x_shade, mean_static, std_dev_static)
gaussian_shade_scaled = gaussian_shade / np.max(gaussian_static) * scaling_gauss
axs[0].fill_between(x_shade, 0, gaussian_shade_scaled, color='green', alpha=0.2)

# Vertical lines for the standrad deviation
it = 0
for x_val in [mean_static - std_dev_static, mean_static + std_dev_static]:
    y_val = gaussian_static_scaled[np.argmin(np.abs(x_range - x_val))]
    if it == 0:
        axs[0].axvline(x_val, ymax=y_val / (1.6*scaling_gauss), color='green', linewidth=1, alpha=0.6)
    else:
        axs[0].axvline(x_val, ymax=y_val / (1.6*scaling_gauss), color='green', linewidth=1, alpha=0.6, label=rf'$\sigma(\mathcal{{X}}_{{{lab_cand}, \text{{static}}}})$: {std_dev_static:.3f}')
    it = it + 1    


y_val = gaussian_static_scaled[np.argmin(np.abs(x_range - mean_static))]
axs[0].axvline(mean_static, ymax=y_val / (1.6*scaling_gauss), color='green', linestyle='dashed', linewidth=2, alpha=1.0, label=rf'$\mu$: {mean_static:.3f}')

# Updated legend with mean and std deviation values
axs[0].legend(fontsize = 25)
#axs[0].set_ylabel(rf"\textbf{{Static case}}", fontsize=25)
axs[0].set_xlim(x_min, x_max)
axs[0].set_ylim(-0.05, 1.6)  # Adjust based on Gaussian curve height
#axs[0].set_title(rf"\textbf{{Candidate {lab_cand}}}", fontsize=25)

axs[0].grid(True)

# ---- Second subplot: K-Means Clustering ----
axs[1].scatter(x_cluster_1, np.zeros_like(x_cluster_1), 
               label=rf"Cluster $\mathcal{{X}}_{{{lab_cand},1, \text{{opt}}}} \rightarrow \mu: {mean_cluster_1:.2f}, \sigma: {std_dev_cluster_1:.2f}$", 
               alpha=0.6, color='red', s=100)

axs[1].scatter(x_cluster_2, np.zeros_like(x_cluster_2), 
               label=rf"Cluster $\mathcal{{X}}_{{{lab_cand},2, \text{{opt}}}} \rightarrow \mu: {mean_cluster_2:.2f}, \sigma: {std_dev_cluster_2:.2f}$", 
               alpha=0.6, color='blue', s=100)

#axs[1].scatter(x_cluster_2, np.zeros_like(x_cluster_2), label=rf"Cluster 2, $\mu$: {mean_cluster_2:.2f}, $\sigma$: {std_dev_cluster_2:.2f}", alpha=0.6, color='blue', s=100)

# Generate Gaussian curves for clusters
for mean, std_dev, color in [(mean_cluster_1, std_dev_cluster_1, 'red'), (mean_cluster_2, std_dev_cluster_2, 'blue')]:
    x_range = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 300)
    gaussian_curve = norm.pdf(x_range, mean, std_dev)
    gaussian_scaled = gaussian_curve / np.max(gaussian_curve) * scaling_gauss  # Scale height
    axs[1].plot(x_range, gaussian_scaled, color=color, linestyle='solid')

    # Shade only between ±1σ
    x_shade = np.linspace(mean - std_dev, mean + std_dev, 200)
    gaussian_shade = norm.pdf(x_shade, mean, std_dev)
    gaussian_shade_scaled = gaussian_shade / np.max(gaussian_curve) * scaling_gauss
    axs[1].fill_between(x_shade, 0, gaussian_shade_scaled, color=color, alpha=0.2)

    # Vertical lines for the standard deviation
    for x_val in [mean - std_dev, mean + std_dev]:
        y_val = gaussian_scaled[np.argmin(np.abs(x_range - x_val))]
        axs[1].axvline(x_val, ymax=y_val / (1.6*scaling_gauss), color=color, linewidth=1, alpha=0.6)

    # Vertical line for the mean value
    y_val = gaussian_scaled[np.argmin(np.abs(x_range - mean))]
    axs[1].axvline(mean, ymax=y_val / (1.6*scaling_gauss), color=color, linestyle='dashed', linewidth=2, alpha=1.0)

# Compute the ratios between std devs
lambda_x_max = max((std_dev_static - std_dev_cluster_1) / std_dev_static, 
               (std_dev_static - std_dev_cluster_2) / std_dev_static)

lambda_x_min = min((std_dev_static - std_dev_cluster_1) / std_dev_static, 
               (std_dev_static - std_dev_cluster_2) / std_dev_static)

# Updated legend with mean and std deviation values for clusters
#extra_label = rf"$\lambda_{{x,{lab_cand},\max}} = {lambda_x_max:.3f}, \quad \lambda_{{x,{lab_cand},\min}} = {lambda_x_min:.3f}$"
#axs[1].plot([], [], ' ', label=extra_label)  # Empty plot with a label
axs[1].legend(fontsize=25)
#axs[1].set_xlabel(r"$x_H$ (m)", fontsize=25)
#axs[1].set_ylabel(rf"\textbf{{Optimal solution}}", fontsize=25)
axs[1].set_ylim(-0.05, 1.6)  # Adjust based on Gaussian curve height
axs[1].set_xlim(x_min, x_max)
axs[1].grid(True)

axs[0].tick_params(axis='both', which='major', labelsize=30)  # Increase size for major ticks
axs[1].tick_params(axis='both', which='major', labelsize=30)  # Increase size for major ticks

plt.tight_layout()
plt.show()
