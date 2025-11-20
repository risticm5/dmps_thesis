import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

# LaTeX settings
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# File paths
file1_path = "experiment1/iteration15.csv"
file2_path = "experiment1/static_test.csv"
script_dir = os.path.dirname(os.path.realpath(__file__))
full_file1_path = os.path.join(script_dir, file1_path)
full_file2_path = os.path.join(script_dir, file2_path)

# Read file1 content
with open(full_file1_path, 'r') as file:
    content1 = file.read()

# Extract values for file1
def extract_values(content, start_label, end_label):
    start_idx = content.find(start_label) + len(start_label)
    end_idx = content.find(end_label) if end_label else len(content)
    values_str = content[start_idx:end_idx].strip().replace("\n", "")
    values = [float(v) for v in values_str.split(',') if v.strip()]
    return np.array(values)

# Convert to meters
x_values1 = extract_values(content1, 'X Values', 'Y Values')  
y_values1 = extract_values(content1, 'Y Values', 'Z Values')  
y_values1 = np.minimum(y_values1, 1)  # Adjust max value to meters
z_values1 = extract_values(content1, 'Z Values', 'QX Values')  
qx_values1 = extract_values(content1, 'QX Values', 'QY Values')
qy_values1 = extract_values(content1, 'QY Values', 'QZ Values')
qz_values1 = extract_values(content1, 'QZ Values', 'QW Values')
qw_values1 = extract_values(content1, 'QW Values', None)

# Filter out points with y > 0.97 meters
mask1 = y_values1 <= 0.97
positions1 = np.column_stack((x_values1[mask1], y_values1[mask1], z_values1[mask1]))
quaternions1 = np.column_stack((qx_values1[mask1], qy_values1[mask1], qz_values1[mask1], qw_values1[mask1]))

# Load file2 (static_test.csv)
data2 = pd.read_csv(full_file2_path)

# Convert to meters
x_values2 = data2['X']  
y_values2 = data2['Y']  
y_values2 = np.minimum(y_values2, 1)  # Adjust max value to meters
z_values2 = data2['Z']  
qx_values2 = data2['Rot_X']
qy_values2 = data2['Rot_Y']
qz_values2 = data2['Rot_Z']
qw_values2 = data2['Rot_W']

# Filter out points with y > 0.97 meters
mask2 = y_values2 <= 0.97
positions2 = np.column_stack((x_values2[mask2], y_values2[mask2], z_values2[mask2]))
quaternions2 = np.column_stack((qx_values2[mask2], qy_values2[mask2], qz_values2[mask2], qw_values2[mask2]))

# Determine axis limits
all_positions = np.vstack((positions1, positions2))
x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
z_min, z_max = all_positions[:, 2].min(), all_positions[:, 2].max()

# Compute the center and range for each axis
x_mid = (x_max + x_min) / 2
y_mid = (y_max + y_min) / 2
z_mid = (z_max + z_min) / 2

x_range = x_max - x_min
y_range = y_max - y_min
z_range = z_max - z_min

'''
# Find the maximum range among all axes
max_range = max(x_range, y_range, z_range)

# Adjust limits to ensure equal scaling for all axes
x_min, x_max = x_mid - max_range / 2, x_mid + max_range / 2
y_min, y_max = y_mid - max_range / 2, y_mid + max_range / 2
z_min, z_max = z_mid - max_range / 2, z_mid + max_range / 2
'''

# Create 3D subplots (inverted order)
fig, axs = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})

# Plot reference frames function
def plot_reference_frames(ax, positions, quaternions, title):
    for pos, quat in zip(positions, quaternions):
        rot = R.from_quat(quat)
        origin = pos
        
        axis_length = 0.01  # Adjusted axis length for meters
        x_axis = rot.apply([axis_length, 0, 0])
        y_axis = rot.apply([0, axis_length, 0])
        z_axis = rot.apply([0, 0, axis_length])
        
        ax.quiver(*origin, *x_axis, color='r', length=axis_length, normalize=True)
        ax.quiver(*origin, *y_axis, color='g', length=axis_length, normalize=True)
        ax.quiver(*origin, *z_axis, color='b', length=axis_length, normalize=True)
    
    # Apply the adjusted limits for equal aspect ratio
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    
    ax.set_xlabel(r"$X$ (m)", fontsize=14)
    ax.set_ylabel(r"$Y$ (m)", fontsize=14)
    ax.set_zlabel(r"$Z$ (m)", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)  # Increase tick label size
    ax.view_init(elev=30, azim=150)  # Adjusted for correct perspective

# Plot with inverted order
plot_reference_frames(axs[0], positions2, quaternions2, "Static Test")
plot_reference_frames(axs[1], positions1, quaternions1, "Experiment 1")

# Show the plots
plt.tight_layout()
plt.show()

# COmpute the standard deviation for x 
x_std1 = np.std(positions1[:, 0])
x_std2 = np.std(positions2[:, 0])
print(f"Standard deviation for Experiment 1 (X): {x_std1:.4f} m")
print(f"Standard deviation for Static Test (X): {x_std2:.4f} m")

# Compute the standard deviation for z
z_std1 = np.std(positions1[:, 2])
z_std2 = np.std(positions2[:, 2])
print(f"Standard deviation for Experiment 1 (Z): {z_std1:.4f} m")
print(f"Standard deviation for Static Test (Z): {z_std2:.4f} m")
