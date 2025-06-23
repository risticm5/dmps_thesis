import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import time

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

# Extract values function
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

# Time-controlled animation settings
total_time_seconds = 30  # Set total animation duration
num_frames = max(len(positions1), len(positions2))
frame_interval = total_time_seconds / num_frames  # Time between frames

# Function to plot reference frames progressively
def plot_reference_frames(ax, positions, quaternions):
    start_time = time.time()
    num_positions = len(positions)
    
    for i, (pos, quat) in enumerate(zip(positions, quaternions)):
        elapsed_time = time.time() - start_time
        remaining_time = max(total_time_seconds - elapsed_time, 0)
        frames_left = max(num_positions - i, 1)
        sleep_time = remaining_time / frames_left

        rot = R.from_quat(quat)
        origin = pos
        
        axis_length = 0.01  # Adjusted axis length for meters
        x_axis = rot.apply([axis_length, 0, 0])
        y_axis = rot.apply([0, axis_length, 0])
        z_axis = rot.apply([0, 0, axis_length])
        
        ax.quiver(*origin, *x_axis, color='r', length=axis_length, normalize=True)
        ax.quiver(*origin, *y_axis, color='g', length=axis_length, normalize=True)
        ax.quiver(*origin, *z_axis, color='b', length=axis_length, normalize=True)
        
        plt.pause(0.001)  # Small pause for rendering
        time.sleep(sleep_time)  # Adaptive timing

### **FIRST FIGURE: STATIC TEST**
fig1 = plt.figure(figsize=(10, 8))  # Increased figure size
ax1 = fig1.add_subplot(111, projection='3d')
ax1.set_title(r"\textbf{Static Test}", fontsize=18)

# Set correct axis orientation (adjusted to match your attached figure)
ax1.view_init(elev=25, azim=135)  

# Apply labels
ax1.set_xlabel(r"$x_H$ (m)", fontsize=16, labelpad=10)
ax1.set_ylabel(r"$y_H$ (m)", fontsize=16, labelpad=10)
ax1.set_zlabel(r"$z_H$ (m)", fontsize=16, labelpad=10)
ax1.set_xlim([x_min, x_max])
ax1.set_ylim([y_min, y_max])
ax1.set_zlim([z_min, z_max])

plt.show(block=False)  # Show only the first figure

# Wait for user input before animating
input("Press Enter to start animating the Static Test...")

# Animate first figure
plot_reference_frames(ax1, positions2, quaternions2)

# Wait before proceeding
input("Press Enter to continue to the Experiment 1 plot...")

### **SECOND FIGURE: EXPERIMENT 1**
fig2 = plt.figure(figsize=(10, 8))  # Increased figure size
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_title(r"\textbf{Experiment 1}", fontsize=18)

# Set the same axis orientation as the first figure
ax2.view_init(elev=25, azim=135)

# Apply labels
ax2.set_xlabel(r"$x_H$ (m)", fontsize=16, labelpad=10)
ax2.set_ylabel(r"$y_H$ (m)", fontsize=16, labelpad=10)
ax2.set_zlabel(r"$z_H$ (m)", fontsize=16, labelpad=10)
ax2.set_xlim([x_min, x_max])
ax2.set_ylim([y_min, y_max])
ax2.set_zlim([z_min, z_max])

plt.show(block=False)  # Show second figure only after confirmation

# Wait for user input before animating the second figure
input("Press Enter to start animating Experiment 1...")

# Animate second figure
plot_reference_frames(ax2, positions1, quaternions1)

# Keep the second figure open
plt.show()

