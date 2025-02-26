import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib as mpl

# Imports for LaTeX plots
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Load the CSV file
path = "../../reference_trajectory_c_shape.csv"  # Replace with the actual path if needed
script_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(script_dir, path)

df = pd.read_csv(file_path)

# Extract position (x, y, z) and orientation (quaternions)
positions = df[['x', 'y', 'z']].values
quaternions = df[['qx', 'qy', 'qz', 'qw']].values

# Set up 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Function to plot reference frames with thinner arrows
def plot_reference_frame(ax, position, rotation, scale=0.02, linewidth=0.5):  # Reduced arrow thickness
    origin = position
    x_dir = rotation.apply([scale, 0, 0])  # X-axis in red
    y_dir = rotation.apply([0, scale, 0])  # Y-axis in green
    z_dir = rotation.apply([0, 0, scale])  # Z-axis in blue

    ax.quiver(*origin, *x_dir, color='r', length=scale, linewidth=linewidth, normalize=True)
    ax.quiver(*origin, *y_dir, color='g', length=scale, linewidth=linewidth, normalize=True)
    ax.quiver(*origin, *z_dir, color='b', length=scale, linewidth=linewidth, normalize=True)

# Plot each point with orientation
for pos, quat in zip(positions, quaternions):
    rotation = R.from_quat([quat[0], quat[1], quat[2], quat[3]])  # Convert quaternion to rotation
    plot_reference_frame(ax, pos, rotation)

# Connect points with a line
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color='black', linewidth=1)

# Set plot labels and title with LaTeX formatting
ax.set_xlabel(r'x$_{\text{des}}$', fontsize = 20)
ax.set_ylabel(r'y$_{\text{des}}$', fontsize = 20)
ax.set_zlabel(r'z$_{\text{des}}$', fontsize = 20)
ax.set_title(r'$\textbf{3D\ Trajectory\ with\ Oriented\ Reference\ Frames}$')

# Ensure equal scaling for all axes while keeping proportions
max_range = np.ptp(positions, axis=0).max() / 2.0  # Half the max range
mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Center the plot and set equal aspect ratio for real-world scaling
ax.set_box_aspect([np.ptp(positions[:, 0]), np.ptp(positions[:, 1]), np.ptp(positions[:, 2])])

# Reduce the number of ticks on the Y-axis
y_ticks = np.linspace(mid_y - max_range, mid_y + max_range, 5)  # Set 5 ticks
ax.set_yticks(y_ticks)

# Show the plot
plt.show()
