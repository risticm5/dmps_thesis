import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# LaTeX settings for formatting
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Same heights and custom labels as before
heights = [182, 168, 192, 192, 155, 195]
custom_labels = [
    ("1", r"$\text{expert}$"),
    ("2", r"$\text{non-expert}$"),
    ("3", r"$\text{non-expert}$"),
    ("4", r"$\text{expert}$"),
    ("5", r"$\text{expert}$"),
    ("6", r"$\text{non-expert}$")
]

# Data from the previous steps (means_list, means_excl_first_5_list, q4_values)
# Use means_list and q4_values directly as the means for the histograms
means_list = [
    [3.57, 3.29, 3.57], [4.00, 3.79, 3.86], [3.86, 3.86, 4.00],
    [3.29, 4.14, 4.07], [4.07, 4.07, 3.86], [2.57, 2.43, 2.71]
]

q4_values = [8.00, 10.00, 8.00, 9.00, 8.00, 7.00]

# Prepare the histogram figure
fig, ax1 = plt.subplots(figsize=(10, 5))

# Define bar width and positions
bar_width = 0.18
index = np.arange(len(means_list))

# Colors for each question (fixed colors as per your request)
q1_color = '#EA3E70'  # Q1 color (coral pink)
q2_color = '#954567'  # Q2 color (darker pink)
q3_color = '#0180B5'  # Q3 color (teal blue)
q4_color = '#F37252'  # Q4 color (orange-red)

# Create the bars for each question (Q1, Q2, Q3)
for i, label in enumerate(custom_labels):
    means = means_list[i]

    # Adjust bar positions more clearly to avoid overlap
    ax1.bar(index[i] - 1.5 * bar_width, means[0], bar_width, color=q1_color, label=r"$\mathbf{Q_1}$" if i == 0 else "")
    ax1.bar(index[i] - 0.5 * bar_width, means[1], bar_width, color=q2_color, label=r"$\mathbf{Q_2}$" if i == 0 else "")
    ax1.bar(index[i] + 0.5 * bar_width, means[2], bar_width, color=q3_color, label=r"$\mathbf{Q_3}$" if i == 0 else "")

# Create another axis for Q4
ax2 = ax1.twinx()  # This creates a second y-axis sharing the same x-axis

# Create the bars for Q4 with the fixed color, adjusting position
for i, label in enumerate(custom_labels):
    ax2.bar(index[i] + 1.5 * bar_width, q4_values[i], bar_width, color=q4_color, label=r"$\mathbf{Q_4}$" if i == 0 else "")

# Set labels for both y-axes
ax1.set_xlabel(r'Participant', fontsize=20)
ax1.set_ylabel('$\mathbf{Duration}$, $\mathbf{Assistance}$, $\mathbf{Comfort}$', fontsize=16)  # Left axis label for Q1, Q2, Q3
ax2.set_ylabel('$\mathbf{Overall experience}$', fontsize=16)  # Right axis label for Q4

# Title
# ax1.set_title('Scores for Questions 1 to 4 by Participant', fontsize=16)

# Add participant labels
ax1.set_xticks(index)
ax1.set_xticklabels([f'P{i+1}' for i in range(len(means_list))], fontsize=16)

# Add legend
ax1.legend(loc='upper left', fontsize = 15)
ax2.legend(loc='upper right', fontsize = 15)

# Set y-limits for both axes
ax1.set_ylim(2.0, 5.5)  # Y-limits for Q1, Q2, Q3
ax2.set_ylim(0, 10)  # Y-limits for Q4

# Show grid
ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

# Remove the black border around the plot
fig.patch.set_facecolor('white')  # Set the figure background color to white
ax1.set_facecolor('white')  # Set the axis background color to white

ax1.tick_params(axis='both', which='major', labelsize=14)  # Increase size for major ticks on the left axis
ax2.tick_params(axis='both', which='major', labelsize=14)  # Increase size for major ticks on the right axis

# Show the plot
plt.tight_layout()
plt.show()
