#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Determine the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(script_dir, "distance.csv")

# Extract the first row
data = np.loadtxt(csv_file, delimiter=",")
not_filtered_distance = data[0, :]
filtered_distance = data[1, :]

# Generate x-axis values
x_values = np.arange(len(not_filtered_distance ))
x_values1 = np.arange(len(filtered_distance ))
print(f"The length of the first row is: {len(not_filtered_distance)}")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(x_values, not_filtered_distance, x_values1, filtered_distance, marker="o", linestyle="-", label="First Row Data")

# Labels and title
plt.xlabel("Index")
plt.ylabel("Values")
plt.title("Plot of First Row Data")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()


# Extract the third and fourth row
third_row = data[2, :]
fourth_row = data[3, :]

# Generate x-axis values for the new rows
x_values2 = np.arange(len(third_row))
x_values3 = np.arange(len(fourth_row))

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot third row
ax1.plot(x_values2, third_row, marker="o", linestyle="-", label="Third Row Data")
ax1.set_xlabel("Index")
ax1.set_ylabel("Ct")
ax1.set_title("Plot of Third Row Data")
ax1.legend()
ax1.grid(True)

# Plot fourth row
ax2.plot(x_values3, fourth_row, marker="o", linestyle="-", label="Fourth Row Data")
ax2.set_xlabel("Index")
ax2.set_ylabel("Cs")
ax2.set_title("Plot of Fourth Row Data")
ax2.legend()
ax2.grid(True)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

# Plot the fifth row
fifth_row = data[4, :]
x_values5 = np.arange(len(fifth_row))
# Plot
plt.figure(figsize=(10, 5))
plt.plot(x_values5, fifth_row, marker="o", linestyle="-", label="Data")

# Labels and title
plt.xlabel("Index")
plt.ylabel("Filtered velocity")
plt.title("PVelocity")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()