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
