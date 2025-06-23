import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations
import statsmodels.stats.multitest as multitest
import matplotlib as mpl

# LaTeX settings for formatting
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# File paths for multiple participants
file_paths = [
    "test1_christian/answers_test1.csv",
    "test3_xenia/answers_test3.csv",
    "test4_mihailo/answers_test4.csv",
    "test5_marko/answers_test5.csv",
    "test6_eleonora/answers_test6.csv",
    "test7_bozidar/answers_test7.csv"
]

# Store results for all participants
all_q1, all_q2, all_q3 = [], [], []
script_dir = os.path.dirname(os.path.realpath(__file__))

# Process each file
for path in file_paths:
    
    file_path = os.path.join(script_dir, path)
    print(f"The script directory is: {file_path}")
    # Read first 3 rows (Q1, Q2, Q3 scores)
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    q1_scores = list(map(float, lines[0].strip().split(',')))
    q2_scores = list(map(float, lines[1].strip().split(',')))
    q3_scores = list(map(float, lines[2].strip().split(',')))

    all_q1.append(q1_scores)
    all_q2.append(q2_scores)
    all_q3.append(q3_scores)

# Convert lists to NumPy arrays
q1_matrix = np.array(all_q1)
q2_matrix = np.array(all_q2)
q3_matrix = np.array(all_q3)

# Perform Friedman test
friedman_stat, p_value = friedmanchisquare(q1_matrix.mean(axis=0), q2_matrix.mean(axis=0), q3_matrix.mean(axis=0))

# Print test result
print(f"Friedman Test Statistic: {friedman_stat:.3f}, p-value: {p_value:.3f}")

# Post-hoc analysis (Wilcoxon signed-rank tests)
if p_value < 0.05:
    print("Significant differences detected! Performing post-hoc pairwise Wilcoxon tests...")
    
    # Perform pairwise Wilcoxon tests for Q1 vs Q2, Q1 vs Q3, Q2 vs Q3
    pairs = [("Q1", q1_matrix.flatten()), ("Q2", q2_matrix.flatten()), ("Q3", q3_matrix.flatten())]
    comparisons = list(combinations(pairs, 2))  # Generate all possible pairs
    p_values = []
    comparisons_labels = []
    
    for (name1, data1), (name2, data2) in comparisons:
        stat, p = wilcoxon(data1, data2)
        p_values.append(p)
        comparisons_labels.append(f"{name1} vs {name2}")
        print(f"{name1} vs {name2}: Wilcoxon test statistic = {stat:.3f}, p-value = {p:.5f}")
    
    # Apply Bonferroni correction for multiple comparisons
    corrected_p = multitest.multipletests(p_values, method="bonferroni")[1]

    # Print adjusted results
    print("\nPost-hoc Test Results (Bonferroni Corrected):")
    for i in range(len(comparisons_labels)):
        print(f"{comparisons_labels[i]}: Adjusted p-value = {corrected_p[i]:.5f}")
    
    # Mark significant comparisons
    sig_comparisons = [comparisons_labels[i] for i, p in enumerate(corrected_p) if p < 0.05]
    
else:
    print("No significant differences found. No post-hoc tests required.")

# Define LaTeX-formatted axis labels
xlabel_iterations = r"Iteration"
ylabel_score = r"$\bar{q}_{i,k}=(\sum_{j=1}^N q_{i,k}^j)/N$"
#title_evolution = r"\textbf{Evolution of Q$_1$, Q$_2$, and Q$_3$ over Iterations}"

# Plot the evolution of Q1, Q2, and Q3 with LaTeX labels
plt.figure(figsize=(10, 5))
iterations = np.arange(1, q1_matrix.shape[1] + 1)

plt.plot(iterations, np.mean(q1_matrix, axis=0), marker='o', linestyle='-', label=r"\textbf{Q$_1$: Duration}", color='#EA3E70')  # Coral Pink
plt.plot(iterations, np.mean(q2_matrix, axis=0), marker='s', linestyle='-', label=r"\textbf{Q$_2$: Assistance}", color='#954567')  # Teal
plt.plot(iterations, np.mean(q3_matrix, axis=0), marker='^', linestyle='-', label=r"\textbf{Q$_3$: Comfort}", color='#0180B5')  # Lavender Gray

# Fill the area between x = 6 and x = 10
ymin, ymax = plt.ylim()

# Fill the area between x=6 and x=10 (on the x-axis)
plt.fill_between(iterations, ymin - 10, ymax + 10, where=((iterations >= 6) & (iterations <= 10)), 
                 color='lightblue', alpha=0.4)

# Draw vertical dashed lines at x = 6 and x = 10
plt.axvline(x=6, color='black', linestyle='--', linewidth=1.5)
plt.axvline(x=10, color='black', linestyle='--', linewidth=1.5)
# Increase the size of the numbers on both the x and y axes
plt.tick_params(axis='both', which='major', labelsize=15)  # Increase size for major ticks


plt.xlabel(xlabel_iterations, fontsize = 20)
plt.ylabel(ylabel_score, fontsize = 20)
plt.ylim(ymin, ymax)
#plt.title(title_evolution, fontsize=16)
plt.legend(fontsize=15)
plt.grid(True)
plt.show()

# Define LaTeX-formatted title and labels for boxplot
#title_boxplot = r"\textbf{Distribution of Responses Across All Participants}"
ylabel_boxplot = r"\textbf{Score}"

# Create boxplots to show score distributions
plt.figure(figsize=(8, 6))
plt.boxplot([q1_matrix.flatten(), q2_matrix.flatten(), q3_matrix.flatten()], labels=[r"\textbf{Q$_1$}", r"\textbf{Q$_2$}", r"\textbf{Q$_3$}"])
#plt.title(title_boxplot, fontsize=16) # Do not print the title
plt.ylabel(ylabel_boxplot, fontsize=14)
plt.grid(True)

'''
# Add significance markers if applicable
if p_value < 0.05:
    y_max = max(q1_matrix.max(), q2_matrix.max(), q3_matrix.max())
    for i, label in enumerate(sig_comparisons):
        plt.text(1.5, y_max - i * 0.5, rf"\textbf{Significant: {label}}", fontsize=12, color="red")

plt.show()'
'''
