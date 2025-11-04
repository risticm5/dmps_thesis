import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations
import statsmodels.stats.multitest as multitest
import matplotlib as mpl

# -----------------------------
# LaTeX settings for formatting
# -----------------------------
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# -----------------------------
# File paths for participants
# -----------------------------
file_paths = [
    "test1_christian/answers_test1.csv",
    "test3_xenia/answers_test3.csv",
    "test4_mihailo/answers_test4.csv",
    "test5_marko/answers_test5.csv",
    "test6_eleonora/answers_test6.csv",
    "test7_bozidar/answers_test7.csv",
    "test1_new_alessandro/answers_test1_new.csv",
    "test2_new_daniele/answers_test1.csv",
    "test3_new_eleonora/answers_test1.csv"
]

# -----------------------------
# Load all data
# -----------------------------
all_q1, all_q2, all_q3 = [], [], []
script_dir = os.path.dirname(os.path.realpath(__file__))

for path in file_paths:
    file_path = os.path.join(script_dir, path)
    print(f"The script directory is: {file_path}")
    with open(file_path, 'r') as file:
        lines = file.readlines()

    q1_scores = list(map(float, lines[0].strip().split(',')))
    q2_scores = list(map(float, lines[1].strip().split(',')))
    q3_scores = list(map(float, lines[2].strip().split(',')))

    all_q1.append(q1_scores)
    all_q2.append(q2_scores)
    all_q3.append(q3_scores)

q1_matrix = np.array(all_q1)
q2_matrix = np.array(all_q2)
q3_matrix = np.array(all_q3)

# -----------------------------
# Friedman + post-hoc tests
# -----------------------------
friedman_stat, p_value = friedmanchisquare(
    q1_matrix.mean(axis=0), q2_matrix.mean(axis=0), q3_matrix.mean(axis=0)
)
print(f"Friedman Test Statistic: {friedman_stat:.3f}, p-value: {p_value:.3f}")

if p_value < 0.05:
    print("Significant differences detected! Performing post-hoc pairwise Wilcoxon tests...")
    pairs = [("Q1", q1_matrix.flatten()), ("Q2", q2_matrix.flatten()), ("Q3", q3_matrix.flatten())]
    comparisons = list(combinations(pairs, 2))
    p_values = []
    comparisons_labels = []
    for (name1, data1), (name2, data2) in comparisons:
        stat, p = wilcoxon(data1, data2)
        p_values.append(p)
        comparisons_labels.append(f"{name1} vs {name2}")
        print(f"{name1} vs {name2}: Wilcoxon test statistic = {stat:.3f}, p-value = {p:.5f}")

    corrected_p = multitest.multipletests(p_values, method="bonferroni")[1]
    print("\nPost-hoc Test Results (Bonferroni Corrected):")
    for i in range(len(comparisons_labels)):
        print(f"{comparisons_labels[i]}: Adjusted p-value = {corrected_p[i]:.5f}")
else:
    print("No significant differences found. No post-hoc tests required.")

# -----------------------------
# Plot: mean trends + dispersion
# -----------------------------
xlabel_iterations = r"Iteration"
ylabel_score = r"$\bar{q}_{i,k}=(\sum_{j=1}^N q_{i,k}^j)/N$"

plt.figure(figsize=(10, 5))
iterations = np.arange(1, q1_matrix.shape[1] + 1)
n = q1_matrix.shape[0]

# Means
m1 = q1_matrix.mean(axis=0)
m2 = q2_matrix.mean(axis=0)
m3 = q3_matrix.mean(axis=0)

# Dispersion choices
sd1, sd2, sd3 = q1_matrix.std(axis=0, ddof=1), q2_matrix.std(axis=0, ddof=1), q3_matrix.std(axis=0, ddof=1)
sem1, sem2, sem3 = sd1 / np.sqrt(n), sd2 / np.sqrt(n), sd3 / np.sqrt(n)
ci1, ci2, ci3 = 1.96 * sem1, 1.96 * sem2, 1.96 * sem3  # 95% CI

# ---- Toggle which band to use: 'ci', 'sem', or 'std'
BAND = 'sem'
if BAND == 'ci':
    b1, b2, b3 = ci1, ci2, ci3
elif BAND == 'sem':
    b1, b2, b3 = sem1, sem2, sem3
elif BAND == 'std':
    b1, b2, b3 = sd1, sd2, sd3
else:
    raise ValueError("BAND must be one of: 'ci', 'sem', 'std'.")

# Intervention window (behind everything)
plt.axvspan(6, 10, color='lightblue', alpha=0.35, zorder=0)

# Plot means (in front)
c1, c2, c3 = '#EA3E70', '#954567', '#0180B5'
plt.plot(iterations, m1, marker='o', linestyle='-', label=r"\textbf{Q$_1$: Duration}",  color=c1, zorder=3)
plt.plot(iterations, m2, marker='s', linestyle='-', label=r"\textbf{Q$_2$: Assistance}", color=c2, zorder=3)
plt.plot(iterations, m3, marker='^', linestyle='-', label=r"\textbf{Q$_3$: Comfort}",   color=c3, zorder=3)

# Dispersion bands (kept out of the legend)
plt.fill_between(iterations, m1 - b1, m1 + b1, color=c1, alpha=0.18, label='_nolegend_', zorder=2)
plt.fill_between(iterations, m2 - b2, m2 + b2, color=c2, alpha=0.18, label='_nolegend_', zorder=2)
plt.fill_between(iterations, m3 - b3, m3 + b3, color=c3, alpha=0.18, label='_nolegend_', zorder=2)

# Axes, labels, grid
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlabel(xlabel_iterations, fontsize=20)
plt.ylabel(ylabel_score, fontsize=20)
plt.legend(fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()


