import numpy as np
import matplotlib.pyplot as plt

# Data categories
categories = ["Mean Routes", "Mean Payoff", "Mean Regret", "Mean Switches"]

# Data for Game A
mwu_values_A = [7.24, 94.52, 17.99, 19.25]
exp3_values_A = [7.28, 94.60, 17.60, 19.58]
sro_values_A = [7.48, 93.62, 17.77, 2.51]
mwu_errors_A = [0.03, 0.15, 0.40, 0.11]
exp3_errors_A = [0.03, 0.16, 0.41, 0.11]
sro_errors_A = [0.41, 1.54, 4.78, 0.29]

# Data for Game B
mwu_values_B = [13.87, 64.95, 14.02, 12.83]
exp3_values_B = [13.81, 65.18, 14.35, 13.15]
sro_values_B = [17.71, 41.72, 0.97, 0.58]
mwu_errors_B = [0.02, 0.16, 0.13, 0.09]
exp3_errors_B = [0.03, 0.15, 0.15, 0.09]
sro_errors_B = [0.03, 0.15, 0.18, 0.07]

# Define colors (Soft blue, pastel teal, muted green)
colors = ['#91B2EF', '#3A606E', '#AAAE8E']

# Set text size to 20 throughout
plt.rcParams.update({'font.size': 20})

# Metrics for plotting
metrics = ["Mean Routes", "Mean Payoff", "Mean Regret", "Mean Switches"]

# Save directory
output_folder = "plots"
import os
os.makedirs(output_folder, exist_ok=True)

# Loop through each metric and create individual plots
for i, metric in enumerate(metrics):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Extract values and errors for the current metric
    mwu_A, exp3_A, sro_A = mwu_values_A[i], exp3_values_A[i], sro_values_A[i]
    mwu_B, exp3_B, sro_B = mwu_values_B[i], exp3_values_B[i], sro_values_B[i]
    
    mwu_err_A, exp3_err_A, sro_err_A = mwu_errors_A[i], exp3_errors_A[i], sro_errors_A[i]
    mwu_err_B, exp3_err_B, sro_err_B = mwu_errors_B[i], exp3_errors_B[i], sro_errors_B[i]

    # X-axis positions
    x = np.array([0, 1])  # 0 for Game A, 1 for Game B
    bar_width = 0.25

    # Plot bars
    ax.bar(x - bar_width, [mwu_A, mwu_B], bar_width, yerr=[mwu_err_A, mwu_err_B], label="MWU", capsize=5, color=colors[0], alpha=0.8)
    ax.bar(x, [exp3_A, exp3_B], bar_width, yerr=[exp3_err_A, exp3_err_B], label="EXP3", capsize=5, color=colors[1], alpha=0.8)
    ax.bar(x + bar_width, [sro_A, sro_B], bar_width, yerr=[sro_err_A, sro_err_B], label="S-RO", capsize=5, color=colors[2], alpha=0.8)

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(["Game A", "Game B"])
    ax.set_ylabel(metric)
    ax.legend()

    # Save figure
    filename = f"{output_folder}/{metric.replace(' ', '_').lower()}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

print(f"Plots saved in: {output_folder}/")
