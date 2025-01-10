import matplotlib.pyplot as plt
import numpy as np
import os

# Data
labels = ['Representation 1', 'Representation 2', 'Representation 3', 
          'Representation 4', 'Representation 5', 'Representation 6', 
          'Human subjects', 'Mixed-strategy']
route_labels = ['(O-L-D)', '(O-R-D)', '(O-L-D)', '(O-R-D)', '(O-L-R-D)']
mean_data = [
    [9.03, 8.98, 6.39, 6.88, 4.74],  # Representation 1
    [8.90, 9.10, 3.47, 3.65, 10.88],  # Representation 2
    [8.79, 9.21, 0.33, 0.58, 17.53],  # Representation 3
    [7.86, 10.14, 0.11, 0.18, 17.72],  # Representation 4
    [8.97, 9.03, 4.50, 5.23, 4.50],  # Representation 5
    [9.34, 8.66, 1.07, 1.03, 15.90],  # Representation 6
    [9.02, 8.98, 1.72, 1.47, 14.82],  # Human subjects
    [9.00, 9.00, 0.00, 0.00, 18.00]   # Mixed-strategy
]
std_dev_data = [
    [7.25, 7.25, 3.69, 1.82, 4.06],  # Representation 1
    [2.47, 2.47, 1.30, 1.17, 1.42],  # Representation 2
    [4.90, 4.90, 0.35, 0.38, 0.22],  # Representation 3
    [1.83, 1.83, 0.43, 0.93, 1.22],  # Representation 4
    [0.64, 0.64, 2.60, 3.20, 2.60],  # Representation 5
    [1.53, 1.53, 1.33, 1.32, 2.37],  # Representation 6
    [2.11, 2.11, 1.64, 1.40, 2.54],  # Human subjects
    [2.12, 2.12, 0.00, 0.00, 0.00]   # Mixed-strategy
]

# Splitting the data
routes_split = [2, 3]  # Number of routes in each split
split_labels = [route_labels[:routes_split[0]], route_labels[routes_split[0]:]]
split_means = [np.array(mean_data)[:, :routes_split[0]], np.array(mean_data)[:, routes_split[0]:]]
split_stds = [np.array(std_dev_data)[:, :routes_split[0]], np.array(std_dev_data)[:, routes_split[0]:]]

# Define hatches for odd representations
odd_hatch = "//"

# Output folder
output_folder = '.'
os.makedirs(output_folder, exist_ok=True)

plt.style.use("seaborn-v0_8")

# Create grouped plots with hatching for odd representations
fig1, ax1 = plt.subplots(figsize=(12, 6))

x1 = np.arange(len(labels))  # One x value per representation
bar_width = 0.15

for i, route_label in enumerate(split_labels[0]):  # First two routes
    means = split_means[0][:, i]
    stds = split_stds[0][:, i]
    for j in range(len(labels)):
        hatch = odd_hatch if j % 2 == 0 else None
        ax1.bar(x1[j] + i * bar_width, means[j], bar_width, yerr=stds[j], capsize=5,
                label=route_label if j == 0 else "", color=f"C{i}", hatch=hatch)

ax1.set_xlabel('Representations', fontsize=14)
ax1.set_ylabel('Mean Number of Agents', fontsize=14)
ax1.set_title('Mean Number of Agents Choosing Routes (Game A)', fontsize=16)
ax1.set_xticks(x1 + bar_width * (len(split_labels[0]) - 1) / 2)
ax1.set_xticklabels(labels, rotation=45, ha="right")
ax1.legend(title="Routes", loc='upper right')
plt.tight_layout()

# Save the first plot
fig1.savefig(os.path.join(output_folder, "mean_routes_game_A.png"), dpi=300)

fig2, ax2 = plt.subplots(figsize=(12, 6))

x2 = np.arange(len(labels))  # One x value per representation

for i, route_label in enumerate(split_labels[1]):  # Last three routes
    means = split_means[1][:, i]
    stds = split_stds[1][:, i]
    for j in range(len(labels)):
        hatch = odd_hatch if j % 2 == 0 else None
        ax2.bar(x2[j] + i * bar_width, means[j], bar_width, yerr=stds[j], capsize=5,
                label=route_label if j == 0 else "", color=f"C{i+2}", hatch=hatch)

ax2.set_xlabel('Representations', fontsize=14)
ax2.set_ylabel('Mean Number of Agents', fontsize=14)
ax2.set_title('Mean Number of Agents Choosing Routes (Game B)', fontsize=16)
ax2.set_xticks(x2 + bar_width * (len(split_labels[1]) - 1) / 2)
ax2.set_xticklabels(labels, rotation=45, ha="right")
ax2.legend(title="Routes", loc='upper right')
plt.tight_layout()

# Save the second plot
fig2.savefig(os.path.join(output_folder, "mean_routes_game_B.png"), dpi=300)

plt.close()
