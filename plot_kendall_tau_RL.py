import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau

# Define expected route frequencies for Games A and B
EXPECTED_FREQUENCIES = {
    "A": {"O-L-D": 9, "O-R-D": 9},
    "B": {"O-L-D": 0, "O-R-D": 0, "O-L-R-D": 18}
}

# Map folder names to algorithm labels
ALGORITHM_LABELS = {
    "exp3A": "EXP3", "exp3B": "EXP3",
    "mwA": "MWU", "mwB": "MWU"
}

def compute_kendalls_tau_per_folder(folders):
    """
    Computes Kendall's Tau correlation between round number and deviation from expected route choices.
    This is done separately for Game A and Game B.

    Args:
        folders (list): List of experiment folders (e.g., ["exp3A", "mwA", "exp3B", "mwB"]).

    Returns:
        dict: { "A": {algorithm_name: list_of_tau_values}, "B": {algorithm_name: list_of_tau_values} }
    """
    game_tau_values = {"A": {}, "B": {}}  # Separate storage for Game A and B

    for folder in folders:
        # Determine if it's Game A or Game B from folder name
        game_type = "A" if folder.endswith("A") else "B"
        algorithm_name = ALGORITHM_LABELS.get(folder, folder)  # Default to folder name if not found

        # Collect Kendall's Tau values for this folder
        tau_values = []

        # Find all CSV files inside this folder
        csv_files = sorted(glob.glob(os.path.join(folder, "results_run_*.csv")))

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)

            # Ensure we have expected frequencies for this game
            if game_type not in EXPECTED_FREQUENCIES:
                print(f"Skipping {csv_file}: Missing expected route frequencies for Game {game_type}.")
                continue

            expected_counts = EXPECTED_FREQUENCIES[game_type]
            round_numbers = []
            deviation_scores = []

            # Compute deviation scores round by round
            for round_num in sorted(df["Round"].unique()):
                round_numbers.append(round_num)

                # Count observed route choices
                observed_counts = df[df["Round"] == round_num]["Route"].value_counts().to_dict()

                # Compute deviation from expected frequencies
                deviation_score = sum(
                    abs(observed_counts.get(route, 0) - expected_counts.get(route, 0))
                    for route in expected_counts
                )
                deviation_scores.append(deviation_score)

            # Ensure there's variation in deviation scores before computing Tau
            if len(set(deviation_scores)) > 1 and len(round_numbers) > 1:
                tau, _ = kendalltau(round_numbers, deviation_scores)
                if not np.isnan(tau):
                    tau_values.append(tau)

        # Store Kendall's Tau values if valid
        if tau_values:
            if algorithm_name not in game_tau_values[game_type]:
                game_tau_values[game_type][algorithm_name] = []
            game_tau_values[game_type][algorithm_name].extend(tau_values)

    return game_tau_values


def plot_kendall_tau_barplots(game_tau_values, output_folder="output_plots"):
    """
    Plots two separate bar charts for Kendall's Tau correlations for Game A and Game B,
    ensuring both plots have the same y-axis range and using algorithm names as x-axis labels.

    Args:
        game_tau_values (dict): {"A": {algorithm: list_of_tau_values}, "B": {algorithm: list_of_tau_values}}.
        output_folder (str): Path to save the plots.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Compute y-axis limits across both Game A and B
    all_values = []
    for game_type in ["A", "B"]:
        for tau_list in game_tau_values[game_type].values():
            all_values.extend(tau_list)

    y_min, y_max = min(all_values), max(all_values)
    y_buffer = (y_max - y_min) * 0.1  # Add 10% buffer to avoid clipping
    y_lim = (y_min - y_buffer, y_max + y_buffer)

    for game_type in ["A", "B"]:
        if not game_tau_values[game_type]:
            print(f"No valid data for Game {game_type}, skipping plot.")
            continue

        # Extract means and standard deviations
        algorithms = list(game_tau_values[game_type].keys())
        means = [np.mean(game_tau_values[game_type][alg]) for alg in algorithms]
        std_devs = [np.std(game_tau_values[game_type][alg]) for alg in algorithms]

        # Create bar plot
        plt.figure(figsize=(7, 5))
        plt.rcParams.update({'font.size': 25})
        plt.bar(algorithms, means, yerr=std_devs, capsize=5, color='cornflowerblue', alpha=0.7, edgecolor='black')

        # Set y-axis limit to be the same for both plots
        plt.ylim(y_lim)

        # Add labels and title
        plt.axhline(y=0, color='black', linewidth=1)  # Ensure zero-line is visible
        plt.ylabel("Mean $\\tau$")
        plt.xlabel("Algorithm")
        # plt.title(f"Kendall’s Tau Correlations for Game {game_type}")

        # Adjust formatting
        plt.xticks(rotation=0, ha="center")  # Centered, since labels are now short
        plt.grid(axis='y', linestyle="--", alpha=0.5)

        # Save the plot
        output_path = os.path.join(output_folder, f"kendall_tau_game{game_type}_barplot.png")
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"Saved plot to {output_path}")

        # Show plot
        plt.show()


# Example Usage
folders = ["exp3A", "mwA", "exp3B", "mwB"]  # Adjust this list to match your actual folder structure

# Compute Kendall’s Tau for each game separately
game_tau_values = compute_kendalls_tau_per_folder(folders)

# Generate and save separate plots for Game A and Game B with synchronized y-axis
plot_kendall_tau_barplots(game_tau_values, output_folder=".")
