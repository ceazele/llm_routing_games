import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm, to_hex

# Define game labels
GAME_LABELS = {
    "game_1": "F-PE",
    "game_2": "S-PE",
    "game_3": "F-RO",
    "game_4": "S-RO",
    "game_5": "F-PO",
    "game_6": "S-PO",
    "game_11": "F-RE",
    "game_12": "S-RE"
}

# Define a custom color gradient
CUSTOM_CMAP = plt.get_cmap("Greys")  # Red to Blue reversed

# Define whether higher or lower values should be blue
BLUE_FOR_HIGHER = {"route": False, "payoff": False, "regret": True, "switches": True}

# Order for output printing
PRINT_ORDER = ["F-PE", "F-RE", "F-PO", "F-RO", "S-PE", "S-RE", "S-PO", "S-RO"]

def compute_statistics_with_colors(game_folders, base_path, statistic_type):
    """
    Computes the mean and standard error (SE) of the selected statistic (route choice, payoff, regret, or switch count)
    across all rounds and trials for Game A and Game B within the given representations.

    Also computes a color hex code for each statistic, normalized within Game A and Game B separately.

    Args:
        game_folders (list): List of specific game_x folder names (e.g., ["game_1", "game_2"]).
        base_path (str): Path to the directory containing all game_x subfolders.
        statistic_type (str): The statistic to compute ("route", "payoff", "regret", "switches").

    Returns:
        dict: Dictionary containing computed statistics and their corresponding hex color codes.
    """

    all_stats = {"Game A": {}, "Game B": {}}  # Store statistics separately for Game A and Game B

    for game_folder in game_folders:
        game_path = os.path.join(base_path, game_folder)

        if not os.path.isdir(game_path):  # Ensure it's a valid directory
            print(f"Skipping invalid folder: {game_path}")
            continue

        game_label = GAME_LABELS.get(game_folder, game_folder)  # Convert to correct label

        for game_variant in ["A", "B"]:
            stat_values = []  # Store statistics for this representation (across runs)

            run_folders = sorted(glob.glob(os.path.join(game_path, "run *")))

            for run_folder in run_folders:
                game_variant_path = os.path.join(run_folder, f"{game_folder}{game_variant}")
                csv_file = os.path.join(game_variant_path, f"{game_folder}{game_variant}.csv")

                if not os.path.exists(csv_file):
                    print(f"Skipping missing file: {csv_file}")
                    continue

                df = pd.read_csv(csv_file)

                # Compute statistic per agent over 40 rounds
                if statistic_type == "route":
                    if game_variant == "A":
                        # Get counts for both O-L-D and O-R-D
                        old_counts = df[df["Route"] == "O-L-D"].groupby("Round")["Agent"].count()
                        ord_counts = df[df["Route"] == "O-R-D"].groupby("Round")["Agent"].count()

                        # Compute the smaller count for each round, then average over all rounds
                        min_route_counts = np.minimum(old_counts, ord_counts)
                        avg_route_choice = min_route_counts.mean()  # Average across rounds

                    else:  # Game B
                        route_counts = df[df["Route"] == "O-L-R-D"].groupby("Round")["Agent"].count()
                        avg_route_choice = route_counts.mean()  # Average across rounds
                    
                    stat_values.append(avg_route_choice)

                elif statistic_type == "payoff":
                    avg_payoff = df.groupby("Agent")["Payoff"].mean().mean()
                    stat_values.append(avg_payoff)

                elif statistic_type == "regret":
                    avg_regret = df.groupby("Agent")["Regret"].mean().mean()
                    stat_values.append(avg_regret)

                elif statistic_type == "switches":
                    # Count switches per agent across 40 rounds
                    switches = df.groupby("Agent")["Route"].apply(lambda x: (x != x.shift()).sum() - 1)
                    avg_switches = switches.mean()
                    stat_values.append(avg_switches)

            if stat_values:
                # Compute final mean and standard error across 5 runs
                final_mean = np.mean(stat_values)
                final_std = np.std(stat_values, ddof=1)
                final_se = final_std / np.sqrt(len(stat_values))  # Standard error computation
                all_stats[f"Game {game_variant}"][game_label] = (final_mean, final_se)

    # Normalize and compute colors for each statistic
    game_a_norm = None  # Store normalization from Game A to apply its inverse to Game B for Payoff
    for game in ["Game A", "Game B"]:
        if not all_stats[game]:  # Skip if no data
            continue
        
        stat_means = np.array([v[0] for v in all_stats[game].values()])  # Extract means
        min_val, max_val = min(stat_means), max(stat_means)
        midpoint = (min_val + max_val) / 2

        # Use TwoSlopeNorm to correctly anchor mid-value
        norm = TwoSlopeNorm(vmin=min_val, vcenter=midpoint, vmax=max_val)

        if statistic_type == "payoff" and game == "Game A":
            game_a_norm = norm  # Store Game A normalization for inversion in Game B

        for rep, (mean, se) in all_stats[game].items():
            if statistic_type == "payoff":
                if game == "Game A":
                    color_value = norm(mean) if BLUE_FOR_HIGHER[statistic_type] else 1 - norm(mean)
                elif game == "Game B" and game_a_norm is not None:
                    color_value = 1 - game_a_norm(mean) if BLUE_FOR_HIGHER[statistic_type] else game_a_norm(mean)
            else:
                color_value = norm(mean) if BLUE_FOR_HIGHER[statistic_type] else 1 - norm(mean)

            color = to_hex(CUSTOM_CMAP(color_value))
            all_stats[game][rep] = (mean, se, color)

    # Print results in an easy-to-copy format with correct labels
    print("\n================== Statistics and Colors ==================\n")
    for game in ["Game A", "Game B"]:
        print(f"\n{game} ({statistic_type.upper()})\n")
        
        # Print in the specific order defined in PRINT_ORDER
        for rep in PRINT_ORDER:
            if rep in all_stats[game]:
                mean, se, color = all_stats[game][rep]
                print(f"{rep}: Mean = {mean:.2f}, SE = {se:.2f}, Color = {color}")

    return all_stats



# Define the game representations
game_folders = ["game_1", "game_2", "game_3", "game_4", "game_5", "game_6", "game_11", "game_12"]
base_path = "."

# Compute statistics for each metric
compute_statistics_with_colors(game_folders, base_path, "route")
compute_statistics_with_colors(game_folders, base_path, "payoff")
compute_statistics_with_colors(game_folders, base_path, "regret")
compute_statistics_with_colors(game_folders, base_path, "switches")
