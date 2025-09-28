import os
import glob
import pandas as pd
import numpy as np

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

# Define expected routes for each game
ROUTES_A = ["O-L-D", "O-R-D"]
ROUTES_B = ["O-L-D", "O-R-D", "O-L-R-D"]

def compute_route_statistics(game_folders, base_path):
    """
    Computes the mean and standard deviation of the number of agents choosing each route 
    per round within each specified game_x folder (representation), separately for Game A and Game B.

    Args:
        game_folders (list): List of specific game_x folder names to process (e.g., ["game_1", "game_2"]).
        base_path (str): Path to the directory containing all game_x subfolders.

    Returns:
        pd.DataFrame: Aggregated table with mean and standard deviation for each route per representation.
    """
    all_results = []

    # Loop through only the specified game_x folders
    for game_folder in game_folders:
        game_path = os.path.join(base_path, game_folder)

        if not os.path.isdir(game_path):  # Ensure it's a valid directory
            print(f"Skipping invalid folder: {game_path}")
            continue

        game_label = GAME_LABELS.get(game_folder, game_folder)  # Use folder name as label

        # Loop through all run_y folders inside game_x
        run_folders = sorted(glob.glob(os.path.join(game_path, "run *")))

        # Initialize data storage for each representation
        for game_variant in ["A", "B"]:
            all_round_counts = []

            for run_folder in run_folders:
                game_variant_path = os.path.join(run_folder, f"{game_folder}{game_variant}")

                if not os.path.isdir(game_variant_path):  # Ensure it's a directory
                    continue

                # Construct CSV file path
                csv_file = os.path.join(game_variant_path, f"{game_folder}{game_variant}.csv")

                if not os.path.exists(csv_file):
                    print(f"Skipping missing file: {csv_file}")
                    continue

                # Read CSV
                df = pd.read_csv(csv_file)

                # Group by Round and count occurrences of each route
                round_counts = df.groupby("Round")["Route"].value_counts().unstack(fill_value=0)
                all_round_counts.append(round_counts)

            if all_round_counts:
                # Concatenate all rounds from different runs
                full_round_counts = pd.concat(all_round_counts)

                # Determine routes based on game type
                routes = ROUTES_A if game_variant == "A" else ROUTES_B

                # Compute mean and std per route for this representation
                for route in routes:
                    mean_agents = full_round_counts[route].mean()
                    std_dev = full_round_counts[route].std()

                    all_results.append({
                        "Game": f"Game {game_variant.upper()}",
                        "Representation": game_label,  # Use game label for clarity
                        "Route": route,
                        "Mean Agents": mean_agents,
                        "Std Dev": std_dev
                    })

    # Convert to DataFrame for easy readability
    summary_df = pd.DataFrame(all_results)

    # Pivot table for better table format
    summary_pivot = summary_df.pivot(index=["Representation", "Game"], columns="Route", values=["Mean Agents", "Std Dev"])

    return summary_pivot


# Define game folders and base path
game_folders = ["game_1", "game_2", "game_3", "game_4", "game_5", "game_6", "game_11", "game_12"]
base_path = "."

# Compute statistics and display results
result_df = compute_route_statistics(game_folders, base_path)

# Print table output
print(result_df)

# Save to CSV for easier analysis
result_df.to_csv("route_statistics.csv")
print("Results saved to route_statistics.csv")
