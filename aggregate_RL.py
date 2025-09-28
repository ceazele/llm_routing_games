import os
import glob
import pandas as pd
import numpy as np

# Define expected routes
EXPECTED_ROUTES = {
    "A": ["O-L-D", "O-R-D"],
    "B": ["O-L-D", "O-R-D", "O-L-R-D"]
}

# Function to compute mean and standard error
def compute_mean_se(values):
    """
    Compute mean and standard error of a list of values.
    """
    if not values:
        return np.nan, np.nan
    
    mean_val = np.mean(values)
    se_val = np.std(values, ddof=1) / np.sqrt(len(values))
    return mean_val, se_val

# Function to process a single algorithm's data
def process_algorithm(folder, game_variant):
    """
    Reads all CSVs in a given folder, extracts route counts by round,
    and computes mean and SE across runs.
    """
    csv_paths = sorted(glob.glob(os.path.join(folder, "results_run_*.csv")))
    route_data = {route: [] for route in EXPECTED_ROUTES[game_variant]}
    
    for path in csv_paths:
        df = pd.read_csv(path)
        
        if "Route" not in df.columns or "Round" not in df.columns:
            print(f"Skipping {path}: Missing required columns.")
            continue
        
        # Count number of agents on each route per round
        route_counts = df.groupby(["Round", "Route"])["Agent"].count().unstack(fill_value=0)

        for route in EXPECTED_ROUTES[game_variant]:
            if route in route_counts.columns:
                route_data[route].append(route_counts[route].mean())  # Aggregate over rounds

    # Compute mean and standard error across trials
    results = {}
    for route, values in route_data.items():
        mean_val, se_val = compute_mean_se(values)
        results[route] = {"Mean": mean_val, "SE": se_val}

    return results

# Function to compute and print statistics
def compute_route_statistics(folder_list):
    """
    Computes route statistics (mean, SE) for EXP3 and MWU folders.
    """
    algorithms = {"EXP3": [], "MWU": []}

    for folder in folder_list:
        game_variant = "A" if folder.endswith("A") else "B"
        alg_name = "EXP3" if "exp3" in folder.lower() else "MWU"
        algorithms[alg_name].append((folder, game_variant))
    
    for game_variant in ["A", "B"]:
        print(f"\nGame {game_variant} Results:\n")

        for alg, folders in algorithms.items():
            relevant_folders = [f for f, g in folders if g == game_variant]
            if not relevant_folders:
                continue

            print(f"{alg}:")
            aggregated_results = {route: {"Mean": [], "SE": []} for route in EXPECTED_ROUTES[game_variant]}

            for folder in relevant_folders:
                results = process_algorithm(folder, game_variant)
                for route in EXPECTED_ROUTES[game_variant]:
                    if route in results:
                        aggregated_results[route]["Mean"].append(results[route]["Mean"])
                        aggregated_results[route]["SE"].append(results[route]["SE"])

            # Print final mean and SE across all folders
            for route in EXPECTED_ROUTES[game_variant]:
                mean_val = np.nanmean(aggregated_results[route]["Mean"])
                se_val = np.nanmean(aggregated_results[route]["SE"])
                print(f"  {route}: Mean = {mean_val:.2f}, SE = {se_val:.2f}")

# Example usage
folder_list = ["exp3A", "mwA", "exp3B", "mwB"]
compute_route_statistics(folder_list)
