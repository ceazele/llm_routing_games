import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau

##############################
# FIRST DATA STRUCTURE (8 labels)
##############################
# This function scans folders like "game_1", "game_2", etc.
# It uses the GAME_LABELS dictionary (8 representations)
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

# Expected route frequencies for Game A and Game B (for the first data structure)
EXPECTED_FREQUENCIES_FIRST = {
    "game_A": {"O-L-D": 9, "O-R-D": 9},
    "game_B": {"O-L-D": 0, "O-R-D": 0, "O-L-R-D": 18}
}

def compute_kendalls_tau_first(input_folder, ordering):
    """
    For each game representation (from GAME_LABELS) the function computes Kendall's τ
    between round number and the deviation from expected route counts.
    
    The CSV files are expected at:
      {input_folder}/{game_folder}/run */{game_folder}{game_variant}/{game_folder}{game_variant}.csv
      
    Returns a dictionary with keys equal to representation labels (e.g. "F-PE") and values
    a dict with keys "game_A" and "game_B" (each is a list of τ values).
    """
    # Initialize dictionary for 8 representations
    tau_dict = {label: {"game_A": [], "game_B": []} for label in ordering}
    
    # Loop over game folders defined in GAME_LABELS
    for game_folder, rep_label in GAME_LABELS.items():
        game_path = os.path.join(input_folder, game_folder)
        run_folders = sorted(glob.glob(os.path.join(game_path, 'run *')))
        
        for run_folder in run_folders:
            # For each game variant (A and B)
            for game_variant in ["A", "B"]:
                csv_path = os.path.join(run_folder, f"{game_folder}{game_variant}", f"{game_folder}{game_variant}.csv")
                if not os.path.exists(csv_path):
                    # Skip missing files
                    print(f"Skipping missing file: {csv_path}")
                    continue
                df = pd.read_csv(csv_path)
                
                # Use the expected frequencies for this game variant:
                expected_counts = EXPECTED_FREQUENCIES_FIRST.get(f"game_{game_variant}", {})
                rounds = sorted(df["Round"].unique())
                deviation_scores = []
                
                for round_num in rounds:
                    observed_counts = df[df["Round"] == round_num]["Route"].value_counts().to_dict()
                    # Sum the absolute differences from expected counts
                    deviation = sum(abs(observed_counts.get(route, 0) - expected_counts.get(route, 0))
                                    for route in expected_counts)
                    deviation_scores.append(deviation)
                
                # Only compute τ if there is variation (and at least 2 rounds)
                if len(rounds) > 1 and len(set(deviation_scores)) > 1:
                    tau, _ = kendalltau(rounds, deviation_scores)
                    if not np.isnan(tau):
                        tau_dict[rep_label][f"game_{game_variant}"].append(tau)
                    else:
                        print(f"NaN τ computed for {csv_path}")
                else:
                    print(f"Not enough variation for τ in {csv_path}")
    
    return tau_dict

##############################
# SECOND DATA STRUCTURE (2 labels)
##############################
# This second function uses folders such as "exp3A", "mwA", "exp3B", "mwB".
# We then map folder names to algorithm labels.
ALGORITHM_LABELS = {
    "exp3A": "EXP3", "exp3B": "EXP3",
    "mwA": "MWU", "mwB": "MWU"
}
# Expected frequencies for the second structure (note: keys here are "A" and "B")
EXPECTED_FREQUENCIES_SECOND = {
    "A": {"O-L-D": 9, "O-R-D": 9},
    "B": {"O-L-D": 0, "O-R-D": 0, "O-L-R-D": 18}
}

def compute_kendalls_tau_second(folders):
    """
    For each experiment folder (e.g. "exp3A", "mwA", etc.) compute Kendall's τ between
    round number and deviation from expected route counts.
    
    Returns a dictionary with keys "A" and "B" (for the two game types). Under each key,
    the value is a dictionary mapping algorithm labels (e.g. "EXP3", "MWU") to lists of τ values.
    """
    tau_dict = {"A": {}, "B": {}}
    
    for folder in folders:
        # Determine game type based on folder name ending: A or B.
        if folder.endswith("A"):
            game_type = "A"
        elif folder.endswith("B"):
            game_type = "B"
        else:
            print(f"Folder {folder} does not end with A or B, skipping.")
            continue
        
        algorithm = ALGORITHM_LABELS.get(folder, folder)
        
        # Find CSV files (assumed to be named "results_run_*.csv")
        csv_files = sorted(glob.glob(os.path.join(folder, "results_run_*.csv")))
        tau_values = []
        
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            expected = EXPECTED_FREQUENCIES_SECOND.get(game_type, {})
            rounds = sorted(df["Round"].unique())
            deviation_scores = []
            
            for round_num in rounds:
                observed = df[df["Round"] == round_num]["Route"].value_counts().to_dict()
                deviation = sum(abs(observed.get(route, 0) - expected.get(route, 0))
                                for route in expected)
                deviation_scores.append(deviation)
            
            if len(rounds) > 1 and len(set(deviation_scores)) > 1:
                tau, _ = kendalltau(rounds, deviation_scores)
                if not np.isnan(tau):
                    tau_values.append(tau)
                else:
                    print(f"NaN τ in {csv_file}")
            else:
                print(f"Not enough variation in {csv_file} to compute τ")
        
        if tau_values:
            # Store under the appropriate algorithm label
            if algorithm not in tau_dict[game_type]:
                tau_dict[game_type][algorithm] = []
            tau_dict[game_type][algorithm].extend(tau_values)
    
    return tau_dict

##############################
# MERGE THE TWO SETS OF RESULTS
##############################
def merge_tau_values(tau_first, tau_second):
    """
    Merges the dictionaries from the two computations into one with keys "game_A" and "game_B".
    The first dictionary (from compute_kendalls_tau_first) has keys like "F-PE", etc.
    The second (from compute_kendalls_tau_second) has keys "A" and "B" whose inner dict keys (e.g. "EXP3")
    are renamed to be added to the combined dictionary.
    
    Returns a dict:
      {
         "game_A": { "F-PE": [...], ..., "EXP3": [...], "MWU": [...] },
         "game_B": { ... similarly ... }
      }
    """
    combined = {"game_A": {}, "game_B": {}}
    
    # First, add all representations from the first dictionary.
    for rep_label, game_dict in tau_first.items():
        for game_variant in ["game_A", "game_B"]:
            # Only add if there is at least one τ value
            if game_dict[game_variant]:
                combined[game_variant][rep_label] = game_dict[game_variant]
    
    # Next, add the algorithm results from the second dictionary.
    # (Rename key "A" -> "game_A", "B" -> "game_B")
    for game_key in ["A", "B"]:
        game_variant = f"game_{game_key}"
        for alg_label, tau_list in tau_second.get(game_key, {}).items():
            if tau_list:
                combined[game_variant][alg_label] = tau_list
    
    return combined

##############################
# PLOTTING FUNCTION (BAR PLOTS with error bars)
##############################
def plot_combined_tau_barplots(combined_tau, ordering, output_folder="."):
    """
    Creates two bar plots (one for Game A and one for Game B) using the merged τ values.
    
    Parameters:
      - combined_tau: dictionary with keys "game_A" and "game_B", each a dict mapping
                      representation label to list of τ values.
      - ordering: list of labels in the desired order. (Should include all 10 labels.)
      - output_folder: where to save the figures.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Gather all τ values across both games (for same y-axis limits)
    all_vals = []
    for game in ["game_A", "game_B"]:
        for label, values in combined_tau.get(game, {}).items():
            all_vals.extend(values)
    if not all_vals:
        print("No τ values found!")
        return
    
    y_min, y_max = min(all_vals), max(all_vals)
    y_buffer = (y_max - y_min) * 0.1
    y_lim = (y_min - y_buffer, y_max + y_buffer)
    
    for game in ["game_A", "game_B"]:
        labels = []
        means = []
        stds = []
        
        # Use the provided ordering.
        for lab in ordering:
            if lab in combined_tau.get(game, {}):
                values = np.array(combined_tau[game][lab])
                labels.append(lab)
                means.append(np.mean(values))
                stds.append(np.std(values))
            else:
                # If no data for this label, you can choose to add a NaN or skip.
                labels.append(lab)
                means.append(np.nan)
                stds.append(0)
        
        plt.figure(figsize=(10, 6))
        plt.rcParams.update({'font.size': 16})
        bar_containers = plt.bar(labels, means, yerr=stds, capsize=5,
                                 color='cornflowerblue', alpha=0.7, edgecolor='black')
        plt.axhline(0, color='black', linewidth=1)
        plt.ylim(y_lim)
        plt.ylabel("Mean Kendall's τ")
        plt.xlabel("Representation / Algorithm")
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis='y', linestyle="--", alpha=0.5)
        
        output_path = os.path.join(output_folder, f"combined_kendall_tau_{game}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
        plt.show()

##############################
# MAIN SCRIPT: COMPUTE, MERGE, AND PLOT
##############################
if __name__ == "__main__":
    # --- Parameters and Ordering ---
    # For the first data structure we expect 8 labels.
    ordering_first = ["F-PE", "F-PO", "F-RE", "F-RO", "S-PE", "S-PO", "S-RE", "S-RO"]
    # For the second data structure we have 2 algorithm labels.
    ordering_second = ["EXP3", "MWU"]
    # Combined ordering (adjust order as desired; here we put the 8 first then the 2 algorithms)
    combined_ordering = ordering_first + ordering_second
    
    # --- Compute τ values from both sources ---
    # Change these paths as needed:
    input_folder_first = "."  # e.g. folder containing "game_1", "game_2", etc.
    tau_first = compute_kendalls_tau_first(input_folder_first, ordering_first)
    
    # For the second structure, list the folders (ensure these folders exist in your directory)
    folders_second = ["exp3A", "mwA", "exp3B", "mwB"]
    tau_second = compute_kendalls_tau_second(folders_second)
    
    # --- Merge the dictionaries ---
    combined_tau = merge_tau_values(tau_first, tau_second)
    
    # --- Plot combined bar plots for Game A and Game B ---
    plot_combined_tau_barplots(combined_tau, combined_ordering, output_folder="output_plots")
