import os
import glob
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###############################################################################
# 1) Reading and Combining Data
###############################################################################

def read_all_runs_in_folder(folder_path):
    """
    Reads all *.csv files in 'folder_path', assuming each CSV is one run.
    Returns a list of DataFrames (one per run).
    """
    csv_paths = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    all_dfs = []
    for path in csv_paths:
        df = pd.read_csv(path)
        # Expect columns: Round, Agent, Route, Payoff, Regret (etc.)
        all_dfs.append(df)
    return all_dfs

def compute_mean_se(list_of_series):
    """
    Given a list of Series (all with matching indices),
    return (mean_series, se_series).
    SE = std / sqrt(N).
    """
    if not list_of_series:
        return None, None
    df = pd.concat(list_of_series, axis=1)
    mean = df.mean(axis=1)
    se = df.std(axis=1, ddof=1) / np.sqrt(len(list_of_series))
    return mean, se

###############################################################################
# 2) Counting Switches
###############################################################################

def count_switches_for_run(df):
    """
    For one run's DataFrame, returns a Series (# of switches by round).
    If the CSV has columns [Round, Agent, Route], we do:
      - For each round t in [2..maxRound],
        count how many agents changed route from round (t-1).
    Index = Rounds (2..maxRound). Value = # of switches
    """
    # We'll store the result in a dict {round -> count}
    switches_dict = {}
    rounds = sorted(df['Round'].unique())
    for round_num in rounds[1:]:
        prev_df = df[df['Round'] == round_num - 1].set_index('Agent')['Route']
        curr_df = df[df['Round'] == round_num].set_index('Agent')['Route']
        # Align on agents
        agents = prev_df.index.intersection(curr_df.index)
        prev_routes = prev_df.loc[agents]
        curr_routes = curr_df.loc[agents]
        num_switches = (prev_routes != curr_routes).sum()
        switches_dict[round_num] = num_switches
    # Convert to a Series, index by the round numbers
    return pd.Series(switches_dict)

###############################################################################
# 3) Generic Plotting Function
###############################################################################

def plot_comparison(
    stat_type, 
    folders, 
    output_path="comparison_plot.png", 
    labels=None
):
    """
    Plots a comparison of 'stat_type' across multiple folders.
    - stat_type: one of {"payoff", "regret", "switches"}.
    - folders: list of folder paths, each containing multiple CSVs.
    - output_path: where to save the plot (PNG).
    - labels: optional list of legend labels, must match length of folders.
    
    Each folder is processed as follows:
    1) Read all CSV files.
    2) For each CSV, create a Series (index=Round, value=[ payoffs, regrets, or switches ]).
       - payoff or regret: groupby("Round") and take .mean().
       - switches: use count_switches_for_run().
    3) Compute mean+SE across runs in that folder.
    4) Plot line + shaded SE region.
    
    Then we combine on a single figure.
    """

    if labels is None:
        # Default to folder names
        labels = [os.path.basename(f.strip("/")) for f in folders]

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(8, 6))

    for i, folder in enumerate(folders):
        runs = read_all_runs_in_folder(folder)
        if not runs:
            print(f"No CSV runs found in {folder}, skipping.")
            continue

        # Build a list of Series for this folder
        series_list = []
        for df in runs:
            if stat_type == "payoff":
                # payoff_by_round: index=Round, value=mean payoff
                payoff_by_round = df.groupby("Round")["Payoff"].mean()
                series_list.append(payoff_by_round)
            elif stat_type == "regret":
                # regret_by_round: index=Round, value=mean regret
                regret_by_round = df.groupby("Round")["Regret"].mean()
                series_list.append(regret_by_round)
            elif stat_type == "switches":
                # switches_by_round: index=Round, value=# of switches
                # (But note index starts from round 2 if there's a previous round)
                switches_by_round = count_switches_for_run(df)
                series_list.append(switches_by_round)
            else:
                raise ValueError(f"Unknown stat_type: {stat_type}")

        # Compute mean & SE across runs
        mean_s, se_s = compute_mean_se(series_list)
        if mean_s is None:
            print(f"No valid data in {folder} for stat={stat_type}, skipping.")
            continue

        # Plot
        rounds = mean_s.index
        plt.plot(rounds, mean_s, label=labels[i])
        plt.fill_between(rounds, mean_s - se_s, mean_s + se_s, alpha=0.2)

    # Axes labels
    plt.xlabel("Round")
    if stat_type == "payoff":
        plt.ylabel("Mean Payoff")
        plt.title("Payoff Comparison")
    elif stat_type == "regret":
        plt.ylabel("Mean Regret")
        plt.title("Regret Comparison")
    elif stat_type == "switches":
        plt.ylabel("Number of Switches")
        plt.title("Switches Comparison")

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Plot for '{stat_type}' saved to {output_path}")

###############################################################################
# Example usage
###############################################################################

if __name__ == "__main__":
    # Suppose we have 3 folders of RL runs, each containing several CSVs:
    folder_list = ["mwB"]
    
    # Compare payoff across them
    plot_comparison("payoff", folder_list, output_path="compare_payoff.png", labels=["MWU"])
    
    # Compare regret across them
    plot_comparison("regret", folder_list, output_path="compare_regret.png")
    
    # Compare # of switches across them
    plot_comparison("switches", folder_list, output_path="compare_switches.png")
