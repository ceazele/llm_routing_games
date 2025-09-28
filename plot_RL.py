import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###############################################################################
# Helper: Identify 'A' vs 'B' from Folder Name
###############################################################################

def identify_game_variant(folder_name):
    """Detect whether a folder corresponds to Game A or Game B based on its name."""
    base = os.path.basename(folder_name.rstrip("/"))
    if base.endswith("A"):
        return "A"
    elif base.endswith("B"):
        return "B"
    else:
        return "A"  # Default to A if unclear

###############################################################################
# Helper: Identify Algorithm Name
###############################################################################

def identify_algorithm(folder_name):
    """Map folder names to algorithm labels (EXP3 or MWU)."""
    if "exp3" in folder_name.lower():
        return "EXP3"
    elif "mw" in folder_name.lower():
        return "MWU"
    else:
        return folder_name  # Preserve original folder name if unclear

###############################################################################
# Helper: Count Switches Per Round
###############################################################################

def count_switches(df):
    """Counts the number of times each agent switches routes across rounds."""
    switches = []
    rounds = sorted(df['Round'].unique())
    for round_num in rounds[1:]:
        prev_df = df[df['Round'] == round_num - 1].set_index('Agent')['Route']
        curr_df = df[df['Round'] == round_num].set_index('Agent')['Route']
        agents = prev_df.index.intersection(curr_df.index)
        switches.append((curr_df.loc[agents] != prev_df.loc[agents]).sum())
    return pd.Series(switches, index=rounds[1:])

###############################################################################
# Helper: Compute Mean and Standard Error
###############################################################################

def compute_mean_se(series_list):
    """Computes the mean and standard error across a list of Series indexed by Round."""
    df = pd.concat(series_list, axis=1)
    mean = df.mean(axis=1)
    se = df.std(axis=1, ddof=1) / np.sqrt(len(series_list))
    return mean, se

###############################################################################
# Core Function: Plot Trends (Reward, Regret, Switches)
###############################################################################

def plot_trends(folders, metric, output_folder=".", ylabel="Metric", filename="trends.png"):
    """
    Generic function to plot trends for a given metric (Payoff, Regret, or Switches).

    Args:
        folders (list): List of folder names to process.
        metric (str): Column name in the CSV to plot (e.g., "Payoff", "Regret", "Switches").
        output_folder (str): Folder where the plot should be saved.
        ylabel (str): Label for the y-axis.
        filename (str): The original filename format to preserve.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rcParams.update({'font.size': 20})

    # Detect game variant (A or B) from the first folder
    game_variant = identify_game_variant(folders[0]) if folders else "A"

    min_y, max_y = float('inf'), float('-inf')

    for folder in folders:
        metric_list = []
        csv_paths = sorted(glob.glob(os.path.join(folder, "results_run_*.csv")))
        for path in csv_paths:
            df = pd.read_csv(path)
            if metric == "Switches":
                values = count_switches(df)
            else:
                values = df.groupby("Round")[metric].mean()
            metric_list.append(values)

        if not metric_list:
            continue

        mean_vals, se_vals = compute_mean_se(metric_list)
        algorithm_label = identify_algorithm(folder)
        ax.plot(mean_vals.index, mean_vals, label=algorithm_label)
        ax.fill_between(mean_vals.index, mean_vals - se_vals, mean_vals + se_vals, alpha=0.2)

        # Track min/max y values for consistent axis limits
        min_y, max_y = min(min_y, mean_vals.min()), max(max_y, mean_vals.max())

    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.grid(True)

    # Set y-axis limits consistently
    ax.set_ylim(min_y - 0.05 * (max_y - min_y), max_y + 0.05 * (max_y - min_y))

    # Place legend below the plot in a horizontal layout
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(folders))

    os.makedirs(output_folder, exist_ok=True)
    out_name = filename.replace(".png", f"_game{game_variant}.png")  # Preserve original naming convention
    out_path = os.path.join(output_folder, out_name)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {metric} trends to: {out_path}")

###############################################################################
# Core Function: Plot Route Trends
###############################################################################

def plot_specific_route_trends(folders, output_folder=".", filename="specific_route_trends.png"):
    """
    Plots trends for specific route usage:
        - For Game A: Plots the min(#O-L-D, #O-R-D) usage per round.
        - For Game B: Plots #O-L-R-D usage per round.

    Args:
        folders (list): List of folders to process.
        output_folder (str): Output directory for plots.
        filename (str): The original filename format to preserve.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rcParams.update({'font.size': 20})

    game_variant = identify_game_variant(folders[0]) if folders else "A"
    min_y, max_y = float('inf'), float('-inf')

    for folder in folders:
        usage_list = []
        csv_paths = sorted(glob.glob(os.path.join(folder, "results_run_*.csv")))
        for path in csv_paths:
            df = pd.read_csv(path)
            route_counts = df.groupby(["Round", "Route"]).size().unstack(fill_value=0)

            if game_variant == "A":
                if "O-L-D" not in route_counts.columns or "O-R-D" not in route_counts.columns:
                    continue
                usage_series = route_counts[["O-L-D", "O-R-D"]].min(axis=1)
            else:
                if "O-L-R-D" not in route_counts.columns:
                    continue
                usage_series = route_counts["O-L-R-D"]

            usage_list.append(usage_series)

        if not usage_list:
            continue

        mean_vals, se_vals = compute_mean_se(usage_list)
        algorithm_label = identify_algorithm(folder)
        ax.plot(mean_vals.index, mean_vals, label=algorithm_label)
        ax.fill_between(mean_vals.index, mean_vals - se_vals, mean_vals + se_vals, alpha=0.2)

        min_y, max_y = min(min_y, mean_vals.min()), max(max_y, mean_vals.max())

    ax.set_xlabel("Round")
    ylabel_text = "Mean of min(#O-L-D, #O-R-D)" if game_variant == "A" else "Mean #O-L-R-D"
    ax.set_ylabel(ylabel_text)
    ax.grid(True)

    ax.set_ylim(min_y - 0.05 * (max_y - min_y), max_y + 0.05 * (max_y - min_y))

    # Place legend below the plot in a horizontal layout
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(folders))

    os.makedirs(output_folder, exist_ok=True)
    out_name = filename.replace(".png", f"_game{game_variant}.png")  # Preserve original filename
    out_path = os.path.join(output_folder, out_name)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved specific route trends to: {out_path}")

###############################################################################
# Example Usage
###############################################################################

if __name__ == "__main__":
    # Define experiment folders (Modify as needed)
    folder_list_A = ["mwA", "exp3A"]  # Game A
    folder_list_B = ["mwB", "exp3B"]  # Game B

    for folders in [folder_list_A, folder_list_B]:
        plot_trends(folders, "Payoff", output_folder=".", ylabel="Mean Payoff", filename="average_reward_trends.png")
        plot_trends(folders, "Regret", output_folder=".", ylabel="Mean Regret", filename="average_regret_trends.png")
        plot_trends(folders, "Switches", output_folder=".", ylabel="Mean Number of Switches", filename="average_switch_trends.png")
        plot_specific_route_trends(folders, output_folder=".", filename="specific_route_trends.png")
