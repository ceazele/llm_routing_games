import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm, to_hex

###############################################################################
# Globals/Constants
###############################################################################

BLUE_FOR_HIGHER = {
    "route": False,
    "payoff": False,
    "regret": True,
    "switches": True
}
CUSTOM_CMAP = plt.get_cmap("coolwarm")

PRINT_ORDER = ["F-PE", "F-RE", "F-PO", "F-RO", "S-PE", "S-RE", "S-PO", "S-RO"]  # If you use these representations

###############################################################################
# Folder logic: "Game A" vs. "Game B"
###############################################################################

def identify_game_variant(folder_name):
    """
    If the folder name ends with 'A', treat it as Game A.
    If it ends with 'B', treat it as Game B.
    Otherwise, default to Game A or adapt as needed.
    """
    base = os.path.basename(folder_name.rstrip("/"))
    if base.endswith("A"):
        return "A"
    elif base.endswith("B"):
        return "B"
    else:
        return "A"

###############################################################################
# Reading all CSV runs in a folder
###############################################################################

def read_all_runs_in_folder(folder_path):
    """
    Reads all CSV files in 'folder_path'. Each CSV is assumed to be one run.
    Returns a list of DataFrames (one per run).
    """
    import glob
    csv_paths = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    runs = []
    for path in csv_paths:
        df = pd.read_csv(path)
        runs.append(df)
    return runs

###############################################################################
# Switch counting helper
###############################################################################

def count_switches_in_run(df):
    """
    For each agent, count how many times they switch routes. Return the average across all agents.
    """
    df_sorted = df.sort_values(["Agent", "Round"])
    # Switches = # times (route != route.shift())
    # Subtract 1 so the first row isn't counted as a switch
    changes = df_sorted.groupby("Agent")["Route"].apply(lambda x: (x != x.shift()).sum() - 1)
    return changes.mean()

###############################################################################
# Main function: compute_statistics_with_colors
###############################################################################

def compute_statistics_with_colors(folders, metric_type="route"):
    """
    For each folder in `folders`, read all CSV runs and compute an average value for:
      - 'route':   if Game A -> min(#O-L-D, #O-R-D) each round, then average
                   if Game B -> #O-L-R-D each round, then average
      - 'payoff':  average payoff across all agents
      - 'regret':  average regret across all agents
      - 'switches': average # of route changes per agent

    Then we color-code each folder's mean from a red/blue scale, with a midpoint-based normalization.
    Returns: dict -> { folder_basename : (meanVal, seVal, colorHex) }
    """

    folder_stats = {}  # folder_name -> (mean, se, color)

    # 1) Gather raw means for each folder
    raw_data = {}

    for folder in folders:
        folder_name = os.path.basename(folder.rstrip("/"))
        game_variant = identify_game_variant(folder)  # "A" or "B"

        # Read all runs
        runs = read_all_runs_in_folder(folder)
        if not runs:
            print(f"No CSV files found in {folder}, skipping.")
            folder_stats[folder_name] = (0, 0, "#AAAAAA")
            continue

        # We'll compute one numeric metric per run, then average
        values = []
        for df in runs:
            if metric_type == "route":
                if game_variant == "A":
                    # min(#O-L-D, #O-R-D) for each round, average across rounds
                    old_counts = df[df["Route"] == "O-L-D"].groupby("Round")["Agent"].count()
                    ord_counts = df[df["Route"] == "O-R-D"].groupby("Round")["Agent"].count()

                    # Align them by round, fill missing with 0
                    old_counts = old_counts.reindex(range(int(df["Round"].min()), int(df["Round"].max()) + 1), fill_value=0)
                    ord_counts = ord_counts.reindex(range(int(df["Round"].min()), int(df["Round"].max()) + 1), fill_value=0)

                    min_counts = np.minimum(old_counts, ord_counts)
                    avg_count = min_counts.mean()  # average across rounds
                    values.append(avg_count)

                else:  # game_variant == "B"
                    # average # of O-L-R-D
                    olrd_counts = df[df["Route"] == "O-L-R-D"].groupby("Round")["Agent"].count()
                    olrd_counts = olrd_counts.reindex(range(int(df["Round"].min()), int(df["Round"].max()) + 1), fill_value=0)
                    avg_count = olrd_counts.mean()
                    values.append(avg_count)

            elif metric_type == "payoff":
                # average payoff across all agents
                avg_payoff = df.groupby("Agent")["Payoff"].mean().mean()
                values.append(avg_payoff)

            elif metric_type == "regret":
                # average regret across all agents
                avg_regret = df.groupby("Agent")["Regret"].mean().mean()
                values.append(avg_regret)

            elif metric_type == "switches":
                # average # of route changes
                avg_sw = count_switches_in_run(df)
                values.append(avg_sw)

            else:
                raise ValueError(f"Unknown metric: {metric_type}")

        if values:
            m = np.mean(values)
            s = np.std(values, ddof=1)
            se = s / np.sqrt(len(values))
            raw_data[folder_name] = (m, se)
        else:
            raw_data[folder_name] = (0, 0)

    # 2) Color-code each folder's mean
    means_only = [v[0] for v in raw_data.values()]
    if not means_only:
        return {}

    min_val, max_val = min(means_only), max(means_only)
    mid_val = 0.5 * (min_val + max_val)

    from matplotlib.colors import TwoSlopeNorm, to_hex
    norm = TwoSlopeNorm(vmin=min_val, vcenter=mid_val, vmax=max_val)

    for folder_name, (mean_val, se_val) in raw_data.items():
        if len(means_only) == 1:
            color_hex = "#AAAAAA"
        else:
            raw_color_val = norm(mean_val)
            # If BLUE_FOR_HIGHER is True => bigger => bluer
            if BLUE_FOR_HIGHER[metric_type]:
                color_val = raw_color_val
            else:
                color_val = 1 - raw_color_val
            color_hex = to_hex(CUSTOM_CMAP(color_val))

        folder_stats[folder_name] = (mean_val, se_val, color_hex)

    # Print results sorted by folder name
    print(f"\n=== {metric_type.upper()} STATISTICS WITH COLORS ===\n")
    sorted_folders = sorted(folder_stats.keys())
    for folder_name in sorted_folders:
        mean_val, se_val, color_hex = folder_stats[folder_name]
        print(f"{folder_name}: mean={mean_val:.2f}, se={se_val:.2f}, color={color_hex}")

    return folder_stats

###############################################################################
# Example usage
###############################################################################

if __name__ == "__main__":
    # Suppose you have folders like "mwA", "mwB", "exp3A", "exp3B",
    # each containing multiple run CSVs:
    folders = ["mwA", "mwB", "exp3A", "exp3B"]

    # Example: compute "route" statistics with the same logic as your original code:
    # - Game A => min(#O-L-D, #O-R-D)
    # - Game B => #O-L-R-D
    compute_statistics_with_colors(folders, metric_type="route")

    # Another example: "payoff"
    compute_statistics_with_colors(folders, metric_type="payoff")

    # Another example: "regret"
    compute_statistics_with_colors(folders, metric_type="regret")

    # Another example: "switches"
    compute_statistics_with_colors(folders, metric_type="switches")
