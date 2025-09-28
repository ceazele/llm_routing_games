import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
import os
import numpy as np
import glob

# --------------------------------------------------------------------------
# Function to count switches between rounds
# --------------------------------------------------------------------------
def count_switches(df):
    switches = []
    rounds = sorted(df['Round'].unique())
    for round_num in rounds[1:]:
        prev_round = df[df['Round'] == round_num - 1].set_index('Agent')['Route']
        current_round = df[df['Round'] == round_num].set_index('Agent')['Route']
        agents = prev_round.index.intersection(current_round.index)
        prev_round = prev_round.loc[agents]
        current_round = current_round.loc[agents]
        switches.append((current_round != prev_round).sum())
    return pd.Series(switches, index=rounds[1:])

# --------------------------------------------------------------------------
# Function to compute mean and standard error across a list of Series/DataFrames
# (ONE Series/DataFrame per RUN).
# --------------------------------------------------------------------------
def compute_mean_se(data_list):
    """
    data_list: list of Series (or DataFrames) indexed by round (or whatever)
    We combine them side-by-side and compute the mean and standard error
    across these runs, *for each index*.
    """
    df = pd.concat(data_list, axis=1)
    mean = df.mean(axis=1)
    se = df.std(axis=1, ddof=1) / np.sqrt(len(data_list))
    return mean, se

# --------------------------------------------------------------------------
# Function to process data and extract relevant statistics (unchanged)
# --------------------------------------------------------------------------
def process_data(input_folder):
    run_folders = sorted(glob.glob(os.path.join(input_folder, 'run *')))

    no_bridge_routes_list = []
    with_bridge_routes_list = []
    no_bridge_payoff_list = []
    with_bridge_payoff_list = []
    no_bridge_switches_list = []
    with_bridge_switches_list = []
    no_bridge_regret_list = []
    with_bridge_regret_list = []

    # Identify valid routes dynamically
    valid_routes_no_bridge = set()
    valid_routes_with_bridge = set()

    for run_folder in run_folders:
        # Infer game folder name from input_folder
        game_folder_name = os.path.basename(input_folder)
        game_A_path = os.path.join(run_folder, f'{game_folder_name}A', f'{game_folder_name}A.csv')
        game_B_path = os.path.join(run_folder, f'{game_folder_name}B', f'{game_folder_name}B.csv')

        if not os.path.exists(game_A_path) or not os.path.exists(game_B_path):
            print(f"Data files not found in {run_folder}, skipping this run.")
            continue

        # --------------------
        # Read data for Game A (No Bridge)
        # --------------------
        no_bridge_df = pd.read_csv(game_A_path)
        valid_routes_no_bridge.update(no_bridge_df['Route'].unique())

        # (Route usage) = # of players on each route per round
        # We'll store this DataFrame for direct usage in the route plot
        no_bridge_routes = no_bridge_df.groupby(['Round', 'Route']).size().unstack(fill_value=0)
        no_bridge_routes_list.append(no_bridge_routes)

        # (Payoff) = average payoff per round
        no_bridge_payoff = no_bridge_df.groupby('Round')['Payoff'].mean()
        no_bridge_payoff_list.append(no_bridge_payoff)

        # (Switches) = # of route switches between consecutive rounds
        no_bridge_switches = count_switches(no_bridge_df)
        no_bridge_switches_list.append(no_bridge_switches)

        # (Regret) = average regret per round
        no_bridge_regret = no_bridge_df.groupby('Round')['Regret'].mean()
        no_bridge_regret_list.append(no_bridge_regret)

        # --------------------
        # Read data for Game B (With Bridge)
        # --------------------
        with_bridge_df = pd.read_csv(game_B_path)
        valid_routes_with_bridge.update(with_bridge_df['Route'].unique())

        with_bridge_routes = with_bridge_df.groupby(['Round', 'Route']).size().unstack(fill_value=0)
        with_bridge_routes_list.append(with_bridge_routes)

        with_bridge_payoff = with_bridge_df.groupby('Round')['Payoff'].mean()
        with_bridge_payoff_list.append(with_bridge_payoff)

        with_bridge_switches = count_switches(with_bridge_df)
        with_bridge_switches_list.append(with_bridge_switches)

        with_bridge_regret = with_bridge_df.groupby('Round')['Regret'].mean()
        with_bridge_regret_list.append(with_bridge_regret)

    return {
        "no_bridge_routes": no_bridge_routes_list,
        "with_bridge_routes": with_bridge_routes_list,
        "no_bridge_payoff": no_bridge_payoff_list,
        "with_bridge_payoff": with_bridge_payoff_list,
        "no_bridge_switches": no_bridge_switches_list,
        "with_bridge_switches": with_bridge_switches_list,
        "no_bridge_regret": no_bridge_regret_list,
        "with_bridge_regret": with_bridge_regret_list,
        "valid_routes_no_bridge": list(valid_routes_no_bridge),
        "valid_routes_with_bridge": list(valid_routes_with_bridge)
    }

# --------------------------------------------------------------------------
# Plot 1: Routes for Game A (No Bridge)
# --------------------------------------------------------------------------
def plot_routes_no_bridge(data, output_folder):
    """
    For each route in 'no_bridge_routes', gather one Series per run (route usage vs. round),
    then compute the across-run mean & SE to plot.
    """
    runs_data = data["no_bridge_routes"]  # list of DataFrames, one per run
    routes = data["valid_routes_no_bridge"]

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(12, 6))

    for route in routes:
        # Collect a list of Series for this route (one Series per run)
        route_series_list = []
        for route_df in runs_data:
            if route in route_df.columns:
                route_series_list.append(route_df[route])

        if not route_series_list:
            continue

        # Mean & SE across runs
        mean_vals, se_vals = compute_mean_se(route_series_list)

        # Plot
        plt.plot(mean_vals.index, mean_vals, marker='o', label=route)
        plt.fill_between(
            mean_vals.index,
            mean_vals - se_vals,
            mean_vals + se_vals,
            alpha=0.2
        )

    plt.xlabel('Round')
    plt.ylabel('Mean Number of Subjects')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'no_bridge_routes.png'))
    plt.close()

# --------------------------------------------------------------------------
# Plot 2: Routes for Game B (With Bridge)
# --------------------------------------------------------------------------
def plot_routes_with_bridge(data, output_folder):
    runs_data = data["with_bridge_routes"]
    routes = data["valid_routes_with_bridge"]

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(12, 6))

    for route in routes:
        route_series_list = []
        for route_df in runs_data:
            if route in route_df.columns:
                route_series_list.append(route_df[route])

        if not route_series_list:
            continue

        mean_vals, se_vals = compute_mean_se(route_series_list)

        plt.plot(mean_vals.index, mean_vals, marker='o', label=route)
        plt.fill_between(
            mean_vals.index,
            mean_vals - se_vals,
            mean_vals + se_vals,
            alpha=0.2
        )

    plt.xlabel('Round')
    plt.ylabel('Mean Number of Subjects')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'with_bridge_routes.png'))
    plt.close()

# --------------------------------------------------------------------------
# Plot 3: Payoff comparison between Game A and Game B
# --------------------------------------------------------------------------
def plot_payoff_comparison(data, output_folder):
    """
    'data["no_bridge_payoff"]' and 'data["with_bridge_payoff"]' are lists of Series.
    Each Series is payoff vs. round for one run. We just do mean & SE across runs.
    """
    no_bridge_payoff_mean, no_bridge_payoff_se = compute_mean_se(data["no_bridge_payoff"])
    with_bridge_payoff_mean, with_bridge_payoff_se = compute_mean_se(data["with_bridge_payoff"])

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(12, 6))

    # Plot Game A
    plt.plot(no_bridge_payoff_mean.index, no_bridge_payoff_mean, marker='o', label='Game A')
    plt.fill_between(
        no_bridge_payoff_mean.index,
        no_bridge_payoff_mean - no_bridge_payoff_se,
        no_bridge_payoff_mean + no_bridge_payoff_se,
        alpha=0.2
    )

    # Plot Game B
    plt.plot(with_bridge_payoff_mean.index, with_bridge_payoff_mean, marker='o', label='Game B')
    plt.fill_between(
        with_bridge_payoff_mean.index,
        with_bridge_payoff_mean - with_bridge_payoff_se,
        with_bridge_payoff_mean + with_bridge_payoff_se,
        alpha=0.2
    )

    plt.xlabel('Round')
    plt.ylabel('Mean Payoff')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'payoff_comparison.png'))
    plt.close()

# --------------------------------------------------------------------------
# Plot 4: Average Regret (Game A vs. Game B)
# --------------------------------------------------------------------------
def plot_average_regret(data, output_folder):
    no_bridge_regret_mean, no_bridge_regret_se = compute_mean_se(data["no_bridge_regret"])
    with_bridge_regret_mean, with_bridge_regret_se = compute_mean_se(data["with_bridge_regret"])

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(12, 6))

    plt.plot(no_bridge_regret_mean.index, no_bridge_regret_mean, marker='o', label='Game A')
    plt.fill_between(
        no_bridge_regret_mean.index,
        no_bridge_regret_mean - no_bridge_regret_se,
        no_bridge_regret_mean + no_bridge_regret_se,
        alpha=0.2
    )

    plt.plot(with_bridge_regret_mean.index, with_bridge_regret_mean, marker='o', label='Game B')
    plt.fill_between(
        with_bridge_regret_mean.index,
        with_bridge_regret_mean - with_bridge_regret_se,
        with_bridge_regret_mean + with_bridge_regret_se,
        alpha=0.2
    )

    plt.xlabel('Round')
    plt.ylabel('Mean Regret')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'average_regret_comparison.png'))
    plt.close()

# --------------------------------------------------------------------------
# Plot 5: Number of Switches (Game A vs. Game B)
# --------------------------------------------------------------------------
def plot_number_of_switches(data, output_folder):
    # data["no_bridge_switches"] and data["with_bridge_switches"] are lists of Series (one per run)
    no_bridge_switches_mean, no_bridge_switches_se = compute_mean_se(data["no_bridge_switches"])
    with_bridge_switches_mean, with_bridge_switches_se = compute_mean_se(data["with_bridge_switches"])

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(12, 6))

    plt.plot(no_bridge_switches_mean.index, no_bridge_switches_mean, marker='o', label='Game A')
    plt.fill_between(
        no_bridge_switches_mean.index,
        no_bridge_switches_mean - no_bridge_switches_se,
        no_bridge_switches_mean + no_bridge_switches_se,
        alpha=0.2
    )

    plt.plot(with_bridge_switches_mean.index, with_bridge_switches_mean, marker='o', label='Game B')
    plt.fill_between(
        with_bridge_switches_mean.index,
        with_bridge_switches_mean - with_bridge_switches_se,
        with_bridge_switches_mean + with_bridge_switches_se,
        alpha=0.2
    )

    plt.xlabel('Round')
    plt.ylabel('Mean Number of Switches')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'number_of_switches.png'))
    plt.close()

# --------------------------------------------------------------------------
# Plot 6: Average Reward Trends for game A or B across multiple folders
# --------------------------------------------------------------------------
def plot_average_reward_trends(game_folders, output_folder, game):
    """
    Plots the average reward trends for the specified game (A or B) across multiple folders.
    Each folder can have multiple runs. We gather payoff Series from each run,
    then compute the mean ± SE across runs.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(16, 9), facecolor="white")

    labels = ["F-PE", "S-PE", "F-RO", "S-RO", "F-PO", "S-PO", "F-RE", "S-RE"]

    for i, game_folder in enumerate(game_folders, start=1):
        run_folders = sorted(glob.glob(os.path.join(game_folder, 'run *')))
        all_rewards = []  # list of Series, one per run

        for run_folder in run_folders:
            game_folder_name = os.path.basename(game_folder)
            game_csv_path = os.path.join(
                run_folder, f'{game_folder_name}{game}', f'{game_folder_name}{game}.csv'
            )
            
            if not os.path.exists(game_csv_path):
                print(f"Data file not found in {run_folder}, skipping this run.")
                continue

            df = pd.read_csv(game_csv_path)
            # For that run, payoff by round
            average_rewards = df.groupby('Round')['Payoff'].mean()
            all_rewards.append(average_rewards)

        # Now compute mean ± SE across runs
        if all_rewards:
            rewards_mean, rewards_se = compute_mean_se(all_rewards)
            label_str = labels[i - 1] if i - 1 < len(labels) else f"Folder {i}"

            plt.plot(rewards_mean.index, rewards_mean, label=label_str)
            plt.fill_between(
                rewards_mean.index,
                rewards_mean - rewards_se,
                rewards_mean + rewards_se,
                alpha=0.2
            )

    plt.xlabel('Round')
    plt.ylabel('Mean Payoff')
    plt.ylim(0, 120)

    # Remove top/right/left spines
    ax = plt.gca()
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    ax.set_xlim(0, 41)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40])

    # Instead of using a fixed ncol, get the handles and set ncol equal to their count.
    handles, leg_labels = ax.get_legend_handles_labels()
    plt.legend(handles, leg_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(handles))
    plt.grid(True)

    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(
        os.path.join(output_folder, f'average_reward_trends_game_{game}.png'),
        facecolor='white',
        bbox_inches='tight'
    )
    plt.close()

# --------------------------------------------------------------------------
# Plot 7: Average Regret Trends for game A or B across multiple folders
# --------------------------------------------------------------------------
def plot_average_regret_trends(game_folders, output_folder, game):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(16, 9), facecolor="white")

    labels = ["F-PE", "S-PE", "F-RO", "S-RO", "F-PO", "S-PO", "F-RE", "S-RE"]

    for i, game_folder in enumerate(game_folders, start=1):
        run_folders = sorted(glob.glob(os.path.join(game_folder, 'run *')))
        all_regrets = []  # list of Series, one per run

        for run_folder in run_folders:
            game_folder_name = os.path.basename(game_folder)
            game_csv_path = os.path.join(
                run_folder, f'{game_folder_name}{game}', f'{game_folder_name}{game}.csv'
            )
            
            if not os.path.exists(game_csv_path):
                print(f"Data file not found in {run_folder}, skipping this run.")
                continue

            df = pd.read_csv(game_csv_path)
            average_regrets = df.groupby('Round')['Regret'].mean()
            all_regrets.append(average_regrets)

        if all_regrets:
            regrets_mean, regrets_se = compute_mean_se(all_regrets)
            label_str = labels[i - 1] if i - 1 < len(labels) else f"Folder {i}"

            plt.plot(regrets_mean.index, regrets_mean, label=label_str)
            plt.fill_between(
                regrets_mean.index,
                regrets_mean - regrets_se,
                regrets_mean + regrets_se,
                alpha=0.2
            )

    plt.xlabel('Round')
    plt.ylabel('Mean Regret')
    plt.ylim(0, 175)

    ax = plt.gca()
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    ax.set_xlim(0, 41)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40])

    # Place legend in one row by setting ncol to the total number of legend handles.
    handles, leg_labels = ax.get_legend_handles_labels()
    plt.legend(handles, leg_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(handles))
    plt.grid(True)

    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(
        os.path.join(output_folder, f'average_regret_trends_game_{game}.png'),
        facecolor='white',
        bbox_inches='tight'
    )
    plt.close()

# --------------------------------------------------------------------------
# Plot 8: Average Switch Trends for game A or B across multiple folders
# --------------------------------------------------------------------------
def plot_average_switch_trends(game_folders, output_folder, game):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(16, 9), facecolor="white")

    labels = ["F-PE", "S-PE", "F-RO", "S-RO", "F-PO", "S-PO", "F-RE", "S-RE"]

    for i, game_folder in enumerate(game_folders, start=1):
        run_folders = sorted(glob.glob(os.path.join(game_folder, 'run *')))
        all_switch_counts = []  # list of Series, one per run

        for run_folder in run_folders:
            game_folder_name = os.path.basename(game_folder)
            game_csv_path = os.path.join(
                run_folder, f'{game_folder_name}{game}', f'{game_folder_name}{game}.csv'
            )

            if not os.path.exists(game_csv_path):
                print(f"Data file not found in {run_folder}, skipping this run.")
                continue

            df = pd.read_csv(game_csv_path)
            switch_counts = count_switches(df)
            all_switch_counts.append(switch_counts)

        if all_switch_counts:
            switch_mean, switch_se = compute_mean_se(all_switch_counts)
            label_str = labels[i - 1] if i - 1 < len(labels) else f"Folder {i}"

            plt.plot(switch_mean.index, switch_mean, label=label_str)
            plt.fill_between(
                switch_mean.index,
                switch_mean - switch_se,
                switch_mean + switch_se,
                alpha=0.2
            )

    plt.xlabel('Round')
    plt.ylabel('Mean Number of Switches')

    ax = plt.gca()
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    ax.set_xlim(0, 41)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40])

    # Force one-row legend (may extend past plot’s right edge)
    handles, leg_labels = ax.get_legend_handles_labels()
    plt.legend(handles, leg_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(handles))
    plt.grid(True)

    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(
        os.path.join(output_folder, f'average_switch_trends_game_{game}.png'),
        facecolor='white',
        bbox_inches='tight'
    )
    plt.close()

# --------------------------------------------------------------------------
# Plot 9: Specific Route Trends (Different logic for A vs B)
# --------------------------------------------------------------------------
def plot_specific_route_trends(game_folders, output_folder, game):
    """
    - Game A: min(#O-L-D, #O-R-D) per round
    - Game B: #O-L-R-D per round

    We gather one Series per run, then do mean ± SE across runs for each round.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(16, 9), facecolor="white")

    labels = ["F-PE", "S-PE", "F-RO", "S-RO", "F-PO", "S-PO", "F-RE", "S-RE"]

    for i, game_folder in enumerate(game_folders, start=1):
        run_folders = sorted(glob.glob(os.path.join(game_folder, 'run *')))
        all_series = []

        for run_folder in run_folders:
            game_folder_name = os.path.basename(game_folder)
            game_csv_path = os.path.join(
                run_folder,
                f"{game_folder_name}{game}",
                f"{game_folder_name}{game}.csv"
            )

            if not os.path.exists(game_csv_path):
                print(f"Data file not found in {run_folder}, skipping this run.")
                continue

            df = pd.read_csv(game_csv_path)
            route_counts = df.groupby(['Round', 'Route']).size().unstack(fill_value=0)

            if game == "A":
                # min(#O-L-D, #O-R-D)
                if "O-L-D" not in route_counts.columns or "O-R-D" not in route_counts.columns:
                    continue
                usage_series = route_counts[["O-L-D", "O-R-D"]].min(axis=1)
            else:
                # "B" -> #O-L-R-D
                if "O-L-R-D" not in route_counts.columns:
                    continue
                usage_series = route_counts["O-L-R-D"]

            all_series.append(usage_series)

        if all_series:
            mean_vals, se_vals = compute_mean_se(all_series)
            label_str = labels[i - 1] if i - 1 < len(labels) else f"Folder {i}"
            plt.plot(mean_vals.index, mean_vals, label=label_str)
            plt.fill_between(
                mean_vals.index,
                mean_vals - se_vals,
                mean_vals + se_vals,
                alpha=0.2
            )

    if game == "A":
        ylabel = "Mean of min(#O-L-D, #O-R-D)"
    else:
        ylabel = "Mean #O-L-R-D"

    plt.xlabel("Round")
    plt.ylabel(ylabel)

    ax = plt.gca()
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    ax.set_xlim(0, 41)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40])

    # Place legend in one row by setting ncol to the number of handles.
    handles, leg_labels = ax.get_legend_handles_labels()
    plt.legend(handles, leg_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(handles))
    plt.grid(True)

    os.makedirs(output_folder, exist_ok=True)
    if game == "A":
        plot_filename = "average_min_OLD_ORD_gameA.png"
    else:
        plot_filename = "average_OLRD_gameB.png"

    plt.savefig(
        os.path.join(output_folder, plot_filename),
        facecolor='white',
        bbox_inches='tight'
    )
    plt.close()

# --------------------------------------------------------------------------
# Utility: compute_average_route_choices (unchanged)
# --------------------------------------------------------------------------
def compute_average_route_choices(input_folder):
    """
    Compute the average and standard deviation of route choices across all rounds
    and trials for both Game A and Game B.
    """
    game_name = os.path.basename(input_folder)
    game_A_filename = f"{game_name}A.csv"
    game_B_filename = f"{game_name}B.csv"

    game_A_paths = glob.glob(os.path.join(input_folder, "run *", game_name + "A", game_A_filename))
    game_B_paths = glob.glob(os.path.join(input_folder, "run *", game_name + "B", game_B_filename))

    if not game_A_paths or not game_B_paths:
        raise FileNotFoundError(
            f"Could not find {game_A_filename} or {game_B_filename} in {input_folder}."
        )

    # Initialize lists to hold route choice data for each run
    game_A_route_choices_list = []
    game_B_route_choices_list = []

    for path in game_A_paths:
        game_A_df = pd.read_csv(path)
        game_A_route_choices = game_A_df.groupby(['Round', 'Route']).size().unstack(fill_value=0)
        game_A_route_choices_list.append(game_A_route_choices)

    for path in game_B_paths:
        game_B_df = pd.read_csv(path)
        game_B_route_choices = game_B_df.groupby(['Round', 'Route']).size().unstack(fill_value=0)
        game_B_route_choices_list.append(game_B_route_choices)

    # Combine data across all runs
    game_A_combined = pd.concat(game_A_route_choices_list).groupby(level=0).mean()
    game_A_avg = game_A_combined.mean(axis=0)
    game_A_std = game_A_combined.std(axis=0, ddof=1)

    game_B_combined = pd.concat(game_B_route_choices_list).groupby(level=0).mean()
    game_B_avg = game_B_combined.mean(axis=0)
    game_B_std = game_B_combined.std(axis=0, ddof=1)

    game_A_stats = pd.DataFrame({
        "Route": game_A_avg.index,
        "Average Players": game_A_avg.values,
        "Standard Deviation": game_A_std.values
    })

    game_B_stats = pd.DataFrame({
        "Route": game_B_avg.index,
        "Average Players": game_B_avg.values,
        "Standard Deviation": game_B_std.values
    })

    output_folder = os.path.join(input_folder, "results")
    os.makedirs(output_folder, exist_ok=True)

    game_A_csv_path = os.path.join(output_folder, f"{game_name}_a_route_stats.csv")
    game_B_csv_path = os.path.join(output_folder, f"{game_name}_b_route_stats.csv")

    game_A_stats.to_csv(game_A_csv_path, index=False)
    game_B_stats.to_csv(game_B_csv_path, index=False)

    print(f"Results saved to:\n  {game_A_csv_path}\n  {game_B_csv_path}")

    return {"game_a": game_A_stats, "game_b": game_B_stats}

# --------------------------------------------------------------------------
# Define expected route frequencies (unchanged)
# --------------------------------------------------------------------------
EXPECTED_FREQUENCIES = {
    "game_A": {"O-L-D": 9, "O-R-D": 9},
    "game_B": {"O-L-D": 0, "O-R-D": 0, "O-L-R-D": 18}
}

# --------------------------------------------------------------------------
# compute_kendalls_tau (unchanged)
# --------------------------------------------------------------------------
def compute_kendalls_tau(input_folder):
    """
    Computes Kendall's Tau correlation between round number and deviation scores for each run.
    """
    run_folders = sorted(glob.glob(os.path.join(input_folder, 'run *')))
    tau_results = {}

    for run_folder in run_folders:
        game_folder_name = os.path.basename(input_folder)
        
        # Paths for Game A and Game B
        game_A_path = os.path.join(run_folder, f'{game_folder_name}A', f'{game_folder_name}A.csv')
        game_B_path = os.path.join(run_folder, f'{game_folder_name}B', f'{game_folder_name}B.csv')

        for game_label, game_path in [("game_A", game_A_path), ("game_B", game_B_path)]:
            if not os.path.exists(game_path):
                print(f"Skipping missing file: {game_path}")
                continue
            
            df = pd.read_csv(game_path)
            expected_counts = EXPECTED_FREQUENCIES[game_label]

            round_numbers = []
            deviation_scores = []

            for round_num in sorted(df["Round"].unique()):
                round_numbers.append(round_num)
                observed_counts = df[df["Round"] == round_num]["Route"].value_counts().to_dict()

                # Sum of absolute differences from expected freq
                deviation_score = sum(
                    abs(observed_counts.get(route, 0) - expected_counts.get(route, 0))
                    for route in expected_counts
                )
                deviation_scores.append(deviation_score)

            # Compute Kendall’s Tau if >1 round
            if len(round_numbers) > 1:
                tau, p_value = kendalltau(round_numbers, deviation_scores)
                tau_results[f"{os.path.basename(run_folder)}_{game_label}"] = (tau, p_value)

    return tau_results

# --------------------------------------------------------------------------
# End of program
# --------------------------------------------------------------------------

# # Example Usage
# input_folder = "game_2"  # Change this to your actual folder path
# tau_results = compute_kendalls_tau(input_folder)

# # Print results
# for run, (tau, p) in tau_results.items():
#     print(f"{run}: Kendall's Tau = {tau:.4f}, p-value = {p:.4f}")




# for i in range (11,13):
#     input_folder = f"game_{i}"
#     # results = compute_average_route_choices(input_folder)
#     data = process_data(input_folder)
#     plot_routes_no_bridge(data, input_folder)
#     plot_routes_with_bridge(data, input_folder)
#     plot_payoff_comparison(data, input_folder)
#     plot_average_regret(data, input_folder)
#     plot_number_of_switches(data, input_folder)
    


plot_average_reward_trends(['game_1', 'game_2', 'game_3', 'game_4', 'game_5', 'game_6', 'game_11', 'game_12'], '.', 'A')
plot_average_reward_trends(['game_1', 'game_2', 'game_3', 'game_4', 'game_5', 'game_6', 'game_11', 'game_12'], '.', 'B')

plot_average_regret_trends(['game_1', 'game_2', 'game_3', 'game_4', 'game_5', 'game_6', 'game_11', 'game_12'], '.', 'A')
plot_average_regret_trends(['game_1', 'game_2', 'game_3', 'game_4', 'game_5', 'game_6', 'game_11', 'game_12'], '.', 'B')

plot_average_switch_trends(['game_1', 'game_2', 'game_3', 'game_4', 'game_5', 'game_6', 'game_11', 'game_12'], '.', 'A')
plot_average_switch_trends(['game_1', 'game_2', 'game_3', 'game_4', 'game_5', 'game_6', 'game_11', 'game_12'], '.', 'B')

plot_specific_route_trends(
    game_folders=['game_1', 'game_2', 'game_3', 'game_4', 'game_5', 'game_6', 'game_11', 'game_12'], output_folder='.', game='A')

plot_specific_route_trends(
    game_folders=['game_1', 'game_2', 'game_3', 'game_4', 'game_5', 'game_6', 'game_11', 'game_12'], output_folder='.', game='B')
