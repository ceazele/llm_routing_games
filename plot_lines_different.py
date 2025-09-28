import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
import os
import numpy as np
import glob


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


def add_side_labels(ax):
    """ Adds labels to the right side of each line in the plot, avoiding overlap and using leader lines. """
    y_margin = 2  # Minimum vertical spacing between labels
    x_margin = 1.0  # Minimum horizontal spacing adjustment to the left
    line_length = 5  # Length of the black leader line extending from the plot line
    
    # Extract lines and sort them by their final y-value
    labels = []
    for line in ax.get_lines():
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        color = line.get_color()  # Get the line color

        if len(x_data) > 0 and len(y_data) > 0:
            label_x = x_data[-1]  # Start the label near the end of the line
            label_y = y_data[-1]
            labels.append([label_x, label_y, line.get_label(), color])

    # Sort labels by their y-values
    labels.sort(key=lambda l: l[1])

    # Adjust labels to prevent overlap
    previous_x = None
    previous_y = None
    adjusted_labels = []
    
    for i, (label_x, label_y, label, color) in enumerate(labels):
        # If too close to the previous label, adjust y
        if previous_y is not None and label_y - previous_y < y_margin:
            label_y = previous_y + y_margin  # Push downward
        
        # If too close in x-direction, move label left
        if previous_x is not None and abs(label_x - previous_x) < x_margin:
            label_x -= x_margin  # Shift left

        adjusted_labels.append((label_x, label_y, label, color))
        previous_x, previous_y = label_x, label_y  # Update last placed label position

    # Plot labels and leader lines after adjustment
    for label_x, label_y, label, color in adjusted_labels:
        # Draw a short black leader line from the curve to the label
        ax.plot([label_x, label_x + line_length], [label_y, label_y], color="black", linewidth=1.2)
        
        # Place the label at the end of the leader line
        ax.text(label_x + line_length + 1, label_y, label, 
                verticalalignment='center', fontsize=14, color=color, fontweight='bold')


# Update only plotting functions to add side labels
def plot_average_reward_trends(game_folders, output_folder, game):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")

    labels = ["F-PE", "S-PE", "F-RO", "S-RO", "F-PO", "S-PO", "F-RE", "S-RE"]

    for i, game_folder in enumerate(game_folders, start=1):
        run_folders = sorted(glob.glob(os.path.join(game_folder, 'run *')))
        all_rewards = []

        for run_folder in run_folders:
            game_folder_name = os.path.basename(game_folder)
            game_csv_path = os.path.join(run_folder, f'{game_folder_name}{game}', f'{game_folder_name}{game}.csv')

            if not os.path.exists(game_csv_path):
                continue

            df = pd.read_csv(game_csv_path)
            average_rewards = df.groupby('Round')['Payoff'].mean()
            all_rewards.append(average_rewards)

        if all_rewards:
            rewards_mean, rewards_se = compute_mean_se(all_rewards)
            label_str = labels[i - 1] if i - 1 < len(labels) else f"Folder {i}"
            ax.plot(rewards_mean.index, rewards_mean, label=label_str)
            ax.fill_between(rewards_mean.index, rewards_mean - rewards_se, rewards_mean + rewards_se, alpha=0.2)

    plt.xlabel('Round')
    plt.ylabel('Mean Payoff')
    plt.ylim(0, 120)
    ax.set_xlim(0, 41)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40])
    plt.grid(True)

    add_side_labels(ax)  # Add labels to the right side
    plt.savefig(os.path.join(output_folder, f'average_reward_trends_game_{game}.png'), facecolor='white', bbox_inches='tight')
    plt.close()

def plot_average_regret_trends(game_folders, output_folder, game):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")

    labels = ["F-PE", "S-PE", "F-RO", "S-RO", "F-PO", "S-PO", "F-RE", "S-RE"]

    for i, game_folder in enumerate(game_folders, start=1):
        run_folders = sorted(glob.glob(os.path.join(game_folder, 'run *')))
        all_regrets = []

        for run_folder in run_folders:
            game_folder_name = os.path.basename(game_folder)
            game_csv_path = os.path.join(run_folder, f'{game_folder_name}{game}', f'{game_folder_name}{game}.csv')

            if not os.path.exists(game_csv_path):
                continue

            df = pd.read_csv(game_csv_path)
            average_regrets = df.groupby('Round')['Regret'].mean()
            all_regrets.append(average_regrets)

        if all_regrets:
            regrets_mean, regrets_se = compute_mean_se(all_regrets)
            label_str = labels[i - 1] if i - 1 < len(labels) else f"Folder {i}"
            ax.plot(regrets_mean.index, regrets_mean, label=label_str)
            ax.fill_between(regrets_mean.index, regrets_mean - regrets_se, regrets_mean + regrets_se, alpha=0.2)

    plt.xlabel('Round')
    plt.ylabel('Mean Regret')
    plt.ylim(0, 175)
    ax.set_xlim(0, 41)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40])
    plt.grid(True)

    add_side_labels(ax)  # Add labels to the right side
    plt.savefig(os.path.join(output_folder, f'average_regret_trends_game_{game}.png'), facecolor='white', bbox_inches='tight')
    plt.close()

def plot_average_switch_trends(game_folders, output_folder, game):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")

    labels = ["F-PE", "S-PE", "F-RO", "S-RO", "F-PO", "S-PO", "F-RE", "S-RE"]

    for i, game_folder in enumerate(game_folders, start=1):
        run_folders = sorted(glob.glob(os.path.join(game_folder, 'run *')))
        all_switch_counts = []

        for run_folder in run_folders:
            game_folder_name = os.path.basename(game_folder)
            game_csv_path = os.path.join(run_folder, f'{game_folder_name}{game}', f'{game_folder_name}{game}.csv')

            if not os.path.exists(game_csv_path):
                continue

            df = pd.read_csv(game_csv_path)
            switch_counts = count_switches(df)
            all_switch_counts.append(switch_counts)

        if all_switch_counts:
            switch_mean, switch_se = compute_mean_se(all_switch_counts)
            label_str = labels[i - 1] if i - 1 < len(labels) else f"Folder {i}"
            ax.plot(switch_mean.index, switch_mean, label=label_str)
            ax.fill_between(switch_mean.index, switch_mean - switch_se, switch_mean + switch_se, alpha=0.2)

    plt.xlabel('Round')
    plt.ylabel('Mean Number of Switches')
    ax.set_xlim(0, 41)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40])
    plt.grid(True)

    add_side_labels(ax)  # Add labels to the right side
    plt.savefig(os.path.join(output_folder, f'average_switch_trends_game_{game}.png'), facecolor='white', bbox_inches='tight')
    plt.close()


def plot_specific_route_trends(game_folders, output_folder, game):
    """
    - Game A: min(#O-L-D, #O-R-D) per round
    - Game B: #O-L-R-D per round
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")

    labels = ["F-PE", "S-PE", "F-RO", "S-RO", "F-PO", "S-PO", "F-RE", "S-RE"]

    for i, game_folder in enumerate(game_folders, start=1):
        run_folders = sorted(glob.glob(os.path.join(game_folder, 'run *')))
        all_series = []

        for run_folder in run_folders:
            game_folder_name = os.path.basename(game_folder)
            game_csv_path = os.path.join(run_folder, f"{game_folder_name}{game}", f"{game_folder_name}{game}.csv")

            if not os.path.exists(game_csv_path):
                continue

            df = pd.read_csv(game_csv_path)
            route_counts = df.groupby(['Round', 'Route']).size().unstack(fill_value=0)

            if game == "A":
                if "O-L-D" not in route_counts.columns or "O-R-D" not in route_counts.columns:
                    continue
                usage_series = route_counts[["O-L-D", "O-R-D"]].min(axis=1)
            else:
                if "O-L-R-D" not in route_counts.columns:
                    continue
                usage_series = route_counts["O-L-R-D"]

            all_series.append(usage_series)

        if all_series:
            mean_vals, se_vals = compute_mean_se(all_series)
            label_str = labels[i - 1] if i - 1 < len(labels) else f"Folder {i}"
            ax.plot(mean_vals.index, mean_vals, label=label_str)
            ax.fill_between(mean_vals.index, mean_vals - se_vals, mean_vals + se_vals, alpha=0.2)

    ylabel = "Mean of min(#O-L-D, #O-R-D)" if game == "A" else "Mean #O-L-R-D"

    plt.xlabel("Round")
    plt.ylabel(ylabel)
    ax.set_xlim(0, 41)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40])
    plt.grid(True)

    add_side_labels(ax)  # Add labels to the right side

    os.makedirs(output_folder, exist_ok=True)
    plot_filename = "average_min_OLD_ORD_gameA.png" if game == "A" else "average_OLRD_gameB.png"
    plt.savefig(os.path.join(output_folder, plot_filename), facecolor='white', bbox_inches='tight')
    plt.close()




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
