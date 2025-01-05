import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import glob


# Function to count switches between rounds
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

# Function to compute mean and standard error across runs
def compute_mean_se(data_list):
    df = pd.concat(data_list, axis=1)
    mean = df.mean(axis=1)
    se = df.std(axis=1, ddof=1) / np.sqrt(len(data_list))
    return mean, se

# Function to process data and extract relevant statistics
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

        # Read data for game A (No Bridge)
        no_bridge_df = pd.read_csv(game_A_path)
        valid_routes_no_bridge.update(no_bridge_df['Route'].unique())

        # Process game A
        no_bridge_routes = no_bridge_df.groupby(['Round', 'Route']).size().unstack().fillna(0)
        no_bridge_routes_list.append(no_bridge_routes)
        no_bridge_payoff = no_bridge_df.groupby('Round')['Payoff'].mean()
        no_bridge_payoff_list.append(no_bridge_payoff)
        no_bridge_switches = count_switches(no_bridge_df)
        no_bridge_switches_list.append(no_bridge_switches)
        no_bridge_regret = no_bridge_df.groupby('Round')['Regret'].mean()
        no_bridge_regret_list.append(no_bridge_regret)

        # Read data for game B (With Bridge)
        with_bridge_df = pd.read_csv(game_B_path)
        valid_routes_with_bridge.update(with_bridge_df['Route'].unique())

        # Process game B
        with_bridge_routes = with_bridge_df.groupby(['Round', 'Route']).size().unstack().fillna(0)
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


# Function to plot routes for Game A (No Bridge)
def plot_routes_no_bridge(data, output_folder):
    no_bridge_routes_all = pd.concat(data["no_bridge_routes"], keys=range(len(data["no_bridge_routes"])), axis=1)

    no_bridge_routes_mean = (
        no_bridge_routes_all
        .T
        .groupby(level=1)
        .mean()
        .T
    )
    no_bridge_routes_se = (
        no_bridge_routes_all
        .T
        .groupby(level=1)
        .apply(lambda x: x.std(ddof=1) / np.sqrt(len(data["no_bridge_routes"])))
        .T
    )

    plt.figure(figsize=(12, 6))
    for route in data["valid_routes_no_bridge"]:
        if route in no_bridge_routes_mean.columns:
            mean_values = no_bridge_routes_mean[route]
            se_values = no_bridge_routes_se[route]
            rounds = no_bridge_routes_mean.index

            plt.plot(rounds, mean_values, marker='o', label=route)
            plt.fill_between(rounds, mean_values - se_values, mean_values + se_values, alpha=0.2)

    plt.xlabel('Round Number')
    plt.ylabel('Number of Subjects')
    plt.title('Number of Subjects per Route (No Bridge)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'no_bridge_routes.png'))
    plt.close()

# Function to plot routes for Game B (With Bridge)
def plot_routes_with_bridge(data, output_folder):
    with_bridge_routes_all = pd.concat(data["with_bridge_routes"], keys=range(len(data["with_bridge_routes"])), axis=1)

    with_bridge_routes_mean = (
        with_bridge_routes_all
        .T
        .groupby(level=1)
        .mean()
        .T
    )
    with_bridge_routes_se = (
        with_bridge_routes_all
        .T
        .groupby(level=1)
        .apply(lambda x: x.std(ddof=1) / np.sqrt(len(data["with_bridge_routes"])))
        .T
    )

    plt.figure(figsize=(12, 6))
    for route in data["valid_routes_with_bridge"]:
        if route in with_bridge_routes_mean.columns:
            mean_values = with_bridge_routes_mean[route]
            se_values = with_bridge_routes_se[route]
            rounds = with_bridge_routes_mean.index

            plt.plot(rounds, mean_values, marker='o', label=route)
            plt.fill_between(rounds, mean_values - se_values, mean_values + se_values, alpha=0.2)

    plt.xlabel('Round Number')
    plt.ylabel('Number of Subjects')
    plt.title('Number of Subjects per Route (With Bridge)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'with_bridge_routes.png'))
    plt.close()

# Function to plot payoff comparison between Game A and Game B
def plot_payoff_comparison(data, output_folder):
    no_bridge_payoff_mean, no_bridge_payoff_se = compute_mean_se(data["no_bridge_payoff"])
    with_bridge_payoff_mean, with_bridge_payoff_se = compute_mean_se(data["with_bridge_payoff"])

    plt.figure(figsize=(12, 6))
    plt.plot(no_bridge_payoff_mean.index, no_bridge_payoff_mean, marker='o', label='No Bridge')
    plt.fill_between(
        no_bridge_payoff_mean.index,
        no_bridge_payoff_mean - no_bridge_payoff_se,
        no_bridge_payoff_mean + no_bridge_payoff_se,
        alpha=0.2
    )

    plt.plot(with_bridge_payoff_mean.index, with_bridge_payoff_mean, marker='o', label='With Bridge')
    plt.fill_between(
        with_bridge_payoff_mean.index,
        with_bridge_payoff_mean - with_bridge_payoff_se,
        with_bridge_payoff_mean + with_bridge_payoff_se,
        alpha=0.2
    )

    plt.xlabel('Round Number')
    plt.ylabel('Mean Payoff')
    plt.title('Mean Payoff Comparison Between Experiments')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'payoff_comparison.png'))
    plt.close()

# Function to plot average regret for Game A and Game B
def plot_average_regret(data, output_folder):
    no_bridge_regret_mean, no_bridge_regret_se = compute_mean_se(data["no_bridge_regret"])
    with_bridge_regret_mean, with_bridge_regret_se = compute_mean_se(data["with_bridge_regret"])

    plt.figure(figsize=(12, 6))
    plt.plot(no_bridge_regret_mean.index, no_bridge_regret_mean, marker='o', label='No Bridge')
    plt.fill_between(
        no_bridge_regret_mean.index,
        no_bridge_regret_mean - no_bridge_regret_se,
        no_bridge_regret_mean + no_bridge_regret_se,
        alpha=0.2
    )

    plt.plot(with_bridge_regret_mean.index, with_bridge_regret_mean, marker='o', label='With Bridge')
    plt.fill_between(
        with_bridge_regret_mean.index,
        with_bridge_regret_mean - with_bridge_regret_se,
        with_bridge_regret_mean + with_bridge_regret_se,
        alpha=0.2
    )

    plt.xlabel('Round Number')
    plt.ylabel('Average Regret')
    plt.title('Average Regret Comparison Between Experiments')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'average_regret_comparison.png'))
    plt.close()

def plot_number_of_switches(data, output_folder):
    # Compute the mean and standard error of switches for both games
    no_bridge_switches_mean, no_bridge_switches_se = compute_mean_se(data["no_bridge_switches"])
    with_bridge_switches_mean, with_bridge_switches_se = compute_mean_se(data["with_bridge_switches"])

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(no_bridge_switches_mean.index, no_bridge_switches_mean, marker='o', label='No Bridge')
    plt.fill_between(
        no_bridge_switches_mean.index,
        no_bridge_switches_mean - no_bridge_switches_se,
        no_bridge_switches_mean + no_bridge_switches_se,
        alpha=0.2
    )

    plt.plot(with_bridge_switches_mean.index, with_bridge_switches_mean, marker='o', label='With Bridge')
    plt.fill_between(
        with_bridge_switches_mean.index,
        with_bridge_switches_mean - with_bridge_switches_se,
        with_bridge_switches_mean + with_bridge_switches_se,
        alpha=0.2
    )

    # Add labels, title, legend, and grid
    plt.xlabel('Round Number')
    plt.ylabel('Number of Switches')
    plt.title('Number of Switches Between Rounds (No Bridge vs. With Bridge)')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(os.path.join(output_folder, 'number_of_switches.png'))
    plt.close()



input_folder = 'game_4'

data = process_data(input_folder)
plot_routes_no_bridge(data, input_folder)
plot_routes_with_bridge(data, input_folder)
plot_payoff_comparison(data, input_folder)
plot_average_regret(data, input_folder)
plot_number_of_switches(data, input_folder)