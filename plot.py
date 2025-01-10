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

        no_bridge_routes = no_bridge_df.groupby(['Round', 'Route']).size().unstack(fill_value=0)


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

    plt.style.use("seaborn-v0_8")
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

    plt.style.use("seaborn-v0_8")
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

    plt.style.use("seaborn-v0_8")
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

    plt.style.use("seaborn-v0_8")
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
    plt.style.use("seaborn-v0_8")
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

def plot_average_reward_trends(game_folders, output_folder, game):
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(12, 6))

    for game_folder in game_folders:
        run_folders = sorted(glob.glob(os.path.join(game_folder, 'run *')))
        all_rewards = []

        for run_folder in run_folders:
            game_folder_name = os.path.basename(game_folder)
            game_csv_path = os.path.join(run_folder, f'{game_folder_name}{game}', f'{game_folder_name}{game}.csv')
            if not os.path.exists(game_csv_path):
                print(f"Data file not found in {run_folder}, skipping this run.")
                continue

            df = pd.read_csv(game_csv_path)
            average_rewards = df.groupby('Round')['Payoff'].mean()
            all_rewards.append(average_rewards)

        if all_rewards:
            rewards_mean, rewards_se = compute_mean_se(all_rewards)
            plt.plot(rewards_mean.index, rewards_mean, marker='o', label=f'{game_folder}')
            plt.fill_between(
                rewards_mean.index,
                rewards_mean - rewards_se,
                rewards_mean + rewards_se,
                alpha=0.2
            )

    plt.xlabel('Round Number')
    plt.ylabel('Average Reward')
    plt.title(f'Average Reward Trends Across Game {game}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, f'average_reward_trends_game_{game}.png'))
    plt.close()

def plot_average_regret_trends(game_folders, output_folder, game):
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(12, 6))

    for game_folder in game_folders:
        run_folders = sorted(glob.glob(os.path.join(game_folder, 'run *')))
        all_regrets = []

        for run_folder in run_folders:
            game_folder_name = os.path.basename(game_folder)
            game_csv_path = os.path.join(run_folder, f'{game_folder_name}{game}', f'{game_folder_name}{game}.csv')
            if not os.path.exists(game_csv_path):
                print(f"Data file not found in {run_folder}, skipping this run.")
                continue

            df = pd.read_csv(game_csv_path)
            average_regrets = df.groupby('Round')['Regret'].mean()
            all_regrets.append(average_regrets)

        if all_regrets:
            regrets_mean, regrets_se = compute_mean_se(all_regrets)
            plt.plot(regrets_mean.index, regrets_mean, marker='o', label=f'{game_folder}')
            plt.fill_between(
                regrets_mean.index,
                regrets_mean - regrets_se,
                regrets_mean + regrets_se,
                alpha=0.2
            )

    plt.xlabel('Round Number')
    plt.ylabel('Average Regret')
    plt.title(f'Average Regret Trends Across Game {game}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, f'average_regret_trends_game_{game}.png'))
    plt.close()


def compute_average_route_choices(input_folder):
    """
    Compute the average and standard deviation of route choices across all rounds
    and trials for both Game A and Game B.

    Args:
        input_folder (str): Path to the input folder containing `run *` subdirectories.

    Returns:
        dict: Containing dataframes for Game A and Game B statistics.
    """

    # Infer game name from the input folder name (e.g., "game_2" -> "game_2A.csv", "game_2B.csv")
    game_name = os.path.basename(input_folder)
    game_A_filename = f"{game_name}A.csv"
    game_B_filename = f"{game_name}B.csv"

    # Locate the paths for game_A.csv and game_B.csv
    game_A_paths = glob.glob(os.path.join(input_folder, "run *", game_name + "A", game_A_filename))
    game_B_paths = glob.glob(os.path.join(input_folder, "run *", game_name + "B", game_B_filename))

    if not game_A_paths or not game_B_paths:
        raise FileNotFoundError(f"Could not find {game_A_filename} or {game_B_filename} in the expected folder structure.")

    # Initialize lists to hold route choice data for each run
    game_A_route_choices_list = []
    game_B_route_choices_list = []

    # Process all runs for Game A
    for path in game_A_paths:
        game_A_df = pd.read_csv(path)
        # Group by 'Round' and 'Route', then count players choosing each route
        game_A_route_choices = game_A_df.groupby(['Round', 'Route']).size().unstack(fill_value=0)
        game_A_route_choices_list.append(game_A_route_choices)

    # Process all runs for Game B
    for path in game_B_paths:
        game_B_df = pd.read_csv(path)
        # Group by 'Round' and 'Route', then count players choosing each route
        game_B_route_choices = game_B_df.groupby(['Round', 'Route']).size().unstack(fill_value=0)
        game_B_route_choices_list.append(game_B_route_choices)

    # Combine data across all runs
    game_A_combined = pd.concat(game_A_route_choices_list).groupby(level=0).mean()
    game_A_avg = game_A_combined.mean(axis=0)
    game_A_std = game_A_combined.std(axis=0, ddof=1)

    game_B_combined = pd.concat(game_B_route_choices_list).groupby(level=0).mean()
    game_B_avg = game_B_combined.mean(axis=0)
    game_B_std = game_B_combined.std(axis=0, ddof=1)

    # Prepare DataFrames for Game A and Game B statistics
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

    # Save the statistics to CSV files
    output_folder = os.path.join(input_folder, "results")
    os.makedirs(output_folder, exist_ok=True)

    game_A_csv_path = os.path.join(output_folder, f"{game_name}_a_route_stats.csv")
    game_B_csv_path = os.path.join(output_folder, f"{game_name}_b_route_stats.csv")

    game_A_stats.to_csv(game_A_csv_path, index=False)
    game_B_stats.to_csv(game_B_csv_path, index=False)

    print(f"Results saved to:\n  {game_A_csv_path}\n  {game_B_csv_path}")

    return {"game_a": game_A_stats, "game_b": game_B_stats}

# for i in range (1,7):
#     input_folder = f"game_{i}"
#     results = compute_average_route_choices(input_folder)

# data = process_data(input_folder)
# plot_routes_no_bridge(data, input_folder)
# plot_routes_with_bridge(data, input_folder)
# plot_payoff_comparison(data, input_folder)
# plot_average_regret(data, input_folder)
# plot_number_of_switches(data, input_folder)
    
plot_average_reward_trends(['game_1', 'game_2', 'game_3', 'game_4', 'game_5', 'game_6'], '.', 'A')
plot_average_reward_trends(['game_1', 'game_2', 'game_3', 'game_4', 'game_5', 'game_6'], '.', 'B')

plot_average_regret_trends(['game_1', 'game_2', 'game_3', 'game_4', 'game_5', 'game_6'], '.', 'A')
plot_average_regret_trends(['game_1', 'game_2', 'game_3', 'game_4', 'game_5', 'game_6'], '.', 'B')