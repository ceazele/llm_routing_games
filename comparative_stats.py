import matplotlib.pyplot as plt
from scipy.stats import levene
import numpy as np
import pandas as pd
import os
import glob
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression



def count_switches(df):
    """
    Counts the number of route switches between consecutive rounds for each agent.

    Args:
        df (pd.DataFrame): DataFrame with columns 'Round', 'Agent', and 'Route'.

    Returns:
        list: Number of switches per round (excluding the first round).
    """
    switches = []
    rounds = sorted(df['Round'].unique())
    for round_num in rounds[1:]:
        prev_round = df[df['Round'] == round_num - 1].set_index('Agent')['Route']
        current_round = df[df['Round'] == round_num].set_index('Agent')['Route']
        agents = prev_round.index.intersection(current_round.index)
        prev_round = prev_round.loc[agents]
        current_round = current_round.loc[agents]
        switches.append((current_round != prev_round).sum())
    return switches

def process_data(input_folder, metric):
    """
    Aggregates data from all rounds, trials, and agents into Game A and Game B for the given metric.

    Args:
        input_folder (str): Path to the input folder containing the runs.
        metric (str): The metric to aggregate (e.g., "Regret", "Payoff", or "Switches").

    Returns:
        dict: A dictionary containing aggregated data for Game A and Game B.
    """
    run_folders = sorted(glob.glob(os.path.join(input_folder, 'run *')))
    
    game_a_metric_data = []
    game_b_metric_data = []

    for run_folder in run_folders:
        # Construct paths to the subfolders
        game_a_subfolder = os.path.join(run_folder, os.path.basename(input_folder) + 'A')
        game_b_subfolder = os.path.join(run_folder, os.path.basename(input_folder) + 'B')
        
        # Construct full paths to the CSV files
        game_a_csv = os.path.join(game_a_subfolder, os.path.basename(input_folder) + 'A.csv')
        game_b_csv = os.path.join(game_b_subfolder, os.path.basename(input_folder) + 'B.csv')
        
        # Check if both files exist
        if not os.path.exists(game_a_csv) or not os.path.exists(game_b_csv):
            print(f"Missing data for {run_folder}. Skipping...")
            continue

        # Read Game A data
        game_a_df = pd.read_csv(game_a_csv)
        if metric == "Switches":
            game_a_metric_data.extend(count_switches(game_a_df))  # Collect switch data across all rounds
        else:
            game_a_metric_data.extend(game_a_df[metric])  # Collect individual values directly

        # Read Game B data
        game_b_df = pd.read_csv(game_b_csv)
        if metric == "Switches":
            game_b_metric_data.extend(count_switches(game_b_df))  # Collect switch data across all rounds
        else:
            game_b_metric_data.extend(game_b_df[metric])  # Collect individual values directly

    return {
        "game_a": game_a_metric_data,
        "game_b": game_b_metric_data
    }


def levene_test_by_game(input_folder_1, input_folder_2, metric="Regret"):
    """
    Runs Levene's test for comparing variances between two datasets across two games (A and B).

    Args:
        input_folder_1 (str): Path to the first input folder.
        input_folder_2 (str): Path to the second input folder.
        metric (str): The metric to compare ('Regret', 'Payoff', or 'Switches').

    Returns:
        None: Prints the results of Levene's test.
    """
    # Process data for both input folders
    data_1 = process_data(input_folder_1, metric)
    data_2 = process_data(input_folder_2, metric)

    # Extract the aggregated data for Game A and Game B
    game_a_data_1 = data_1["game_a"]
    game_a_data_2 = data_2["game_a"]
    game_b_data_1 = data_1["game_b"]
    game_b_data_2 = data_2["game_b"]

    # Run Levene's test for Game A
    game_a_stat, game_a_p_value = levene(game_a_data_1, game_a_data_2, center="median")

    # Run Levene's test for Game B
    game_b_stat, game_b_p_value = levene(game_b_data_1, game_b_data_2, center="median")

    # Print results
    print(f"--- LEVENE'S TEST RESULTS ---")
    print(f"Metric: {metric}")
    
    print(f"\nGAME A:")
    print(f"Levene’s test statistic: {game_a_stat:.4f}")
    print(f"p-value: {game_a_p_value:.4e}")
    if game_a_p_value < 0.05:
        print("Interpretation: Significant difference in variance for Game A")
    else:
        print("Interpretation: No significant difference in variance for Game A")

    print(f"\nGAME B:")
    print(f"Levene’s test statistic: {game_b_stat:.4f}")
    print(f"p-value: {game_b_p_value:.4e}")
    if game_b_p_value < 0.05:
        print("Interpretation: Significant difference in variance for Game B")
    else:
        print("Interpretation: No significant difference in variance for Game B")

    # # Print the variances for additional context
    # print("\nAdditional Debugging Information:")
    # print(f"Game A Variances: Folder 1: {pd.Series(game_a_data_1).var(ddof=1):.4f}, Folder 2: {pd.Series(game_a_data_2).var(ddof=1):.4f}")
    # print(f"Game B Variances: Folder 1: {pd.Series(game_b_data_1).var(ddof=1):.4f}, Folder 2: {pd.Series(game_b_data_2).var(ddof=1):.4f}")

        
def count_total_switches_by_game(input_folder):
    """
    Counts the total number of switches across all agents, rounds, and runs for both Game A and Game B.

    Args:
        input_folder (str): Path to the input folder containing the runs.

    Returns:
        dict: Total number of switches for Game A and Game B.
    """
    run_folders = sorted(glob.glob(os.path.join(input_folder, 'run *')))
    total_switches_a = 0
    total_switches_b = 0

    for run_folder in run_folders:
        # Construct paths to the subfolders
        game_a_subfolder = os.path.join(run_folder, os.path.basename(input_folder) + 'A')
        game_b_subfolder = os.path.join(run_folder, os.path.basename(input_folder) + 'B')
        
        # Construct full paths to the CSV files
        game_a_csv = os.path.join(game_a_subfolder, os.path.basename(input_folder) + 'A.csv')
        game_b_csv = os.path.join(game_b_subfolder, os.path.basename(input_folder) + 'B.csv')
        
        # Check if both files exist
        if not os.path.exists(game_a_csv) or not os.path.exists(game_b_csv):
            print(f"Missing data for {run_folder}. Skipping...")
            continue

        # Count switches for Game A
        game_a_df = pd.read_csv(game_a_csv)
        total_switches_a += sum(count_switches(game_a_df))

        # Count switches for Game B
        game_b_df = pd.read_csv(game_b_csv)
        total_switches_b += sum(count_switches(game_b_df))

    return {
        "game_a_total_switches": total_switches_a,
        "game_b_total_switches": total_switches_b
    }

# def plot_variances_by_round(data, metric="Regret", game_type="no_bridge"):
#     """
#     Plots the variance of a given metric for each round across the 40 rounds for the specified game type.

#     Args:
#         data (dict): Processed data dictionary from `process_data`.
#         metric (str): The metric to compute variances for ("Regret", "Payoff", "Switches").
#         game_type (str): Either "no_bridge" (Game A) or "with_bridge" (Game B).
#     """
#     # Determine which metric and game to use
#     if metric == "Regret":
#         data_list = data["no_bridge_regret"] if game_type == "no_bridge" else data["with_bridge_regret"]
#     elif metric == "Payoff":
#         data_list = data["no_bridge_payoff"] if game_type == "no_bridge" else data["with_bridge_payoff"]
#     elif metric == "Switches":
#         data_list = data["no_bridge_switches"] if game_type == "no_bridge" else data["with_bridge_switches"]
#     else:
#         raise ValueError("Invalid metric specified. Choose from 'Regret', 'Payoff', or 'Switches'.")

#     # Combine the data from all runs
#     combined_data = pd.concat(data_list, axis=1)

#     # Compute variance for each round across all runs
#     round_variances = combined_data.var(axis=1, ddof=1)

#     # Ensure that all 40 rounds are included
#     all_rounds = range(1, 41)  # Assuming the game always has 40 rounds
#     round_variances = round_variances.reindex(all_rounds, fill_value=0)

#     # Plot the variances
#     plt.figure(figsize=(12, 6))
#     plt.plot(round_variances.index, round_variances.values, marker='o', label=f"{game_type.capitalize()} - {metric}")
#     plt.xlabel("Round Number")
#     plt.ylabel("Variance")
#     plt.title(f"Variance of {metric} Across Rounds ({game_type.capitalize()})")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# rep_1 = 'game_1'
# rep_2 = 'game_2'
# levene_test_by_game(rep_1, rep_2, 'Regret')
# levene_test_by_game(rep_1, rep_2, 'Switches')
# levene_test_by_game(rep_1, rep_2, 'Payoff')

# print(compute_total_switches_by_game(rep_1))
# print(compute_total_switches_by_game(rep_2))






# # Process data for both input folders
# data_game_3 = process_data("game_3")
# data_game_4 = process_data("game_4")

# # Plot variances for Game A (No Bridge) and Regret metric
# plot_variances_by_round(data_game_3, metric="Regret", game_type="no_bridge")
# plot_variances_by_round(data_game_4, metric="Regret", game_type="no_bridge")

# # Optionally, plot variances for Game B (With Bridge) and Regret metric
# plot_variances_by_round(data_game_3, metric="Regret", game_type="with_bridge")
# plot_variances_by_round(data_game_4, metric="Regret", game_type="with_bridge")



# Extended function to compute variances and means
def compute_variances_and_means(input_folder, metric):
    run_folders = sorted(glob.glob(os.path.join(input_folder, 'run *')))
    
    game_a_data = []
    game_b_data = []

    for run_folder in run_folders:
        # Construct paths to the subfolders
        game_a_subfolder = os.path.join(run_folder, os.path.basename(input_folder) + 'A')
        game_b_subfolder = os.path.join(run_folder, os.path.basename(input_folder) + 'B')
        
        # Construct full paths to the CSV files
        game_a_csv = os.path.join(game_a_subfolder, os.path.basename(input_folder) + 'A.csv')
        game_b_csv = os.path.join(game_b_subfolder, os.path.basename(input_folder) + 'B.csv')
        
        # Check if both files exist
        if not os.path.exists(game_a_csv) or not os.path.exists(game_b_csv):
            print(f"Missing data for {run_folder}. Skipping...")
            continue

        # Read Game A data
        game_a_df = pd.read_csv(game_a_csv)
        if metric == "Switches":
            switches = count_switches(game_a_df)
            game_a_data.extend(switches)  # Collect switch data across all rounds
        else:
            game_a_data.extend(game_a_df[metric])  # Collect individual values

        # Read Game B data
        game_b_df = pd.read_csv(game_b_csv)
        if metric == "Switches":
            switches = count_switches(game_b_df)
            game_b_data.extend(switches)  # Collect switch data across all rounds
        else:
            game_b_data.extend(game_b_df[metric])  # Collect individual values

    # Compute means
    game_a_mean = pd.Series(game_a_data).mean() if game_a_data else None
    game_b_mean = pd.Series(game_b_data).mean() if game_b_data else None

    # Compute variances
    game_a_variance = pd.Series(game_a_data).var(ddof=1) if game_a_data else None  # Sample variance
    game_b_variance = pd.Series(game_b_data).var(ddof=1) if game_b_data else None  # Sample variance

    return {
        "game_a_mean": game_a_mean,
        "game_b_mean": game_b_mean,
        "game_a_variance": game_a_variance,
        "game_b_variance": game_b_variance
    }

# Example usage
game_1 = 'game_2' 
game_2 = 'game_6'

print("\n")
print("COMPARISON OF PAYOFF MEAN AND VARIANCE")
print("\n")

results_1 = compute_variances_and_means(game_1, 'Payoff')
results_2 = compute_variances_and_means(game_2, 'Payoff')
print("First Game A Mean:", results_1["game_a_mean"])
print("Second Game A Mean:", results_2["game_a_mean"])
print("First Game A Variance:", results_1["game_a_variance"])
print("Second Game A Variance:", results_2["game_a_variance"])
print("First Game B Mean:", results_1["game_b_mean"])
print("Second Game B Mean:", results_2["game_b_mean"])
print("First Game B Variance:", results_1["game_b_variance"])
print("Second Game B Variance:", results_2["game_b_variance"])
print("\n")
levene_test_by_game(game_1, game_2, 'Payoff')

print("\n")
print("COMPARISON OF REGRET MEAN AND VARIANCE")
print("\n")

results_1 = compute_variances_and_means(game_1, 'Regret')
results_2 = compute_variances_and_means(game_2, 'Regret')
print("First Game A Mean:", results_1["game_a_mean"])
print("Second Game A Mean:", results_2["game_a_mean"])
print("First Game A Variance:", results_1["game_a_variance"])
print("Second Game A Variance:", results_2["game_a_variance"])
print("First Game B Mean:", results_1["game_b_mean"])
print("Second Game B Mean:", results_2["game_b_mean"])
print("First Game B Variance:", results_1["game_b_variance"])
print("Second Game B Variance:", results_2["game_b_variance"])
print("\n")
levene_test_by_game(game_1, game_2, 'Regret')

print("\n")
print("COMPARISON OF TOTAL SWITCH COUNT")
print("\n")

game_1_switches = count_total_switches_by_game(game_1)
game_2_switches = count_total_switches_by_game(game_2)

print("First Game A Total Switches", game_1_switches['game_a_total_switches'])
print("Second Game A Total Switches", game_2_switches['game_a_total_switches'])
print("First Game B Total Switches", game_1_switches['game_b_total_switches'])
print("Second Game B Total Switches", game_2_switches['game_b_total_switches'])



def aggregate_regret_data(input_folder):
    run_folders = sorted(glob.glob(os.path.join(input_folder, "run *")))
    regret_data_a = []
    regret_data_b = []

    for run_folder in run_folders:
        # Construct paths to Game A and Game B subfolders and CSV files
        game_a_subfolder = os.path.join(run_folder, os.path.basename(input_folder) + 'A')
        game_b_subfolder = os.path.join(run_folder, os.path.basename(input_folder) + 'B')
        game_a_csv = os.path.join(game_a_subfolder, os.path.basename(input_folder) + 'A.csv')
        game_b_csv = os.path.join(game_b_subfolder, os.path.basename(input_folder) + 'B.csv')

        if not os.path.exists(game_a_csv) or not os.path.exists(game_b_csv):
            print(f"Missing CSV files in {run_folder}, skipping.")
            continue

        # Read Game A data
        game_a_df = pd.read_csv(game_a_csv)
        game_a_regrets = game_a_df.groupby("Round")["Regret"].mean()
        regret_data_a.append(game_a_regrets)

        # Read Game B data
        game_b_df = pd.read_csv(game_b_csv)
        game_b_regrets = game_b_df.groupby("Round")["Regret"].mean()
        regret_data_b.append(game_b_regrets)

    # Combine all runs and average across repetitions
    combined_a = pd.concat(regret_data_a, axis=1).mean(axis=1)
    combined_b = pd.concat(regret_data_b, axis=1).mean(axis=1)

    return combined_a, combined_b


# Exponential decay model
def exponential_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

# Function for log-linear regression
def log_linear_regression(x, y):
    # Fit a linear regression model to log-transformed x
    log_x = np.log(x + 1)  # Log-transform x with +1 to avoid log(0)
    model = LinearRegression()
    model.fit(log_x.reshape(-1, 1), y)
    predicted = model.predict(log_x.reshape(-1, 1))
    return model.coef_[0], model.intercept_, predicted

# Function to compare models
def compare_decay_models_log_linear(input_folder):
    # Aggregate data
    regret_a, regret_b = aggregate_regret_data(input_folder)

    for game, regrets in zip(["Game A", "Game B"], [regret_a, regret_b]):
        rounds = regrets.index.to_numpy()
        regrets_values = regrets.values

        # Fit exponential decay model
        try:
            popt_exp, _ = curve_fit(exponential_decay, rounds, regrets_values, maxfev=5000)
            exp_pred = exponential_decay(rounds, *popt_exp)
            exp_r2 = r2_score(regrets_values, exp_pred)
        except RuntimeError:
            popt_exp = [None, None, None]
            exp_r2 = float("nan")

        # Fit log-linear model
        slope, intercept, log_pred = log_linear_regression(rounds, regrets_values)
        log_r2 = r2_score(regrets_values, log_pred)

        # Print results for each game
        print(f"--- {game} ---")
        print(f"Exponential Decay Model: R² = {exp_r2:.4f}, Parameters: {popt_exp}")
        print(f"Log-Linear Model: R² = {log_r2:.4f}, Slope = {slope:.4f}, Intercept = {intercept:.4f}")

        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, regrets_values, "o", label="Original Data")
        if popt_exp[0] is not None:
            plt.plot(rounds, exp_pred, "-", label=f"Exponential Decay (R²={exp_r2:.4f})")
        plt.plot(rounds, log_pred, "--", label=f"Log-Linear Model (R²={log_r2:.4f})")
        plt.xlabel("Round Number")
        plt.ylabel("Average Regret")
        plt.title(f"Model Comparison: {game}")
        plt.legend()
        plt.show()

# compare_decay_models_log_linear('game_1')