import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau, wilcoxon
import numpy as np
import pandas as pd
import glob
import os

# Define expected route frequencies
EXPECTED_FREQUENCIES = {
    "game_A": {"O-L-D": 9, "O-R-D": 9},
    "game_B": {"O-L-D": 0, "O-R-D": 0, "O-L-R-D": 18}
}

# Define game labels
GAME_LABELS = {
    "game_1": "F-APO",
    "game_2": "S-APO",
    "game_3": "F-AR",
    "game_4": "S-AR",
    "game_5": "F-AP",
    "game_6": "S-AP"
}


def compute_mean_metric(input_folder, metric="Regret", ordering=None, game="B"):
    """
    Computes the mean value of a given metric for each game condition.

    Args:
        input_folder (str): Path to the folder containing multiple game subfolders (game_1, game_2, ...).
        metric (str): The metric to compute ('Regret', 'Payoff', or 'Switches').
        ordering (list): The explicit ordering of conditions.
        game (str): Either "A" or "B" to indicate which game to process.

    Returns:
        dict: Mapping of condition names (e.g., "F-APO") to mean metric values.
    """
    game_labels = {
        "game_1": "F-APO",
        "game_2": "S-APO",
        "game_3": "F-AR",
        "game_4": "S-AR",
        "game_5": "F-AP",
        "game_6": "S-AP"
    }

    mean_metrics = {}

    for game_folder, label in game_labels.items():
        game_path = os.path.join(input_folder, game_folder)
        run_folders = sorted(glob.glob(os.path.join(game_path, 'run *')))
        
        all_values = []

        for run_folder in run_folders:
            game_csv_path = os.path.join(run_folder, f"{game_folder}{game}", f"{game_folder}{game}.csv")
            if not os.path.exists(game_csv_path):
                print(f"Skipping missing file: {game_csv_path}")
                continue

            df = pd.read_csv(game_csv_path)

            if metric == "Switches":
                all_values.append(count_total_switches(df))  # Count total switches
            else:
                all_values.append(df[metric].mean())  # Compute mean metric value

        if all_values:
            mean_metrics[label] = np.mean(all_values)

    if ordering is not None:
        mean_metrics = {label: mean_metrics[label] for label in ordering}

    print(f"Mean metrics for Game {game}: {mean_metrics}")
    return mean_metrics


def create_metric_difference_matrix(input_folder, metric="Regret", ordering=None, game="B"):
    """
    Constructs a **fully populated** difference matrix for a selected metric.

    Args:
        input_folder (str): Path to the folder containing all game_x subfolders.
        metric (str): The metric to analyze ('Regret', 'Payoff', or 'Switches').
        ordering (list): The explicit ordering of conditions.
        game (str): Either "A" or "B" to indicate which game to process.

    Returns:
        pd.DataFrame: A matrix where each cell is (row metric - column metric).
    """
    if ordering is None:
        raise ValueError("Ordering must be specified for consistency.")

    mean_metrics = compute_mean_metric(input_folder, metric, ordering, game)

    # Initialize empty matrix
    matrix = np.full((len(ordering), len(ordering)), np.nan)

    # Compute row-wise difference for both upper and lower triangles
    for i, row_label in enumerate(ordering):
        for j, col_label in enumerate(ordering):
            if i != j:  # Fill all cells except diagonal
                matrix[i, j] = mean_metrics[row_label] - mean_metrics[col_label]

    # Convert to DataFrame with explicit ordering
    df = pd.DataFrame(matrix, index=ordering, columns=ordering)

    return df


def plot_metric_difference_matrix(metric_matrix, metric="Regret", ordering=None, game="B", output_path="metric_matrix.png"):
    """
    Creates a heatmap visualization for a **fully populated** metric difference matrix.

    Args:
        metric_matrix (pd.DataFrame): A DataFrame containing differences (row - column).
        metric (str): The metric being visualized ('Regret', 'Payoff', or 'Switches').
        ordering (list): The explicit ordering of conditions.
        game (str): Either "A" or "B" to indicate which game to process.
        output_path (str): Path to save the plot.
    """
    if ordering is None:
        raise ValueError("Ordering must be specified for consistency.")

    # Define color scale: Blue (high values) → White (neutral) → Red (low values)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)  # Reversed gradient

    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(
        metric_matrix,
        annot=True, fmt=".2f", cmap=cmap, center=0,
        linewidths=0.5, linecolor='black',
        cbar_kws={'label': f'{metric} Difference'}
    )

    # Add diagonal 'X' marks to indicate self-comparison
    for i in range(len(metric_matrix)):
        ax.text(i + 0.5, i + 0.35, 'x', fontsize=12, color='black', ha='center', va='center', fontweight='bold')

    # Improve readability
    # plt.title(f"Game {game}: Difference in Mean {metric}", fontsize=14, fontweight='bold')
    plt.xlabel("")
    plt.ylabel("")

    # Save figure
    output_path = output_path.replace(".png", f"_game_{game}.png")  # Differentiate Game A/B plots
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def count_total_switches(df):
    """
    Counts the total number of switches across all agents and rounds.

    Args:
        df (pd.DataFrame): DataFrame containing 'Round', 'Agent', and 'Route'.

    Returns:
        int: Total number of switches.
    """
    total_switches = 0
    rounds = sorted(df['Round'].unique())

    for round_num in rounds[1:]:
        prev_round = df[df['Round'] == round_num - 1].set_index('Agent')['Route']
        current_round = df[df['Round'] == round_num].set_index('Agent')['Route']
        agents = prev_round.index.intersection(current_round.index)
        total_switches += (current_round.loc[agents] != prev_round.loc[agents]).sum()

    return total_switches


def compute_kendalls_tau(input_folder, ordering):
    """
    Computes Kendall's Tau correlation for each representation across runs and games.

    Args:
        input_folder (str): Path to the folder containing all game_x subfolders.
        ordering (list): List specifying the ordering of representations.

    Returns:
        dict: Mapping of each representation to a list of Kendall's Tau values for game_A and game_B.
    """
    game_tau_values = {label: {"game_A": [], "game_B": []} for label in ordering}

    for game_folder, game_representation_label in GAME_LABELS.items():
        if game_representation_label not in ordering:
            continue

        game_path = os.path.join(input_folder, game_folder)
        run_folders = sorted(glob.glob(os.path.join(game_path, 'run *')))

        for run_folder in run_folders:
            game_folder_name = os.path.basename(game_path)

            # Construct paths for Game A and Game B
            game_A_path = os.path.join(run_folder, f'{game_folder_name}A', f'{game_folder_name}A.csv')
            game_B_path = os.path.join(run_folder, f'{game_folder_name}B', f'{game_folder_name}B.csv')

            for game_label_key, game_file_path in [("game_A", game_A_path), ("game_B", game_B_path)]:
                if not os.path.exists(game_file_path):
                    print(f"Skipping missing file: {game_file_path}")
                    continue

                df = pd.read_csv(game_file_path)

                if game_label_key not in EXPECTED_FREQUENCIES:
                    print(f"Skipping {game_label_key} in {run_folder}: Missing expected frequencies.")
                    continue

                expected_counts = EXPECTED_FREQUENCIES[game_label_key]
                round_numbers = []
                deviation_scores = []

                for round_num in sorted(df["Round"].unique()):
                    round_numbers.append(round_num)

                    # Count observed route choices
                    observed_counts = df[df["Round"] == round_num]["Route"].value_counts().to_dict()

                    # Compute deviation score
                    deviation_score = sum(
                        abs(observed_counts.get(route, 0) - expected_counts.get(route, 0))
                        for route in expected_counts
                    )
                    deviation_scores.append(deviation_score)

                # Skip if all deviation scores are the same
                if len(set(deviation_scores)) == 1:
                    print(f"Skipping NaN tau for {game_label_key} in {run_folder}: Constant deviation scores.")
                    continue

                # Compute Kendall's Tau correlation
                if len(round_numbers) > 1:
                    tau, _ = kendalltau(round_numbers, deviation_scores)
                    if not np.isnan(tau):
                        game_tau_values[game_representation_label][game_label_key].append(tau)
                    else:
                        print(f"Skipping NaN tau for {game_label_key} in {run_folder}")

    return game_tau_values


def create_mean_difference_matrix(game_tau_values, game_type, ordering):
    """
    Constructs a mean difference matrix for Kendall's Tau correlations.

    Args:
        game_tau_values (dict): Mapping of game representations to lists of Kendall's Tau values per run.
        game_type (str): Either "game_A" or "game_B".
        ordering (list): The explicit ordering of conditions.

    Returns:
        pd.DataFrame: A matrix of Kendall's Tau mean differences.
    """
    matrix = np.zeros((len(ordering), len(ordering)))

    for i, row_label in enumerate(ordering):
        for j, col_label in enumerate(ordering):
            if i != j:
                # Compute mean difference by first computing per-run differences
                common_length = min(len(game_tau_values[row_label][game_type]), len(game_tau_values[col_label][game_type]))
                
                if common_length > 0:
                    per_run_differences = [
                        game_tau_values[row_label][game_type][k] - game_tau_values[col_label][game_type][k]
                        for k in range(common_length)
                    ]
                    matrix[i, j] = np.mean(per_run_differences)  # Mean of per-run differences
                else:
                    matrix[i, j] = np.nan  # No valid data to compare

    return pd.DataFrame(matrix, index=ordering, columns=ordering)

def plot_kendall_tau_matrix(matrix, ordering, metric="Kendall's Tau", output_path="kendall_tau_matrix.png"):
    """
    Creates a heatmap visualization for Kendall's Tau difference matrix, leaving the diagonal uncolored.

    Args:
        matrix (pd.DataFrame): A DataFrame containing differences (row - column).
        ordering (list): The explicit ordering of conditions.
        metric (str): The metric being visualized.
        output_path (str): Path to save the plot.
    """
    if ordering is None:
        raise ValueError("Ordering must be specified for consistency.")

    # Create a mask to hide the diagonal
    mask = np.eye(len(matrix), dtype=bool)

    # Define color scale: Blue (low values) → White (neutral) → Red (high values)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)  

    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(
        matrix,
        annot=True, fmt=".2f", cmap=cmap, center=0,
        linewidths=0.5, linecolor='black',
        cbar_kws={'label': 'Difference in Mean Correlation'},
        mask=mask  # Hide diagonal
    )

    # Add diagonal 'X' marks to indicate self-comparison
    for i in range(len(matrix)):
        ax.text(i + 0.5, i + 0.35, 'x', fontsize=12, color='black', ha='center', va='center', fontweight='bold')

    # Improve readability
    # plt.title(f"Game B: Difference in Mean {metric}", fontsize=14, fontweight='bold')
    plt.xlabel("")
    plt.ylabel("")

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def compute_wilcoxon_p_matrix(game_tau_values, game_type, ordering):
    """
    Computes a Wilcoxon signed-rank test p-value matrix for Kendall's Tau differences.

    Args:
        game_tau_values (dict): Mapping of game representations to lists of Kendall's Tau values per run.
        game_type (str): Either "game_A" or "game_B".
        ordering (list): The explicit ordering of conditions.

    Returns:
        pd.DataFrame: A matrix of Wilcoxon signed-rank test p-values.
    """
    matrix = np.full((len(ordering), len(ordering)), np.nan)

    for i, row_label in enumerate(ordering):
        for j, col_label in enumerate(ordering):
            if i != j:
                # Get Kendall's Tau values for both representations
                tau_1 = game_tau_values[row_label][game_type]
                tau_2 = game_tau_values[col_label][game_type]

                print(f"Comparing {row_label} vs {col_label} for {game_type}")
                print(f"Tau values for {row_label}: {tau_1}")
                print(f"Tau values for {col_label}: {tau_2}")

                common_length = min(len(tau_1), len(tau_2))
                if common_length > 0:
                    tau_differences = [tau_1[k] - tau_2[k] for k in range(common_length)]
                    print(f"Tau differences: {tau_differences}")

                    # Perform Wilcoxon signed-rank test if valid
                    try:
                        _, p_value = wilcoxon(tau_differences, alternative="two-sided")
                        print(f"Wilcoxon p-value: {p_value}\n")
                    except ValueError:
                        p_value = np.nan  # Occurs if all values are the same
                        print("Skipping Wilcoxon test due to constant differences.\n")

                    matrix[i, j] = p_value

    return pd.DataFrame(matrix, index=ordering, columns=ordering)


def plot_p_value_heatmap(p_matrix, game_type, ordering, output_path="p_value_matrix.png"):
    """
    Plots a heatmap of Wilcoxon p-values for statistical significance.

    Args:
        p_matrix (pd.DataFrame): Matrix containing p-values from the Wilcoxon test.
        game_type (str): Either "game_A" or "game_B".
        ordering (list): The explicit ordering of conditions.
        output_path (str): File path to save the figure.
    """
    plt.figure(figsize=(10, 7))
    
    # Use a log scale to emphasize small p-values
    mask = np.triu(np.ones_like(p_matrix, dtype=bool))
    sns.heatmap(
        p_matrix,
        annot=True, fmt=".3f", cmap="coolwarm", center=0.05,
        linewidths=0.5, linecolor="black",
        cbar_kws={"label": "Wilcoxon p-value"},
        mask=mask  # Hide upper triangle
    )

    plt.title(f"Game {game_type[-1]}: Wilcoxon p-values for Kendall's Tau Differences", fontsize=14, fontweight='bold')
    plt.xlabel("")
    plt.ylabel("")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_bar_with_error_bars(game_tau_values, ordering, game_type, output_folder="output_plots"):
    """
    Plots and saves a bar chart with error bars for Kendall's Tau correlations across representations.

    Args:
        game_tau_values (dict): Mapping of game representations to lists of Kendall's Tau values per run.
        ordering (list): List of representations in the desired order.
        game_type (str): Either "game_A" or "game_B".
        output_folder (str): Directory to save the plots.
    """
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Extract means and standard deviations
    means = [np.mean(game_tau_values[label][game_type]) for label in ordering]
    std_devs = [np.std(game_tau_values[label][game_type]) for label in ordering]

    # Create bar plot with error bars
    plt.figure(figsize=(10, 6))
    plt.bar(ordering, means, yerr=std_devs, capsize=5, color='cornflowerblue', alpha=0.7, edgecolor='black')

    # Add labels and title
    plt.axhline(y=0, color='black', linewidth=1)  # Ensure zero-line is visible
    plt.ylabel("Mean Correlation Coefficent")
    plt.xlabel("Representation")
    # plt.title(f"Kendall’s Tau Correlations for {game_type.replace('_', ' ').title()}")
    
    # Adjust formatting
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle="--", alpha=0.5)

    # Save the plot
    output_path = os.path.join(output_folder, f"{game_type}_tau_barplot.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Saved plot to {output_path}")

    # Show plot
    plt.show()




# # Example Usage
input_folder = "."  # Change this to your actual folder path

# # Define ordering explicitly
ordering = ["F-APO", "F-AP", "S-APO", "S-AP", "F-AR", "S-AR"]
# ordering = ["F-APO", "F-AP", "F-AR", "S-APO", "S-AP", "S-AR"]
# expected_frequencies = EXPECTED_FREQUENCIES

# # Compute mean Kendall’s Tau per game representation
# game_tau_values = compute_kendalls_tau(input_folder, ordering)

# # Example usage:
# plot_bar_with_error_bars(game_tau_values, ordering, "game_A", input_folder)
# plot_bar_with_error_bars(game_tau_values, ordering, "game_B", input_folder)


# # Compute Wilcoxon signed-rank p-value matrices for both game A and game B
# p_matrix_A = compute_wilcoxon_p_matrix(game_tau_values, "game_A", ordering)
# p_matrix_B = compute_wilcoxon_p_matrix(game_tau_values, "game_B", ordering)

# # Plot heatmaps of statistical significance
# plot_p_value_heatmap(p_matrix_A, "game_A", ordering, output_path="game_A_p_values.png")
# plot_p_value_heatmap(p_matrix_B, "game_B", ordering, output_path="game_B_p_values.png")


# # Generate and plot matrices for Game A and Game B
# for game_type in ["game_A", "game_B"]:
#     tau_matrix = create_mean_difference_matrix(mean_tau_values, game_type, ordering)
#     plot_kendall_tau_matrix(tau_matrix, ordering, metric="Kendall's Tau", output_path=f"{game_type}_tau_matrix.png")


# Generate and plot for different metrics and for both Game A and Game B
for game in ["A", "B"]:
    for metric in ["Regret", "Payoff", "Switches"]:
        metric_matrix = create_metric_difference_matrix(input_folder, metric, ordering, game)
        plot_metric_difference_matrix(metric_matrix, metric, ordering, game, output_path=f"{metric.lower()}_matrix.png")