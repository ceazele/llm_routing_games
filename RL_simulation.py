import os
import csv
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# The network + regret code you already have
import networkx as nx
from operator import itemgetter
from network import TrafficNetwork
from regret import compute_regret_for_agents
from collections import Counter

###############################################################################
# SINGLE SIMULATION CODE (Scaled-Payoff MW/EXP3)
###############################################################################

def compute_probs(weights):
    wsum = sum(weights)
    if wsum == 0:
        return [1.0 / len(weights)] * len(weights)
    return [w / wsum for w in weights]

def sample_from_dist(prob_dist):
    r = random.random()
    cum = 0.0
    for i, p in enumerate(prob_dist):
        cum += p
        if r < cum:
            return i
    return len(prob_dist) - 1

def update_mw(agent, full_scaled_payoffs):
    eps = agent['epsilon']
    for i in range(len(agent['weights'])):
        agent['weights'][i] *= math.exp(eps * full_scaled_payoffs[i])

def update_exp3(agent, chosen_action, chosen_scaled_payoff):
    eps = agent['epsilon']
    p_chosen = agent['probs'][chosen_action]
    if p_chosen <= 0:
        return
    est_payoff = chosen_scaled_payoff / p_chosen
    agent['weights'][chosen_action] *= math.exp(eps * est_payoff)

def initialize_agents(num_agents, num_actions, algorithm, epsilon):
    agents = []
    for _ in range(num_agents):
        weights = [1.0] * num_actions
        agents.append({
            "algorithm": algorithm,
            "weights": weights,
            "epsilon": epsilon,
            "probs": [1.0 / num_actions] * num_actions
        })
    return agents

def run_simulation(
    num_agents,
    num_rounds,
    has_bridge,
    output_csv_path,
    algorithm="mw",
    epsilon=0.1
):
    """
    Run ONE repeated congestion-game simulation using scaled payoffs.
    Saves a CSV of round-by-round (Round, Agent, Route, Payoff, Regret).
    """
    # Build traffic network
    network = TrafficNetwork(has_bridge)
    available_routes = network.get_avail_routes()
    num_actions = len(available_routes)

    # Initialize agents
    agent_info = initialize_agents(num_agents, num_actions, algorithm, epsilon)
    all_agent_histories = [[] for _ in range(num_agents)]

    MAX_PAYOFF = 400.0
    route_to_idx = {r: i for i, r in enumerate(available_routes)}
    idx_to_route = {i: r for i, r in enumerate(available_routes)}

    for round_num in range(num_rounds):
        chosen_actions = [None] * num_agents

        # Each agent picks a route
        for i in range(num_agents):
            agent_info[i]['probs'] = compute_probs(agent_info[i]['weights'])
            action_idx = sample_from_dist(agent_info[i]['probs'])
            chosen_actions[i] = action_idx

        # Mark chosen routes
        for action_idx in chosen_actions:
            route_str = idx_to_route[action_idx]
            network.add_player_to_path(network.path_to_edges(route_str))

        # Compute actual payoffs + scaled payoffs for each route
        full_route_payoffs = []
        full_route_scaled_payoffs = []
        for r in available_routes:
            cost = network.calculate_total_cost(network.path_to_edges(r))
            payoff = MAX_PAYOFF - cost
            scaled = payoff / MAX_PAYOFF
            full_route_payoffs.append(payoff)
            full_route_scaled_payoffs.append(scaled)

        # Update weights
        for i in range(num_agents):
            action_idx = chosen_actions[i]
            chosen_scaled = full_route_scaled_payoffs[action_idx]
            if agent_info[i]['algorithm'] == "mw":
                update_mw(agent_info[i], full_route_scaled_payoffs)
            elif agent_info[i]['algorithm'] == "exp3":
                update_exp3(agent_info[i], action_idx, chosen_scaled)
            else:
                raise ValueError("Unknown algorithm")

        # Store results
        for i in range(num_agents):
            action_idx = chosen_actions[i]
            payoff = full_route_payoffs[action_idx]
            route_str = idx_to_route[action_idx]
            all_agent_histories[i].append({
                'round_num': round_num + 1,
                'decision': route_str,
                'payoff': payoff,
                'regret': None
            })

        # Compute regrets
        compute_regret_for_agents(all_agent_histories, round_num, network)
        network.reset_player_counts()

    # Write CSV
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["Round", "Agent", "Route", "Payoff", "Regret"])
        writer.writeheader()
        for agent_id, history in enumerate(all_agent_histories):
            for row in history:
                writer.writerow({
                    "Round": row['round_num'],
                    "Agent": agent_id,
                    "Route": row['decision'],
                    "Payoff": row['payoff'],
                    "Regret": row['regret']
                })
    print(f"One simulation done -> '{output_csv_path}'")


###############################################################################
# MULTIPLE RUNS + POST-PROCESSING & PLOT
###############################################################################

def multi_run_simulations(
    folder_path,
    num_sims,
    num_agents,
    num_rounds,
    has_bridge,
    algorithm="mw",
    epsilon=0.1
):
    """
    Runs the simulation 'num_sims' times, saving each CSV into 'folder_path'.
    Then aggregates the CSV data to plot a mean +/- std trend line of how many
    agents chose each route per round.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 1) Run multiple simulations, writing e.g. results_run_1.csv, etc.
    for i in range(num_sims):
        csv_name = f"results_run_{i+1}.csv"
        output_csv_path = os.path.join(folder_path, csv_name)
        print(f"\n--- Starting Simulation #{i+1} ---")
        run_simulation(
            num_agents=num_agents,
            num_rounds=num_rounds,
            has_bridge=has_bridge,
            output_csv_path=output_csv_path,
            algorithm=algorithm,
            epsilon=epsilon
        )

    print("\nAll simulations complete. Now aggregating results...")

    # 2) Aggregate the route choices from each CSV to compute average & std.
    #    We'll parse each run's CSV and count how many times each route is picked in each round.

    # route_counts_for_all_runs[route] = an array of shape (num_sims, num_rounds)
    # we'll fill it with the count of how many agents chose that route in each run/round.
    route_counts_for_all_runs = defaultdict(lambda: np.zeros((num_sims, num_rounds), dtype=int))

    # We'll parse each CSV in turn.
    for i in range(num_sims):
        csv_name = f"results_run_{i+1}.csv"
        csv_path = os.path.join(folder_path, csv_name)

        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                r = row['Route']
                rnd = int(row['Round']) - 1  # zero-based index for array
                route_counts_for_all_runs[r][i, rnd] += 1

    # 3) Plot the mean + std as shaded error bars
    import matplotlib.pyplot as plt

    rounds_axis = np.arange(1, num_rounds+1)

    plt.figure(figsize=(8, 6))
    for route, counts_array in route_counts_for_all_runs.items():
        # counts_array has shape (num_sims, num_rounds)
        mean_counts = counts_array.mean(axis=0)          # shape (num_rounds,)
        std_counts = counts_array.std(axis=0)           # shape (num_rounds,)

        # Plot the mean line
        plt.plot(rounds_axis, mean_counts, label=route)
        # Fill between mean ± std
        plt.fill_between(rounds_axis,
                         mean_counts - std_counts,
                         mean_counts + std_counts,
                         alpha=0.2)  # alpha for shading

    plt.title(f"Average # of Agents per Route (±1 std)\n"
              f"{num_sims} simulations, {num_rounds} rounds each")
    plt.xlabel("Round")
    plt.ylabel("# of Agents on Route")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # If you want to save the figure:
    # plt.savefig(os.path.join(folder_path, "aggregated_plot.png"))
    # Otherwise, just show it:
    plt.show()
    print("Aggregation & plotting complete.")


###############################################################################
# EXAMPLE USAGE
###############################################################################

if __name__ == "__main__":
    multi_run_simulations(
        folder_path="mwB",
        num_sims=50,
        num_agents=18,
        num_rounds=40,
        has_bridge=True,
        algorithm="mw",
        epsilon=0.75
    )
