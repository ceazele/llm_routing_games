def compute_regret_for_agents(all_agent_histories, round_num, network):
    """
    Computes the regret for each agent and updates their histories.

    Args:
        all_agent_histories: List of agent histories. Each agent's history is a list of dictionaries.
        round_num: The current round number (zero-indexed).
        network: The TrafficNetwork instance.
    """
    num_agents = len(all_agent_histories)

    for session_id in range(num_agents):
        # Reset the network
        network.reset_player_counts()

        # Add other agents' routes to the network
        for other_id in range(num_agents):
            if other_id != session_id:
                other_route = all_agent_histories[other_id][round_num]['decision']
                if other_route in network.get_avail_routes():
                    network.add_player_to_path(network.path_to_edges(other_route))

        # Compute potential costs for each possible route for the current agent
        potential_costs = {}
        for route in network.get_avail_routes():
            # Add the current agent to the route
            network.add_player_to_path(network.path_to_edges(route))
            cost = network.calculate_total_cost(network.path_to_edges(route))
            potential_costs[route] = cost
            # Remove the current agent from the route
            network.remove_player_from_path(network.path_to_edges(route))

        # Find the best response (minimum cost)
        best_route = min(potential_costs, key=potential_costs.get)
        best_cost = potential_costs[best_route]
        potential_payoff = 400 - best_cost

        # Get the actual payoff for the current round
        actual_payoff = all_agent_histories[session_id][round_num]['payoff']

        # Compute regret
        regret = potential_payoff - actual_payoff

        # Store the regret in the agent's history
        all_agent_histories[session_id][round_num]['regret'] = regret
