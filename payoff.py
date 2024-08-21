from langchain_core.tools import tool
from network import TrafficNetwork

def create_payoff_func(num_agents, network: TrafficNetwork):
    @tool
    def calculate_payoff(route_distribution: dict, chosen_route: str) -> int:
        """
        Calculate the payoff for choosing a specific route given a distribution of agents on routes. You may try any distribution of agents you think is likely to occur.

        Args:
        route_distribution (dict): A dictionary where keys are route names and values are the number of agents on that route. The total number of agents across all routes must equal the number of participats in the experiment, including yourself. Here is an example of the formatting where 7 players choose O-A-D and 11 players choose O-B-D: {"O-A-D": 7, "O-B-D": 11}
        chosen_route (str): The route you are testing, whose payoff will be calculated. 
        """
        
        # Check if the chosen route has at least one player
        if chosen_route not in route_distribution or route_distribution[chosen_route] < 1:
            raise ValueError(f"The chosen route '{chosen_route}' must have at least one player.")

        # Ensure the total number of agents equals num_agents
        total_agents = sum(route_distribution.values())
        if total_agents != num_agents:
            raise ValueError(f"Total number of agents {total_agents} does not equal num_agents {num_agents}.")

        # Save the original player counts to restore later
        original_player_counts = {
            edge: data["players"] for edge, data in network.graph.edges.items()
        }

        network.reset_player_counts()

        # Simulate the distribution of players across the network
        for route, player_count in route_distribution.items():
            edges = network.path_to_edges(route)
            for _ in range(player_count):
                network.add_player_to_path(edges)

        # Calculate the total cost for the chosen route
        edges = network.path_to_edges(chosen_route)
        total_cost = network.calculate_total_cost(edges)

        # Calculate the payoff
        payoff = 400 - total_cost

        # Restore the original player counts
        for edge, players in original_player_counts.items():
            network.graph.edges[edge]["players"] = players

        return payoff
    
    return calculate_payoff
