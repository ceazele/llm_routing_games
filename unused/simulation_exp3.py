import networkx as nx
import math
import csv
import random

# Helper functions for EXP3
def distr(weights, gamma=0.0):
    theSum = float(sum(weights))
    return tuple((1.0 - gamma) * (w / theSum) + (gamma / len(weights)) for w in weights)

def draw(probabilityDistribution):
    return random.choices(range(len(probabilityDistribution)), weights=probabilityDistribution)[0]

# Generate the available routes from the graph
def get_avail_routes(G):
    paths = list(nx.all_simple_paths(G, source='O', target='D'))
    formatted_paths = ['-'.join(path) for path in paths]
    return formatted_paths

# Cost functions for the network
def cost_OA(flow):
    return 10 * flow

def cost_BD(flow):
    return 10 * flow

def cost_AD(flow):
    return 210

def cost_OB(flow):
    return 210

def cost_AB(flow):
    return 0

# Create the network graph
def create_network(has_bridge):
    G = nx.DiGraph()
    G.add_nodes_from(['O', 'A', 'B', 'D'])
    G.add_edge('O', 'A', cost_func=cost_OA, cost_formula="10 * X", players=0)
    G.add_edge('B', 'D', cost_func=cost_BD, cost_formula="10 * X", players=0)
    G.add_edge('A', 'D', cost_func=cost_AD, cost_formula="210", players=0)
    G.add_edge('O', 'B', cost_func=cost_OB, cost_formula="210", players=0)
    if has_bridge:
        G.add_edge('A', 'B', cost_func=cost_AB, cost_formula="0", players=0)
    return G

# Helper functions for updating player paths
def path_to_edges(path):
    nodes = path.split('-')
    return [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]

def add_player_to_path(path, G):
    for edge in path:
        G.edges[edge]["players"] += 1

def calculate_total_cost(path, G):
    total_cost = 0
    for edge in path:
        edge_data = G.edges[edge]
        cost_func = edge_data["cost_func"]
        players = edge_data["players"]
        total_cost += cost_func(players)
    return total_cost

def reset_player_counts(G):
    for edge in G.edges():
        G.edges[edge]["players"] = 0

# Function to run the simulation with EXP3 agents
def run_exp3_simulation(has_bridge, num_rounds, num_agents, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "Agent", "Route", "Cost"])
        
        # Create the network graph and get available routes
        G = create_network(has_bridge)
        avail_routes = get_avail_routes(G)
        num_routes = len(avail_routes)
        
        gamma = 0.5
        exp3_agents = [{'weights': [1.0] * num_routes, 'history': []} for _ in range(num_agents)]
        
        for round_num in range(1, num_rounds + 1):
            print(f"Round {round_num}")
            # Initialize the route distribution as a dictionary
            route_distribution = {route: 0 for route in avail_routes}
            chosen_routes = []

            # Each agent selects a route using their independent EXP3 algorithm
            for agent_id, agent_data in enumerate(exp3_agents):
                probability_distribution = distr(agent_data['weights'], gamma)
                choice_index = draw(probability_distribution)
                chosen_route = avail_routes[choice_index]  # Map index to route name
                chosen_routes.append(chosen_route)
                route_distribution[chosen_route] += 1  # Update the dictionary count
                add_player_to_path(path_to_edges(chosen_route), G)  # Update the network with the agent's choice

            # Calculate costs and rewards after all agents have made their choices
            rewards = []
            for agent_id, chosen_route in enumerate(chosen_routes):
                path_edges = path_to_edges(chosen_route)
                total_cost = calculate_total_cost(path_edges, G)
                payoff = 400 - total_cost  # Higher payoffs for lower costs
                
                # Normalize the reward to [0, 1]
                normalized_reward = (payoff - 10) / (390 - 10)
                rewards.append(normalized_reward)

            # Update weights for each agent using the normalized rewards
            for agent_id, chosen_route in enumerate(chosen_routes):
                agent_data = exp3_agents[agent_id]
                probability_distribution = distr(agent_data['weights'], gamma)
                choice_index = avail_routes.index(chosen_route)  # Find the index for the chosen route
                estimated_reward = rewards[agent_id] / probability_distribution[choice_index]
                agent_data['weights'][choice_index] *= math.exp(estimated_reward * gamma / num_routes)
                
                # Log agent's choice and the resulting cost
                writer.writerow([round_num, agent_id, chosen_route, total_cost])
                print(f"Agent {agent_id} chose route {chosen_route} with cost {total_cost}")

            reset_player_counts(G)

print("Starting EXP3 simulations...")
run_exp3_simulation(True, 1000, 18, 'exp3_routes_bridge_1000_18.csv')
print("Simulations completed and routes saved")
