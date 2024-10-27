import networkx as nx

# TrafficNetwork class to manage the traffic network graph
class TrafficNetwork:
    def __init__(self, has_bridge):
        self.graph = self.create_network(has_bridge)

    def create_network(self, has_bridge):
        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes
        G.add_nodes_from(['O', 'L', 'R', 'D'])

        # Add edges with cost functions as attributes
        G.add_edge('O', 'L', cost_func=self.cost_OA, cost_formula="10 * X", players=0)
        G.add_edge('R', 'D', cost_func=self.cost_BD, cost_formula="10 * X", players=0)
        G.add_edge('L', 'D', cost_func=self.cost_AD, cost_formula="210", players=0)
        G.add_edge('O', 'R', cost_func=self.cost_OB, cost_formula="210", players=0)
        if has_bridge:
            G.add_edge('L', 'R', cost_func=self.cost_AB, cost_formula="0", players=0)

        return G

    # Cost functions for each segment
    def cost_OA(self, flow):
        return 10 * flow

    def cost_BD(self, flow):
        return 10 * flow

    def cost_AD(self, flow):
        return 210

    def cost_OB(self, flow):
        return 210

    def cost_AB(self, flow):
        return 0

    def path_to_edges(self, path):
        nodes = path.split('-')
        return [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]

    def add_player_to_path(self, path):
        for edge in path:
            self.graph.edges[edge]["players"] += 1    

    def remove_player_from_path(self, path):
        for edge in path:
            self.graph.edges[edge]["players"] -= 1

    def calculate_total_cost(self, route):
        total_cost = 0
        for edge in route:
            edge_data = self.graph.edges[edge]
            cost_func = edge_data["cost_func"]
            players = edge_data["players"]
            total_cost += cost_func(players)
        return total_cost

    def reset_player_counts(self):
        for edge in self.graph.edges():
            self.graph.edges[edge]["players"] = 0

    def describe_graph(self):
        description = []
        description.append("Nodes:")
        nodes = " ".join(self.graph.nodes)
        description.append(nodes)
        description.append("Segments and associated costs:")
        for u, v, data in self.graph.edges(data=True):
            cost_formula = data['cost_formula']
            description.append(f"Segment {u}-{v}, cost function: {cost_formula}")
        return "\n".join(description)

    def get_avail_routes(self):
        paths = list(nx.all_simple_paths(self.graph, source='O', target='D'))
        return ['-'.join(path) for path in paths]
