import networkx as nx
import matplotlib.pyplot as plt

def build_decision_tree():
    """
    Build the decision tree for the history representation configurations.
    Returns:
        G (nx.DiGraph): The decision tree graph.
    """
    G = nx.DiGraph()

    # Add nodes and edges for decision points
    G.add_node("Root", subset=0)
    G.add_node("Choice Representation", subset=1)
    G.add_node("My Choice", subset=2)
    G.add_node("Everyone's Choices", subset=2)
    G.add_node("Payoff Representation", subset=1)
    G.add_node("My Payoff", subset=2)
    G.add_node("Everyone's Payoff", subset=2)
    G.add_node("Regret Representation", subset=1)
    G.add_node("My Regret", subset=2)
    G.add_node("Everyone's Regret", subset=2)

    # Example branches
    G.add_node("Include My Choice?", subset=3)
    G.add_node("Include Everyone's Choices?", subset=3)
    G.add_node("Include My Payoff?", subset=3)
    G.add_node("Include Everyone's Payoff?", subset=3)
    G.add_node("Include My Regret?", subset=3)
    G.add_node("Include Everyone's Regret?", subset=3)

    # Connect nodes
    G.add_edges_from([
        ("Root", "Choice Representation"),
        ("Root", "Payoff Representation"),
        ("Root", "Regret Representation"),
        ("Choice Representation", "My Choice"),
        ("Choice Representation", "Everyone's Choices"),
        ("Payoff Representation", "My Payoff"),
        ("Payoff Representation", "Everyone's Payoff"),
        ("Regret Representation", "My Regret"),
        ("Regret Representation", "Everyone's Regret"),
        ("My Choice", "Include My Choice?"),
        ("Everyone's Choices", "Include Everyone's Choices?"),
        ("My Payoff", "Include My Payoff?"),
        ("Everyone's Payoff", "Include Everyone's Payoff?"),
        ("My Regret", "Include My Regret?"),
        ("Everyone's Regret", "Include Everyone's Regret?")
    ])

    return G

def visualize_tree():
    """
    Visualize the decision tree using NetworkX and Matplotlib.
    """
    G = build_decision_tree()

    # Ensure subset attribute is properly set for multipartite layout
    for node, data in G.nodes(data=True):
        if "subset" not in data:
            raise ValueError(f"Node {node} is missing the 'subset' attribute.")

    # Generate the multipartite layout
    pos = nx.multipartite_layout(G, subset_key="subset")

    # Draw the graph
    plt.figure(figsize=(12, 8))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color="lightblue",
        node_size=3000,
        font_size=10,
        font_weight="bold",
        edge_color="gray",
        arrowsize=10
    )
    plt.title("Decision Tree for History Representation Configurations")
    plt.show()

if __name__ == "__main__":
    visualize_tree()