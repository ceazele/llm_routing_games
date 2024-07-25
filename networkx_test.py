import networkx as nx
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI
import json
import os
import csv

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCAHIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "bluesky"
os.environ["OPENAI_API_KEY"] = "sk-proj-2YIRzXSkGvD5mXF5KSwST3BlbkFJ32aEm9aV3nrL6gFGCr97"

model = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=1)

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)


store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# Specify the Pydantic format of output
class Route(BaseModel):
    route: str = Field(description="choice of route")

# Set up a parser
parser = PydanticOutputParser(pydantic_object=Route)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Imagine you are John Smith, a 40-year-old individual living in San Francisco, California.
You have been asked to participate in an experiment on route selection in traffic networks.
There are {num_agents} drivers, including yourself, who will be asked to choose a route to travel.
Each driver will choose one of {num_routes} routes to travel from the starting point O to the final destination D.
Here is the description of the network: 
{description}
Your total cost for a route is the sum of the costs incurred on each segment of the route, where x denotes the number of users (including you) who travel on that segment.
Your payoff per round will be 400 - cost.
All the drivers make their route choices independently of one another and leave point O at the same time.
""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


chain = prompt | model


# session_id is a string denoting an agent's conversation session id
def prompt_route(session_id, num_agents, num_routes, description, avail_routes, prev_routes, cost):
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="messages",
    )

    parsed_chain = with_message_history | parser

    config = {"configurable": {"session_id": f"session_{session_id}"}}
    format_instructions = parser.get_format_instructions()


    messages = [HumanMessage(f"The available routes are: {avail_routes}\n{format_instructions}")]
    if prev_routes != None and cost != None:
        messages.insert(0, HumanMessage(f"Last round, the number of players on each route was: {prev_routes}\nYour cost was {cost}"))

    response = parsed_chain.invoke(
        {"messages": messages, "num_agents": num_agents, "num_routes": num_routes, "description": description},
        config=config,
    )

    return response






# Define the cost functions
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


def create_network(has_bridge):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes
    G.add_nodes_from(['O', 'A', 'B', 'D'])

    # Add edges with cost functions as attributes
    G.add_edge('O', 'A', cost_func=cost_OA, cost_formula="10 * x", players=0)
    G.add_edge('B', 'D', cost_func=cost_BD, cost_formula="10 * x", players=0)
    G.add_edge('A', 'D', cost_func=cost_AD, cost_formula="210", players=0)
    G.add_edge('O', 'B', cost_func=cost_OB, cost_formula="210", players=0)
    if has_bridge:
        G.add_edge('A', 'B', cost_func=cost_AB, cost_formula="0", players=0)

    return G



def path_to_edges(path):
    nodes = path.split('-')
    edges = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
    return edges


def add_player_to_path(path, G):
    for edge in path:
        G.edges[edge]["players"] += 1    


def calculate_total_cost(route, G):
    total_cost = 0
    for edge in route:
        edge_data = G.edges[edge]
        cost_func = edge_data["cost_func"]
        players = edge_data["players"]
        total_cost += cost_func(players)
    return total_cost


def reset_player_counts(G):
    for edge in G.edges():
        G.edges[edge]["players"] = 0


def describe_graph(G):
    description = []
    description.append("Nodes:")
    nodes = " ".join(G.nodes)
    description.append(nodes)
    description.append("Edges and associated costs:")
    for u, v, data in G.edges(data=True):
        cost_formula = data['cost_formula']
        description.append(f"Edge {u} to {v}, cost function: {cost_formula}")
    return "\n".join(description)


# Assumes source node O and desination node D
def get_avail_routes(G):
    paths = list(nx.all_simple_paths(G, source='O', target='D'))
    formatted_paths = ['-'.join(path) for path in paths]
    return formatted_paths

# Run the simulation for a specific network with num_agents agents and num_rounds rounds
def run_simulation(has_bridge, num_rounds, num_agents, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "Agent", "Route", "Cost"])

        G = create_network(has_bridge)

        prev_costs = [0] * num_agents
        agent_routes = [0] * num_agents

        str_routes = get_avail_routes(G)
        prev_routes = {route: 0 for route in str_routes}

        for round_num in range(1, num_rounds + 1):
            print(f"Round {round_num}")

            for session_id in range(0, num_agents):
                description = describe_graph(G)
                if round_num == 1:
                    decision = prompt_route(session_id, num_agents, len(str_routes), description, str_routes, None, None)
                else:
                    decision = prompt_route(session_id, num_agents, len(str_routes), description, str_routes, prev_routes, prev_costs[session_id])

                chosen_route = decision.route
                agent_routes[session_id] = chosen_route
                add_player_to_path(path_to_edges(chosen_route), G)

            prev_routes = {route: 0 for route in str_routes}

            for session_id in range(0, num_agents):

                chosen_route = agent_routes[session_id]
                prev_routes[chosen_route] += 1

                cost = calculate_total_cost(path_to_edges(chosen_route), G)
                prev_costs[session_id] = cost

                writer.writerow([round_num, session_id, chosen_route, cost])

                print(f"Agent {session_id} chose route {chosen_route} with cost {cost}")

            reset_player_counts(G)
        

print("Starting simulations...")
# Run the first simulation without the bridge
# run_simulation(False, 5, 18, 'routes_no_bridge_5_18.csv')

# Run the second simulation with the bridge
run_simulation(True, 5, 18, 'routes_with_bridge_5_18.csv')

print("Simulations completed and routes saved")
