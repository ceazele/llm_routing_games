import networkx as nx
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.globals import set_verbose
from langchain.globals import set_debug
from contextlib import redirect_stdout
import sys
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


def add_system_message_to_history(session_id: str, system_message: str):
    history = get_session_history(session_id)
    history.add_message(SystemMessage(system_message))


def save_session_history(session_id: str):
    history = store.get(session_id, None)
    if history:
        with open(f"session_{session_id}_history.txt", "w", encoding="utf-8") as file:
            for message in history.messages:
                if isinstance(message, HumanMessage):
                    file.write(f"Human: {message.content}\n\n")
                elif isinstance(message, SystemMessage):
                    file.write(f"System: {message.content}\n\n")
                elif isinstance(message, AIMessage):
                    file.write(f"AI: {message.content}\n\n")

class PayoffCalculator:
    def __init__(self, num_agents):
        self.num_agents = num_agents

    def calculate_payoff(self, route_distribution: dict, chosen_route: str) -> int:
        """
        Calculate the payoff for choosing a specific route given a distribution of agents.

        Args:
        route_distribution (dict): A dictionary where keys are route names and values are the number of agents on that route.
        chosen_route (str): The route to calculate the payoff of.
        """
        
        # Check if the chosen route has at least one player
        if chosen_route not in route_distribution or route_distribution[chosen_route] < 1:
            raise ValueError(f"The chosen route '{chosen_route}' must have at least one player.")

        # Ensure the total number of agents equals num_agents
        total_agents = sum(route_distribution.values())
        if total_agents != self.num_agents:
            raise ValueError(f"Total number of agents {total_agents} does not equal num_agents {self.num_agents}.")

        # Calculate the cost for the chosen route based on the distribution
        if chosen_route == "O-A-D":
            cost = 10 * route_distribution['O-A'] + 210
        elif chosen_route == "O-B-D":
            cost = 210 + 10 * route_distribution['B-D']
        elif chosen_route == "O-A-B-D":
            cost = 10 * route_distribution['O-A'] + 10 * route_distribution['B-D']
        else:
            raise ValueError(f"Unknown route: {chosen_route}")

        payoff = 400 - cost
        return payoff



# Specify the Pydantic format of output
class Route(BaseModel):
    route: str = Field(description="choice of route")

# Set up a parser
parser = PydanticOutputParser(pydantic_object=Route)


def generate_system_message(agent_id, num_rounds, num_agents, num_routes, str_routes, has_bridge, demographics, description):
# Imagine you are an incredibly rational economics professor.
    instructions = f"""
You're {demographics[agent_id]['name']}, a {demographics[agent_id]['age']}-year-old {demographics[agent_id]['occupation']}.
You will be participating in an experiment on route selection in traffic networks.
During this experiment you'll be asked to make many decisions about route selection in a traffic network game.
Your payoff will depend on the decisions you make as well as the decisions made by the other participants.

There are {num_agents} participants in this experiment, including yourself, who will be asked to serve as drivers and choose a route to travel in a traffic network game that is described below.
You will play the game for {num_rounds} identical rounds.

Consider the very simple traffic network described below.
{description}
Each driver is required to choose one of {num_routes} routes to travel from the starting point, denoted by O, to the final destination, denoted by D.
There are {num_routes} alternative routes and they are denoted by {str_routes}. 

Travel is always costly in terms of the time needed to complete a segment of the road, tolls, fuel etc.
The travel costs are written near each segment of the route you choose.
For example, if you choose route O-A-D, you will be charged a total cost of 10X + 210 where X indicates the number of participants who choose segment O-A to travel from O to A plus a fixed cost of 210 for traveling on segment A-D.
Similarly, if you choose route O-B-D, you will be charged a total travel cost of 210 + 10Y, where Y indicates the number of participants who choose the segment B-D to drive from O to D.
Please note that the cost charged for segments O-A and B-D depends on the number of drivers choosing them.
In contrast, the cost charged for traveling on segments A-D and O-B is fixed at 210 and does not depend on the number of drivers choosing them.
All the drivers make their route choices independently of one another and leave point O at the same time.
"""
    
    if not has_bridge:
        example = f"""
Example.
If you happen to be the only driver who chooses route O-A-D, and all other 17 drivers choose route O-B-D, then your travel cost from point O to point D is equal to (10 X 1) + 210 = 220.
If, on another round, you and 2 more drivers choose route O-B-D and 15 other drivers choose route O-A-D, then your travel cost for that round will be 210 + (10 X 3) = 240.
"""
    else:
        example = f"""
Example.
Supposing that you choose route O-A-B-D, 3 other drivers choose route O-A-D, and 14 additional drivers choose route O-B-D.
Then, your total travel cost for that period is equal to (10 X 4) + 0 + (10 X 15) = 190.
Note that in this example, 4 drivers (including you) traveled on the segment O-A and 15 drivers (again, including you) traveled the segment B-D to go from O to D.
Each of the 3 drivers choosing route O-A-D will be charged a travel cost of (10 X 4) + (210) = 250, and each of the 14 drivers choosing the route O-B-D will be charged a travel cost of (210) + (10 X 15) = 360.
"""
    instructions += example
    instructions += f"""
At the beginning of each round, you will receive an endowment of 400 points.
Your payoff for each round will be determined by subtracting your travel cost from your endowment.
At the end of each round, you will be informed of the number of drivers who chose each route and your payoff for that round. 
All {num_rounds} rounds have exactly the same structure.
"""
    
    return instructions

prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="messages")
    ]
)


chain = prompt | model


def prompt_route(session_id, num_rounds, num_agents, num_routes, demographics, description, avail_routes, has_bridge, prev_routes, cost):
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="messages",
    )

    parsed_chain = with_message_history | parser

    config = {"configurable": {"session_id": f"session_{session_id}"}}
    format_instructions = parser.get_format_instructions()

    messages = []
    if prev_routes is None and cost is None:
        system_message = generate_system_message(session_id, num_rounds, num_agents, num_routes, avail_routes, has_bridge, demographics, description)
        messages.append(SystemMessage(system_message))
    else:
        # Report only the actual cost the player incurred
        message_content = f"Last round, the number of players on each route was: {prev_routes}\n" \
                          f"Your payoff was {400 - cost}"

        messages.append(HumanMessage(message_content))

    messages.append(HumanMessage(f"The available routes are: {avail_routes}\n{format_instructions}\nThink step-by-step before making your decision."))

    response = parsed_chain.invoke(
        {"messages": messages, "num_agents": num_agents, "num_routes": num_routes, "description": description, "name": demographics[session_id]["name"], "age": demographics[session_id]["age"], "occupation": demographics[session_id]["occupation"]},
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
    G.add_edge('O', 'A', cost_func=cost_OA, cost_formula="10 * X", players=0)
    G.add_edge('B', 'D', cost_func=cost_BD, cost_formula="10 * X", players=0)
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


def remove_player_from_path(path, G):
    for edge in path:
        G.edges[edge]["players"] -= 1


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
    description.append("Segments and associated costs:")
    for u, v, data in G.edges(data=True):
        cost_formula = data['cost_formula']
        description.append(f"Segment {u}-{v}, cost function: {cost_formula}")
    return "\n".join(description)


# Assumes source node O and desination node D
def get_avail_routes(G):
    paths = list(nx.all_simple_paths(G, source='O', target='D'))
    formatted_paths = ['-'.join(path) for path in paths]
    return formatted_paths


# Run the simulation for a specific network with num_agents agents and num_rounds rounds
def run_simulation(has_bridge, num_rounds, num_agents, demographics, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "Agent", "Route", "Cost"])

        G = create_network(has_bridge)
        payoff_calculator = PayoffCalculator(num_agents)

        prev_costs = [{} for _ in range(num_agents)]
        agent_routes = [0] * num_agents
        # segment_counts = {}

        str_routes = get_avail_routes(G)
        prev_routes = {route: 0 for route in str_routes}

        for round_num in range(1, num_rounds + 1):
            print(f"Round {round_num}")

            for session_id in range(0, num_agents):
                description = describe_graph(G)
                if round_num == 1:
                    decision = prompt_route(session_id, num_rounds, num_agents, len(str_routes), demographics, description, str_routes, has_bridge, None, None)
                else:
                    decision = prompt_route(session_id, num_rounds, num_agents, len(str_routes), demographics, description, str_routes, has_bridge, prev_routes, prev_costs[session_id])

                chosen_route = decision.route
                agent_routes[session_id] = chosen_route
                add_player_to_path(path_to_edges(chosen_route), G)

            prev_routes = {route: 0 for route in str_routes}

            for session_id in range(0, num_agents):
                chosen_route = agent_routes[session_id]
                cost = calculate_total_cost(path_to_edges(chosen_route), G)
                writer.writerow([round_num, session_id, chosen_route, cost])
                print(f"Agent {session_id} chose route {chosen_route} with cost {cost}")
                prev_costs[session_id] = cost
                prev_routes[chosen_route] += 1

            # segment_counts = {f"{u}-{v}": data["players"] for u, v, data in G.edges(data=True)}
            reset_player_counts(G)

        for session_id in range(0, num_agents):
            save_session_history(f"session_{session_id}")



print("Starting simulations...")

demographics = {
    0: {"name": "John Smith", "age": 40, "occupation": "teacher"},
    1: {"name": "Jane Doe", "age": 35, "occupation": "software developer"},
    2: {"name": "Michael Brown", "age": 45, "occupation": "doctor"},
    3: {"name": "Emily White", "age": 30, "occupation": "nurse"},
    4: {"name": "David Wilson", "age": 50, "occupation": "lawyer"},
    5: {"name": "Laura Green", "age": 28, "occupation": "engineer"},
    6: {"name": "James Taylor", "age": 55, "occupation": "architect"},
    7: {"name": "Sarah Miller", "age": 33, "occupation": "data scientist"},
    8: {"name": "Robert Davis", "age": 42, "occupation": "bartender"},
    9: {"name": "Linda Martinez", "age": 29, "occupation": "sales representative"},
    10: {"name": "William Anderson", "age": 37, "occupation": "electrician"},
    11: {"name": "Karen Thomas", "age": 41, "occupation": "accountant"},
    12: {"name": "Christopher Jackson", "age": 38, "occupation": "journalist"},
    13: {"name": "Patricia Lee", "age": 32, "occupation": "photographer"},
    14: {"name": "Daniel Harris", "age": 43, "occupation": "graphic designer"},
    15: {"name": "Barbara Clark", "age": 36, "occupation": "chef"},
    16: {"name": "Matthew Lewis", "age": 31, "occupation": "construction worker"},
    17: {"name": "Jessica Young", "age": 27, "occupation": "librarian"}
}



run_simulation(True, 3, 3, demographics, 'routes_with_bridge_10_18.csv')
print("Simulations completed and routes saved")