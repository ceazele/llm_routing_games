from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from typing import List, Optional
import json
import os
import re
import csv
from agent import Agent
from network import TrafficNetwork
from payoff import create_payoff_func

# Ensure environment variables are set
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCAHIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "bluesky"
os.environ["OPENAI_API_KEY"] = "sk-proj-2YIRzXSkGvD5mXF5KSwST3BlbkFJ32aEm9aV3nrL6gFGCr97"

# Define Pydantic model for output
class Route(BaseModel):
    route: str = Field(description="choice of route")

# Set up a parser
parser = PydanticOutputParser(pydantic_object=Route)

# Store for chat histories
chat_histories = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]

def save_session_history(session_id: str):
    history = chat_histories.get(session_id, None)
    if history:
        with open(f"{session_id}_history.txt", "w", encoding="utf-8") as file:
            for message in history.messages:
                if isinstance(message, HumanMessage):
                    file.write(f"Human: {message.content}\n\n")
                elif isinstance(message, SystemMessage):
                    file.write(f"System: {message.content}\n\n")
                elif isinstance(message, AIMessage):
                    file.write(f"AI: {message.content}\n\n")


# Custom parser
def extract_json(message: AIMessage) -> List[dict]:
    """Extracts JSON content from a string where JSON is embedded between ```json and ``` tags.

    Parameters:
        text (str): The text containing the JSON content.

    Returns:
        list: A list of extracted JSON strings.
    """
    text = message.content
    # Define the regular expression pattern to match JSON blocks
    pattern = r"```json(.*?)```"

    # Find all non-overlapping matches of the pattern in the string
    matches = re.findall(pattern, text, re.DOTALL)

    # Return the list of matched JSON strings, stripping any leading or trailing whitespace
    try:
        return [json.loads(match.strip()) for match in matches]
    except Exception:
        raise ValueError(f"Failed to parse: {message}")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from network import TrafficNetwork
from langchain_core.tools import tool


def generate_few_shot_examples(num_agents, routes, network: TrafficNetwork):
    # Create the payoff function using the provided tool
    calculate_payoff = create_payoff_func(num_agents, network)

    examples = []

    # Example 1: Simple distribution, all agents on one route
    simple_dist = {route: num_agents if i == 0 else 0 for i, route in enumerate(routes)}
    chosen_route_1 = routes[0]
    
    # Calculate the payoff using the tool
    payoff_1 = calculate_payoff(simple_dist, chosen_route_1)
    
    examples.append(
        HumanMessage(
            content="What is the payoff if I choose the route O-L-D and everyone else chooses O-R-D?",
            name="example_user"
        )
    )
    examples.append(
        AIMessage(
            content="",
            name="example_assistant",
            tool_calls=[
                {"name": "calculate_payoff", "args": {"route_distribution": simple_dist, "chosen_route": chosen_route_1}, "id": "1"}
            ]
        )
    )
    examples.append(
        ToolMessage(
            content=str(payoff_1),
            tool_call_id="1"
        )
    )
    examples.append(
        AIMessage(
            content=f"The payoff for choosing the route O-L-D is {payoff_1} when all other agents choose O-R-D.",
            name="example_assistant"
        )
    )

    # Example 2: Split distribution, agents divided across routes
    base_agents_per_route = num_agents // len(routes)
    remainder_agents = num_agents % len(routes)
    
    # Distribute remainder agents across the first few routes
    split_dist = {route: base_agents_per_route + (1 if i < remainder_agents else 0) for i, route in enumerate(routes)}
    chosen_route_2 = routes[1]
    
    # Calculate the payoff using the tool
    payoff_2 = calculate_payoff(split_dist, chosen_route_2)
    
    examples.append(
        HumanMessage(
            content=f"What is the payoff if I choose the route {chosen_route_2} and the agents are evenly distributed across all routes?",
            name="example_user"
        )
    )
    examples.append(
        AIMessage(
            content="",
            name="example_assistant",
            tool_calls=[
                {"name": "calculate_payoff", "args": {"route_distribution": split_dist, "chosen_route": chosen_route_2}, "id": "2"}
            ]
        )
    )
    examples.append(
        ToolMessage(
            content=str(payoff_2),
            tool_call_id="2"
        )
    )
    examples.append(
        AIMessage(
            content=f"The payoff for choosing the route {chosen_route_2} is {payoff_2} when the agents are evenly distributed.",
            name="example_assistant"
        )
    )
    return examples


# Define the system message generator
def generate_system_message(num_rounds, num_agents, num_routes, str_routes, has_bridge, description):
    instructions = f"""
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
For example, if you choose route O-L-D, you will be charged a total cost of 10X + 210 where X indicates the number of participants who choose segment O-L to travel from O to L plus a fixed cost of 210 for traveling on segment L-D.
Similarly, if you choose route O-R-D, you will be charged a total travel cost of 210 + 10Y, where Y indicates the number of participants who choose the segment R-D to drive from O to D.
Please note that the cost charged for segments O-L and R-D depends on the number of drivers choosing them.
In contrast, the cost charged for traveling on segments L-D and O-R is fixed at 210 and does not depend on the number of drivers choosing them.
All the drivers make their route choices independently of one another and leave point O at the same time.
"""
    
    if not has_bridge:
        example = f"""
Example.
If you happen to be the only driver who chooses route O-L-D, and all other 17 drivers choose route O-R-D, then your travel cost from point O to point D is equal to (10 X 1) + 210 = 220.
If, on another round, you and 2 more drivers choose route O-R-D and 15 other drivers choose route O-L-D, then your travel cost for that round will be 210 + (10 X 3) = 240.
"""
    else:
        example = f"""
Example.
Supposing that you choose route O-L-R-D, 3 other drivers choose route O-L-D, and 14 additional drivers choose route O-R-D.
Then, your total travel cost for that period is equal to (10 X 4) + 0 + (10 X 15) = 190.
Note that in this example, 4 drivers (including you) traveled on the segment O-L and 15 drivers (again, including you) traveled the segment R-D to go from O to D.
Each of the 3 drivers choosing route O-L-D will be charged a travel cost of (10 X 4) + (210) = 250, and each of the 14 drivers choosing the route O-R-D will be charged a travel cost of (210) + (10 X 15) = 360.
"""
    instructions += example
    instructions += f"""
At the beginning of each round, you will receive an endowment of 400 points.
Your payoff for each round will be determined by subtracting your travel cost from your endowment.
Your goal is to maximize your payoff (likewise minimize your cost).
At the end of each round, you will be informed of the number of drivers who chose each route and your payoff for that round. 
All {num_rounds} rounds have exactly the same structure. Use past tool usage as an example of how to correctly use the tool.
When calling the tool, remember to provide both your chosen route AND the route distribution.
"""
    
    return instructions


# Update the prompt_route function to use the Agent class
def prompt_route(agent, examples, num_rounds, num_routes, description, avail_routes, prev_routes, cost):
    session_id = agent.thread_id
    history = get_session_history(session_id)

    # Prepare initial messages
    if prev_routes is None and cost is None:
        system_message = generate_system_message(num_rounds, agent.num_agents, num_routes, avail_routes, True, description)
        messages = [SystemMessage(content=system_message)]
        messages.extend(examples)
    else:
        message_content = f"Last round, the number of players on each route was: {prev_routes}\n" \
                          f"Your payoff was {400 - cost}"
        messages = [HumanMessage(content=message_content)]

    format_instructions = parser.get_format_instructions()
    messages.append(HumanMessage(content=f"The available routes are: {avail_routes}\nThink step-by-step before making your decision."))
    messages.append(HumanMessage(content=format_instructions))

    # Record initial messages to history
    for msg in messages:
        history.add_message(msg)

    # Invoke the agent's workflow
    response = agent.call({"messages": messages})
    
    # Capture the AI response and save to history
    ai_message = AIMessage(content=response["messages"][-1].content)
    history.add_message(ai_message)

    return response



# Simulation function updated to use the TrafficNetwork class
def run_simulation(has_bridge, num_rounds, num_agents, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "Agent", "Route", "Cost"])

        network = TrafficNetwork(has_bridge)

        agents = [Agent(thread_id=str(i), num_agents=num_agents, network=network) for i in range(num_agents)]

        prev_costs = [{} for _ in range(num_agents)]
        agent_routes = [0] * num_agents

        str_routes = network.get_avail_routes()
        prev_routes = {route: 0 for route in str_routes}

        for round_num in range(1, num_rounds + 1):
            print(f"Round {round_num}")

            for session_id, agent in enumerate(agents):
                description = network.describe_graph()
                examples = generate_few_shot_examples(num_agents, str_routes, network)
                if round_num == 1:
                    response = prompt_route(agent, examples, num_rounds, len(str_routes), description, str_routes, None, None)
                else:
                    response = prompt_route(agent, examples, num_rounds, len(str_routes), description, str_routes, prev_routes, prev_costs[session_id])

                # parsed_response = extract_json(response)
                ai_message = response["messages"][-1].content
                parsed_response = parser.parse(ai_message)
                chosen_route = parsed_response.route
                agent_routes[session_id] = chosen_route
                network.add_player_to_path(network.path_to_edges(chosen_route))

            prev_routes = {route: 0 for route in str_routes}

            for session_id, agent in enumerate(agents):
                chosen_route = agent_routes[session_id]
                cost = network.calculate_total_cost(network.path_to_edges(chosen_route))
                writer.writerow([round_num, session_id, chosen_route, cost])
                print(f"Agent {session_id} chose route {chosen_route} with cost {cost}")
                prev_costs[session_id] = cost
                prev_routes[chosen_route] += 1

            network.reset_player_counts()
        
        # Save histories for each agent after simulation
        for agent in agents:
            save_session_history(agent.thread_id)

# Running the simulation
print("Starting simulations...")
run_simulation(False, 3, 3, 'LR_10_18_no_bridge.csv')
print("Simulations completed and routes saved")