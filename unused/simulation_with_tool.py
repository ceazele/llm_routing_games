import os
import re
import csv
import json
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
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

def extract_json(message: AIMessage) -> List[dict]:
    """Extracts JSON content from a string where JSON is embedded between ```json and ``` tags."""
    text = message.content
    pattern = r"```json(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    try:
        return [json.loads(match.strip()) for match in matches]
    except Exception:
        raise ValueError(f"Failed to parse: {message}")

# def generate_few_shot_examples(num_agents, routes, network: TrafficNetwork):
#     # Create the payoff function using the provided tool
#     calculate_payoff = create_payoff_func(num_agents, network)

#     examples = []

#     # Example 1: Simple distribution, all agents on one route
#     simple_dist = {route: num_agents if i == 0 else 0 for i, route in enumerate(routes)}
#     chosen_route_1 = routes[0]

#     # Calculate the payoff using the tool
#     payoff_1 = calculate_payoff(simple_dist, chosen_route_1)

#     examples.append(
#         HumanMessage(
#             content="What is the payoff if I choose the route O-L-D and everyone else chooses O-R-D?",
#             name="example_user"
#         )
#     )
#     examples.append(
#         AIMessage(
#             content="",
#             name="example_assistant",
#             tool_calls=[
#                 {"name": "calculate_payoff", "args": {"route_distribution": simple_dist, "chosen_route": chosen_route_1}, "id": "1"}
#             ]
#         )
#     )
#     examples.append(
#         ToolMessage(
#             content=str(payoff_1),
#             tool_call_id="1"
#         )
#     )
#     examples.append(
#         AIMessage(
#             content=f"The payoff for choosing the route O-L-D is {payoff_1} when all other agents choose O-R-D.",
#             name="example_assistant"
#         )
#     )

#     # Example 2: Split distribution, agents divided across routes
#     base_agents_per_route = num_agents // len(routes)
#     remainder_agents = num_agents % len(routes)

#     # Distribute remainder agents across the first few routes
#     split_dist = {route: base_agents_per_route + (1 if i < remainder_agents else 0) for i, route in enumerate(routes)}
#     chosen_route_2 = routes[1]

#     # Calculate the payoff using the tool
#     payoff_2 = calculate_payoff(split_dist, chosen_route_2)

#     examples.append(
#         HumanMessage(
#             content=f"What is the payoff if I choose the route {chosen_route_2} and the agents are evenly distributed across all routes?",
#             name="example_user"
#         )
#     )
#     examples.append(
#         AIMessage(
#             content="",
#             name="example_assistant",
#             tool_calls=[
#                 {"name": "calculate_payoff", "args": {"route_distribution": split_dist, "chosen_route": chosen_route_2}, "id": "2"}
#             ]
#         )
#     )
#     examples.append(
#         ToolMessage(
#             content=str(payoff_2),
#             tool_call_id="2"
#         )
#     )
#     examples.append(
#         AIMessage(
#             content=f"The payoff for choosing the route {chosen_route_2} is {payoff_2} when the agents are evenly distributed.",
#             name="example_assistant"
#         )
#     )
#     return examples

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
When calling the tool, remember to provide both your chosen route AND the distribution of players on routes.
"""
    return instructions

def summarize_agent_history(agent_history):
    """
    Generates a summary for an agent that includes all previous rounds.

    Args:
    - agent_history (list): A list of dictionaries, one for each previous round, each containing:
        - 'round_num' (int): The round number.
        - 'agent_route' (str): The route chosen by the agent.
        - 'agent_payoff' (float): The payoff received by the agent.
        - 'route_distribution' (dict): A dict of {route: number of agents}.

    Returns:
    - str: A summary of all previous rounds for the agent.
    """
    summary_lines = []
    for round_data in agent_history:
        round_num = round_data['round_num']
        agent_route = round_data['agent_route']
        agent_payoff = round_data['agent_payoff']
        route_distribution = round_data['route_distribution']

        route_summary = "\n        ".join([f"{route}: {count} agents" for route, count in route_distribution.items()])
        summary = (
            f"Round {round_num}:\n"
            f"- You chose route {agent_route} and received a payoff of {agent_payoff}.\n"
            f"- The number of agents on each route was:\n        {route_summary}\n"
        )
        summary_lines.append(summary)
    full_summary = "\n".join(summary_lines)
    return full_summary

# Update the prompt_route function to use the Agent class
def prompt_route(agent, round_num, avail_routes, summary, system_message): # examples,
    format_instructions = parser.get_format_instructions()
    messages = []
    messages.append(SystemMessage(system_message))
    # messages.extend(examples)

    # Construct the human message
    human_message_content = ""
    if summary is not None:
        human_message_content += f"Summary of previous rounds:\n{summary}\n\n"
    human_message_content += (
        f"Welcome to round {round_num}. The available routes are: {avail_routes}\n"
        f"{format_instructions}\n"
        f"Think step-by-step before making your decision."
    )
    messages.append(HumanMessage(human_message_content))

    # Invoke the agent's workflow
    response = agent.call({"messages": messages})
    llm_output = response["messages"]
    answer = llm_output[-1].content

    # Parse the output
    try:
        parsed_response = parser.parse(answer)
    except Exception as e:
        # Handle parsing errors
        parsed_response = None
        print(f"Error parsing LLM output for agent {agent.thread_id} in round {round_num}: {e}")
    
    return parsed_response, llm_output

# Simulation function updated to use the TrafficNetwork class
def run_simulation(has_bridge, num_rounds, num_agents, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "Agent", "Route", "Cost"])

        network = TrafficNetwork(has_bridge)

        agents = [Agent(thread_id=str(i), num_agents=num_agents, network=network) for i in range(num_agents)]

        agent_histories = [[] for _ in range(num_agents)]  # List of histories for each agent
        agent_routes = [""] * num_agents
        agent_llm_outputs = [[] for _ in range(num_agents)]  # To store full LLM outputs per agent

        str_routes = network.get_avail_routes()
        prev_routes = {route: 0 for route in str_routes}

        description = network.describe_graph()
        # examples = generate_few_shot_examples(num_agents, str_routes, network)

        # Generate system message once
        system_message = generate_system_message(
            num_rounds,
            num_agents,
            len(str_routes),
            str_routes,
            has_bridge,
            description
        )

        for round_num in range(1, num_rounds + 1):
            print(f"Round {round_num}")

            # Collect agent decisions
            for session_id, agent in enumerate(agents):
                if round_num == 1:
                    summary = None  # No previous rounds
                else:
                    # Generate the summary for the agent
                    summary = summarize_agent_history(agent_histories[session_id])

                parsed_response, llm_output = prompt_route(
                    agent,
                    # examples,
                    round_num,  # Pass the current round number
                    str_routes,
                    summary,
                    system_message
                )

                if parsed_response is None:
                    # Handle the case where parsing failed
                    chosen_route = "Invalid"
                else:
                    chosen_route = parsed_response.route

                agent_routes[session_id] = chosen_route
                # Only add to path if the route is valid
                if chosen_route in str_routes:
                    network.add_player_to_path(network.path_to_edges(chosen_route))
                else:
                    print(f"Agent {session_id} provided an invalid route: {chosen_route}")

                # Store the full LLM output after discarding the first 10 messages
                round_llm_output = llm_output[2:]  # Discard the first 10 messages
                round_messages_content = [msg.content for msg in round_llm_output]
                agent_llm_outputs[session_id].append(round_messages_content)

            # Update route counts
            prev_routes = {route: 0 for route in str_routes}
            for route in agent_routes:
                if route in str_routes:
                    prev_routes[route] += 1

            # Calculate costs and update agent histories
            for session_id in range(num_agents):
                chosen_route = agent_routes[session_id]
                if chosen_route in str_routes:
                    cost = network.calculate_total_cost(network.path_to_edges(chosen_route))
                    payoff = 400 - cost
                    writer.writerow([round_num, session_id, chosen_route, cost])
                    print(f"Agent {session_id} chose route {chosen_route} with cost {cost}")
                else:
                    cost = 400  # Max cost if invalid
                    payoff = 0
                    writer.writerow([round_num, session_id, "Invalid", cost])
                    print(f"Agent {session_id} had invalid choice with cost {cost}")

                # Update agent's history
                agent_histories[session_id].append({
                    'round_num': round_num,
                    'agent_route': chosen_route,
                    'agent_payoff': payoff,
                    'route_distribution': prev_routes.copy()  # Copy to capture state at this round
                })
            network.reset_player_counts()
        
        # After the simulation, save the required information per agent
        for session_id in range(num_agents):
            final_summary = summarize_agent_history(agent_histories[session_id])
            with open(f"agent_{session_id}_history.txt", "w", encoding="utf-8") as f:
                # Write the system message
                f.write("System Message:\n")
                f.write(system_message)
                f.write("\n\n")
                # Write the final summary
                f.write("Final Summary of All Rounds:\n")
                f.write(final_summary)
                f.write("\n\n")
                # Write all messages output by the LLM
                f.write("LLM Outputs:\n")
                for round_num, round_messages in enumerate(agent_llm_outputs[session_id], start=1):
                    f.write(f"Round {round_num}:\n")
                    for message in round_messages:
                        f.write(message)
                        f.write("\n\n")

    print("Simulations completed and routes saved")

# Running the simulation
print("Starting simulations...")
run_simulation(False, 5, 18, 'test.csv')
