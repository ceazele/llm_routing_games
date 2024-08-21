from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from typing import List, Optional
import json
import os
import re
import csv
from agent import Agent
from network import TrafficNetwork

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




# Define the system message generator
def generate_system_message(agent_id, num_rounds, num_agents, num_routes, str_routes, has_bridge, demographics, description):
    agent_id = int(agent_id)
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

# Update the prompt_route function to use the Agent class
def prompt_route(agent, num_rounds, num_routes, demographics, description, avail_routes, prev_routes, cost):
    # Prepare the initial message if this is the first round
    if prev_routes is None and cost is None:
        system_message = generate_system_message(agent.thread_id, num_rounds, agent.num_agents, num_routes, avail_routes, True, demographics, description)
        messages = [SystemMessage(content=system_message)]
    else:
        # Report only the actual cost the player incurred
        message_content = f"Last round, the number of players on each route was: {prev_routes}\n" \
                          f"Your payoff was {400 - cost}"
        messages = [HumanMessage(content=message_content)]

    format_instructions = parser.get_format_instructions()
    messages.append(HumanMessage(content=f"The available routes are: {avail_routes}"))
    messages.append(HumanMessage(content=format_instructions)) # Try using a SystemMessage also

    # Invoke the agent's workflow with the prepared messages
    response = agent.call({"messages": messages})
    return response


# Simulation function updated to use the TrafficNetwork class
def run_simulation(has_bridge, num_rounds, num_agents, demographics, filename):
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
                if round_num == 1:
                    response = prompt_route(agent, num_rounds, len(str_routes), demographics, description, str_routes, None, None)
                else:
                    response = prompt_route(agent, num_rounds, len(str_routes), demographics, description, str_routes, prev_routes, prev_costs[session_id])

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

        # for agent in agents:
        #     save_session_history(agent.thread_id)

# Running the simulation
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

run_simulation(False, 3, 3, demographics, 'routes_with_bridge_10_18.csv')
print("Simulations completed and routes saved")
