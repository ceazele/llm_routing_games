import networkx as nx
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from network import TrafficNetwork
import yaml
import os
import csv
from regret import compute_regret_for_agents
import re  # Regular expression usage
from collections import Counter

# Set your API keys and environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "false" # change to true when langsmith credits are back
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCAHIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "bluesky"
os.environ["OPENAI_API_KEY"] = "sk-proj-2YIRzXSkGvD5mXF5KSwST3BlbkFJ32aEm9aV3nrL6gFGCr97"

# Initialize the model
model = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=1)

# Placeholder store for chat histories
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def add_system_message_to_history(session_id: str, system_message: str):
    history = get_session_history(session_id)
    history.add_message(SystemMessage(system_message))

def save_session_history(session_id: str, folder_name: str):
    history = store.get(session_id, None)
    if history:
        with open(os.path.join(folder_name, f"{session_id}_history.txt"), "w", encoding="utf-8") as file:
            for message in history.messages:
                if isinstance(message, HumanMessage):
                    file.write(f"Human: {message.content}\n\n")
                elif isinstance(message, SystemMessage):
                    file.write(f"System Message: {message.content}\n\n")
                elif isinstance(message, AIMessage):
                    file.write(f"LLM Output: {message.content}\n\n")

# Specify the Pydantic format of output
class Route(BaseModel):
    route: str = Field(description="choice of route")

# Set up a parser
parser = PydanticOutputParser(pydantic_object=Route)

def setup_folders(base_folder, current_run_folder=None):
    """
    Create the folder structure as `game_x/run_x/game_xA` and `game_x/run_x/game_xB`.

    Args:
        base_folder (str): Base folder name in the format `game_xA` or `game_xB`.
        current_run_folder (str): The current `run_x` folder to reuse if provided.

    Returns:
        tuple: Paths to `game_x/run_x/game_xA` and `game_x/run_x/game_xB`.
    """
    # Extract the game ID (e.g., "game_1" from "game_1A")
    base_game_folder = base_folder[:-1]  # Remove the trailing "A" or "B"

    # Create the base game folder
    if not os.path.exists(base_game_folder):
        os.makedirs(base_game_folder)

    # Use the provided `current_run_folder`, or create a new one
    if current_run_folder:
        run_folder = current_run_folder
    else:
        existing_runs = [f for f in os.listdir(base_game_folder) if f.startswith("run")]
        if existing_runs:
            next_run = f"run {len(existing_runs) + 1}"
        else:
            next_run = "run 1"

        run_folder = os.path.join(base_game_folder, next_run)
        os.makedirs(run_folder, exist_ok=True)

    # Create the game_xA and game_xB folders inside the run folder
    game_a_folder = os.path.join(run_folder, f"{base_folder[:-1]}A")
    game_b_folder = os.path.join(run_folder, f"{base_folder[:-1]}B")

    os.makedirs(game_a_folder, exist_ok=True)
    os.makedirs(game_b_folder, exist_ok=True)

    return game_a_folder, game_b_folder, run_folder


def generate_system_message(num_rounds, num_agents, num_routes, str_routes, has_bridge, description):
    format_instructions = parser.get_format_instructions()
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
If you happen to be the only driver who chooses route O-L-D, and all other {num_agents - 1} drivers choose route O-R-D, then your travel cost from point O to point D is equal to (10 x 1) + 210 = 220.
If, on another round, you and 2 more drivers choose route O-R-D and {num_agents - 3} other drivers choose route O-L-D, then your travel cost for that round will be 210 + (10 x 3) = 240.
"""
    else:
        example = f"""
Example.
Supposing that you choose route O-L-R-D, 3 other drivers choose route O-L-D, and {num_agents - 4} additional drivers choose route O-R-D.
Then, your total travel cost for that period is equal to (10 x 4) + 0 + (10 x {num_agents - 4}) = {10 * 4 + 0 + 10 * (num_agents - 4)}.
Note that in this example, 4 drivers (including you) traveled on the segment O-L and {num_agents - 4} drivers (again, including you) traveled the segment R-D to go from O to D.
Each of the 3 drivers choosing route O-L-D will be charged a travel cost of (10 x 4) + 210 = {10 * 4 + 210}.
Each of the {num_agents - 4} drivers choosing the route O-R-D will be charged a travel cost of 210 + (10 x {num_agents - 4}) = {210 + 10 * (num_agents - 4)}.
"""
    instructions += example
    instructions += f"""
At the beginning of each round, you will receive an endowment of 400 points.
Your payoff for each round will be determined by subtracting your travel cost from your endowment.
Your goal is to maximize your payoff (likewise minimize your cost).
All {num_rounds} rounds have exactly the same structure.

{format_instructions}
"""
    return instructions

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="messages")
    ]
)

# Define the full chat chain
full_chain = prompt | model

def create_prompt(config, system_message, all_agent_histories, session_id, round_num, available_routes):
    """
    Create a prompt for the LLM based on the provided configuration.

    Args:
        config (dict): Configuration file specifying the simulation and history representation settings.
        all_agent_histories (list): List of all agents' histories.
        session_id (int): The ID of the agent being prompted.
        round_num (int): The current round number (zero-indexed internally).
        available_routes (list): List of available routes.

    Returns:
        list: A list of message objects to serve as the prompt for the LLM.
    """
    prompting_style = config['history_representation']['prompting_style']['type']
    messages = []

    if prompting_style == "consecutive":
        # Add the system message for the first round only
        if round_num == 0:
            messages.append(SystemMessage(content=system_message))
        else:
            # Add the summary for the most recent round
            round_summary = generate_consecutive_prompt(config, all_agent_histories, session_id, round_num)
            messages.append(HumanMessage(content=round_summary))

    elif prompting_style == "zero-shot":
        messages.append(SystemMessage(content=system_message))
        # Generate the zero-shot summary
        if round_num > 0:
            summary = summarize_history(config, all_agent_histories, session_id, round_num)
            messages.append(HumanMessage(content=summary))

    # Add the prompt for the next decision
    messages.append(HumanMessage(content=f"The available routes are: {', '.join(available_routes)}.\n"
                                         f"Think step-by-step before making your decision."))
    return messages

def generate_consecutive_prompt(config, all_agent_histories, session_id, round_num):
    """
    Generates a round summary for consecutive prompting based on the most recent round.

    Args:
        config (dict): Configuration file specifying history representation.
        all_agent_histories (list): List of all agents' histories.
        session_id (int): ID of the agent being prompted.
        round_num (int): Current round number (zero-indexed internally).

    Returns:
        str: A summary string for the most recent round.
    """
    agent_history = all_agent_histories[session_id]
    last_round_data = agent_history[round_num - 1]

    round_summary = f"You are agent {session_id}.\nSummary of previous round:\n"

    # Include choices if configured
    choices_config = config['history_representation']['choice_representation']
    if choices_config['my_choice']['include']:
        round_summary += f"  Your Choice: {last_round_data['decision']}\n"
    if choices_config['everyone_choices']['include']:
        type_of_choices = choices_config['everyone_choices']['type']
        if type_of_choices == "aggregate":
            distribution = compute_route_distribution(all_agent_histories, round_num - 1)
            round_summary += f"  Route Choice Distribution: {distribution}\n"
        elif type_of_choices == "individual":
            for agent_id, history in enumerate(all_agent_histories):
                round_summary += f"  Agent {agent_id}: {history[round_num - 1]['decision']}\n"

    # Include payoffs if configured
    payoff_config = config['history_representation']['payoff_representation']
    if payoff_config['my_payoff']['include']:
        round_summary += f"  Your Payoff: {last_round_data['payoff']}\n"
    if payoff_config['everyone_payoff']['include']:
        type_of_payoff = payoff_config['everyone_payoff']['type']
        if type_of_payoff == "aggregate":
            avg_payoff = compute_average_payoff(all_agent_histories, round_num - 1)
            round_summary += f"  Average Payoff: {avg_payoff}\n"
        elif type_of_payoff == "individual":
            for agent_id, history in enumerate(all_agent_histories):
                round_summary += f"  Agent {agent_id} Payoff: {history[round_num - 1]['payoff']}\n"

    # Include regrets if configured
    regret_config = config['history_representation']['regret_representation']
    if regret_config['my_regret']['include']:
        round_summary += f"  Your Regret: {last_round_data['regret']}\n"
    if regret_config['everyone_regret']['include']:
        type_of_regret = regret_config['everyone_regret']['type']
        if type_of_regret == "aggregate":
            avg_regret = compute_average_regret(all_agent_histories, round_num - 1)
            round_summary += f"  Average Regret: {avg_regret}\n"
        elif type_of_regret == "individual":
            for agent_id, history in enumerate(all_agent_histories):
                round_summary += f"  Agent {agent_id} Regret: {history[round_num - 1]['regret']}\n"

    return round_summary

def summarize_history(config, all_agent_histories, session_id, round_num):
    """
    Generate a summary of the agent's history based on the provided configuration.

    Args:
        config (dict): Configuration file specifying history representation.
        all_agent_histories (list): List of all agents' histories.
        session_id (int): The ID of the agent being prompted.
        round_num (int): The current round number (zero-indexed internally).

    Returns:
        str: A summary string including aggregate statistics or per-round details.
    """
    summary = f"You are agent {session_id}.\nSummary of previous rounds:\n"

    # Prepare configurations
    choices_config = config['history_representation']['choice_representation']
    payoff_config = config['history_representation']['payoff_representation']
    regret_config = config['history_representation']['regret_representation']

    # Per-round details for each round
    for r in range(round_num):
        round_summary = []

        # My Choice (per-round)
        if choices_config['my_choice']['include'] and choices_config['my_choice']['rounds'] == "per-round":
            my_choice = all_agent_histories[session_id][r]['decision']
            round_summary.append(f"    Your Choice: {my_choice}")

        # Everyone's Choices (per-round)
        if choices_config['everyone_choices']['include'] and choices_config['everyone_choices']['rounds'] == "per-round":
            if choices_config['everyone_choices']['type'] == "aggregate":
                route_distribution = compute_route_distribution(all_agent_histories, r)
                round_summary.append(f"    Route Choice Distribution: {route_distribution}")
            elif choices_config['everyone_choices']['type'] == "individual":
                round_summary.append("    Choices by Agent:")
                for agent_id, history in enumerate(all_agent_histories):
                    agent_choice = history[r]['decision']
                    round_summary.append(f"      Agent {agent_id}: {agent_choice}")

        # My Payoff (per-round)
        if payoff_config['my_payoff']['include'] and payoff_config['my_payoff']['rounds'] == "per-round":
            my_payoff = all_agent_histories[session_id][r]['payoff']
            round_summary.append(f"    Your Payoff: {my_payoff}")

        # Everyone's Payoff (per-round)
        if payoff_config['everyone_payoff']['include'] and payoff_config['everyone_payoff']['rounds'] == "per-round":
            if payoff_config['everyone_payoff']['type'] == "aggregate":
                avg_payoff = compute_average_payoff(all_agent_histories, r)
                round_summary.append(f"    Average Payoff of All Agents: {avg_payoff}")
            elif payoff_config['everyone_payoff']['type'] == "individual":
                round_summary.append("    Payoffs by Agent:")
                for agent_id, history in enumerate(all_agent_histories):
                    agent_payoff = history[r]['payoff']
                    round_summary.append(f"      Agent {agent_id}: {agent_payoff}")

        # My Regret (per-round)
        if regret_config['my_regret']['include'] and regret_config['my_regret']['rounds'] == "per-round":
            my_regret = all_agent_histories[session_id][r].get('regret', 0)
            round_summary.append(f"    Your Regret: {my_regret}")

        # Everyone's Regret (per-round)
        if regret_config['everyone_regret']['include'] and regret_config['everyone_regret']['rounds'] == "per-round":
            if regret_config['everyone_regret']['type'] == "aggregate":
                avg_regret = compute_average_regret(all_agent_histories, r)
                round_summary.append(f"    Average Regret of All Agents: {avg_regret}")
            elif regret_config['everyone_regret']['type'] == "individual":
                round_summary.append("    Regrets by Agent:")
                for agent_id, history in enumerate(all_agent_histories):
                    agent_regret = history[r].get('regret', 0)
                    round_summary.append(f"      Agent {agent_id}: {agent_regret}")

        # Only include the round header if there's per-round information
        if round_summary:
            summary += f"  Round {r + 1}:\n" + "\n".join(round_summary) + "\n"

    # Aggregate statistics
    aggregate_summary = []

    # My Choice (aggregate)
    if choices_config['my_choice']['include'] and choices_config['my_choice']['rounds'] == "aggregate":
        my_choice_counts = compute_individual_aggregate_choices(all_agent_histories[session_id][:round_num])
        aggregate_summary.append(f"  Your Aggregated Choices From Previous Rounds: {my_choice_counts}")

    # Everyone's Choices (aggregate)
    if choices_config['everyone_choices']['include'] and choices_config['everyone_choices']['rounds'] == "aggregate":
        if choices_config['everyone_choices']['type'] == "aggregate":
            overall_distribution = compute_aggregate_route_distribution(all_agent_histories, round_num)
            aggregate_summary.append(f"  Aggregated Choices of All Agents From Previous Rounds: {overall_distribution}")
        elif choices_config['everyone_choices']['type'] == "individual":
            aggregate_summary.append("  Aggregated Choices From Previous Rounds by Agent:")
            for agent_id, history in enumerate(all_agent_histories):
                agent_aggregate = compute_individual_aggregate_choices(history[:round_num])
                aggregate_summary.append(f"    Agent {agent_id}: {agent_aggregate}")

    # My Payoff (averaged)
    if payoff_config['my_payoff']['include'] and payoff_config['my_payoff']['rounds'] == "averaged":
        my_payoffs = [round['payoff'] for round in all_agent_histories[session_id][:round_num]]
        avg_my_payoff = sum(my_payoffs) / len(my_payoffs) if my_payoffs else 0
        aggregate_summary.append(f"  Your Average Payoff Per-Round: {avg_my_payoff}")

    # Everyone's Payoff (averaged)
    if payoff_config['everyone_payoff']['include'] and payoff_config['everyone_payoff']['rounds'] == "averaged":
        if payoff_config['everyone_payoff']['type'] == "aggregate":
            avg_payoff = compute_average_payoff(all_agent_histories, round_num)
            aggregate_summary.append(f"  Average Payoff of All Agents Per-Round: {avg_payoff}")
        elif payoff_config['everyone_payoff']['type'] == "individual":
            aggregate_summary.append("  Average Payoff Per-Round by Agent:")
            for agent_id, history in enumerate(all_agent_histories):
                agent_payoffs = [round['payoff'] for round in history[:round_num]]
                avg_agent_payoff = sum(agent_payoffs) / len(agent_payoffs) if agent_payoffs else 0
                aggregate_summary.append(f"    Agent {agent_id}: {avg_agent_payoff}")

    # My Regret (averaged)
    if regret_config['my_regret']['include'] and regret_config['my_regret']['rounds'] == "averaged":
        my_regrets = [round.get('regret', 0) for round in all_agent_histories[session_id][:round_num]]
        avg_my_regret = sum(my_regrets) / len(my_regrets) if my_regrets else 0
        aggregate_summary.append(f"  Your Average Regret: {avg_my_regret}")

    # Everyone's Regret (averaged)
    if regret_config['everyone_regret']['include'] and regret_config['everyone_regret']['rounds'] == "averaged":
        if regret_config['everyone_regret']['type'] == "aggregate":
            avg_regret = compute_average_regret(all_agent_histories, round_num)
            aggregate_summary.append(f"  Average Regret of All Agents Per-Round: {avg_regret}")
        elif regret_config['everyone_regret']['type'] == "individual":
            aggregate_summary.append("  Average Regret Per-Round by Agent:")
            for agent_id, history in enumerate(all_agent_histories):
                agent_regrets = [round.get('regret', 0) for round in history[:round_num]]
                avg_agent_regret = sum(agent_regrets) / len(agent_regrets) if agent_regrets else 0
                aggregate_summary.append(f"    Agent {agent_id}: {avg_agent_regret}")

    # Add aggregate summary if present
    if aggregate_summary:
        summary += "Aggregate statistics:\n" + "\n".join(aggregate_summary)

    return summary

def prompt_llm(prompting_style, avail_routes, session_id, prompt_messages):
    """
    Helper function to send a prompt to the LLM and parse its response.

    Args:
        session_id (int): ID of the current agent.
        prompt_messages (list): List of SystemMessage and HumanMessage objects.

    Returns:
        str: The route chosen by the agent.
    """

    config = {"configurable": {"session_id": f"session_{session_id}"}}

    if prompting_style == "consecutive":
        with_message_history = RunnableWithMessageHistory(
            full_chain,
            get_session_history,
            input_messages_key="messages",
        )


        response = with_message_history.invoke(
        {"messages": prompt_messages},
        config=config,
        )

    elif prompting_style == "zero-shot":
        response = full_chain.invoke(
        {"messages": prompt_messages},
        config=config,
        )

    choice = parse_route_choice(response.content, avail_routes, session_id)
    return choice, response.content

# Helper function to parse the route choice
def parse_route_choice(llm_output, str_routes, session_id):
    """
    Parses the route choice from the LLM's response.

    Args:
        llm_output: The raw LLM output as a string.
        str_routes: List of valid route strings.
        session_id: The agent's session ID (for logging purposes).

    Returns:
        The chosen route as a string.
    """
    try:
        parsed_response = parser.parse(llm_output)
        return parsed_response.route
    except:
        # Parsing failed, implement fallback
        print(f"Agent {session_id}: Parsing failed, attempting to extract route from LLM output.")
        # Fallback parsing
        chosen_route = "Invalid"
        # Create a regex pattern to match any of the valid routes
        pattern = r'\b(' + '|'.join(re.escape(route) for route in str_routes) + r')\b'
        matches = re.findall(pattern, llm_output)
        if matches:
            # Take the last match
            chosen_route = matches[-1]
            print(f"Agent {session_id}: Extracted route '{chosen_route}' from LLM output.")
        else:
            print(f"Agent {session_id}: Could not extract route from LLM output.")
    return chosen_route


def compute_individual_aggregate_choices(history):
    """
    Computes the aggregated number of times each route was chosen by an individual agent.

    Args:
        history (list): List of decision histories for an individual agent.

    Returns:
        dict: A dictionary with the count of times each route was chosen.
    """
    route_counts = Counter(round['decision'] for round in history)
    return dict(route_counts)


def compute_route_distribution(all_agent_histories, round_num):
    """
    Computes the distribution of routes chosen in a specific round.

    Args:
        all_agent_histories (list): List of all agents' histories.
        round_num (int): The round number to analyze (zero-indexed).

    Returns:
        dict: A dictionary with route counts.
    """
    distribution = {}
    for history in all_agent_histories:
        route = history[round_num]['decision']
        if route not in distribution:
            distribution[route] = 0
        distribution[route] += 1
    return distribution

def compute_aggregate_route_distribution(all_agent_histories, round_num):
    """
    Computes the aggregate distribution of routes over all rounds up to the current round.

    Args:
        all_agent_histories (list): List of all agents' histories.
        round_num (int): Current round number (zero-indexed internally).

    Returns:
        dict: A dictionary with aggregate route counts.
    """
    distribution = {}
    for history in all_agent_histories:
        for i in range(round_num):
            route = history[i]['decision']
            if route not in distribution:
                distribution[route] = 0
            distribution[route] += 1
    return distribution

def compute_average_payoff(all_agent_histories, round_num):
    """
    Computes the average payoff of all agents for a specific round.

    Args:
        all_agent_histories (list): List of all agents' histories.
        round_num (int): The round number to analyze (zero-indexed).

    Returns:
        float: The average payoff for the round.
    """
    total_payoff = 0
    count = 0
    for history in all_agent_histories:
        total_payoff += history[round_num]['payoff']
        count += 1
    return total_payoff / count if count > 0 else 0

def compute_average_regret(all_agent_histories, round_num):
    """
    Computes the average regret of all agents for a specific round.

    Args:
        all_agent_histories (list): List of all agents' histories.
        round_num (int): The round number to analyze (zero-indexed).

    Returns:
        float: The average regret for the round.
    """
    total_regret = 0
    count = 0
    for history in all_agent_histories:
        total_regret += history[round_num]['regret']
        count += 1
    return total_regret / count if count > 0 else 0

# Simulation execution
def run_simulation(config, current_run_folder=None):
    num_agents = config['simulation']['num_agents']
    num_rounds = config['simulation']['num_rounds']
    has_bridge = config['simulation']['has_bridge']
    folder_name = config['simulation']['folder_name']
    file_name = config['simulation']['file_name']
    prompting_style = config['history_representation']['prompting_style']['type']

    # Setup the folder structure
    game_a_folder, game_b_folder, current_run_folder = setup_folders(folder_name, current_run_folder)

    # File paths for game_xA and game_xB
    file_path_a = os.path.join(game_a_folder, file_name)
    file_path_b = os.path.join(game_b_folder, file_name)

    network = TrafficNetwork(has_bridge)
    all_agent_histories = [[] for _ in range(num_agents)]
    available_routes = network.get_avail_routes()
    description = network.describe_graph()

    # Generate system message once
    system_message = generate_system_message(
        num_rounds,
        num_agents,
        len(available_routes),
        available_routes,
        has_bridge,
        description
    )

    # Initialize storage for zero-shot message logs
    zero_shot_logs = [[] for _ in range(num_agents)] if prompting_style == "zero-shot" else None

    for round_num in range(num_rounds):
        print(f"Starting round {round_num + 1}")

        for session_id in range(num_agents):
            # Create the prompt for the current agent
            prompt_messages = create_prompt(
                config=config,
                system_message=system_message,
                all_agent_histories=all_agent_histories,
                session_id=session_id,
                round_num=round_num,
                available_routes=available_routes
            )

            # Get the chosen route from the LLM
            chosen_route, response = prompt_llm(prompting_style, available_routes, session_id, prompt_messages)

            if chosen_route not in available_routes:
                print(f"Invalid route selected by Agent {session_id}: {chosen_route}")
                chosen_route = "Invalid"

            # Save prompt and response for zero-shot prompting
            if prompting_style == "zero-shot":
                round_log = []
                if round_num == 0:
                    round_log.append(f"System Message: {system_message}")
                for i, msg in enumerate(prompt_messages):
                    if isinstance(msg, HumanMessage):
                        round_log.append(f"Human: {msg.content}")
                round_log.append(f"LLM Output: {response}")
                zero_shot_logs[session_id].append(round_log)

            # Append the decision to the agent's history
            all_agent_histories[session_id].append({
                'round_num': round_num + 1,  # Reported as 1-indexed
                'decision': chosen_route,
                'payoff': None,  # To be updated later
                'regret': None   # To be updated later
            })

            if chosen_route != "Invalid":
                network.add_player_to_path(network.path_to_edges(chosen_route))

        # Compute payoffs and regrets after all agents make decisions
        route_counts = compute_route_distribution(all_agent_histories, round_num)

        for session_id in range(num_agents):
            chosen_route = all_agent_histories[session_id][round_num]['decision']

            if chosen_route == "Invalid":
                all_agent_histories[session_id][round_num]['payoff'] = 0
                continue

            cost = network.calculate_total_cost(network.path_to_edges(chosen_route))
            payoff = 400 - cost
            all_agent_histories[session_id][round_num]['payoff'] = payoff

        compute_regret_for_agents(all_agent_histories, round_num, network)

        network.reset_player_counts()

    # Write simulation results to CSV
    with open(file_path_a if "A" in folder_name else file_path_b, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["Round", "Agent", "Route", "Payoff", "Regret"])
        writer.writeheader()
        for agent_id, history in enumerate(all_agent_histories):
            for round_data in history:
                writer.writerow({
                    "Round": round_data['round_num'],
                    "Agent": agent_id,
                    "Route": round_data['decision'],
                    "Payoff": round_data['payoff'],
                    "Regret": round_data['regret']
                })

    # Save chat history
    if prompting_style == "consecutive":
        for session_id in range(num_agents):
            save_session_history(f"session_{session_id}", game_a_folder if "A" in folder_name else game_b_folder)
    if prompting_style == "zero-shot":
        for agent_id, log in enumerate(zero_shot_logs):
            target_folder = game_a_folder if "A" in folder_name else game_b_folder
            with open(os.path.join(target_folder, f"agent_{agent_id}_history.txt"), "w", encoding="utf-8") as file:
                for round_log in log:
                    for line in round_log:
                        file.write(f"{line}\n")
                    file.write("\n")

    print(f"Simulation complete.")

if __name__ == "__main__":
    # Initialize the run folder for this game
    current_run_folder = None

    # Run simulation for game_xA
    config_path = f'configs/config12B.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        game_a_folder, game_b_folder, current_run_folder = setup_folders(config['simulation']['folder_name'])
        run_simulation(config, current_run_folder=current_run_folder)
