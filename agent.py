from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import operator
from langgraph.graph import END, StateGraph, START
from typing import Annotated, Sequence, TypedDict
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import tool
from payoff import create_payoff_tool
from langchain_openai import ChatOpenAI
from network import TrafficNetwork
import os


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCAHIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "bluesky"
os.environ["OPENAI_API_KEY"] = "sk-proj-2YIRzXSkGvD5mXF5KSwST3BlbkFJ32aEm9aV3nrL6gFGCr97"

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

class Agent:
    def __init__(self, thread_id: str, num_agents: int, network: TrafficNetwork):
        self.thread_id = thread_id
        self.num_agents = num_agents
        self.network = network
        self.tools = []
        self.model = self._initialize_model()
        self.memory = MemorySaver()
        self.runnable = None

        # Create the workflow for this agent
        self.workflow = self._create_workflow()

    def _initialize_model(self):
        # Initialize the model and bind the tools
        calculate_payoff = create_payoff_tool(self.num_agents, self.network)

        model = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=1)
        tools = [calculate_payoff]
        self.tools = tools
        return model.bind_tools(tools)

    def _create_workflow(self):
        # Define the function that determines whether to continue or not
        def should_continue(state):
            messages = state["messages"]
            last_message = messages[-1]
            # If there are no tool calls, then we finish
            if not last_message.tool_calls:
                return "end"
            # Otherwise if there is, we continue
            else:
                return "continue"

        # Define the function that calls the model
        def call_model(state):
            messages = state["messages"]
            response = self.model.invoke(messages)
            return {"messages": [response]}

        # Define a new graph
        workflow = StateGraph(AgentState)

        # Define the two nodes we will cycle between
        workflow.add_node("agent", call_model)
        tool_node = ToolNode(self.tools)
        workflow.add_node("action", tool_node)

        # Set the entry point as `agent`
        workflow.add_edge(START, "agent")

        # Add a conditional edge
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {"continue": "action", "end": END},
        )

        # Add a normal edge from `action` to `agent`
        workflow.add_edge("action", "agent")

        # Compile the workflow
        return workflow.compile(checkpointer=self.memory)

    def call(self, inputs):
        config = {"configurable": {"thread_id": self.thread_id}}
        return self.workflow.invoke(inputs, config=config)