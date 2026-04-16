from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

# Load environment
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("Missing GROQ_API_KEY")

# LLM
llm = init_chat_model(
    model="llama-3.1-8b-instant",
    model_provider="groq",
    temperature=0
)

# State
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def make_tool_graph():

    # Tool
    @tool
    def add(a: float, b: float):
        """Add two numbers"""
        return a + b

    tools = [add]

    # Bind tools
    llm_with_tools = llm.bind_tools(tools)

    # LLM Node
    def call_llm_model(state: State):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # Graph
    builder = StateGraph(State)

    builder.add_node("tool_calling_llm", call_llm_model)
    builder.add_node("tools", ToolNode(tools))

    # Flow
    builder.add_edge(START, "tool_calling_llm")

    builder.add_conditional_edges(
        "tool_calling_llm",
        tools_condition
    )

    builder.add_edge("tools", "tool_calling_llm")

    graph = builder.compile()

    return graph


tool_agent = make_tool_graph()