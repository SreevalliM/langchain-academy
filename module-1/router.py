"""
Router Example: LangGraph with ToolNode and Conditional Routing
"""
import os
from getpass import getpass
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# Set environment variable for API key
# GROQ_API_KEY will be loaded from .env automatically

# Define tool function
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

# Initialize LLM and bind tool
tool_list = [multiply]
llm = ChatGroq(model="qwen/qwen3-32b")
llm_with_tools = llm.bind_tools(tool_list)

# Node function for tool-calling LLM
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tool_list))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition,
)
builder.add_edge("tools", END)
graph = builder.compile()

# Example invocation
if __name__ == "__main__":
    messages = [HumanMessage(content="Hello, what is 2 multiplied by 2?")]
    result = graph.invoke({"messages": messages})
    for m in result['messages']:
        m.pretty_print()

