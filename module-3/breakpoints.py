import os
import ssl
import httpx
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.graph import MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from IPython import get_ipython
from dotenv import load_dotenv

load_dotenv()

# Workaround for corporate SSL proxy
os.environ["SSL_CERT_FILE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a by b."""
    return a / b

tools: List = [add, multiply, divide]
http_client = httpx.Client(verify=False)
llm = ChatGroq(model="llama-3.3-70b-versatile", http_client=http_client)
llm_with_tools = llm.bind_tools(tools)
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
memory = MemorySaver()
graph = builder.compile(interrupt_before=["tools"], checkpointer=memory)

def run_basic_breakpoint():
    print("=== Basic breakpoint (thread 1) ===")
    initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}
    thread = {"configurable": {"thread_id": "1"}}
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()
    state = graph.get_state(thread)
    print("Next node (should show interruption point):", state.next)

def run_continue():
    print("=== Continue from breakpoint (thread 1) ===")
    thread = {"configurable": {"thread_id": "1"}}
    for event in graph.stream(None, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

def run_user_approval():
    print("=== Human approval flow (thread 2) ===")
    initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}
    thread = {"configurable": {"thread_id": "2"}}
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()
    user_approval = input("Do you want to call the tool? (yes/no): ")
    if user_approval.lower().startswith("y"):
        for event in graph.stream(None, thread, stream_mode="values"):
            event["messages"][-1].pretty_print()
    else:
        print("Operation cancelled by user.")

def main():
       run_basic_breakpoint()
       run_continue()
       run_user_approval()

if __name__ == "__main__":
       main()
