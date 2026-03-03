
import os, httpx
from typing import List

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv

# Environment setup
load_dotenv("/Users/L107127/Library/CloudStorage/OneDrive-EliLillyandCompany/Desktop/langchain-academy/.env", override=True)

CA_BUNDLE = "/Users/L107127/Library/CloudStorage/OneDrive-EliLillyandCompany/Desktop/langchain-academy/ca-bundle.pem"
os.environ["SSL_CERT_FILE"] = CA_BUNDLE
os.environ["REQUESTS_CA_BUNDLE"] = CA_BUNDLE
http_client = httpx.Client(verify=CA_BUNDLE)


def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b


def add(a: int, b: int) -> int:
    """Add a and b."""
    return a + b


def divide(a: int, b: int) -> float:
    """Divide a and b."""
    return a / b


def build_graph(llm_with_tools):
    # System message
    sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

    # Node function
    def assistant(state: MessagesState):
        return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode([add, multiply, divide]))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    return builder


def run_without_memory(graph):
    print("\n=== No Memory Example ===")
    messages: List[BaseMessage] = [HumanMessage(content="Add 3 and 4.")]
    result = graph.invoke({"messages": messages})
    for m in result["messages"]:
        print(m)
    # New turn loses context
    messages = [HumanMessage(content="Multiply that by 2.")]
    result = graph.invoke({"messages": messages})
    for m in result["messages"]:
        print(m)


def run_with_memory(builder):
    print("\n=== With Memory Example ===")
    memory = MemorySaver()
    graph_with_memory = builder.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "1"}}

    messages: List[BaseMessage] = [HumanMessage(content="Add 3 and 4.")]
    result = graph_with_memory.invoke({"messages": messages}, config)
    for m in result["messages"]:
        print(m)

    # Now the second turn keeps context
    messages = [HumanMessage(content="Multiply that by 2.")]
    result = graph_with_memory.invoke({"messages": messages}, config)
    for m in result["messages"]:
        print(m)


def main():
    

    tools = [add, multiply, divide]
    llm = ChatGroq(model="qwen/qwen3-32b", http_client=http_client)
    llm_with_tools = llm.bind_tools(tools)

    builder = build_graph(llm_with_tools)
    graph = builder.compile()

    run_without_memory(graph)
    run_with_memory(builder)


if __name__ == "__main__":
    main()
