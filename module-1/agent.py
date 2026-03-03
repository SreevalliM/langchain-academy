import os, httpx
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_groq import ChatGroq
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

from dotenv import load_dotenv

# Environment setup
load_dotenv("/Users/L107127/Library/CloudStorage/OneDrive-EliLillyandCompany/Desktop/langchain-academy/.env", override=True)

CA_BUNDLE = "/Users/L107127/Library/CloudStorage/OneDrive-EliLillyandCompany/Desktop/langchain-academy/ca-bundle.pem"
os.environ["SSL_CERT_FILE"] = CA_BUNDLE
os.environ["REQUESTS_CA_BUNDLE"] = CA_BUNDLE
http_client = httpx.Client(verify=CA_BUNDLE)

def multiply(a: int, b: int) -> int:
    """Multiply a by b and return the product."""
    return a * b

def add(a: int, b: int) -> int:
    """Add a and b and return the sum."""
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a by b and return the quotient."""
    return a / b

TOOLS = [add, multiply, divide]

llm = ChatGroq(model="qwen/qwen3-32b", http_client=http_client)
llm_with_tools = llm.bind_tools(TOOLS, parallel_tool_calls=False)

SYS_MESSAGE = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
)

def assistant(state: MessagesState):
    messages: List[BaseMessage] = state["messages"]
    response = llm_with_tools.invoke([SYS_MESSAGE] + messages)
    return {"messages": [response]}

def build_react_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    return builder.compile()

def run_query(query: str):
    graph = build_react_graph()
    state = graph.invoke({"messages": [HumanMessage(content=query)]})
    return state

if __name__ == "__main__":
    demo_query = "Add 3 and 4. Multiply the output by 2. Divide the output by 5"
    print(f"Query: {demo_query}\n---")
    final_state = run_query(demo_query)
    for m in final_state["messages"]:
        try:
            m.pretty_print()
        except AttributeError:
            print(m)
