"""
Router Example: LangGraph with ToolNode and Conditional Routing
"""
import os, httpx
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv("/Users/L107127/Library/CloudStorage/OneDrive-EliLillyandCompany/Desktop/langchain-academy/.env", override=True)

CA_BUNDLE = "/Users/L107127/Library/CloudStorage/OneDrive-EliLillyandCompany/Desktop/langchain-academy/ca-bundle.pem"
os.environ["SSL_CERT_FILE"] = CA_BUNDLE
os.environ["REQUESTS_CA_BUNDLE"] = CA_BUNDLE
http_client = httpx.Client(verify=CA_BUNDLE)

# Define tool function
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

# Initialize LLM and bind tool
tool_list = [multiply]
llm = ChatGroq(model="qwen/qwen3-32b", http_client=http_client)
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
    messages = [HumanMessage(content="Hello, what is genai and can you multiply 6 and 7 for me?")]
    result = graph.invoke({"messages": messages})
    for m in result['messages']:
        m.pretty_print()

