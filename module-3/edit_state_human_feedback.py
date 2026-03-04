import os
import httpx
from typing import Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
load_dotenv()

http_client = httpx.Client(verify=False)

def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

def add(a: int, b: int) -> int:
    """Add a and b."""
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a by b."""
    return a / b

TOOLS = [add, multiply, divide]
LLM = ChatGroq(model="llama-3.3-70b-versatile", http_client=http_client)
LLM_WITH_TOOLS = LLM.bind_tools(TOOLS)
SYS_MSG = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

def _assistant_node(state: MessagesState):
    return {"messages": [LLM_WITH_TOOLS.invoke([SYS_MSG] + state["messages"])]}

def build_local_graph_with_breakpoint():
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", _assistant_node)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    memory = MemorySaver()
    return builder.compile(interrupt_before=["assistant"], checkpointer=memory)

def run_local_state_edit_demo():
    print("=== Local State Edit Demo ===")
    graph = build_local_graph_with_breakpoint()
    thread = {"configurable": {"thread_id": "local-demo-1"}}
    initial_input = {"messages": "Multiply 2 and 3"}
    print("-- Streaming until first breakpoint (before assistant) --")
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()
    state = graph.get_state(thread)
    print("Next nodes to execute:", state.next)
    print("-- Updating state: user changes mind (multiply 3 and 3) --")
    graph.update_state(
        thread,
        {"messages": [HumanMessage(content="No, actually multiply 3 and 3!")]},
    )
    print("-- Current messages after update --")
    for m in graph.get_state(thread).values["messages"]:
        m.pretty_print()
    print("-- Resume execution (pass None) --")
    for event in graph.stream(None, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()
    print("-- Resume again (tool call result & completion) --")
    for event in graph.stream(None, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

def _assistant_hf(state: MessagesState):
    return {"messages": [LLM_WITH_TOOLS.invoke([SYS_MSG] + state["messages"])]}

def _human_feedback_node(state: MessagesState):
    pass

def build_human_feedback_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", _assistant_hf)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_node("human_feedback", _human_feedback_node)
    builder.add_edge(START, "human_feedback")
    builder.add_edge("human_feedback", "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "human_feedback")
    memory = MemorySaver()
    return builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)

def run_human_feedback_demo(user_override: Optional[str] = None):
    print("=== Human Feedback Node Demo ===")
    graph = build_human_feedback_graph()
    thread = {"configurable": {"thread_id": "human-feedback-1"}}
    initial_input = {"messages": "Multiply 2 and 3"}
    print("-- Initial run until interruption at human_feedback --")
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()
    if user_override is None:
        try:
            user_override = input("Tell me how you want to update the state: ")
        except EOFError:
            user_override = "No, actually multiply 3 and 3!"
    print(f"-- Applying human feedback: {user_override!r} --")
    graph.update_state(thread, {"messages": user_override}, as_node="human_feedback")
    print("-- Continue execution --")
    for event in graph.stream(None, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()
    print("-- Final continuation (if additional tool / assistant cycles) --")
    for event in graph.stream(None, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

if __name__ == "__main__":
    run_local_state_edit_demo()
    run_human_feedback_demo()
