"""
State Reducers - Module 2
Demonstrates default overwriting, branching, custom reducers, and message handling.
"""
from operator import add
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.errors import InvalidUpdateError
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage, RemoveMessage


# ============================================================
# 1. Default Overwriting State
# ============================================================

class OverwriteState(TypedDict):
    foo: int

def overwrite_node_1(state):
    print("---Node 1---")
    return {"foo": state['foo'] + 1}

def demo_overwrite():
    print("\n=== 1. Default Overwriting State ===")
    builder = StateGraph(OverwriteState)
    builder.add_node("node_1", overwrite_node_1)
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", END)
    graph = builder.compile()
    result = graph.invoke({"foo": 1})
    print("Result:", result)


# ============================================================
# 2. Branching (shows the problem without reducers)
# ============================================================

class BranchState(TypedDict):
    foo: int

def branch_node_1(state):
    print("---Node 1---")
    return {"foo": state['foo'] + 1}

def branch_node_2(state):
    print("---Node 2---")
    return {"foo": state['foo'] + 1}

def branch_node_3(state):
    print("---Node 3---")
    return {"foo": state['foo'] + 1}

def demo_branching_error():
    print("\n=== 2. Branching Without Reducer (InvalidUpdateError) ===")
    builder = StateGraph(BranchState)
    builder.add_node("node_1", branch_node_1)
    builder.add_node("node_2", branch_node_2)
    builder.add_node("node_3", branch_node_3)
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", "node_2")
    builder.add_edge("node_1", "node_3")
    builder.add_edge("node_2", END)
    builder.add_edge("node_3", END)
    graph = builder.compile()
    try:
        graph.invoke({"foo": 1})
    except InvalidUpdateError as e:
        print(f"InvalidUpdateError occurred: {e}")


# ============================================================
# 3. Reducers (operator.add for list concatenation)
# ============================================================

class ReducerState(TypedDict):
    foo: Annotated[list[int], add]

def reducer_node_1(state):
    print("---Node 1---")
    return {"foo": [state['foo'][-1] + 1]}

def reducer_node_2(state):
    print("---Node 2---")
    return {"foo": [state['foo'][-1] + 1]}

def reducer_node_3(state):
    print("---Node 3---")
    return {"foo": [state['foo'][-1] + 1]}

def demo_reducer():
    print("\n=== 3. Reducer with operator.add ===")
    # Simple single-node graph
    builder = StateGraph(ReducerState)
    builder.add_node("node_1", reducer_node_1)
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", END)
    graph = builder.compile()
    result = graph.invoke({"foo": [1]})
    print("Single node result:", result)

    # Branching graph with reducer
    print()
    builder = StateGraph(ReducerState)
    builder.add_node("node_1", reducer_node_1)
    builder.add_node("node_2", reducer_node_2)
    builder.add_node("node_3", reducer_node_3)
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", "node_2")
    builder.add_edge("node_1", "node_3")
    builder.add_edge("node_2", END)
    builder.add_edge("node_3", END)
    graph = builder.compile()
    result = graph.invoke({"foo": [1]})
    print("Branching result:", result)

    # None input causes TypeError with operator.add
    print()
    try:
        graph.invoke({"foo": None})
    except TypeError as e:
        print(f"TypeError with None input: {e}")


# ============================================================
# 4. Custom Reducers
# ============================================================

def reduce_list(left: list | None, right: list | None) -> list:
    """Safely combine two lists, handling None inputs."""
    if not left:
        left = []
    if not right:
        right = []
    return left + right

class DefaultState(TypedDict):
    foo: Annotated[list[int], add]

class CustomReducerState(TypedDict):
    foo: Annotated[list[int], reduce_list]

def custom_node_1(state):
    print("---Node 1---")
    return {"foo": [2]}

def demo_custom_reducer():
    print("\n=== 4. Custom Reducer ===")

    # Default reducer fails with None
    builder = StateGraph(DefaultState)
    builder.add_node("node_1", custom_node_1)
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", END)
    graph = builder.compile()
    try:
        print(graph.invoke({"foo": None}))
    except TypeError as e:
        print(f"Default reducer TypeError: {e}")

    # Custom reducer handles None
    builder = StateGraph(CustomReducerState)
    builder.add_node("node_1", custom_node_1)
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", END)
    graph = builder.compile()
    try:
        result = graph.invoke({"foo": None})
        print(f"Custom reducer result: {result}")
    except TypeError as e:
        print(f"TypeError occurred: {e}")


# ============================================================
# 5. Messages and add_messages reducer
# ============================================================

def demo_messages():
    print("\n=== 5. Messages with add_messages ===")

    # Appending messages
    initial_messages = [
        AIMessage(content="Hello! How can I assist you?", name="Model"),
        HumanMessage(content="I'm looking for information on marine biology.", name="Lance"),
    ]
    new_message = AIMessage(content="Sure, I can help with that. What specifically are you interested in?", name="Model")
    result = add_messages(initial_messages, new_message)
    print("After appending:")
    for m in result:
        print(f"  {m.name}: {m.content}")

    # Rewriting messages by ID
    print("\nRewriting by ID:")
    initial_messages = [
        AIMessage(content="Hello! How can I assist you?", name="Model", id="1"),
        HumanMessage(content="I'm looking for information on marine biology.", name="Lance", id="2"),
    ]
    new_message = HumanMessage(content="I'm looking for information on whales, specifically", name="Lance", id="2")
    result = add_messages(initial_messages, new_message)
    for m in result:
        print(f"  [{m.id}] {m.name}: {m.content}")

    # Removing messages
    print("\nRemoving messages:")
    messages = [
        AIMessage("Hi.", name="Bot", id="1"),
        HumanMessage("Hi.", name="Lance", id="2"),
        AIMessage("So you said you were researching ocean mammals?", name="Bot", id="3"),
        HumanMessage("Yes, I know about whales. But what others should I learn about?", name="Lance", id="4"),
    ]
    delete_messages = [RemoveMessage(id=m.id) for m in messages[:-2]]
    print(f"Deleting IDs: {[m.id for m in delete_messages]}")
    result = add_messages(messages, delete_messages)
    print("Remaining messages:")
    for m in result:
        print(f"  [{m.id}] {m.name}: {m.content}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    demo_overwrite()
    demo_branching_error()
    demo_reducer()
    demo_custom_reducer()
    demo_messages()
