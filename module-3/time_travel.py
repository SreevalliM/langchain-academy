"""
Time Travel with LangGraph

Demonstrates three key capabilities:
1. Browsing - View the full state history of an agent's execution
2. Replaying - Re-run the agent from any past checkpoint (cached, no LLM calls)
3. Forking  - Modify state at a past checkpoint and run the agent fresh from there
"""

import os
import httpx
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode

# ──────────────────────────────────────────────
# Environment setup
# ──────────────────────────────────────────────

load_dotenv(
    "/Users/L107127/Library/CloudStorage/OneDrive-EliLillyandCompany/Desktop/langchain-academy/.env",
    override=True,
)

CA_BUNDLE = "/Users/L107127/Library/CloudStorage/OneDrive-EliLillyandCompany/Desktop/langchain-academy/ca-bundle.pem"
os.environ["SSL_CERT_FILE"] = CA_BUNDLE
os.environ["REQUESTS_CA_BUNDLE"] = CA_BUNDLE

http_client = httpx.Client(verify=CA_BUNDLE)
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ──────────────────────────────────────────────
# Tool definitions
# ──────────────────────────────────────────────

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


def divide(a: int, b: int) -> float:
    """Divide a by b.

    Args:
        a: first int
        b: second int
    """
    return a / b


tools = [add, multiply, divide]

# ──────────────────────────────────────────────
# LLM setup
# ──────────────────────────────────────────────

llm = ChatGroq(model="llama-3.3-70b-versatile", http_client=http_client)
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
)


# ──────────────────────────────────────────────
# Graph construction
# ──────────────────────────────────────────────

def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


builder = StateGraph(MessagesState)

# Nodes
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Edges
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,  # routes to "tools" on tool call, END otherwise
)
builder.add_edge("tools", "assistant")

# Compile with an in-memory checkpointer (enables time travel)
graph = builder.compile(checkpointer=MemorySaver())


# ──────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────

def print_separator(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ──────────────────────────────────────────────
# 1. Run the agent
# ──────────────────────────────────────────────

if __name__ == "__main__":

    print_separator("STEP 1: Run the agent — 'Multiply 2 and 3'")

    initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}
    thread = {"configurable": {"thread_id": "1"}}

    for event in graph.stream(initial_input, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

    # ──────────────────────────────────────────
    # 2. Browse history
    # ──────────────────────────────────────────

    print_separator("STEP 2: Browse state history")

    current_state = graph.get_state({"configurable": {"thread_id": "1"}})
    print(f"Current state next node: {current_state.next}")

    all_states = list(graph.get_state_history(thread))
    print(f"Total checkpoints: {len(all_states)}")

    for i, state in enumerate(all_states):
        msgs = state.values.get("messages", [])
        print(f"  [{i}] next={state.next}, messages={len(msgs)}")

    # ──────────────────────────────────────────
    # 3. Replay from the initial human input
    # ──────────────────────────────────────────

    print_separator("STEP 3: Replay from the initial human input checkpoint")

    to_replay = all_states[-2]  # the checkpoint after the human message was received
    print(f"Replaying from checkpoint: {to_replay.config['configurable']['checkpoint_id']}")
    print(f"State values: {to_replay.values}")
    print(f"Next node: {to_replay.next}")
    print()

    for event in graph.stream(None, to_replay.config, stream_mode="values"):
        event["messages"][-1].pretty_print()

    # ──────────────────────────────────────────
    # 4. Fork — change input and re-run
    # ──────────────────────────────────────────

    print_separator("STEP 4: Fork — change 'Multiply 2 and 3' to 'Multiply 5 and 3'")

    to_fork = all_states[-2]
    original_msg_id = to_fork.values["messages"][0].id
    print(f"Overwriting message ID: {original_msg_id}")

    # Overwrite the human message at this checkpoint (same ID = overwrite, not append)
    fork_config = graph.update_state(
        to_fork.config,
        {"messages": [HumanMessage(content="Multiply 5 and 3", id=original_msg_id)]},
    )
    print(f"Forked checkpoint config: {fork_config}")
    print()

    # Run from the forked checkpoint — this is a fresh run, not a replay
    for event in graph.stream(None, fork_config, stream_mode="values"):
        event["messages"][-1].pretty_print()

    # ──────────────────────────────────────────
    # 5. Verify final state
    # ──────────────────────────────────────────

    print_separator("STEP 5: Final state after fork")

    final_state = graph.get_state({"configurable": {"thread_id": "1"}})
    print(f"Next node: {final_state.next}")
    for msg in final_state.values["messages"]:
        msg.pretty_print()
