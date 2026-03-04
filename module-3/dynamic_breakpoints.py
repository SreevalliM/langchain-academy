from typing import Any, Iterable
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from langgraph.graph import START, END, StateGraph

class State(TypedDict):
    input: str

def step_1(state: State) -> State:
    print("---Step 1---")
    return state

def step_2(state: State) -> State:
    # Conditionally interrupt if the length of the input is longer than 5 characters
    if len(state["input"]) > 5:
        interrupt(
            f"Received input that is longer than 5 characters: {state['input']}"
        )
    print("---Step 2---")
    return state

def step_3(state: State) -> State:
    print("---Step 3---")
    return state

def build_graph() -> Any:
    builder = StateGraph(State)
    builder.add_node("step_1", step_1)
    builder.add_node("step_2", step_2)
    builder.add_node("step_3", step_3)
    builder.add_edge(START, "step_1")
    builder.add_edge("step_1", "step_2")
    builder.add_edge("step_2", "step_3")
    builder.add_edge("step_3", END)
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


def run_with_interrupt_demo() -> None:
    print("\n=== Dynamic Breakpoint (Internal Interrupt) Demo ===\n")
    graph = build_graph()

    initial_input: State = {"input": "hello world"}  # > 5 chars triggers interrupt in step_2
    thread_config = {"configurable": {"thread_id": "demo-thread"}}

    print("Streaming until first interruption (or completion):")
    for event in graph.stream(initial_input, thread_config, stream_mode="values"):
        print(event)

    state = graph.get_state(thread_config)
    print("\nState 'next' after interruption:", state.next)
    print("Logged interrupted tasks (state.tasks):", state.tasks)

    print("\nAttempting to resume without changing state (will hit interrupt again):")
    for event in graph.stream(None, thread_config, stream_mode="values"):
        print(event)

    state = graph.get_state(thread_config)
    print("State 'next' still:", state.next)

    print("\nUpdating state to shorter input to bypass interrupt...")
    graph.update_state(thread_config, {"input": "hi"})
    print("Updated state:", graph.get_state(thread_config).next)

    print("Resuming execution:")
    for event in graph.stream(None, thread_config, stream_mode="values"):
        print(event)

    final_state = graph.get_state(thread_config)
    print("\nFinal state next (should be END or empty):", final_state.next)
    print("Done.\n")

if __name__ == "__main__":
    run_with_interrupt_demo()
  
