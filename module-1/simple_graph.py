from typing_extensions import TypedDict
from typing import Literal
import random
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    graph_state: str

def node_1(state: State):
   
    return {"graph_state": state["graph_state"] + " I am"}

def node_2(state: State):
    
    return {"graph_state": state["graph_state"] + " happy!"}

def node_3(state: State):
   
    return {"graph_state": state["graph_state"] + " sad!"}

def decide_mood(state: State) -> Literal["node_2", "node_3"]:
    if random.random() < 0.5:
        return "node_2"
    return "node_3"

def build_graph():
    builder = StateGraph(State)
    builder.add_node("node_1", node_1)
    builder.add_node("node_2", node_2)
    builder.add_node("node_3", node_3)
    builder.add_edge(START, "node_1")
    builder.add_conditional_edges("node_1", decide_mood)
    builder.add_edge("node_2", END)
    builder.add_edge("node_3", END)
    return builder.compile()

def main():
    graph = build_graph()
    result = graph.invoke({"graph_state": "Hi, this is Lance."})
    print(result)
    
if __name__ == "__main__":
    main()
