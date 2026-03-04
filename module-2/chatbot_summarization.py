import os
import httpx
import getpass
from typing import Literal
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv("/Users/L107127/Library/CloudStorage/OneDrive-EliLillyandCompany/Desktop/langchain-academy/.env", override=True)

CA_BUNDLE = "/Users/L107127/Library/CloudStorage/OneDrive-EliLillyandCompany/Desktop/langchain-academy/ca-bundle.pem"
os.environ["SSL_CERT_FILE"] = CA_BUNDLE
os.environ["REQUESTS_CA_BUNDLE"] = CA_BUNDLE
http_client = httpx.Client(verify=CA_BUNDLE)

class State(MessagesState):
    summary: str

model = ChatGroq(model="qwen/qwen3-32b", http_client=http_client)


def call_model(state: State):
    print("Calling model with messages:")
    for msg in state["messages"]:
        print(f" - {msg.content}")
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
    response = model.invoke(messages)
    print("Model response:")
    print(f" - {response.content}")
    return {"messages": response}


def summarize_conversation(state: State):
    print("Summarizing conversation...")
    summary = state.get("summary", "")
    print(f"Existing summary: {summary}")
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    print("New summary:")
    print(f" - {response.content}")
    return {"summary": response.content, "messages": delete_messages}


def should_continue(state: State):
    print("Checking if conversation should continue...")
    messages = state["messages"]
    if len(messages) > 6:
        print("Conversation is long, summarizing...")
        return "summarize_conversation"
    print("Conversation is short, continuing...")
    return END


workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)


def pretty_print_messages(msgs):
    print("Pretty printing messages:")
    for m in msgs:
        try:
            m.pretty_print()
        except AttributeError:
            role = getattr(m, 'type', 'message')
            print(f"[{role}] {getattr(m, 'content', m)}")


def run_demo():
    print("Starting demo thread...\n")
    config = {"configurable": {"thread_id": "demo-thread"}}

    user_inputs = [
        "hi! I'm Lance",
        "what's my name?",
        "i like the 49ers!",
        "i like Nick Bosa, isn't he the highest paid defensive player?",
        "also, I'm planning to watch the game this weekend",
        "do you remember what team I said I like?",
    ]

    for idx, text in enumerate(user_inputs, 1):
        print(f"\n--- User Turn {idx}: {text}")
        output = graph.invoke({"messages": [HumanMessage(content=text)]}, config)
        pretty_print_messages(output["messages"])
        summary = graph.get_state(config).values.get("summary", "")
        if summary:
            print("\nCurrent Summary:\n", summary)

    print("\nFinal State Keys:", list(graph.get_state(config).values.keys()))
    print("Summary Present:", bool(graph.get_state(config).values.get("summary")))


if __name__ == "__main__":
    run_demo()
