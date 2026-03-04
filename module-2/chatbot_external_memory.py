import os
import sys
import sqlite3
import getpass
import argparse
import httpx
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv("/Users/L107127/Library/CloudStorage/OneDrive-EliLillyandCompany/Desktop/langchain-academy/.env", override=True)

CA_BUNDLE = "/Users/L107127/Library/CloudStorage/OneDrive-EliLillyandCompany/Desktop/langchain-academy/ca-bundle.pem"
os.environ["SSL_CERT_FILE"] = CA_BUNDLE
os.environ["REQUESTS_CA_BUNDLE"] = CA_BUNDLE
http_client = httpx.Client(verify=CA_BUNDLE)

# Single model initialization (removed duplicate)
model = ChatGroq(model="qwen/qwen3-32b", http_client=http_client)

class State(MessagesState):
    summary: str

# Optional debug flag
DEBUG = os.getenv("DEBUG_CHATBOT", "0") not in {"0", "false", "False", ""}

def _dbg(msg: str):
    if DEBUG:
        print(f"[DEBUG] {msg}")


def call_model(state: State):
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
    _dbg(f"Calling model with {len(messages)} messages (summary present={bool(summary)})")
    response = model.invoke(messages)
    return {"messages": response}


def summarize_conversation(state: State):
    summary = state.get("summary", "")
    if summary:
        summary_prompt = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_prompt = "Create a summary of the conversation above:"
    messages = state["messages"] + [HumanMessage(content=summary_prompt)]
    _dbg("Summarizing conversation")
    response = model.invoke(messages)
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


def should_continue(state: State):
    # After more than 6 messages, trigger summarization
    decision = "summarize_conversation" if len(state["messages"]) > 6 else END
    _dbg(f"should_continue -> {decision}")
    return decision


def build_graph(db_path: str | Path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _dbg(f"Using database at: {db_path}")
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    memory = SqliteSaver(conn)
    workflow = StateGraph(State)
    workflow.add_node("conversation", call_model)
    workflow.add_node(summarize_conversation)
    workflow.add_edge(START, "conversation")
    workflow.add_conditional_edges("conversation", should_continue)
    workflow.add_edge("summarize_conversation", END)
    return workflow.compile(checkpointer=memory)


def _print_messages(msgs):
    # Print only the latest AI message (or all if debug)
    to_print = msgs if DEBUG else msgs[-1:]
    for m in to_print:
        try:
            m.pretty_print()
        except Exception:
            role = getattr(m, 'type', 'message')
            print(f"[{role}] {getattr(m, 'content', m)}")


def show_state(graph, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    st = graph.get_state(config)
    print("State keys:", list(st.values.keys()))
    if st.values.get("summary"):
        print("\nCurrent Summary:\n", st.values["summary"])
    print("\nMessage Count:", len(st.values.get("messages", [])))


def run_demo(graph, thread_id: str):
    turns: List[str] = [
        "hi! I'm Lance",
        "what's my name?",
        "i like the 49ers!",
        "i like Nick Bosa, isn't he the highest paid defensive player?",
        "also, I'm planning to watch the game this weekend",
        "do you remember what team I said I like?",
    ]
    config = {"configurable": {"thread_id": thread_id}}
    for t in turns:
        print(f"\n[User] {t}")
        output = graph.invoke({"messages": [HumanMessage(content=t)]}, config)
        _print_messages(output["messages"])
    show_state(graph, thread_id)


def run_interactive(graph, thread_id: str):
    print("Interactive chat. Type /exit to quit, /state to inspect state.")
    config = {"configurable": {"thread_id": thread_id}}
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not user:
            continue
        if user.lower() in {"/exit", "/quit"}:
            break
        if user.lower() == "/state":
            show_state(graph, thread_id)
            continue
        output = graph.invoke({"messages": [HumanMessage(content=user)]}, config)
        _print_messages(output["messages"])


def parse_args():
    p = argparse.ArgumentParser(description="Chatbot with external memory and summarization")
    p.add_argument("--db", default="module-2/state_db/example.db", help="Path to sqlite database file (will be created if missing)")
    p.add_argument("--thread", default="external-demo", help="Thread id for conversation (persists across runs)")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--demo", action="store_true", help="Run canned demo conversation")
    group.add_argument("--interactive", action="store_true", help="Run interactive chat loop")
    p.add_argument("--show-state", action="store_true", help="Show state then exit (can combine with --demo/--interactive after)")
    return p.parse_args()


def main():
    args = parse_args()
    graph = build_graph(args.db)
    if args.show_state:
        show_state(graph, args.thread)
        if not (args.demo or args.interactive):
            return
    if args.demo:
        run_demo(graph, args.thread)
    elif args.interactive:
        run_interactive(graph, args.thread)
    else:
        # One-off stdin mode
        if not sys.stdin.isatty():
            user_input = sys.stdin.read().strip()
            if user_input:
                config = {"configurable": {"thread_id": args.thread}}
                output = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)
                _print_messages(output["messages"])
                return
        # Default to demo
        run_demo(graph, args.thread)


if __name__ == "__main__":
    main()
