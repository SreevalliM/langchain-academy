import os, httpx
from pprint import pprint
from typing import List

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    trim_messages,
)
from langchain_groq import ChatGroq
from langgraph.graph import MessagesState, StateGraph, START, END
from dotenv import load_dotenv

# Environment setup
load_dotenv("/Users/L107127/Library/CloudStorage/OneDrive-EliLillyandCompany/Desktop/langchain-academy/.env", override=True)

CA_BUNDLE = "/Users/L107127/Library/CloudStorage/OneDrive-EliLillyandCompany/Desktop/langchain-academy/ca-bundle.pem"
os.environ["SSL_CERT_FILE"] = CA_BUNDLE
os.environ["REQUESTS_CA_BUNDLE"] = CA_BUNDLE
http_client = httpx.Client(verify=CA_BUNDLE)

llm = ChatGroq(model="qwen/qwen3-32b", http_client=http_client)

def build_basic_graph(llm: ChatGroq):
    def chat_model_node(state: MessagesState):
        return {"messages": llm.invoke(state["messages"])}
    builder = StateGraph(MessagesState)
    builder.add_node("chat_model", chat_model_node)
    builder.add_edge(START, "chat_model")
    builder.add_edge("chat_model", END)
    return builder.compile()

def demo_basic(llm: ChatGroq):
    print("\n=== 1. Basic Chat Graph ===")
    messages: List = [AIMessage("So you said you were researching ocean mammals?", name="Bot")]
    messages.append(HumanMessage("Yes, I know about whales. But what others should I learn about?", name="Lance"))
    graph = build_basic_graph(llm)
    output = graph.invoke({"messages": messages})
    for m in output["messages"]:
        m.pretty_print()
    return output["messages"]

def build_filter_graph(llm: ChatGroq):
    def filter_messages(state: MessagesState):
        delete_msgs = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        print(f'deleted messages: {[m.id for m in delete_msgs]}')
        return {"messages": delete_msgs}
    def chat_model_node(state: MessagesState):
        return {"messages": [llm.invoke(state["messages"])]}
    builder = StateGraph(MessagesState)
    builder.add_node("filter", filter_messages)
    builder.add_node("chat_model", chat_model_node)
    builder.add_edge(START, "filter")
    builder.add_edge("filter", "chat_model")
    builder.add_edge("chat_model", END)
    return builder.compile()

def demo_filter(llm: ChatGroq):
    print("\n=== 2. Filter Messages with RemoveMessage ===")
    graph = build_filter_graph(llm)
    messages: List = [
        AIMessage("Hi.", name="Bot", id="1"),
        HumanMessage("Hi.", name="Lance", id="2"),
        AIMessage("So you said you were researching ocean mammals?", name="Bot", id="3"),
        HumanMessage(
            "Yes, I know about whales. But what others should I learn about?",
            name="Lance",
            id="4",
        ),
    ]
    output = graph.invoke({"messages": messages})
    for m in output["messages"]:
        m.pretty_print()
    return output["messages"]

def build_slice_graph(llm: ChatGroq):
    def chat_model_node(state: MessagesState):
        print(f'passing messages: {[m.id for m in state["messages"][-1:]]}')
        return {"messages": [llm.invoke(state["messages"][-1:])]}
    builder = StateGraph(MessagesState)
    builder.add_node("chat_model", chat_model_node)
    builder.add_edge(START, "chat_model")
    builder.add_edge("chat_model", END)
    return builder.compile()

def demo_slice_filter(llm: ChatGroq, prior_messages: List):
    print("\n=== 3. Slice Filtering (pass only last msg to model) ===")
    graph = build_slice_graph(llm)
    messages = list(prior_messages)
    messages.append(HumanMessage("Tell me more about Narwhals!", name="Lance"))
    for m in messages:
        m.pretty_print()
    output = graph.invoke({"messages": messages})
    print("-- Model Output --")
    for m in output["messages"]:
        m.pretty_print()
    return messages + output["messages"]

def build_trim_graph(llm: ChatGroq):
    def chat_model_node(state: MessagesState):
        limited = trim_messages(
            state["messages"],
            max_tokens=100,
            strategy="last",
            token_counter=llm,
            allow_partial=False,
        )
        return {"messages": [llm.invoke(limited)]}
    builder = StateGraph(MessagesState)
    builder.add_node("chat_model", chat_model_node)
    builder.add_edge(START, "chat_model")
    builder.add_edge("chat_model", END)
    return builder.compile()

def demo_trim(llm: ChatGroq, prior_messages: List):
    print("\n=== 4. Trim Messages (token-based) ===")
    graph = build_trim_graph(llm)
    messages = list(prior_messages)
    messages.append(HumanMessage("Tell me where Orcas live!", name="Lance"))
    output = graph.invoke({"messages": messages})
    for m in output["messages"]:
        m.pretty_print()
    return output["messages"]

def main():

    basic_messages = demo_basic(llm)
    filtered_messages = demo_filter(llm)
    combined_state = filtered_messages
    slice_state = demo_slice_filter(llm, combined_state)
    demo_trim(llm, slice_state)

if __name__ == "__main__":
    main()
