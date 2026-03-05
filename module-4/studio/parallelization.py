import os, httpx
import operator
from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults

from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END

### Environment setup
env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
load_dotenv(env_path, override=True)

CA_BUNDLE = os.path.join(os.path.dirname(__file__), "..", "..", "ca-bundle.pem")
os.environ["SSL_CERT_FILE"] = CA_BUNDLE
os.environ["REQUESTS_CA_BUNDLE"] = CA_BUNDLE
http_client = httpx.Client(verify=CA_BUNDLE)

# Use llm_heavy for answer generation (only LLM call in this graph)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, http_client=http_client)

class State(TypedDict):
    question: str
    answer: str
    context: Annotated[list, operator.add]

def search_web(state):
    
    """ Retrieve docs from web search """

    try:
        # Search
        tavily_search = TavilySearchResults(max_results=3)
        search_docs = tavily_search.invoke(state['question'])

        # Handle case where tavily returns a string instead of list of dicts
        if isinstance(search_docs, str):
            formatted_search_docs = f'<Document href="tavily_search"/>\n{search_docs}\n</Document>'
        else:
            formatted_search_docs = "\n\n---\n\n".join(
                [
                    f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
                    for doc in search_docs
                ]
            )
    except Exception as e:
        formatted_search_docs = f'<Document href="web_search_error"/>\n{str(e)}\n</Document>'

    return {"context": [formatted_search_docs]} 

def search_wikipedia(state):
    
    """ Retrieve docs from wikipedia """

    try:
        # Search
        search_docs = WikipediaLoader(query=state['question'], 
                                      load_max_docs=2).load()

        # Format
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
                for doc in search_docs
            ]
        )
    except Exception as e:
        formatted_search_docs = f'<Document source="wikipedia_error"/>\n{str(e)}\n</Document>'

    return {"context": [formatted_search_docs]} 

def generate_answer(state):
    
    """ Node to answer a question """

    # Get state
    context = state["context"]
    question = state["question"]

    # Template
    answer_template = """Answer the question {question} using this context: {context}"""
    answer_instructions = answer_template.format(question=question, 
                                                       context=context)    
    
    # Answer
    answer = llm.invoke([SystemMessage(content=answer_instructions)]+[HumanMessage(content=f"Answer the question.")])
      
    # Append it to state
    return {"answer": answer}

# Add nodes
builder = StateGraph(State)

# Initialize each node with node_secret 
builder.add_node("search_web",search_web)
builder.add_node("search_wikipedia", search_wikipedia)
builder.add_node("generate_answer", generate_answer)

# Flow
builder.add_edge(START, "search_wikipedia")
builder.add_edge(START, "search_web")
builder.add_edge("search_wikipedia", "generate_answer")
builder.add_edge("search_web", "generate_answer")
builder.add_edge("generate_answer", END)
graph = builder.compile()
