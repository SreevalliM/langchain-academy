import os, httpx
from groq import Groq
from langsmith import traceable
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults


# Load environment variables
load_dotenv(dotenv_path="/Users/L107127/Library/CloudStorage/OneDrive-EliLillyandCompany/Desktop/Foundation-Introduction-to-LangGraph---Python/.env", override=True)

CA_BUNDLE = "/Users/L107127/Library/CloudStorage/OneDrive-EliLillyandCompany/Desktop/Foundation-Introduction-to-LangGraph---Python/ca-bundle.pem"
os.environ["SSL_CERT_FILE"] = CA_BUNDLE
os.environ["REQUESTS_CA_BUNDLE"] = CA_BUNDLE
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"


# Initialize web search tool
web_search_tool = TavilySearchResults(max_results=1)

# Define prompt template
prompt = """You are a professor and expert in explaining complex topics in a way that is easy to understand. 
Your job is to answer the provided question so that even a 5 year old can understand it. 
You have provided with relevant background context to answer the question.

Question: {question} 

Context: {context}

Answer:"""
# print("Prompt Template: ", prompt)


# Create Application
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

@traceable
def search(question):
    web_docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in web_docs])
    return web_results
    
@traceable(run_type="llm")
def explain(question, context):
    formatted = prompt.format(question=question, context=context)
    
    completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": formatted},
            {"role": "user", "content": question},
        ],
        model="llama-3.3-70b-versatile",
    )
    return completion.choices[0].message.content

@traceable
def eli5(question):
    context = search(question)
    answer = explain(question, context)
    return answer

# Run the application
question = "What is trustcall?"
print(eli5(question))