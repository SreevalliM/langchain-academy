import json
from operator import add
from typing import List, Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

def _ensure_dict(item):
    """Ensure a log item is a dict (handles JSON-string serialization from Studio API)."""
    if isinstance(item, dict):
        return item
    if isinstance(item, str) and item.strip():
        try:
            parsed = json.loads(item)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return parsed  # caller will handle
        except (json.JSONDecodeError, ValueError):
            pass
    return {"id": str(item), "question": "", "answer": "", "docs": None, "grade": None, "grader": None, "feedback": None}

def _ensure_log_list(state_val):
    """Ensure a state value is a list of dicts.
    
    Handles multiple formats from Studio API:
    - Already a list of dicts
    - A dict with integer-string keys ("0", "1", ...) mapping to log dicts
    - A single dict that IS a log (has 'id' key)
    - A dict wrapping a list in one of its values (double-nested from API)
    - A JSON string encoding any of the above
    """
    # If it's a string, try to parse as JSON first
    if isinstance(state_val, str) and state_val.strip():
        try:
            state_val = json.loads(state_val)
        except (json.JSONDecodeError, ValueError):
            return []
    
    # If it's a dict
    if isinstance(state_val, dict):
        # Check if it looks like an indexed collection (keys are ints or stringified ints)
        try:
            int_keys = sorted([(int(k), v) for k, v in state_val.items()])
            return [_ensure_dict(v) for _, v in int_keys]
        except (ValueError, TypeError):
            pass
        
        # Check if it's a single log entry (has 'id' key)
        if "id" in state_val:
            return [state_val]
        
        # Otherwise it may be double-nested — look for a list value inside
        for v in state_val.values():
            if isinstance(v, list):
                return [_ensure_dict(item) for item in v]
        
        return [state_val]
    
    if isinstance(state_val, list):
        return [_ensure_dict(item) for item in state_val]
    
    return []

# The structure of the logs
class Log(TypedDict):
    id: str
    question: str
    docs: Optional[List]
    answer: str
    grade: Optional[int]
    grader: Optional[str]
    feedback: Optional[str]

# Failure Analysis Sub-graph
class FailureAnalysisState(TypedDict):
    cleaned_logs: List[Log]
    failures: List[Log]
    fa_summary: str
    processed_logs: List[str]

class FailureAnalysisOutputState(TypedDict):
    fa_summary: str
    processed_logs: List[str]

def get_failures(state):
    """ Get logs that contain a failure """
    cleaned_logs = _ensure_log_list(state["cleaned_logs"])
    failures = [log for log in cleaned_logs if "grade" in log]
    return {"failures": failures}

def generate_summary(state):
    """ Generate summary of failures """
    failures = _ensure_log_list(state["failures"])
    # Add fxn: fa_summary = summarize(failures)
    fa_summary = "Poor quality retrieval of Chroma documentation."
    return {"fa_summary": fa_summary, "processed_logs": [f"failure-analysis-on-log-{failure['id']}" for failure in failures]}

fa_builder = StateGraph(FailureAnalysisState,output_schema=FailureAnalysisOutputState)
fa_builder.add_node("get_failures", get_failures)
fa_builder.add_node("generate_summary", generate_summary)
fa_builder.add_edge(START, "get_failures")
fa_builder.add_edge("get_failures", "generate_summary")
fa_builder.add_edge("generate_summary", END)

# Summarization subgraph
class QuestionSummarizationState(TypedDict):
    cleaned_logs: List[Log]
    qs_summary: str
    report: str
    processed_logs: List[str]

class QuestionSummarizationOutputState(TypedDict):
    report: str
    processed_logs: List[str]

def generate_summary(state):
    cleaned_logs = _ensure_log_list(state["cleaned_logs"])
    # Add fxn: summary = summarize(generate_summary)
    summary = "Questions focused on usage of ChatOllama and Chroma vector store."
    return {"qs_summary": summary, "processed_logs": [f"summary-on-log-{log['id']}" for log in cleaned_logs]}

def send_to_slack(state):
    qs_summary = state["qs_summary"]
    # Add fxn: report = report_generation(qs_summary)
    report = "foo bar baz"
    return {"report": report}

qs_builder = StateGraph(QuestionSummarizationState,output_schema=QuestionSummarizationOutputState)
qs_builder.add_node("generate_summary", generate_summary)
qs_builder.add_node("send_to_slack", send_to_slack)
qs_builder.add_edge(START, "generate_summary")
qs_builder.add_edge("generate_summary", "send_to_slack")
qs_builder.add_edge("send_to_slack", END)

# Entry Graph
class EntryGraphState(TypedDict):
    raw_logs: List[Log]
    cleaned_logs: List[Log]
    fa_summary: str # This will only be generated in the FA sub-graph
    report: str # This will only be generated in the QS sub-graph
    processed_logs:  Annotated[List[int], add] # This will be generated in BOTH sub-graphs

def clean_logs(state):
    # Get logs — ensure they're proper dicts after API deserialization
    raw_val = state["raw_logs"]
    print(f"[DEBUG clean_logs] type={type(raw_val).__name__}, repr={repr(raw_val)[:500]}")
    raw_logs = _ensure_log_list(raw_val)
    print(f"[DEBUG clean_logs] after ensure: len={len(raw_logs)}")
    if raw_logs:
        print(f"[DEBUG clean_logs] first item: {repr(raw_logs[0])[:300]}")
    # Data cleaning raw_logs -> docs 
    cleaned_logs = raw_logs
    return {"cleaned_logs": cleaned_logs}

entry_builder = StateGraph(EntryGraphState)
entry_builder.add_node("clean_logs", clean_logs)
entry_builder.add_node("question_summarization", qs_builder.compile())
entry_builder.add_node("failure_analysis", fa_builder.compile())

entry_builder.add_edge(START, "clean_logs")
entry_builder.add_edge("clean_logs", "failure_analysis")
entry_builder.add_edge("clean_logs", "question_summarization")
entry_builder.add_edge("failure_analysis", END)
entry_builder.add_edge("question_summarization", END)

graph = entry_builder.compile()