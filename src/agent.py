import os
import operator
import logging
from typing import Annotated, List, TypedDict, Union

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.retrieval import RetrievalEngine
from src.ingestion_runner import IngestionRunner

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ── Engines Initialization ───────────────────────────────────────────────────

# We initialize these globally so they are loaded once
_retrieval_engine = RetrievalEngine()
_ingestion_runner = IngestionRunner(
    dense_encoder=_retrieval_engine.dense_model,
    embeddings_wrapper=None # It will create its own Specter2Embeddings wrapper
)

# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def rag_search(query: str):
    """
    Search the existing database of ArXiv research papers. 
    Use this as the first option for scientific or technical questions.
    """
    results = _retrieval_engine.query(query, top_k=5)
    if not results:
        return "No relevant papers found in the local database."
    
    context = ""
    for i, res in enumerate(results):
        payload = res.payload
        context += f"--- SOURCE {i+1} (Local DB) ---\n"
        context += f"Title: {payload.get('title')}\n"
        context += f"Authors: {payload.get('authors')}\n"
        context += f"Content: {payload.get('text')}\n\n"
    return context

@tool
def arxiv_search_and_ingest(query: str):
    """
    Search ArXiv for new papers, download them, and add them to the database.
    Use this if rag_search doesn't provide enough information or if the user asks for a specific paper not in the DB.
    This process takes about 20-30 seconds.
    """
    results = _ingestion_runner.ingest_and_retrieve(query, max_results=2)
    if not results:
        return "No new papers found on ArXiv for this query."
    
    context = ""
    for i, res in enumerate(results):
        context += f"--- SOURCE {i+1} (Newly Ingested) ---\n"
        context += f"Title: {res.title}\n"
        context += f"Authors: {res.authors}\n"
        context += f"Content: {res.text}\n\n"
    return context

# Fallback web search
web_search_tool = DuckDuckGoSearchRun()

tools = [rag_search, arxiv_search_and_ingest, web_search_tool]
tool_node = ToolNode(tools)

# ── Graph State ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# ── Nodes & Logic ────────────────────────────────────────────────────────────

model = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
model_with_tools = model.bind_tools(tools)

def call_model(state: AgentState):
    messages = state["messages"]
    
    # System prompt injection
    if len(messages) == 1 and isinstance(messages[0], HumanMessage):
        system_message = (
            "You are a helpful scientific research assistant. You have access to three tools:\n"
            "1. rag_search: Search your existing database of ArXiv papers.\n"
            "2. arxiv_search_and_ingest: Fetch and index NEW papers from ArXiv if needed.\n"
            "3. duckduckgo_search: General web search for non-scientific topics or recent news.\n\n"
            "Always prefer rag_search first. If it's insufficient, try arxiv_search_and_ingest. "
            "Use duckduckgo_search as a final fallback.\n"
            "Cite your sources using [SOURCE X] or [Web Search] notation."
        )
        # We don't want to modify history directly, so we prepend to the local list for the model call
        messages = [HumanMessage(content=system_message)] + messages
        
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

# ── Graph Definition ──────────────────────────────────────────────────────────

workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
)

workflow.add_edge("tools", "agent")

app = workflow.compile()

# ── Helper for API/UI ─────────────────────────────────────────────────────────

def run_agent_query(query: str):
    inputs = {"messages": [HumanMessage(content=query)]}
    result = app.invoke(inputs)
    content = result["messages"][-1].content
    
    if isinstance(content, list):
        # Join text blocks if it's a list of content blocks
        text_parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)
        return "\n".join(text_parts)
    
    return content

if __name__ == "__main__":
    # Test
    # print(run_agent_query("What is the core idea of the Attention is all you need paper?"))
    # print(run_agent_query("Tell me about the recent paper 'Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets'"))
    print(run_agent_query("Who won the last world cup?"))
