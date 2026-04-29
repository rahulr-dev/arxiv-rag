# ArXiv Semantic Search Engine

A modular Retrieval-Augmented Generation (RAG) system that uses an AI agent to navigate scientific research. Unlike static RAG, this system can autonomously decide to fetch and index new papers from ArXiv if the local database doesn't have the answer.

## Features
- Agentic Routing: Uses LangGraph to decide between Local RAG, ArXiv Ingestion, or Web Search.
- Dynamic Ingestion: Synchronously downloads, processes, and embeds new ArXiv papers on-the-fly.
- Hybrid Search: Combines Dense (SPECTER2) and Sparse (BM25) vectors with RRF fusion.
- Chat UI: Simple, ChatGPT-like interface for research exploration.

## Tech Stack
- LLM/Agent: Google Gemini Flash, LangGraph, LangChain
- Vector DB: Qdrant (Hybrid Indexing)
- Embeddings: SPECTER2 (Scientific specialized)
- Backend: FastAPI
- Frontend: Streamlit
- Processing: PyMuPDF, SemanticChunker

## Architecture
1. User Query: Sent to the LangGraph Agent.
2. Decision Node: The agent checks the local Qdrant DB via rag_search.
3. Tool Routing:
    - If found: Returns answer with citations.
    - If missing: Triggers arxiv_search_and_ingest to fetch papers matching the query.
    - Fallback: Uses web_search (DuckDuckGo) for general context.
4. Synthesis: Gemini generates a final response citing the specific sources used.

## How to Run


### 1. Environment Setup
Create a `.env` file in the root:
```env
GEMINI_API_KEY=your_google_ai_studio_key
QDRANT_URL=http://localhost:6333
```

### 2. Run with Docker (Recommended)
```bash
docker-compose up --build
```
- **UI**: `http://localhost:8501`
- **API**: `http://localhost:8000`

### 3. Run Locally
1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Start API**: `python src/api.py`
3. **Start UI**: `streamlit run src/ui.py`

*Note: If no Qdrant server is running, the local setup will automatically use embedded mode in `qdrant_storage/`.*
