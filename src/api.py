import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from src.agent import run_agent_query

# ── Logging ───────────────────────────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ── Models ────────────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    answer: str


# ── App Initialization ────────────────────────────────────────────────────────

app = FastAPI(title="ArXiv RAG API")

# Add CORS middleware to allow Streamlit to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        # Run the agent (this handles retrieval, ingestion if needed, and generation)
        answer = run_agent_query(request.query)

        return QueryResponse(query=request.query, answer=answer)

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
