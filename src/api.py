import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from src.retrieval import RetrievalEngine
from src.generation import GenerationEngine

# ── Logging ───────────────────────────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ── Models ────────────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class SourceChunk(BaseModel):
    title: str
    authors: str
    text: str
    score: float
    pdf_url: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceChunk]


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

# Singleton instances for engines (lazy loaded to avoid start-up delay if models are large)
_retrieval_engine = None
_generation_engine = None


def get_engines():
    global _retrieval_engine, _generation_engine
    if _retrieval_engine is None:
        _retrieval_engine = RetrievalEngine()
    if _generation_engine is None:
        _generation_engine = GenerationEngine()
    return _retrieval_engine, _generation_engine


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        retrieval_engine, generation_engine = get_engines()

        # 1. Retrieve & Rerank
        relevant_chunks = retrieval_engine.query(request.query, top_k=request.top_k)

        # 2. Generate Answer
        answer = generation_engine.generate_answer(request.query, relevant_chunks)

        # 3. Format Sources
        sources = [
            SourceChunk(
                title=c.payload.get("title", "Unknown"),
                authors=c.payload.get("authors", "Unknown"),
                text=c.payload.get("text", ""),
                score=getattr(c, "score", 0.0),
                pdf_url=c.payload.get("pdf_url"),
            )
            for c in relevant_chunks
        ]

        return QueryResponse(query=request.query, answer=answer, sources=sources)

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
