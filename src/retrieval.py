import os
import sqlite3
import logging
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Prefetch,
    SearchRequest,
    SparseVector,
    FusionQuery,
    Fusion,
    NamedVector,
)

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

DB_PATH = Path("data/arxiv_papers.db")
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
DENSE_MODEL = "allenai/specter2_base"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "arxiv_papers"

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ── Retrieval Engine ──────────────────────────────────────────────────────────

class RetrievalEngine:
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        logger.info(f"Loading dense encoder: {DENSE_MODEL}")
        self.dense_model = SentenceTransformer(DENSE_MODEL)
        
        logger.info(f"Loading reranker model: {RERANK_MODEL}")
        self.rerank_model = CrossEncoder(RERANK_MODEL)
        
        self.bm25 = self._initialize_bm25()
        logger.info("Retrieval Engine initialized")

    def _initialize_bm25(self) -> BM25Okapi:
        """Fetch all chunks from SQLite and build BM25 index for sparse encoding."""
        logger.info("Initializing BM25 index from SQLite chunks...")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.execute("SELECT text FROM chunks")
        texts = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if not texts:
            logger.warning("No chunks found in database. BM25 will be empty.")
            return BM25Okapi([])
            
        tokenized = [text.lower().split() for text in texts]
        return BM25Okapi(tokenized)

    def encode_dense(self, query: str) -> list[float]:
        return self.dense_model.encode(query, normalize_embeddings=True).tolist()

    def encode_sparse(self, query: str) -> SparseVector:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        nonzero_idx = np.nonzero(scores)[0].tolist()
        nonzero_val = scores[nonzero_idx].tolist()
        return SparseVector(indices=nonzero_idx, values=nonzero_val)

    def hybrid_search(self, query: str, limit: int = 20) -> list:
        """
        Perform hybrid search (dense + sparse) using Qdrant.
        Combines results using Reciprocal Rank Fusion (RRF).
        """
        dense_vec = self.encode_dense(query)
        sparse_vec = self.encode_sparse(query)

        # ── Refinement: Retrieve a much larger pool for the reranker ──
        fetch_limit = 100 
        
        # ── Refinement: Add a dedicated title-boosted prefetch if keywords look like a title ──
        # We'll use the sparse encoder on the query but we could also add a filter.
        
        try:
            results = self.client.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=[
                    Prefetch(
                        query=dense_vec, 
                        using="dense", 
                        limit=fetch_limit
                    ),
                    Prefetch(
                        query=sparse_vec, 
                        using="sparse", 
                        limit=fetch_limit
                    ),
                    # Extra boost for chunks where the title itself matches the query
                    Prefetch(
                        query=sparse_vec,
                        using="sparse",
                        filter=models.Filter(
                            should=[
                                models.FieldCondition(
                                    key="title",
                                    match=models.MatchText(text=query)
                                )
                            ]
                        ),
                        limit=fetch_limit
                    )
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=fetch_limit,
                with_payload=True,
            ).points
        except Exception as e:
            logger.warning(f"Native RRF query failed: {e}")
            results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=NamedVector(name="dense", vector=dense_vec),
                limit=fetch_limit,
                with_payload=True,
            )

        return results

    def rerank(self, query: str, results: list, top_k: int = 5) -> list:
        """
        Rerank retrieved chunks using a Cross-Encoder.
        """
        if not results:
            return []

        # Prepare pairs for cross-encoder: (query, chunk_text)
        pairs = [[query, r.payload["text"]] for r in results]
        scores = self.rerank_model.predict(pairs)

        # Attach scores and sort
        for i, r in enumerate(results):
            r.score = float(scores[i])

        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        return sorted_results[:top_k]

    def query(self, query: str, top_k: int = 5) -> list:
        """Full retrieval pipeline: Hybrid Search -> Reranking."""
        logger.info(f"Processing query: {query}")
        raw_results = self.hybrid_search(query, limit=20)
        reranked = self.rerank(query, raw_results, top_k=top_k)
        logger.info(f"Retrieved {len(reranked)} chunks after reranking")
        return reranked

if __name__ == "__main__":
    # Quick test
    engine = RetrievalEngine()
    test_query = "How does Retrieval Augmented Generation work?"
    results = engine.query(test_query)
    for i, res in enumerate(results):
        print(f"\n[{i+1}] Score: {res.score:.4f} | Paper: {res.payload['title']}")
        print(f"Snippet: {res.payload['text'][:200]}...")
