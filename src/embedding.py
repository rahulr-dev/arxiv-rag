import os
import sqlite3
import logging
import numpy as np

from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
)

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

DB_PATH = Path("data/arxiv_papers.db")
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
DENSE_MODEL = "allenai/specter2_base"
EMBEDDING_DIM = 768
COLLECTION_NAME = "arxiv_papers"
BATCH_SIZE = 64

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ── Data Model ────────────────────────────────────────────────────────────────


@dataclass
class ChunkRecord:
    chunk_id: str
    arxiv_id: str
    chunk_index: int
    text: str  # raw text — goes into Qdrant payload for display
    embed_text: str  # title [SEP] chunk — used for encoding
    section: str
    strategy: str
    token_count: int
    title: str  # joined from papers table
    authors: str  # JSON string
    published: str
    pdf_url: str


# ── Database ──────────────────────────────────────────────────────────────────


def fetch_all_chunks(conn: sqlite3.Connection) -> list[ChunkRecord]:
    """
    Join chunks with papers to get full metadata.
    Everything needed at query time lives in Qdrant payload —
    no SQLite dependency at runtime.
    """
    cursor = conn.execute("""
        SELECT
            c.chunk_id,
            c.arxiv_id,
            c.chunk_index,
            c.text,
            c.embed_text,
            c.section,
            c.strategy,
            c.token_count,
            p.title,
            p.authors,
            p.published_date,
            p.pdf_url
        FROM chunks c
        JOIN papers p ON c.arxiv_id = p.arxiv_id
        ORDER BY c.arxiv_id, c.chunk_index
    """)
    return [
        ChunkRecord(
            chunk_id=row[0],
            arxiv_id=row[1],
            chunk_index=row[2],
            text=row[3],
            embed_text=row[4],
            section=row[5],
            strategy=row[6],
            token_count=row[7],
            title=row[8],
            authors=row[9],
            published=row[10],
            pdf_url=row[11],
        )
        for row in cursor.fetchall()
    ]


# ── Qdrant Setup ──────────────────────────────────────────────────────────────


def get_qdrant_client() -> QdrantClient:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    logger.info(f"Connected to Qdrant at {QDRANT_URL}")
    return client


def init_collection(client: QdrantClient) -> None:
    """
    Create Qdrant collection with:
    - dense vector config  : 768-dim cosine similarity
    - sparse vector config : BM25 keyword vectors
    Skips creation if collection already exists.
    """
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        logger.info(
            f"Collection '{COLLECTION_NAME}' already exists — skipping creation"
        )
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
        },
    )
    logger.info(f"Collection '{COLLECTION_NAME}' created with dense + sparse vectors")


def get_indexed_ids(client: QdrantClient) -> set[str]:
    """
    Fetch all chunk_ids already in Qdrant.
    Used for resumability — avoids re-indexing on reruns.
    """
    indexed = set()
    offset = None

    while True:
        results, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            with_payload=["chunk_id"],
            with_vectors=False,
            limit=1000,
            offset=offset,
        )
        for point in results:
            indexed.add(point.payload["chunk_id"])

        if next_offset is None:
            break
        offset = next_offset

    logger.info(f"Found {len(indexed)} already-indexed chunks in Qdrant")
    return indexed


# ── Dense Encoding ────────────────────────────────────────────────────────────


def load_dense_encoder() -> SentenceTransformer:
    logger.info(f"Loading dense encoder: {DENSE_MODEL}")
    model = SentenceTransformer(DENSE_MODEL)
    logger.info("Dense encoder loaded")
    return model


def encode_dense(
    model: SentenceTransformer,
    texts: list[str],
) -> np.ndarray:
    """
    Encode texts using SPECTER2.
    Uses embed_text (title [SEP] chunk) — not raw text.
    normalize_embeddings=True required for cosine similarity.
    """
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


# ── Sparse Encoding (BM25) ────────────────────────────────────────────────────


def build_bm25_index(texts: list[str]) -> BM25Okapi:
    """Build BM25 index over all chunk texts for sparse vector generation."""
    logger.info("Building BM25 index over all chunks")
    tokenized = [text.lower().split() for text in texts]
    return BM25Okapi(tokenized)


def encode_sparse(bm25: BM25Okapi, text: str) -> SparseVector:
    """
    Generate a sparse BM25 vector for a single chunk.
    Only keeps non-zero terms — Qdrant sparse format requires indices + values.
    """
    tokens = text.lower().split()
    scores = bm25.get_scores(tokens)
    nonzero_idx = np.nonzero(scores)[0].tolist()
    nonzero_val = scores[nonzero_idx].tolist()

    return SparseVector(
        indices=nonzero_idx,
        values=nonzero_val,
    )


def encode_sparse_batch(bm25: BM25Okapi, texts: list[str]) -> list[SparseVector]:
    """
    Score all texts against BM25 index in one vectorized pass.
    Significantly faster than calling get_scores() per chunk.
    """
    sparse_vectors = []

    for text in texts:
        tokens = text.lower().split()
        scores = bm25.get_scores(tokens)
        nonzero_idx = np.nonzero(scores)[0].tolist()
        nonzero_val = scores[nonzero_idx].tolist()
        sparse_vectors.append(SparseVector(indices=nonzero_idx, values=nonzero_val))

    return sparse_vectors


# ── Upsert ────────────────────────────────────────────────────────────────────


def upsert_batch(
    client: QdrantClient,
    chunks: list[ChunkRecord],
    dense_vectors: np.ndarray,
    sparse_vectors: list[SparseVector],
) -> None:
    """Build PointStructs and upsert to Qdrant in one batch."""
    points = []

    for i, chunk in enumerate(chunks):
        points.append(
            PointStruct(
                id=abs(hash(chunk.chunk_id)) % (2**63),  # Qdrant needs int/uuid id
                vector={
                    "dense": dense_vectors[i].tolist(),
                    "sparse": sparse_vectors[i],
                },
                payload={
                    "chunk_id": chunk.chunk_id,
                    "arxiv_id": chunk.arxiv_id,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "section": chunk.section,
                    "token_count": chunk.token_count,
                    "title": chunk.title,
                    "authors": chunk.authors,
                    "published": chunk.published,
                    "pdf_url": chunk.pdf_url,
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)


# ── Pipeline ──────────────────────────────────────────────────────────────────


def run_embedding_pipeline(db_path: Path = DB_PATH) -> dict:
    """
    Full embedding + indexing pipeline:
      1. Load all chunks from SQLite
      2. Skip already-indexed chunks
      3. Encode dense vectors (SPECTER2) in batches
      4. Build BM25 index over all chunks for sparse vectors
      5. Upsert to Qdrant in batches
    """
    # ── Setup ──
    conn = sqlite3.connect(db_path)
    client = get_qdrant_client()
    init_collection(client)

    # ── Load chunks ──
    all_chunks = fetch_all_chunks(conn)
    conn.close()
    logger.info(f"Loaded {len(all_chunks)} chunks from SQLite")

    # ── Resumability ──
    indexed_ids = get_indexed_ids(client)
    chunks = [c for c in all_chunks if c.chunk_id not in indexed_ids]
    logger.info(f"{len(chunks)} chunks to index ({len(indexed_ids)} already done)")

    if not chunks:
        logger.info("Nothing to index — all chunks already in Qdrant")
        return {"total": len(all_chunks), "indexed": 0, "skipped": len(indexed_ids)}

    # ── Build BM25 over ALL chunks (not just new ones) for consistent vocab ──
    all_texts = [c.text for c in all_chunks]
    bm25 = build_bm25_index(all_texts)

    # ── Load encoder ──
    encoder = load_dense_encoder()

    # ── Process in batches ──
    total_indexed = 0

    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

        logger.info(f"Batch {batch_num}/{total_batches} — encoding {len(batch)} chunks")

        # Dense
        embed_texts = [c.embed_text for c in batch]
        dense_vectors = encode_dense(encoder, embed_texts)

        # Sparse
        sparse_vectors = [encode_sparse(bm25, c.text) for c in batch]

        # Upsert
        upsert_batch(client, batch, dense_vectors, sparse_vectors)
        total_indexed += len(batch)
        logger.info(f"Batch {batch_num}/{total_batches} — upserted {len(batch)} points")

    summary = {
        "total_chunks": len(all_chunks),
        "indexed": total_indexed,
        "skipped": len(indexed_ids),
    }
    logger.info(f"Embedding pipeline complete: {summary}")
    return summary


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_embedding_pipeline()
