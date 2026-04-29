import logging
import sqlite3
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

from src.data_pipeline import (
    init_db,
    fetch_papers,
    filter_papers,
    build_pdf_path,
    download_pdf,
    save_paper,
    DB_PATH,
    PDF_BASE_DIR,
)
from src.data_processing import (
    extract_text_from_pdf,
    build_chunks,
    save_chunks,
    Specter2Embeddings,
    build_chunker,
    ChunkStrategy,
)
from src.embedding import (
    get_qdrant_client,
    init_collection,
    load_dense_encoder,
    build_bm25_index,
    encode_dense,
    encode_sparse,
    upsert_batch,
    ChunkRecord,
)

# ── Logging ───────────────────────────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ── Ingestion Runner ──────────────────────────────────────────────────────────

class IngestionRunner:
    def __init__(self, dense_encoder=None, embeddings_wrapper=None, client: Optional[get_qdrant_client] = None):
        self.dense_encoder = dense_encoder or load_dense_encoder()
        self.embeddings_wrapper = embeddings_wrapper or Specter2Embeddings()
        self.chunker = build_chunker(ChunkStrategy.SEMANTIC, embeddings=self.embeddings_wrapper)
        self.qdrant_client = client or get_qdrant_client()
        init_collection(self.qdrant_client)

    def ingest_and_retrieve(self, query: str, max_results: int = 3) -> list:
        """
        Synchronously fetch, process, index, and return chunks for a new query.
        """
        logger.info(f"Starting dynamic ingestion for query: '{query}'")
        
        # 1. Fetch metadata
        papers = fetch_papers(query=query, max_results=max_results)
        # Low strictness filter for dynamic ingestion
        papers = filter_papers(papers, date_from="2010-01-01", keywords=[]) 
        
        if not papers:
            logger.warning(f"No papers found on ArXiv for query: '{query}'")
            return []

        conn = init_db(DB_PATH)
        new_chunks = []
        
        # 2. Process each paper
        for paper in papers:
            # We don't skip here even if exists, to ensure it's in Qdrant for this session
            # but in a real scenario we'd check existence in Qdrant too.
            
            dest_path = build_pdf_path(PDF_BASE_DIR, paper.categories[0] if paper.categories else "unknown", paper.arxiv_id)
            success = download_pdf(paper.pdf_url, dest_path)
            
            if not success:
                continue
                
            paper.pdf_path = str(dest_path)
            save_paper(conn, paper)
            
            # Extract
            text = extract_text_from_pdf(paper.pdf_path)
            if not text:
                continue
                
            # Chunk
            try:
                chunks = build_chunks(
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    text=text,
                    chunker=self.chunker,
                    strategy=ChunkStrategy.SEMANTIC,
                )
                if not chunks:
                    continue
                    
                save_chunks(conn, chunks)
                
                # 3. Index to Qdrant immediately
                # For BM25 we need a corpus. In dynamic mode, we'll just use these chunks as a micro-corpus
                # or better, fetch a small sample from DB. For now, we use these chunks.
                bm25 = build_bm25_index([c.text for c in chunks])
                
                # Encode
                dense_vecs = encode_dense(self.dense_encoder, [c.embed_text for c in chunks])
                sparse_vecs = [encode_sparse(bm25, c.text) for c in chunks]
                
                # Build ChunkRecords for upsert
                records = [
                    ChunkRecord(
                        chunk_id=c.chunk_id,
                        arxiv_id=c.arxiv_id,
                        chunk_index=c.chunk_index,
                        text=c.text,
                        embed_text=c.embed_text,
                        section=c.section,
                        strategy=c.strategy,
                        token_count=c.token_count,
                        title=paper.title,
                        authors=str(paper.authors),
                        published=paper.published_date,
                        pdf_url=paper.pdf_url
                    ) for c in chunks
                ]
                
                upsert_batch(self.qdrant_client, records, dense_vecs, sparse_vecs)
                new_chunks.extend(records)
                logger.info(f"Successfully ingested and indexed: {paper.title}")
                
            except Exception as e:
                logger.error(f"Failed to process {paper.arxiv_id}: {e}")
                continue
                
        conn.close()
        return new_chunks

if __name__ == "__main__":
    # Test
    runner = IngestionRunner()
    results = runner.ingest_and_retrieve("LoRA: Low-Rank Adaptation of Large Language Models", max_results=1)
    print(f"Ingested {len(results)} chunks.")
