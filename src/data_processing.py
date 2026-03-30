import re
import json
import sqlite3
import logging

from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum

import fitz  # pymupdf

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

# ── Constants ─────────────────────────────────────────────────────────────────

DB_PATH = Path("data/arxiv_papers.db")
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
DENSE_MODEL = "allenai/specter2_base"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
CHARS_PER_TOKEN = 4
MIN_CHUNK_CHARS = 100

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ── Enums ─────────────────────────────────────────────────────────────────────


class ChunkStrategy(str, Enum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class Section(str, Enum):
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED = "related_work"
    METHODS = "methods"
    EXPERIMENTS = "experiments"
    RESULTS = "results"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    OTHER = "other"


# ── Config ────────────────────────────────────────────────────────────────────

STRATEGY = ChunkStrategy.SEMANTIC  # primary strategy

# ── Data Model ────────────────────────────────────────────────────────────────


@dataclass
class Chunk:
    chunk_id: str
    arxiv_id: str
    chunk_index: int
    text: str
    embed_text: str  # title [SEP] chunk — SPECTER2 format
    section: str
    strategy: str
    token_count: int
    created_at: str


# ── SPECTER2 LangChain Wrapper ────────────────────────────────────────────────


class Specter2Embeddings(Embeddings):
    """
    Wraps allenai/specter2 as a LangChain Embeddings class.
    Required by SemanticChunker which expects a LangChain-compatible embedder.
    """

    def __init__(self):
        logger.info(f"Loading embedding model: {DENSE_MODEL}")
        self.model = SentenceTransformer(DENSE_MODEL)
        logger.info("Embedding model loaded")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(
            text,
            normalize_embeddings=True,
        ).tolist()


# ── Chunker Factory ───────────────────────────────────────────────────────────


def build_chunker(
    strategy: ChunkStrategy, embeddings: Optional[Specter2Embeddings] = None
):
    """
    Returns the appropriate LangChain chunker for the given strategy.
    SemanticChunker requires embeddings — passed in to avoid reloading model.
    """
    if strategy == ChunkStrategy.RECURSIVE:
        return RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE * CHARS_PER_TOKEN,
            chunk_overlap=CHUNK_OVERLAP * CHARS_PER_TOKEN,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    elif strategy == ChunkStrategy.SEMANTIC:
        if embeddings is None:
            raise ValueError("SemanticChunker requires embeddings instance")
        return SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",  # split where similarity drops most
            breakpoint_threshold_amount=90,  # top 10% sharpest drops = boundaries
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ── Database ──────────────────────────────────────────────────────────────────


def init_chunks_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id    TEXT PRIMARY KEY,
            arxiv_id    TEXT NOT NULL REFERENCES papers(arxiv_id),
            chunk_index INTEGER NOT NULL,
            text        TEXT NOT NULL,
            embed_text  TEXT NOT NULL,
            section     TEXT,
            strategy    TEXT,
            token_count INTEGER,
            created_at  TEXT
        )
    """)
    conn.commit()


def paper_already_chunked(conn: sqlite3.Connection, arxiv_id: str) -> bool:
    cursor = conn.execute(
        "SELECT 1 FROM chunks WHERE arxiv_id = ? LIMIT 1", (arxiv_id,)
    )
    return cursor.fetchone() is not None


def fetch_papers_to_process(conn: sqlite3.Connection) -> list[dict]:
    cursor = conn.execute("""
        SELECT arxiv_id, title, pdf_path
        FROM papers
        WHERE pdf_path IS NOT NULL
    """)
    return [
        {"arxiv_id": row[0], "title": row[1], "pdf_path": row[2]}
        for row in cursor.fetchall()
    ]


def save_chunks(conn: sqlite3.Connection, chunks: list[Chunk]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO chunks
            (chunk_id, arxiv_id, chunk_index, text, embed_text,
             section, strategy, token_count, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        [
            (
                c.chunk_id,
                c.arxiv_id,
                c.chunk_index,
                c.text,
                c.embed_text,
                c.section,
                c.strategy,
                c.token_count,
                c.created_at,
            )
            for c in chunks
        ],
    )
    conn.commit()


# ── PDF Extraction ────────────────────────────────────────────────────────────


def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    try:
        doc = fitz.open(pdf_path)
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return clean_text("\n".join(pages))

    except Exception as e:
        logger.error(f"PDF extraction failed for {pdf_path}: {e}")
        return None


def clean_text(text: str) -> str:
    # Fix hyphenated line breaks: "transfor-\nmer" → "transformer"
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # Collapse single newlines into spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Collapse 3+ newlines into paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove isolated page numbers
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Collapse multiple spaces/tabs
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Drop micro-fragments under 20 chars
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]

    return "\n\n".join(paragraphs)


# ── Section Detection ─────────────────────────────────────────────────────────

SECTION_PATTERNS: list[tuple[Section, re.Pattern]] = [
    (Section.ABSTRACT, re.compile(r"\babstract\b", re.I)),
    (Section.INTRODUCTION, re.compile(r"\bintroduction\b", re.I)),
    (Section.RELATED, re.compile(r"\brelated\s+work\b", re.I)),
    (
        Section.METHODS,
        re.compile(r"\b(method|methodology|approach|framework|architecture)\b", re.I),
    ),
    (Section.EXPERIMENTS, re.compile(r"\b(experiment|experimental\s+setup)\b", re.I)),
    (
        Section.RESULTS,
        re.compile(r"\b(result|evaluation|performance|analysis)\b", re.I),
    ),
    (
        Section.CONCLUSION,
        re.compile(r"\b(conclusion|discussion|future\s+work|summary)\b", re.I),
    ),
    (Section.REFERENCES, re.compile(r"\breferences\b", re.I)),
]


def detect_section(text: str) -> str:
    header_zone = text[:120]
    for section, pattern in SECTION_PATTERNS:
        if pattern.search(header_zone):
            return section.value
    return Section.OTHER.value


# ── Chunk Builder ─────────────────────────────────────────────────────────────


def build_chunks(
    arxiv_id: str,
    title: str,
    text: str,
    chunker,
    strategy: ChunkStrategy,
) -> list[Chunk]:
    """
    Split text using LangChain chunker, tag sections, build embed_text.
    SemanticChunker returns LangChain Document objects.
    RecursiveCharacterTextSplitter returns strings.
    Handles both.
    """
    raw_chunks = chunker.create_documents([text])

    chunks = []
    for i, doc in enumerate(raw_chunks):
        # LangChain returns Document objects with .page_content
        chunk_text = doc.page_content.strip()

        if len(chunk_text) < MIN_CHUNK_CHARS:
            continue

        section = detect_section(chunk_text)
        embed_text = f"{title} [SEP] {chunk_text}"  # SPECTER2 input format
        token_est = len(chunk_text) // CHARS_PER_TOKEN

        chunks.append(
            Chunk(
                chunk_id=f"{arxiv_id}_chunk_{i}",
                arxiv_id=arxiv_id,
                chunk_index=i,
                text=chunk_text,
                embed_text=embed_text,
                section=section,
                strategy=strategy.value,
                token_count=token_est,
                created_at=datetime.utcnow().isoformat(),
            )
        )

    return chunks


# ── Pipeline ──────────────────────────────────────────────────────────────────


def run_processing_pipeline(
    db_path: Path = DB_PATH,
    strategy: ChunkStrategy = STRATEGY,
) -> dict:
    conn = sqlite3.connect(db_path)
    init_chunks_table(conn)

    papers = fetch_papers_to_process(conn)
    total = len(papers)

    # ── Load embeddings once — expensive, don't reload per paper ──
    embeddings = Specter2Embeddings()
    chunker = build_chunker(strategy, embeddings=embeddings)

    processed = 0
    skipped = 0
    failed = 0
    total_chunks = 0

    logger.info(f"Processing {total} papers | strategy: {strategy.value}")

    for i, paper in enumerate(papers, start=1):
        arxiv_id = paper["arxiv_id"]
        title = paper["title"]
        pdf_path = paper["pdf_path"]

        # ── Resumability ──
        if paper_already_chunked(conn, arxiv_id):
            logger.info(f"[{i}/{total}] [SKIP] {arxiv_id}")
            skipped += 1
            continue

        # ── Extract ──
        text = extract_text_from_pdf(pdf_path)
        if not text:
            logger.error(f"[{i}/{total}] [FAIL] {arxiv_id} — extraction failed")
            failed += 1
            continue

        # ── Chunk ──
        try:
            chunks = build_chunks(
                arxiv_id=arxiv_id,
                title=title,
                text=text,
                chunker=chunker,
                strategy=strategy,
            )
        except Exception as e:
            logger.error(f"[{i}/{total}] [FAIL] {arxiv_id} — chunking failed: {e}")
            failed += 1
            continue

        if not chunks:
            logger.warning(f"[{i}/{total}] [WARN] {arxiv_id} — zero chunks")
            failed += 1
            continue

        # ── Persist ──
        save_chunks(conn, chunks)
        processed += 1
        total_chunks += len(chunks)
        logger.info(
            f"[{i}/{total}] [OK] {arxiv_id} — {len(chunks)} chunks — {title[:50]}"
        )

    conn.close()

    summary = {
        "total_papers": total,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "total_chunks": total_chunks,
        "avg_chunks": round(total_chunks / processed, 1) if processed else 0,
    }
    logger.info(f"Processing complete: {summary}")
    return summary


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_processing_pipeline(strategy=STRATEGY)
