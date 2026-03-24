import arxiv
import requests
import sqlite3
import logging
import json
import time

from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


DB_PATH = Path("data/arxiv_papers.db")
PDF_BASE_DIR = Path("data/pdfs")
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
REQUEST_TIMEOUT = 30
BATCH_SLEEP = 3
MAX_RETRIES = 3
RETRY_BACKOFF = 2
MIN_ABSTRACT_LENGTH = 100
MIN_PDF_SIZE_BYTES = 51_200


logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published_date: str
    pdf_url: str
    pdf_path: Optional[str] = None
    ingested_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


FETCH_CONFIGS = [
    {
        "query": "retrieval augmented generation",
        "category": "cs.CL",
        "max_results": 150,
        "keywords": ["retrieval", "augmented", "RAG", "generation"],
    },
    {
        "query": "dense retrieval embeddings vector search",
        "category": "cs.IR",
        "max_results": 100,
        "keywords": ["embedding", "vector", "retrieval", "dense"],
    },
    {
        "query": "large language models transformers",
        "category": "cs.LG",
        "max_results": 150,
        "keywords": ["language model", "transformer", "attention", "LLM"],
    },
    {
        "query": "semantic search reranking information retrieval",
        "category": "cs.IR",
        "max_results": 100,
        "keywords": ["reranking", "semantic", "retrieval", "search"],
    },
]
DATE_FROM = "2020-01-01"

LANDMARK_PAPERS = [
    # ── Foundational Transformers & Attention ──
    "1706.03762",  # Attention Is All You Need (Vaswani et al.)
    "1409.0473",  # Neural Machine Translation + Attention (Bahdanau et al.)
    "2005.14165",  # GPT-3: Language Models are Few-Shot Learners
    "1810.04805",  # BERT (Devlin et al.)
    "2010.11929",  # ViT: Image is Worth 16x16 Words
    "1301.3666",  # Word2Vec (Mikolov et al.)
    # ── LLMs ──
    "2302.13971",  # LLaMA (Touvron et al.)
    "2307.09288",  # LLaMA 2
    "2303.08774",  # GPT-4 Technical Report
    "1910.01108",  # T5: Exploring Limits of Transfer Learning
    "2109.01652",  # FLAN: Finetuned Language Models are Zero-Shot Learners
    "2203.02155",  # InstructGPT: Training LMs to Follow Instructions
    "2205.01068",  # OPT: Open Pre-trained Transformer Language Models
    "2112.11446",  # WebGPT: Browser-assisted QA with Human Feedback
    # ── RAG & Retrieval Systems ──
    "2005.11401",  # RAG: Retrieval-Augmented Generation (Lewis et al.)
    "2208.09257",  # Atlas: Few-shot Learning with Retrieval (Izacard et al.)
    "2212.10560",  # Self-RAG
    "2310.11511",  # RAG for LLMs: A Survey
    "2004.04906",  # DPR: Dense Passage Retrieval (Karpukhin et al.)
    "2112.09118",  # Improving Retrieval with LLMs (HyDE)
    "2009.02439",  # ColBERT: Efficient Late Interaction Retrieval
    "2108.00573",  # BEIR: Heterogeneous Retrieval Benchmark
    # ── Diffusion & Generative Models ──
    "2006.11239",  # DDPM: Denoising Diffusion Probabilistic Models
    "2112.10752",  # Latent Diffusion Models (Stable Diffusion)
    "2204.06125",  # DALL-E 2
    "1406.2661",  # GANs: Generative Adversarial Networks (Goodfellow)
    "1312.6114",  # VAE: Auto-Encoding Variational Bayes (Kingma & Welling)
    "2103.00020",  # CLIP (Radford et al.)
    "2112.09118",  # Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE) — Gao et al.
    "2004.12832",  # ColBERT: Efficient and Effective Passage Search (Khattab & Zaharia)
    "2109.02048",  # BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation
    "2210.11610",  # Atlas: Few-shot Learning with Retrieval Augmented Language Models
    "1910.10683",  # T5: Exploring the Limits of Transfer Learning (Raffel et al.)
]


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            arxiv_id        TEXT PRIMARY KEY,
            title           TEXT NOT NULL,
            authors         TEXT NOT NULL,
            abstract        TEXT,
            categories      TEXT,
            published_date  TEXT,
            pdf_url         TEXT,
            pdf_path        TEXT,
            ingested_at     TEXT
        )
    """)
    conn.commit()
    logger.info(f"Database ready at {db_path}")
    return conn


def paper_exists(conn: sqlite3.Connection, arxiv_id: str) -> bool:
    cursor = conn.execute("SELECT 1 FROM papers WHERE arxiv_id = ?", (arxiv_id,))
    return cursor.fetchone() is not None


def save_paper(conn: sqlite3.Connection, paper: Paper) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO papers
            (arxiv_id, title, authors, abstract, categories,
             published_date, pdf_url, pdf_path, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            paper.arxiv_id,
            paper.title,
            json.dumps(paper.authors),
            paper.abstract,
            json.dumps(paper.categories),
            paper.published_date,
            paper.pdf_url,
            paper.pdf_path,
            paper.ingested_at,
        ),
    )
    conn.commit()


def fetch_papers(query: str, max_results: int) -> list[Paper]:
    client = arxiv.Client(
        page_size=100,
        delay_seconds=BATCH_SLEEP,
        num_retries=MAX_RETRIES,
    )

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = []
    logger.info(f"Fetching up to {max_results} papers — query: '{query}'")

    for result in client.results(search):
        papers.append(
            Paper(
                arxiv_id=result.entry_id.split("/")[-1],
                title=result.title.strip().replace("\n", " "),
                authors=[a.name for a in result.authors],
                abstract=result.summary.strip().replace("\n", " "),
                categories=result.categories,
                published_date=result.published.strftime("%Y-%m-%d"),
                pdf_url=result.pdf_url,
            )
        )

    logger.info(f"Fetched {len(papers)} records")
    return papers


def filter_papers(
    papers: list[Paper],
    date_from: str,
    keywords: list[str],
) -> list[Paper]:
    filtered = []

    for paper in papers:
        # Abstract quality
        if len(paper.abstract.strip()) < MIN_ABSTRACT_LENGTH:
            logger.info(f"[FILTER] {paper.arxiv_id} — abstract too short")
            continue

        # Date
        if paper.published_date < date_from:
            logger.info(f"[FILTER] {paper.arxiv_id} — too old ({paper.published_date})")
            continue

        # Keyword match in title or abstract
        text = (paper.title + " " + paper.abstract).lower()
        if not any(kw.lower() in text for kw in keywords):
            logger.info(f"[FILTER] {paper.arxiv_id} — no keyword match")
            continue

        filtered.append(paper)

    logger.info(f"Filter pass: {len(filtered)}/{len(papers)} passed")
    return filtered


def build_pdf_path(pdf_base_dir: Path, category: str, arxiv_id: str) -> Path:
    dest_dir = pdf_base_dir / category.replace(".", "_")
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir / f"{arxiv_id}.pdf"


def download_pdf(pdf_url: str, dest_path: Path) -> bool:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(pdf_url, timeout=REQUEST_TIMEOUT, stream=True)
            response.raise_for_status()

            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size = dest_path.stat().st_size
            if file_size < MIN_PDF_SIZE_BYTES:
                raise ValueError(f"PDF too small ({file_size} bytes)")

            return True

        except (requests.RequestException, ValueError) as e:
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES} failed — {pdf_url}: {e}")
            if dest_path.exists():
                dest_path.unlink()
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
            else:
                logger.error(f"All retries exhausted — {pdf_url}")
                return False


def fetch_landmark_papers(arxiv_ids: list[str]) -> list[Paper]:
    """Fetch papers one at a time to avoid rate limiting."""
    papers = []
    logger.info(f"Fetching {len(arxiv_ids)} landmark papers one at a time")

    for i, arxiv_id in enumerate(arxiv_ids, start=1):
        try:
            client = arxiv.Client(num_retries=2, delay_seconds=5)
            search = arxiv.Search(id_list=[arxiv_id])  # one at a time

            results = list(client.results(search))
            if not results:
                logger.warning(f"[{i}/{len(arxiv_ids)}] No result for ID: {arxiv_id}")
                continue

            result = results[0]
            papers.append(
                Paper(
                    arxiv_id=result.entry_id.split("/")[-1],
                    title=result.title.strip().replace("\n", " "),
                    authors=[a.name for a in result.authors],
                    abstract=result.summary.strip().replace("\n", " "),
                    categories=result.categories,
                    published_date=result.published.strftime("%Y-%m-%d"),
                    pdf_url=result.pdf_url,
                )
            )
            logger.info(f"[{i}/{len(arxiv_ids)}] Fetched: {result.title[:70]}")

        except Exception as e:
            logger.error(f"[{i}/{len(arxiv_ids)}] Failed for {arxiv_id}: {e}")
            continue

        time.sleep(5)

    logger.info(f"Landmark fetch complete — {len(papers)} papers retrieved")
    return papers


def run_pipeline(
    query: str,
    max_results: int,
    category: str,
    keywords: list[str],
    date_from: str = DATE_FROM,
    db_path: Path = DB_PATH,
    pdf_dir: Path = PDF_BASE_DIR,
) -> dict:
    conn = init_db(db_path)
    papers = fetch_papers(query=query, max_results=max_results)
    papers = filter_papers(papers, date_from=date_from, keywords=keywords)

    skipped = downloaded = failed = 0

    for paper in papers:
        if paper_exists(conn, paper.arxiv_id):
            logger.info(f"[SKIP] {paper.arxiv_id}")
            skipped += 1
            continue

        dest_path = build_pdf_path(pdf_dir, category, paper.arxiv_id)
        success = download_pdf(paper.pdf_url, dest_path)

        if success:
            paper.pdf_path = str(dest_path)
            downloaded += 1
            logger.info(f"[OK]  {paper.arxiv_id} — {paper.title[:60]}")
        else:
            failed += 1
            logger.error(f"[FAIL] {paper.arxiv_id} — metadata saved, PDF missing")

        save_paper(conn, paper)

    conn.close()

    summary = {
        "query": query,
        "total_fetched": len(papers),
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
    }
    logger.info(f"Done: {summary}")
    return summary


def run_landmark_pipeline(
    arxiv_ids: list[str],
    category: str = "landmark",
    db_path: Path = DB_PATH,
    pdf_dir: Path = PDF_BASE_DIR,
) -> dict:
    """
    Fetch and download landmark papers directly by Arxiv ID.
    No quality filtering — these are pre-validated.
    """
    conn = init_db(db_path)
    papers = fetch_landmark_papers(arxiv_ids)

    skipped = downloaded = failed = 0

    for paper in papers:
        if paper_exists(conn, paper.arxiv_id):
            logger.info(f"[SKIP] {paper.arxiv_id} — {paper.title[:60]}")
            skipped += 1
            continue

        dest_path = build_pdf_path(pdf_dir, category, paper.arxiv_id)
        success = download_pdf(paper.pdf_url, dest_path)

        if success:
            paper.pdf_path = str(dest_path)
            downloaded += 1
            logger.info(f"[OK]  {paper.arxiv_id} — {paper.title[:60]}")
        else:
            failed += 1
            logger.error(f"[FAIL] {paper.arxiv_id} — metadata saved, PDF missing")

        save_paper(conn, paper)

    conn.close()

    summary = {
        "query": "landmark_papers",
        "total_fetched": len(papers),
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
    }
    logger.info(f"Landmark pipeline complete — {summary}")
    return summary


if __name__ == "__main__":
    total_summary = {"downloaded": 0, "skipped": 0, "failed": 0}

    # ── Regular query-based fetch ──
    for config in FETCH_CONFIGS:
        result = run_pipeline(**config)
        total_summary["downloaded"] += result["downloaded"]
        total_summary["skipped"] += result["skipped"]
        total_summary["failed"] += result["failed"]

    # ── Landmark papers fetch ──
    landmark_result = run_landmark_pipeline(arxiv_ids=LANDMARK_PAPERS)
    total_summary["downloaded"] += landmark_result["downloaded"]
    total_summary["skipped"] += landmark_result["skipped"]
    total_summary["failed"] += landmark_result["failed"]

    logger.info(f"All pipelines complete — {total_summary}")
