import os
import logging
from src.retrieval import RetrievalEngine
from dotenv import load_dotenv

load_dotenv()

# ── Setup Logging ─────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.ERROR) # Only show errors to keep output clean
logger = logging.getLogger(__name__)

# ── Test Cases ────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "query": "What is the core idea of the Attention Is All You Need paper?",
        "expected_keywords": ["transformer", "attention", "parallel", "recurrence"],
    },
    {
        "query": "How does Retrieval-Augmented Generation (RAG) combine retrieval and generation?",
        "expected_keywords": ["retrieval", "generation", "context", "knowledge"],
    },
    {
        "query": "What is the purpose of the SPECTER2 model?",
        "expected_keywords": ["embedding", "scientific", "papers", "specter"],
    },
    {
        "query": "Explain the difference between dense and sparse retrieval.",
        "expected_keywords": ["embedding", "keyword", "vector", "bm25"],
    }
]

def evaluate():
    engine = RetrievalEngine()
    print("\n" + "="*80)
    print(f"{'QUERY EVALUATION REPORT':^80}")
    print("="*80)

    for case in TEST_CASES:
        query = case["query"]
        expected = case["expected_keywords"]
        
        print(f"\n[QUERY]: {query}")
        results = engine.query(query, top_k=5)
        
        if not results:
            print("  ❌ FAILURE: No results retrieved.")
            continue

        found_any_keyword = False
        print(f"  {'Rank':<5} | {'Score':<8} | {'Title'}")
        print(f"  {'-'*5} | {'-'*8} | {'-'*60}")
        
        for i, res in enumerate(results):
            payload = res.payload
            title = payload.get('title', 'Unknown')
            text = payload.get('text', '').lower()
            score = getattr(res, 'score', 0.0)
            
            # Simple keyword check for evaluation
            matched = [kw for kw in expected if kw.lower() in text or kw.lower() in title.lower()]
            status = "✅" if matched else "❌"
            if matched: found_any_keyword = True
            
            print(f"  {i+1:<5} | {score:<8.4f} | {title[:60]}")
            if matched:
                print(f"        └─ Keywords found: {', '.join(matched)}")

        if not found_any_keyword:
            print(f"  ⚠️  WARNING: No expected keywords found in the top 5 results.")

if __name__ == "__main__":
    evaluate()
