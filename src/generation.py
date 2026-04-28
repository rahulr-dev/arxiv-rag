import os
import logging
import time
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
GEMINI_MODEL = "gemini-flash-latest"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MAX_RETRIES = 3
RETRY_DELAY = 5

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ── Generation Engine ─────────────────────────────────────────────────────────

class GenerationEngine:
    def __init__(self):
        if not GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY not found in environment variables.")
            raise ValueError("GEMINI_API_KEY is required.")
        
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        logger.info(f"Generation Engine initialized with {GEMINI_MODEL}")

    def _build_prompt(self, query: str, context_chunks: list) -> str:
        """
        Construct a structured prompt for the LLM.
        """
        context_text = ""
        for i, chunk in enumerate(context_chunks):
            payload = chunk.payload
            context_text += f"--- SOURCE {i+1} ---\n"
            context_text += f"Title: {payload.get('title', 'Unknown')}\n"
            context_text += f"Authors: {payload.get('authors', 'Unknown')}\n"
            context_text += f"Content: {payload.get('text', '')}\n\n"

        system_instruction = (
            "You are an expert scientific research assistant specializing in Machine Learning and AI. "
            "Your goal is to answer the user's question accurately based ONLY on the provided research paper snippets. "
            "If the answer is not contained within the snippets, state that you do not have enough information. "
            "Always cite your sources by using the [SOURCE X] notation corresponding to the snippets. "
            "Maintain a formal, academic tone."
        )

        prompt = (
            f"{system_instruction}\n\n"
            f"RELEVANT RESEARCH SNIPPETS:\n{context_text}\n"
            f"USER QUESTION: {query}\n"
            f"SCIENTIFIC ANSWER:"
        )
        return prompt

    def generate_answer(self, query: str, context_chunks: list) -> str:
        """
        Generate a response using Gemini based on the provided context with retry logic.
        """
        if not context_chunks:
            return "I couldn't find any relevant research papers to answer your question."

        prompt = self._build_prompt(query, context_chunks)
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except exceptions.ResourceExhausted as e:
                wait_time = RETRY_DELAY * (attempt + 1)
                logger.warning(f"Quota exceeded (attempt {attempt+1}/{MAX_RETRIES}). Waiting {wait_time}s... Error: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait_time)
                else:
                    return "I'm currently experiencing high traffic and have reached my API quota. Please try again in a few minutes."
            except Exception as e:
                logger.error(f"Gemini generation failed: {e}")
                return f"An error occurred while generating the answer: {str(e)}"
        
        return "Failed to generate an answer after multiple attempts."

if __name__ == "__main__":
    # Quick test (requires GEMINI_API_KEY)
    # We would normally pass actual Qdrant result objects here.
    class MockChunk:
        def __init__(self, payload):
            self.payload = payload

    mock_chunks = [
        MockChunk({
            "title": "Attention Is All You Need",
            "authors": "Vaswani et al.",
            "text": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks..."
        })
    ]
    
    try:
        engine = GenerationEngine()
        answer = engine.generate_answer("What are traditional sequence models based on?", mock_chunks)
        print(f"Answer: {answer}")
    except Exception as e:
        print(e)
