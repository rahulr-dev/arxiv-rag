import streamlit as st
import requests
import json

# ── Configuration ─────────────────────────────────────────────────────────────

API_URL = "http://localhost:8000/query"

st.set_page_config(
    page_title="ArXiv RAG Explorer",
    page_icon="📚",
    layout="wide"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("📚 ArXiv RAG Settings")
st.sidebar.markdown("---")
top_k = st.sidebar.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=5)
st.sidebar.info(
    "This explorer uses a hybrid search (Dense + Sparse) and a "
    "Cross-Encoder reranker to find the most relevant snippets from "
    "scientific papers."
)

# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("ArXiv Research Assistant")
st.markdown(
    "Ask a question about Machine Learning, RAG, or Large Language Models "
    "based on the papers in our database."
)

query = st.text_input("Enter your question:", placeholder="e.g., How does the attention mechanism improve transformers?")

if st.button("Query Pipeline") or (query and st.session_state.get('last_query') != query):
    if not query:
        st.warning("Please enter a question.")
    else:
        st.session_state['last_query'] = query
        
        with st.spinner("Retrieving sources and generating answer..."):
            try:
                payload = {"query": query, "top_k": top_k}
                response = requests.post(API_URL, json=payload, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # 1. Display Answer
                    st.subheader("🤖 Assistant's Answer")
                    st.markdown(data["answer"])
                    
                    st.markdown("---")
                    
                    # 2. Display Sources
                    st.subheader("📄 Referenced Sources")
                    for i, source in enumerate(data["sources"]):
                        with st.expander(f"[{i+1}] {source['title']} (Score: {source['score']:.4f})"):
                            st.markdown(f"**Authors:** {source['authors']}")
                            st.markdown(f"**Snippet:**\n> {source['text']}")
                            if source.get("pdf_url"):
                                st.link_button("View Paper", source["pdf_url"])
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API. Make sure the FastAPI server is running at localhost:8000.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Powered by SPECTER2, Qdrant, and Google Gemini.")
