import streamlit as st
import requests
import json

# ── Configuration ─────────────────────────────────────────────────────────────

API_URL = "http://localhost:8000/query"

st.set_page_config(page_title="ArXiv AI Assistant", layout="centered")

# ── Main Title ────────────────────────────────────────────────────────────────

st.markdown("# **ArXiv AI Assistant**")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("ArXiv Agent")
    st.markdown("---")
    st.markdown("""
    ### About
    This is an agentic RAG assistant specialized in ArXiv research papers.

    **Capabilities:**
    - 🔍 **Search**: Queries the local vector database.
    - 📥 **Ingest**: Fetches and indexes new papers from ArXiv.
    - 🌐 **Web**: Falls back to web search for general queries.
    """)

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ── Chat Logic ────────────────────────────────────────────────────────────────

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask about machine learning research..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🔍 *Thinking...*")

        try:
            # Call API
            payload = {"query": prompt}
            # We use a long timeout because dynamic ingestion can take 30s+
            response = requests.post(API_URL, json=payload, timeout=120)

            if response.status_code == 200:
                answer = response.json()["answer"]
                message_placeholder.markdown(answer)
                # Add assistant response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            else:
                error_msg = (
                    f"Error: Received status code {response.status_code} from API."
                )
                message_placeholder.markdown(error_msg)
                st.error(response.text)

        except requests.exceptions.ConnectionError:
            error_msg = "Error: Could not connect to the API. Please ensure the FastAPI server is running."
            message_placeholder.markdown(error_msg)
        except Exception as e:
            error_msg = f"An unexpected error occurred: {e}"
            message_placeholder.markdown(error_msg)

# # ── Footer ────────────────────────────────────────────────────────────────────
# st.markdown("---")
# st.caption("ArXiv RAG Assistant | 2026")
