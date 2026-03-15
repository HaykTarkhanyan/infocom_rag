"""Streamlit UI for the Telegram RAG Chatbot."""

import streamlit as st
from rag import RAG
from config import (
    TOP_K,
    MAX_DISTANCE,
    GEMINI_TEMPERATURE,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MODEL,
    SYSTEM_PROMPT,
)

st.set_page_config(
    page_title="Telegram RAG Chat",
    page_icon="💬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — hyperparameters
# ---------------------------------------------------------------------------

st.sidebar.title("Settings")

st.sidebar.header("Retrieval")
top_k = st.sidebar.slider("Top K (number of chunks to retrieve)", 1, 50, TOP_K)
max_distance = st.sidebar.slider(
    "Max distance threshold",
    0.0, 2.0, MAX_DISTANCE, 0.05,
    help="Discard results with cosine distance above this value. Lower = stricter.",
)

st.sidebar.header("Generation")
gemini_model = st.sidebar.selectbox(
    "Gemini model",
    ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-lite"],
    index=0,
)
temperature = st.sidebar.slider(
    "Temperature", 0.0, 2.0, GEMINI_TEMPERATURE, 0.05,
    help="Lower = more deterministic, higher = more creative.",
)
max_output_tokens = st.sidebar.slider(
    "Max output tokens", 64, 8192, GEMINI_MAX_OUTPUT_TOKENS, 64,
)

st.sidebar.header("System Prompt")
system_prompt = st.sidebar.text_area(
    "System prompt",
    value=SYSTEM_PROMPT,
    height=200,
)

st.sidebar.divider()
st.sidebar.caption(
    "These settings apply in real-time to every new query. "
    "Chunking & ingestion settings are configured in `config.py` / `.env` "
    "and require re-running `python ingest.py`."
)

# ---------------------------------------------------------------------------
# Initialize RAG (cached so it persists across reruns)
# ---------------------------------------------------------------------------


@st.cache_resource
def get_rag():
    return RAG()


rag = get_rag()

# Apply sidebar settings to the live RAG instance
rag.top_k = top_k
rag.max_distance = max_distance
rag.temperature = temperature
rag.max_output_tokens = max_output_tokens
rag.model = gemini_model
rag.system_prompt = system_prompt

# ---------------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------------

st.title("Telegram RAG Chatbot")
st.caption("Ask questions about your Telegram chat history")

# Chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])} chunks)"):
                for i, src in enumerate(msg["sources"], 1):
                    dist_str = f"{src['distance']:.4f}" if src["distance"] is not None else "N/A"
                    st.markdown(f"**{i}.** (distance: {dist_str})")
                    st.code(src["text"], language=None)

# Chat input
if prompt := st.chat_input("Ask a question about the chat history..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            answer, sources = rag.answer(prompt)
        st.markdown(answer)
        if sources:
            with st.expander(f"Sources ({len(sources)} chunks)"):
                for i, src in enumerate(sources, 1):
                    dist_str = f"{src['distance']:.4f}" if src["distance"] is not None else "N/A"
                    st.markdown(f"**{i}.** (distance: {dist_str})")
                    st.code(src["text"], language=None)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
