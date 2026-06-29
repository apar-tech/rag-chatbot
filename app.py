import streamlit as st
import traceback

from rag import ask_question
from utils import (
    page_config,
    load_css,
    show_header,
    sidebar,
    show_sources
)

# ==========================================================
# Page Setup
# ==========================================================

try:
    page_config()
    load_css()
except Exception as e:
    st.error(f"Page initialization failed: {e}")
    st.stop()

# ==========================================================
# Header
# ==========================================================

try:
    show_header()
except Exception as e:
    st.error(f"Header error: {e}")

# ==========================================================
# Sidebar
# ==========================================================

try:
    sidebar()
except Exception as e:
    st.error(f"Sidebar error: {e}")

# ==========================================================
# Session State
# ==========================================================

try:

    if "messages" not in st.session_state:
        st.session_state.messages = []

except Exception as e:
    st.error(f"Session initialization failed: {e}")

# ==========================================================
# Display Chat History
# ==========================================================

try:

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

except Exception as e:
    st.error(f"Chat history error: {e}")

# ==========================================================
# Chat Input
# ==========================================================

try:

    question = st.chat_input("Ask a Networking Question...")

    if question:

        # -------------------------
        # User Message
        # -------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        # -------------------------
        # Assistant
        # -------------------------

        with st.chat_message("assistant"):

            with st.spinner("Searching knowledge base..."):

                result = ask_question(question)

                answer = result["answer"]

                sources = result["sources"]

                st.markdown(answer)

                show_sources(sources)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

except Exception as e:

    st.error("Something went wrong.")

    st.code(traceback.format_exc())

# ==========================================================
# Footer
# ==========================================================

st.divider()

st.caption(
    "Networking RAG Assistant | "
    "Hybrid Search | Better Chunking | "
    "Gemini Embeddings | Groq Llama"
)
