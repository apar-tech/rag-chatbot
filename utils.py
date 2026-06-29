import streamlit as st


# ==========================================================
# Page Configuration
# ==========================================================

def page_config():
    """Configure Streamlit page."""
    st.set_page_config(
        page_title="Networking RAG Assistant",
        page_icon="🌐",
        layout="wide"
    )


# ==========================================================
# Custom CSS
# ==========================================================

def load_css():
    """Load custom CSS."""

    st.markdown("""
    <style>

    .main {
        padding-top: 1rem;
    }

    .header{
        background: linear-gradient(135deg,#2563eb,#1e3a8a);
        padding:30px;
        border-radius:15px;
        text-align:center;
        color:white;
        margin-bottom:20px;
    }

    .header h1{
        font-size:40px;
        margin-bottom:10px;
    }

    .header p{
        font-size:18px;
    }

    .source-box{
        background:#f8f9fa;
        padding:15px;
        border-radius:10px;
        border-left:5px solid #2563eb;
        margin-bottom:10px;
    }

    .stChatMessage{
        border-radius:15px;
    }

    </style>
    """, unsafe_allow_html=True)


# ==========================================================
# Header
# ==========================================================

def show_header():

    st.markdown("""
    <div class="header">

    <h1>🌐 Networking RAG Assistant</h1>

    <p>
    Better Chunking • Hybrid Search • Gemini Embeddings • Groq LLM
    </p>

    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# Sidebar
# ==========================================================

def sidebar():

    with st.sidebar:

        st.title("📘 About")

        st.write(
            """
This chatbot answers networking questions using

✅ Better Chunking

✅ Hybrid Search

✅ ChromaDB

✅ Gemini Embeddings

✅ Groq Llama 3.3
            """
        )

        st.divider()

        if st.button("🗑️ Clear Chat"):

            st.session_state.messages = []

            st.rerun()


# ==========================================================
# Display Sources
# ==========================================================

def show_sources(sources):

    with st.expander("📚 Retrieved Sources"):

        if len(sources) == 0:

            st.info("No sources found.")

        else:

            for i, source in enumerate(sources, start=1):

                st.markdown(f"### Source {i}")

                st.markdown(
                    f"""
                    <div class="source-box">

                    {source}

                    </div>
                    """,
                    unsafe_allow_html=True
                )
