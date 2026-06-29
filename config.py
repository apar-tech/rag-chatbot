import streamlit as st

# ==========================
# API Keys
# ==========================
GEMINI_API_KEY = st.secrets["geminikey"]
GROQ_API_KEY = st.secrets["groqkey"]

# ==========================
# File Paths
# ==========================
CHROMA_DB_PATH = "networking_chromadb_phase4"
CHUNKS_FILE = "improved_chunks.pkl"

# ==========================
# ChromaDB Collection
# ==========================
COLLECTION_NAME = "networking_chromadb_phase4"

# ==========================
# Models
# ==========================
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "llama-3.3-70b-versatile"

# ==========================
# Retrieval Settings
# ==========================
TOP_K = 5
