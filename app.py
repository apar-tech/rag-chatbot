
import streamlit as st
import pickle
import chromadb
from google import genai
from groq import Groq
from rank_bm25 import BM25Okapi
import os
import traceback

# Page config
try:
    st.set_page_config(page_title="Networking RAG", page_icon="🌐", layout="wide")
except Exception as e:
    print(f"Page config error: {e}")

# CSS Styling
try:
    st.markdown("""
    <style>
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .badge {
            background: #4CAF50;
            color: white;
            padding: 0.3rem 1rem;
            border-radius: 20px;
            margin: 0.2rem;
            display: inline-block;
        }
        .error-box {
            background: #ffebee;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #f44336;
            margin: 1rem 0;
        }
        .success-box {
            background: #e8f5e9;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #4CAF50;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"CSS loading error: {str(e)}")

# Load RAG system with exception handling
@st.cache_resource
def load_rag():
    try:
        # Get API keys from secrets
        try:
            gemini_key = st.secrets["GEMINI_API_KEY"]
            groq_key = st.secrets["GROQ_API_KEY"]
        except Exception as e:
            st.error(f"⚠️ API keys not found in secrets: {str(e)}")
            st.info("Please add GEMINI_API_KEY and GROQ_API_KEY to .streamlit/secrets.toml")
            return None

        # Initialize clients
        try:
            gemini = genai.Client(api_key=gemini_key)
            groq = Groq(api_key=groq_key)
        except Exception as e:
            st.error(f"⚠️ Failed to initialize API clients: {str(e)}")
            return None

        # Load ChromaDB
        try:
            chroma = chromadb.PersistentClient(path="./networking_chromadb_phase4")
            collection = chroma.get_collection("networking_docs_phase4")
        except Exception as e:
            st.error(f"⚠️ Failed to load ChromaDB: {str(e)}")
            st.info("Make sure 'networking_chromadb_phase4' folder exists")
            return None

        # Load chunks
        try:
            with open("improved_chunks.pkl", "rb") as f:
                chunks = pickle.load(f)
        except FileNotFoundError:
            st.error("⚠️ improved_chunks.pkl not found!")
            st.info("Please upload improved_chunks.pkl to the app directory")
            return None
        except Exception as e:
            st.error(f"⚠️ Failed to load chunks: {str(e)}")
            return None

        # Create BM25 index
        try:
            tokenized = [c.lower().split() for c in chunks]
            bm25 = BM25Okapi(tokenized)
        except Exception as e:
            st.error(f"⚠️ Failed to create BM25 index: {str(e)}")
            return None

        # Hybrid retrieval function
        def retrieve(question, k=5):
            try:
                # Vector search
                emb = gemini.models.embed_content(
                    model="models/gemini-embedding-001",
                    contents=question
                ).embeddings[0].values

                vector_results = collection.query(
                    query_embeddings=[emb],
                    n_results=k
                )["documents"][0]

                # BM25 search
                bm25_scores = bm25.get_scores(question.lower().split())
                bm25_indices = sorted(range(len(bm25_scores)),
                                     key=lambda i: bm25_scores[i], reverse=True)[:k]
                bm25_chunks = [chunks[i] for i in bm25_indices]

                # Combine and deduplicate
                combined = []
                for c in vector_results + bm25_chunks:
                    if c not in combined:
                        combined.append(c)
                return combined[:k]

            except Exception as e:
                st.error(f"⚠️ Retrieval error: {str(e)}")
                return []

        # Answer generation
        def generate(question, context):
            try:
                prompt = f"""
You are a networking assistant. Answer only using the provided context.
If answer not in context, say "I could not find that information."

Context: {' '.join(context)}

Question: {question}
"""
                response = groq.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                return response.choices[0].message.content

            except Exception as e:
                return f"⚠️ Generation error: {str(e)}"

        return {"retrieve": retrieve, "generate": generate}

    except Exception as e:
        st.error(f"❌ Unexpected error loading RAG: {str(e)}")
        st.code(traceback.format_exc())
        return None

# Main app with exception handling
try:
    # Header
    st.markdown('''
    <div class="header">
        <h1>🌐 Networking RAG Assistant</h1>
        <p>Better Chunking + Hybrid Search</p>
        <div>
            <span class="badge">✅ Better Chunking</span>
            <span class="badge">✅ Hybrid Search</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Initialize session
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load RAG
    if "rag" not in st.session_state:
        with st.spinner("🔄 Loading RAG system..."):
            st.session_state.rag = load_rag()
            if st.session_state.rag:
                st.success("✅ RAG system loaded successfully!")
            else:
                st.error("❌ Failed to load RAG system")

    # Sidebar
    with st.sidebar:
        try:
            st.image("https://img.icons8.com/color/96/000000/network.png", width=80)
        except:
            st.write("🌐")

        st.markdown("---")
        st.markdown("### 🚀 Improvements")
        st.success("✅ Better Chunking")
        st.success("✅ Hybrid Search")

        st.markdown("---")
        st.markdown("### 📊 Status")
        if st.session_state.rag:
            st.success("✅ System Ready")
        else:
            st.error("❌ System Offline")

        st.markdown(f"**Queries:** {len(st.session_state.messages)//2}")

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # Chat interface
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input
    if prompt := st.chat_input("Ask a networking question..."):
        try:
            # User message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            # Assistant response
            with st.chat_message("assistant"):
                with st.spinner("🧠 Thinking..."):
                    if st.session_state.rag:
                        rag = st.session_state.rag
                        chunks = rag["retrieve"](prompt)
                        answer = rag["generate"](prompt, chunks)
                        st.write(answer)

                        with st.expander("📚 View Sources"):
                            if chunks:
                                for i, c in enumerate(chunks, 1):
                                    st.text(f"Source {i}: {c[:200]}...")
                            else:
                                st.info("No sources retrieved")

                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        st.error("⚠️ RAG system not ready. Please check configuration.")
        except Exception as e:
            st.error(f"❌ Error processing question: {str(e)}")

except Exception as e:
    st.error(f"❌ App error: {str(e)}")
    st.code(traceback.format_exc())
