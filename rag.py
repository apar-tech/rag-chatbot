import pickle
import chromadb
from google import genai
from groq import Groq
from rank_bm25 import BM25Okapi

from config import (
    GEMINI_API_KEY,
    GROQ_API_KEY,
    CHROMA_DB_PATH,
    CHUNKS_FILE,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K
)

# ==========================================================
# Initialize API Clients
# ==========================================================

try:
    gemini = genai.Client(api_key=GEMINI_API_KEY)
    groq = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    raise Exception(f"Failed to initialize API clients: {e}")


# ==========================================================
# Load ChromaDB
# ==========================================================

try:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(digital_doc_phase4)
except Exception as e:
    raise Exception(f"Failed to load ChromaDB: {e}")


# ==========================================================
# Load Chunks
# ==========================================================

try:
    with open(CHUNKS_FILE, "rb") as file:
        chunks = pickle.load(file)
except Exception as e:
    raise Exception(f"Failed to load chunk file: {e}")


# ==========================================================
# Create BM25 Index
# ==========================================================

try:
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
except Exception as e:
    raise Exception(f"Failed to create BM25 index: {e}")


# ==========================================================
# Generate Embedding
# ==========================================================

def get_embedding(text):
    """
    Generate Gemini embedding.
    """
    try:
        response = gemini.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text
        )

        return response.embeddings[0].values

    except Exception as e:
        raise Exception(f"Embedding generation failed: {e}")


# ==========================================================
# Vector Search
# ==========================================================

def vector_search(question, top_k=TOP_K):
    """
    Retrieve documents using ChromaDB.
    """
    try:
        embedding = get_embedding(question)

        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )

        return results["documents"][0]

    except Exception as e:
        raise Exception(f"Vector search failed: {e}")


# ==========================================================
# BM25 Search
# ==========================================================

def bm25_search(question, top_k=TOP_K):
    """
    Retrieve documents using BM25.
    """
    try:
        scores = bm25.get_scores(question.lower().split())

        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        return [chunks[i] for i in ranked]

    except Exception as e:
        raise Exception(f"BM25 search failed: {e}")


# ==========================================================
# Hybrid Search
# ==========================================================

def hybrid_search(question, top_k=TOP_K):
    """
    Combine Vector Search and BM25 Search.
    """
    try:
        vector_docs = vector_search(question, top_k)
        bm25_docs = bm25_search(question, top_k)

        combined = []

        for doc in vector_docs + bm25_docs:
            if doc not in combined:
                combined.append(doc)

        return combined[:top_k]

    except Exception as e:
        raise Exception(f"Hybrid search failed: {e}")


# ==========================================================
# Prompt Builder
# ==========================================================

def build_prompt(question, context):
    """
    Create prompt for Groq.
    """
    try:
        context_text = "\n\n".join(context)

        prompt = f"""
You are an expert Networking Assistant.

Answer ONLY using the provided context.

If the answer is not present in the context, reply:

"I could not find that information in the provided documents."

------------------------
Context:
{context_text}
------------------------

Question:
{question}

Answer:
"""

        return prompt

    except Exception as e:
        raise Exception(f"Prompt creation failed: {e}")


# ==========================================================
# Generate Answer
# ==========================================================

def generate_answer(question):
    """
    Generate answer using Hybrid RAG.
    """
    try:
        retrieved_docs = hybrid_search(question)

        if not retrieved_docs:
            return (
                "I could not find relevant information.",
                []
            )

        prompt = build_prompt(question, retrieved_docs)

        response = groq.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3
        )

        answer = response.choices[0].message.content

        return answer, retrieved_docs

    except Exception as e:
        raise Exception(f"Answer generation failed: {e}")


# ==========================================================
# Main Function
# ==========================================================

def ask_question(question):
    """
    Complete RAG Pipeline.
    """
    try:

        if not question.strip():
            return {
                "question": question,
                "answer": "Please enter a valid question.",
                "sources": []
            }

        answer, sources = generate_answer(question)

        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        return {
            "question": question,
            "answer": f"⚠️ {str(e)}",
            "sources": []
        }
