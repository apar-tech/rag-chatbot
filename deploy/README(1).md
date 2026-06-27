
# 🌐 Networking RAG Assistant

A production-ready RAG system with **Better Chunking** and **Hybrid Search**.

## 📊 Improvements

### 1️⃣ Better Chunking
- Chunk Size: 1000 characters
- Overlap: 200 characters
- Smart boundaries

### 2️⃣ Hybrid Search
- Vector Search + BM25
- Combined results

## 📊 Evaluation Results

| Metric | Phase 2 | Phase 4 | Improvement |
|--------|---------|---------|-------------|
| Faithfulness | 0.9167 | 1.0000 | +0.0833 |
| Answer Relevancy | 0.7908 | 0.8040 | +0.0132 |
| Context Recall | 1.0000 | 1.0000 | 0.0000 |

## 🚀 Deployment

1. Push to GitHub
2. Deploy on Streamlit Cloud
3. Add secrets: GEMINI_API_KEY, GROQ_API_KEY

## 🔗 Live Demo
[https://networking-rag.streamlit.app](https://networking-rag.streamlit.app)
