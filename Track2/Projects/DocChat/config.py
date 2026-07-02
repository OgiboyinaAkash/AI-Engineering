import streamlit as st

GROQ_API_KEY        = st.secrets.get("CHAT_GROQ_API_KEY", "")
ANTHROPIC_API_KEY   = st.secrets.get("ANTHROPIC_API_KEY", "")
PINECONE_API_KEY    = st.secrets.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME", "general-rag")

GROQ_MODEL = "llama-3.3-70b-versatile"
ANTHROPIC_MODEL = "claude-sonnet-4-6"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIMENSION = 768

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K_INITIAL = 16   # candidates fetched before reranking
TOP_K_RESULTS = 5    # kept after reranking

SYSTEM_PROMPT = """You are a knowledgeable and helpful AI assistant.
You help users find accurate information based on the documents they have uploaded.

Guidelines:
- Answer based ONLY on the provided context documents
- If the answer is not in the context, clearly state that
- Use clear, accessible language
- Cite relevant information from the context when answering
- Be concise, accurate, and helpful"""
