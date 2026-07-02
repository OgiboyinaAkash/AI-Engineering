# DocChat

A Retrieval-Augmented Generation (RAG) chatbot. Upload PDF or TXT files, paste a web URL, and ask questions — answers are streamed in real time, grounded in your documents with source citations and chunk previews. Supports conversational follow-ups across multiple turns.

Supports two switchable LLM backends: **Groq (Llama3.3-70B)** and **Anthropic Claude Haiku**.

---

## Features

| Feature | Detail |
|---|---|
| **Streaming responses** | Tokens appear as they arrive via `st.write_stream` |
| **Conversational memory** | Last 4 exchanges are passed as history; ambiguous follow-ups are reformulated before retrieval |
| **Hybrid search** | BM25 keyword + Pinecone dense vectors (ensemble 0.4 / 0.6) — better recall for names, acronyms, exact terms |
| **Cross-encoder reranking** | FlashrankRerank re-scores 16 candidates, keeps top 5 before passing to LLM |
| **Strong embedding model** | `BAAI/bge-base-en-v1.5` (768-dim, higher MTEB scores than MiniLM) |
| **Source chunk preview** | Each citation expander shows the actual retrieved text, not just filename + page |
| **URL ingestion** | Paste any web page URL — fetched, parsed, and indexed alongside uploaded files |
| **Per-document delete** | Remove a single document from the index without clearing everything |
| **Document summarization** | Map-reduce summary of any indexed document, triggered from the sidebar |
| **Follow-up question pills** | 3 clickable suggested follow-ups appear after each grounded answer |
| **Chat export** | Download the full session as a Markdown file |
| **Parallel file processing** | Files parsed concurrently via `ThreadPoolExecutor`; embedding/indexing in one batch |
| **Clear error messages** | API key errors, rate limits, and bad files surface as readable `st.error()` messages |

---

## Project Structure

```
DocChat/
│
├── app.py                  # Streamlit frontend — all UI, session state, streaming
├── config.py               # Central config — API keys, model names, chunk settings
├── requirements.txt        # Python dependencies
│
├── core/                   # RAG pipeline internals
│   ├── __init__.py             # Public API exports
│   ├── document_processor.py   # PDF/TXT loading, URL fetching, chunking
│   ├── vector_store.py         # Pinecone + BM25 hybrid retrieval, per-doc delete
│   └── rag_pipeline.py         # Streaming LCEL chains, reranking, follow-ups, summarization
│
└── .streamlit/
    └── secrets.toml        # API keys — gitignored, never commit
```

---

## Files You Must Create Manually

### `.streamlit/secrets.toml`

Create the file at `DocChat/.streamlit/secrets.toml` (already in `.gitignore`):

```toml
CHAT_GROQ_API_KEY   = "your_groq_api_key_here"
ANTHROPIC_API_KEY   = "your_anthropic_api_key_here"
PINECONE_API_KEY    = "your_pinecone_api_key_here"
PINECONE_INDEX_NAME = "general-rag"
```

- Groq API key: https://console.groq.com
- Anthropic API key: https://console.anthropic.com
- Pinecone API key: https://console.pinecone.io

> This file is never committed to git. Keep it private.

---

## Setup & Installation

### 1. Navigate to the project folder

```bash
cd DocChat
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

> The first run downloads two HuggingFace models: the embedding model `BAAI/bge-base-en-v1.5` (~440 MB) and the FlashrankRerank cross-encoder (~85 MB). Both are cached locally after the first download.

### 3. Create your `secrets.toml` file

As described above — add your Groq, Anthropic, and Pinecone API keys.

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

> The Pinecone index (`general-rag`) is auto-created on first run. If you are upgrading from a previous version that used `all-MiniLM-L6-v2` (384-dim), the index will be automatically deleted and recreated at 768-dim — you will need to re-index your documents.

---

## How to Use

1. **Index documents**
   - Upload PDF or TXT files via the sidebar file uploader, or paste a web page URL into "Add from URL" and click **Fetch & Index URL**.
   - Click **Process Documents** to embed and store all selected files. Files are parsed in parallel.

2. **Manage the index**
   - The **Indexed Documents** section lists every source in a scrollable panel. Each row shows the filename and the number of chunks indexed from it in this session.
   - Click **≡** next to any document to generate a map-reduce summary in the main area.
   - Click **✕** to delete that document from the index without affecting others.
   - Use **Clear Index** to delete all vectors and start fresh.

3. **Select model and response length**
   - **Groq (Llama3.3-70B)** — fast, cost-efficient.
   - **Anthropic (Claude Haiku)** — higher quality with adaptive thinking.
   - Choose **Short / Medium / Long** to control answer length.

4. **Ask questions**
   - Toggle **"Use indexed documents"** to switch between RAG-grounded answers and direct LLM knowledge.
   - Answers stream token-by-token. Expand **Referenced Sources** to see the exact retrieved chunks.
   - After each grounded answer, three **follow-up question pills** appear — click one to continue the thread.
   - Follow-up questions work naturally across turns: "what did it say about that?" is automatically reformulated before retrieval.

5. **Export**
   - Once the conversation has messages, an **Export Chat** button appears at the bottom of the sidebar. Downloads the full session as a `.md` file.

---

## Architecture

```
User Question (+ chat history)
        │
        ▼
[History-aware reformulation]  ←── LLM (if prior turns exist)
        │
        ▼
[Hybrid Retriever]
  ├── BM25 (keyword, in-memory)     weight 0.4
  └── Pinecone MMR (dense vectors)  weight 0.6
        │
        ▼  top-16 candidates
[FlashrankRerank cross-encoder]
        │
        ▼  top-5 chunks
[LLM (Groq / Claude)]  ◄── context + question + history (LCEL)
        │
        ▼
  Streaming answer + Sources + Follow-up questions
```

- **Embeddings**: `BAAI/bge-base-en-v1.5` (768-dim, runs locally, no API needed)
- **Vector DB**: Pinecone serverless (cloud-hosted, auto-created index)
- **Retrieval**: Hybrid BM25 + Pinecone MMR ensemble → FlashrankRerank cross-encoder
- **LLMs**: Groq Llama3.3-70B or Anthropic Claude Haiku (`claude-haiku-4-5-20251001`, adaptive thinking)
- **Chain**: LangChain LCEL with `.stream()` for real-time token delivery

---

## Demo Example — LLM Research Papers

The following example uses a set of research papers on large language models and mechanistic interpretability to demonstrate cross-document retrieval and synthesis.

### Documents used

| Paper |
|---|
| Circuit Tracer |
| Fine-Grained Neuron Analysis |
| Information Flow Routes in LLMs |
| LLMs Explain Themselves |
| LLM Multilingualism |
| Mechanistic Interpretability |
| Multilingual Safety |
| Preference Tuning |
| Transformer Feed-Forward Layers |

> These papers span post-2022 research that Groq (Llama3-70B) cannot answer from training data alone — making RAG grounding essential and clearly demonstrable.

### Recommended demo query

> *"How do feed-forward layers and attention heads store and retrieve factual knowledge inside transformer models?"*

This query pulls from multiple papers in a single retrieval pass — a strong cross-document synthesis example. The hybrid retriever handles both semantic similarity (dense vectors) and exact author/term matching (BM25), while the reranker filters to the most relevant 5 chunks before the LLM sees them.

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI and streaming (`st.write_stream`) |
| `langchain` + `langchain-community` | RAG orchestration, EnsembleRetriever, BM25 retriever |
| `langchain-core` | LCEL primitives — prompts, runnables, parsers, message types |
| `langchain-text-splitters` | Document chunking |
| `langchain-groq` | Groq LLM integration |
| `langchain-anthropic` | Anthropic Claude integration |
| `langchain-huggingface` | HuggingFace embeddings |
| `langchain-pinecone` | Pinecone vector store integration |
| `pinecone` | Pinecone SDK |
| `pypdf` | PDF loading |
| `sentence-transformers` | Embedding model (`BAAI/bge-base-en-v1.5`) |
| `anthropic` | Anthropic SDK |
| `groq` | Groq SDK |
| `rank_bm25` | BM25 keyword retriever for hybrid search |
| `flashrank` | Cross-encoder reranking (FlashrankRerank) |
| `beautifulsoup4` | HTML parsing for URL ingestion (WebBaseLoader) |
