from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    TOP_K_INITIAL,
)

_embeddings = None
_pc = None

# In-memory state for BM25 hybrid search and per-document management.
# Populated when add_documents() is called; survives Streamlit reruns within
# the same process but resets on server restart (Pinecone data persists).
_indexed_docs: list = []
_indexed_sources: set = set()


def get_pinecone_client() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)
    return _pc


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def ensure_index_exists():
    pc = get_pinecone_client()
    existing_names = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_names:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    else:
        # If dimension changed (e.g. embedding model upgrade), recreate the index.
        try:
            desc = pc.describe_index(PINECONE_INDEX_NAME)
            if desc.dimension != EMBEDDING_DIMENSION:
                pc.delete_index(PINECONE_INDEX_NAME)
                pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
        except Exception:
            pass


def get_vector_store() -> PineconeVectorStore:
    ensure_index_exists()
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=get_embeddings(),
        pinecone_api_key=PINECONE_API_KEY,
    )


def add_documents(documents: list, batch_size: int = 50, progress_callback=None) -> int:
    global _indexed_docs, _indexed_sources
    vs = get_vector_store()
    total = len(documents)
    for start in range(0, total, batch_size):
        batch = documents[start:start + batch_size]
        vs.add_documents(batch)
        _indexed_docs.extend(batch)
        for doc in batch:
            src = doc.metadata.get("source", "")
            if src:
                _indexed_sources.add(src)
        if progress_callback:
            progress_callback(min(start + batch_size, total), total)
    return total


def get_retriever(search_type: str = "mmr"):
    """Return a hybrid BM25+Pinecone ensemble retriever when in-memory docs exist,
    otherwise fall back to Pinecone-only."""
    vs = get_vector_store()
    pinecone_retriever = vs.as_retriever(
        search_type=search_type,
        search_kwargs={"k": TOP_K_INITIAL},
    )

    if _indexed_docs:
        try:
            from langchain_community.retrievers import BM25Retriever
            from langchain.retrievers import EnsembleRetriever

            bm25 = BM25Retriever.from_documents(_indexed_docs)
            bm25.k = TOP_K_INITIAL
            return EnsembleRetriever(
                retrievers=[bm25, pinecone_retriever],
                weights=[0.4, 0.6],
            )
        except ImportError:
            pass

    return pinecone_retriever


def get_collection_count() -> int:
    try:
        pc = get_pinecone_client()
        index = pc.Index(PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        return stats.get("total_vector_count", 0)
    except Exception as e:
        print(f"[vector_store] Could not fetch index stats: {e}")
        return 0


def clear_collection():
    global _indexed_docs, _indexed_sources
    try:
        pc = get_pinecone_client()
        index = pc.Index(PINECONE_INDEX_NAME)
        index.delete(delete_all=True)
        _indexed_docs.clear()
        _indexed_sources.clear()
    except Exception as e:
        print(f"[vector_store] Could not clear collection: {e}")


def delete_document_by_source(source: str):
    global _indexed_docs, _indexed_sources
    pc = get_pinecone_client()
    index = pc.Index(PINECONE_INDEX_NAME)
    index.delete(filter={"source": {"$eq": source}})
    _indexed_docs[:] = [d for d in _indexed_docs if d.metadata.get("source") != source]
    _indexed_sources.discard(source)


def get_indexed_sources() -> list:
    return sorted(_indexed_sources)


def get_source_chunk_counts() -> dict:
    counts = {}
    for doc in _indexed_docs:
        src = doc.metadata.get("source", "")
        if src:
            counts[src] = counts.get(src, 0) + 1
    return counts


def get_docs_by_source(source: str) -> list:
    return [d for d in _indexed_docs if d.metadata.get("source") == source]
