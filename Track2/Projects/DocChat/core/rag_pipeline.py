from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from .vector_store import get_retriever, get_docs_by_source
from config import (
    GROQ_API_KEY,
    ANTHROPIC_API_KEY,
    GROQ_MODEL,
    ANTHROPIC_MODEL,
    SYSTEM_PROMPT,
    TOP_K_RESULTS,
)

RESPONSE_LENGTH_INSTRUCTIONS = {
    "Short":  "Respond in 2-3 sentences. Be direct and concise.",
    "Medium": "Respond in a few short paragraphs. Cover the key points without over-explaining.",
    "Long":   "Respond in detail. Use paragraphs or bullet points to cover the topic thoroughly.",
}

# Used to reformulate ambiguous follow-up questions into standalone queries
# before passing to the retriever, so retrieval is always context-independent.
CONTEXTUALIZE_Q_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Given a chat history and the latest user question which might reference "
        "context in the chat history, formulate a standalone question which can be "
        "understood without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful AI assistant. Use the following context excerpts from the "
        "provided documents to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Instructions:\n"
        "- Base your answer strictly on the provided context\n"
        "- If the context does not contain enough information, say so clearly\n"
        "- {length_instruction}"
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

DIRECT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}\n\nResponse length instruction: {length_instruction}"),
])

FOLLOWUP_PROMPT = ChatPromptTemplate.from_template(
    "Based on this Q&A exchange, generate exactly 3 short follow-up questions "
    "a researcher might ask next. Return only the 3 questions, one per line, "
    "with no numbering, bullets, or extra text.\n\n"
    "Question: {question}\nAnswer: {answer}"
)

MAP_PROMPT = ChatPromptTemplate.from_template(
    "Summarize the key points from this document excerpt in 2-3 sentences:\n\n{text}"
)

REDUCE_PROMPT = ChatPromptTemplate.from_template(
    "You are given a series of partial summaries from a single document. "
    "Combine them into one coherent summary covering the main themes, key findings, "
    "and important details:\n\n{text}"
)


def _format_docs_with_labels(docs: list) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")
        label = f"[{i}] {src}" + (f" p.{page + 1}" if page != "" else "")
        parts.append(f"{label}\n{doc.page_content}")
    return "\n\n".join(parts)


def _get_llm(provider: str):
    if provider == "Groq (Llama3.3-70B)":
        if not GROQ_API_KEY:
            raise ValueError("CHAT_GROQ_API_KEY is not set in .streamlit/secrets.toml.")
        return ChatGroq(api_key=GROQ_API_KEY, model_name=GROQ_MODEL, temperature=0.1)
    elif provider == "Anthropic (Claude-Sonnet-4.6)":
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set in .streamlit/secrets.toml.")
        return ChatAnthropic(
            api_key=ANTHROPIC_API_KEY,
            model=ANTHROPIC_MODEL,
            max_tokens=2048,
            thinking={"type": "adaptive"},
        )
    raise ValueError(f"Unknown provider: {provider}")


def _get_groq_llm():
    if not GROQ_API_KEY:
        raise ValueError("CHAT_GROQ_API_KEY is not set in .streamlit/secrets.toml.")
    return ChatGroq(api_key=GROQ_API_KEY, model_name=GROQ_MODEL, temperature=0.3)


def _rerank_docs(docs: list, query: str) -> list:
    """Rerank with a FlashrankRerank cross-encoder; falls back to top-N slice."""
    if not docs:
        return []
    top_n = min(TOP_K_RESULTS, len(docs))
    try:
        from langchain_community.document_compressors import FlashrankRerank
        compressor = FlashrankRerank(top_n=top_n)
        reranked = compressor.compress_documents(docs, query)
        return list(reranked) if reranked else docs[:top_n]
    except Exception:
        return docs[:top_n]


def _extract_source_chunks(docs: list) -> list:
    """Return [{label, content}] deduplicated by label."""
    seen = set()
    result = []
    for doc in docs:
        src = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")
        label = f"{src} (page {page + 1})" if page != "" else src
        if label not in seen:
            seen.add(label)
            result.append({"label": label, "content": doc.page_content})
    return result


def query_rag(
    question: str,
    provider: str,
    response_length: str = "Medium",
    use_rag: bool = True,
    chat_history: list = None,
    stream: bool = False,
) -> dict:
    chat_history = chat_history or []
    llm = _get_llm(provider)
    length_instruction = RESPONSE_LENGTH_INSTRUCTIONS.get(response_length, RESPONSE_LENGTH_INSTRUCTIONS["Medium"])

    def _make_direct(inputs):
        chain = DIRECT_PROMPT | llm | StrOutputParser()
        if stream:
            return {"stream": chain.stream(inputs), "sources": [], "source_chunks": [], "grounded": False}
        return {"answer": chain.invoke(inputs), "sources": [], "source_chunks": [], "grounded": False}

    if not use_rag:
        return _make_direct({"question": question, "length_instruction": length_instruction, "chat_history": chat_history})

    # Retrieve — reformulate ambiguous follow-ups into standalone questions before retrieval
    base_retriever = get_retriever()
    try:
        if chat_history:
            condense_chain = CONTEXTUALIZE_Q_PROMPT | llm | StrOutputParser()
            standalone_question = condense_chain.invoke({"input": question, "chat_history": chat_history})
            source_docs = base_retriever.invoke(standalone_question)
        else:
            source_docs = base_retriever.invoke(question)
    except Exception as e:
        raise RuntimeError(f"Retrieval failed: {e}")

    if not source_docs:
        return _make_direct({"question": question, "length_instruction": length_instruction, "chat_history": chat_history})

    source_docs = _rerank_docs(source_docs, question)
    context = _format_docs_with_labels(source_docs)
    source_chunks = _extract_source_chunks(source_docs)
    sources = [s["label"] for s in source_chunks]

    chain = (
        {
            "context": lambda _: context,
            "question": lambda _: question,
            "length_instruction": lambda _: length_instruction,
            "chat_history": lambda _: chat_history,
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    if stream:
        return {"stream": chain.stream(question), "sources": sources, "source_chunks": source_chunks, "grounded": True}
    return {"answer": chain.invoke(question), "sources": sources, "source_chunks": source_chunks, "grounded": True}


def generate_followup_questions(question: str, answer: str) -> list:
    """Generate 3 follow-up questions from a Q&A pair using a fast Groq call."""
    try:
        llm = _get_groq_llm()
        chain = FOLLOWUP_PROMPT | llm | StrOutputParser()
        raw = chain.invoke({"question": question, "answer": answer[:600]})
        return [q.strip() for q in raw.strip().splitlines() if q.strip()][:3]
    except Exception:
        return []


def summarize_document(source: str, provider: str) -> str:
    """Map-reduce summarization over all indexed chunks for a given source."""
    docs = get_docs_by_source(source)
    if not docs:
        return "No indexed content found for this document. Re-index it to enable summarization."

    docs_sorted = sorted(docs, key=lambda d: d.metadata.get("page", 0))[:20]

    try:
        llm = _get_llm(provider)
        map_chain = MAP_PROMPT | llm | StrOutputParser()
        reduce_chain = REDUCE_PROMPT | llm | StrOutputParser()

        chunk_summaries = [map_chain.invoke({"text": d.page_content}) for d in docs_sorted]
        return reduce_chain.invoke({"text": "\n\n".join(chunk_summaries)})
    except Exception as e:
        raise RuntimeError(f"Summarization failed: {e}")
