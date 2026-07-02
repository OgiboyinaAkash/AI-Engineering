import numpy as np
import faiss


def sparse_search(query, vectorizer, doc_vectors, document_ids, titles, documents, k=3):
    from sklearn.metrics.pairwise import cosine_similarity
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, doc_vectors)[0]
    ranked_indices = scores.argsort()[-k:][::-1]
    results = []
    for idx in ranked_indices:
        results.append({
            "id": document_ids[idx],
            "title": titles[idx],
            "text": documents[idx],
            "score": float(scores[idx])
        })
    return results


def dense_search(query, embed_model, faiss_index, document_ids, titles, documents, k=3):
    query_vec = np.asarray(embed_model.encode([query], convert_to_numpy=True), dtype="float32")
    faiss.normalize_L2(query_vec)
    scores, indices = faiss_index.search(query_vec, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "id": document_ids[idx],
            "title": titles[idx],
            "text": documents[idx],
            "score": float(score)
        })
    return results


def chroma_search(query, embed_model, chroma_collection, k=3):
    q_vec = np.asarray(embed_model.encode([query], convert_to_numpy=True), dtype="float32")
    faiss.normalize_L2(q_vec)
    res = chroma_collection.query(query_embeddings=q_vec.tolist(), n_results=k)
    return [
        {"title": m["title"], "doc_id": m["doc_id"], "distance": d, "text": t}
        for m, d, t in zip(
            res["metadatas"][0], res["distances"][0], res["documents"][0]
        )
    ]


def build_context(results):
    blocks = []
    for item in results:
        blocks.append(f"[Doc: {item['title']} | score={item['score']:.3f}]\n{item['text']}")
    return "\n\n".join(blocks)


def build_prompt(query, context):
    return f"""You are a helpful assistant for an AI engineering course.
Use the context to answer the question. If the context is not enough, say so clearly.

Context:
{context}

Question:
{query}

Answer:
"""


def retrieve(query, embed_model, faiss_index, document_ids, titles, documents, k=3):
    return dense_search(query, embed_model, faiss_index, document_ids, titles, documents, k=k)


def rag_pipeline(query, embed_model, faiss_index, document_ids, titles, documents, k=3):
    results = retrieve(query, embed_model, faiss_index, document_ids, titles, documents, k=k)
    context = build_context(results)
    prompt = build_prompt(query, context)
    return {
        "query": query,
        "results": results,
        "context": context,
        "prompt": prompt,
    }


def non_rag_answer(query, llm):
    if llm is None:
        return "Set CHAT_GROQ_API_KEY to run the live baseline answer."
    return llm.invoke(query).content


def rag_answer(query, llm, embed_model, faiss_index, document_ids, titles, documents, k=3):
    if llm is None:
        return "Set CHAT_GROQ_API_KEY to run the live RAG answer."
    payload = rag_pipeline(query, embed_model, faiss_index, document_ids, titles, documents, k=k)
    response = llm.invoke(payload["prompt"])
    return response.content
