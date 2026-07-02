from pathlib import Path
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict

import numpy as np


def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())


def cosine_similarity(vec_a, vec_b):
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-12
    return float(np.dot(vec_a, vec_b) / denom)


class LocalSemanticEmbedder:
    """Deterministic offline embedder — covers technical, finance, education, healthcare, and legal vocabulary."""

    def __init__(self, dim=32):
        self.dim = dim
        self.semantic_groups = {
            # Technical / retrieval
            0:  {'python', 'async', 'api', 'request', 'requests', 'concurrent',
                 'concurrency', 'nonblocking', 'event', 'loop'},
            1:  {'java', 'thread', 'threads', 'worker', 'multithreading'},
            2:  {'vector', 'vectors', 'embedding', 'embeddings', 'chroma', 'pinecone',
                 'collection', 'collections', 'persist', 'persistence', 'database',
                 'index', 'indexes'},
            3:  {'metadata', 'source', 'date', 'category', 'author', 'filter',
                 'filters', 'filtering'},
            4:  {'hybrid', 'sparse', 'dense', 'bm25', 'keyword', 'rerank',
                 'reranking', 'cross', 'encoder'},
            5:  {'recall', 'precision', 'mrr', 'ndcg', 'rank', 'ranking', 'metric',
                 'metrics', 'evaluation'},
            6:  {'disk', 'local', 'persistent', 'storage', 'survive', 'restart',
                 'restarts'},
            7:  {'cloud', 'hosted', 'serverless', 'remote'},
            # Finance
            8:  {'portfolio', 'diversification', 'diversify', 'investment', 'invest',
                 'asset', 'assets', 'equity', 'equities', 'debt', 'stock', 'stocks',
                 'bond', 'bonds', 'commodity', 'commodities'},
            9:  {'risk', 'credit', 'default', 'financial', 'finance', 'ratio',
                 'ratios', 'earnings', 'income', 'revenue', 'profit', 'eps', 'roi',
                 'ebitda', 'valuation', 'shares', 'outstanding'},
            10: {'inflation', 'interest', 'rate', 'rates', 'market', 'trading',
                 'algorithmic', 'quantitative', 'arbitrage', 'coupon', 'audit',
                 'sox', 'basel', 'aml', 'regulatory', 'compliance', 'transaction',
                 'monitoring'},
            # Education
            11: {'learning', 'education', 'student', 'students', 'teaching',
                 'teacher', 'curriculum', 'lesson', 'course', 'instruction',
                 'instructional'},
            12: {'assessment', 'bloom', 'taxonomy', 'objectives', 'formative',
                 'summative', 'feedback', 'engagement', 'adaptive', 'performance',
                 'competency', 'competencies', 'standards', 'pacing', 'difficulty'},
            # Healthcare
            13: {'patient', 'clinical', 'health', 'medical', 'treatment',
                 'diagnosis', 'ehr', 'records', 'guidelines', 'evidence',
                 'clinician', 'drug', 'interaction', 'allergy', 'medications',
                 'lab', 'longitudinal'},
            # Legal / compliance
            14: {'contract', 'legal', 'clause', 'obligation', 'obligations',
                 'penalty', 'indemnification', 'regulation', 'policy', 'policies',
                 'law', 'deadline', 'terms', 'rag', 'extract', 'review'},
            # E-commerce / analytics
            15: {'recommendation', 'recommendations', 'sentiment', 'customer',
                 'product', 'review', 'reviews', 'collaborative', 'purchase',
                 'browsing', 'history', 'positive', 'negative', 'neutral'},
        }

    def _vector(self, text):
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text)
        for token in tokens:
            matched = False
            for dim_idx, vocab in self.semantic_groups.items():
                if token in vocab:
                    vec[dim_idx] += 1.0
                    matched = True
            if not matched:
                digest = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)
                vec[16 + (digest % 16)] += 0.25
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return np.vstack([self._vector(text) for text in texts])


class SimpleCollection:
    def __init__(self, name):
        self.name  = name
        self.records = []

    def clear(self):
        self.records = []

    def add(self, ids, documents, embeddings, metadatas=None):
        metadatas = metadatas or [{} for _ in ids]
        for doc_id, document, embedding, metadata in zip(ids, documents, embeddings, metadatas):
            self.records = [r for r in self.records if r['id'] != doc_id]
            self.records.append({
                'id': doc_id, 'document': document,
                'embedding': np.array(embedding, dtype=np.float32),
                'metadata': metadata
            })

    def query(self, query_embeddings, n_results=3, where=None):
        qvec = np.array(query_embeddings[0], dtype=np.float32)
        scored = []
        for r in self.records:
            if where and not all(r['metadata'].get(k) == v for k, v in where.items()):
                continue
            scored.append((cosine_similarity(qvec, r['embedding']), r))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:n_results]
        return {
            'ids':       [[r['id']       for _, r in top]],
            'documents': [[r['document'] for _, r in top]],
            'metadatas': [[r['metadata'] for _, r in top]],
            'distances': [[1.0 - s      for s, _ in top]],
        }


class SimplePersistentClient:
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.collections = {}

    def get_or_create_collection(self, name):
        if name not in self.collections:
            self.collections[name] = SimpleCollection(name)
        return self.collections[name]

    def delete_collection(self, name):
        self.collections.pop(name, None)


class MockPineconeIndex:
    def __init__(self, name, dimension):
        self.name = name
        self.dimension = dimension
        self.vectors = {}

    def upsert(self, vectors):
        for doc_id, vector, metadata in vectors:
            self.vectors[doc_id] = {
                'vector': np.array(vector, dtype=np.float32),
                'metadata': metadata,
            }

    def query(self, vector, top_k=3, include_metadata=True):
        qvec = np.array(vector, dtype=np.float32)
        matches = []
        for doc_id, payload in self.vectors.items():
            score = cosine_similarity(qvec, payload['vector'])
            m = {'id': doc_id, 'score': score}
            if include_metadata:
                m['metadata'] = payload['metadata']
            matches.append(m)
        matches.sort(key=lambda x: x['score'], reverse=True)
        return {'matches': matches[:top_k]}


def dump_jsonl(path, rows):
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')


def dump_csv(path, rows):
    fields = ['id', 'title', 'body', 'source', 'date', 'category', 'author']
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})


def build_corpus_docs():
    """Returns the 29-document multi-domain corpus (id, title, body, source, date, category, author, text)."""
    corpus_docs = [
        # ── Technical / retrieval ──────────────────────────────────────────────────
        {'id': 'c2_001', 'title': 'Python async concurrency',
         'body': 'Python async programming keeps API calls non-blocking while handling many requests at once.',
         'source': 'api/engineering-notes', 'date': '2026-03-01', 'category': 'python',    'author': 'Alice'},
        {'id': 'c2_002', 'title': 'Java multithreading',
         'body': 'Java multithreading handles concurrent API requests in worker threads.',
         'source': 'api/engineering-notes', 'date': '2026-03-02', 'category': 'java',      'author': 'Bob'},
        {'id': 'c2_003', 'title': 'ChromaDB persistence',
         'body': 'A persistent ChromaDB client writes embeddings to disk so collections survive restarts.',
         'source': 'vector-db/playbook',    'date': '2026-03-03', 'category': 'vector_db', 'author': 'Chen'},
        {'id': 'c2_004', 'title': 'Pinecone hosted indexes',
         'body': 'Pinecone is a hosted vector database with serverless indexes managed in the cloud.',
         'source': 'vector-db/playbook',    'date': '2026-03-04', 'category': 'vector_db', 'author': 'Dana'},
        {'id': 'c2_005', 'title': 'Metadata filtering',
         'body': 'Metadata filters on source, date, category, and author raise precision by removing off-topic chunks.',
         'source': 'retrieval/guide',       'date': '2026-03-05', 'category': 'retrieval', 'author': 'Alice'},
        {'id': 'c2_006', 'title': 'Hybrid search',
         'body': 'Hybrid search combines sparse keyword retrieval with dense vector retrieval and reranks the merged candidates.',
         'source': 'retrieval/guide',       'date': '2026-03-06', 'category': 'retrieval', 'author': 'Eli'},
        {'id': 'c2_007', 'title': 'Cross encoder reranking',
         'body': 'A cross encoder scores a query and document together, making it slower but often more accurate than bi-encoder retrieval.',
         'source': 'retrieval/guide',       'date': '2026-03-07', 'category': 'retrieval', 'author': 'Sam'},
        {'id': 'c2_008', 'title': 'Retrieval quality metrics',
         'body': 'Recall@k checks coverage, precision@k checks returned quality, MRR rewards the first hit, and NDCG uses graded relevance and rank.',
         'source': 'evaluation/metrics',    'date': '2026-03-08', 'category': 'evaluation','author': 'Priya'},
        {'id': 'c2_009', 'title': 'BM25 sparse search',
         'body': 'BM25 and TF-IDF are sparse methods that rely on exact token overlap and work well for keyword-heavy queries.',
         'source': 'retrieval/guide',       'date': '2026-03-09', 'category': 'retrieval', 'author': 'Noa'},
        {'id': 'c2_010', 'title': 'Dense embeddings',
         'body': 'Dense vector embeddings capture semantic similarity and can match paraphrases even when exact keywords differ.',
         'source': 'retrieval/guide',       'date': '2026-03-10', 'category': 'retrieval', 'author': 'Noa'},
        {'id': 'c2_011', 'title': 'Query rewriting',
         'body': 'Query expansion helps recall, while segmentation and scoping improve precision when the wording is ambiguous.',
         'source': 'retrieval/guide',       'date': '2026-03-11', 'category': 'retrieval', 'author': 'Mina'},
        {'id': 'c2_012', 'title': 'Python observability',
         'body': 'Python logging records structured events and is commonly filtered by source and author in observability systems.',
         'source': 'python/ops',            'date': '2026-03-12', 'category': 'python',    'author': 'Alice'},
        # ── Finance ────────────────────────────────────────────────────────────────
        {'id': 'c2_013', 'title': 'Portfolio diversification',
         'body': 'Modern portfolio theory optimises risk-adjusted returns by diversifying investments across uncorrelated asset classes such as equities, bonds, and commodities.',
         'source': 'finance/investment-guide', 'date': '2026-03-13', 'category': 'finance', 'author': 'Riya'},
        {'id': 'c2_014', 'title': 'Credit risk scoring',
         'body': 'Credit risk models estimate the probability of borrower default using financial ratios like debt-to-equity, interest coverage, and operating cash flow.',
         'source': 'finance/risk-management', 'date': '2026-03-14', 'category': 'finance', 'author': 'Omar'},
        {'id': 'c2_015', 'title': 'Earnings per share analysis',
         'body': 'Earnings per share (EPS) divides net income by outstanding shares and is a primary metric analysts use to compare company profitability across periods.',
         'source': 'finance/investment-guide', 'date': '2026-03-15', 'category': 'finance', 'author': 'Riya'},
        {'id': 'c2_016', 'title': 'Bond pricing and inflation',
         'body': 'Rising inflation erodes the purchasing power of fixed coupon payments, pushing bond prices lower as market interest rates increase.',
         'source': 'finance/investment-guide', 'date': '2026-03-16', 'category': 'finance', 'author': 'Omar'},
        {'id': 'c2_017', 'title': 'Financial regulatory compliance',
         'body': 'Financial institutions must maintain audit trails, transaction monitoring logs, and regulatory reports to satisfy SOX, Basel III, and AML requirements.',
         'source': 'finance/compliance',      'date': '2026-03-17', 'category': 'finance', 'author': 'Priya'},
        {'id': 'c2_018', 'title': 'Algorithmic trading strategies',
         'body': 'Algorithmic trading executes orders using quantitative signals derived from historical price data, volume patterns, and statistical arbitrage models.',
         'source': 'finance/trading-guide',   'date': '2026-03-18', 'category': 'finance', 'author': 'Dana'},
        # ── Education ──────────────────────────────────────────────────────────────
        {'id': 'c2_019', 'title': "Bloom's taxonomy",
         'body': "Bloom's taxonomy organises learning objectives into six cognitive levels: remember, understand, apply, analyse, evaluate, and create, guiding curriculum design and assessment.",
         'source': 'education/curriculum-guide', 'date': '2026-03-19', 'category': 'education', 'author': 'Mina'},
        {'id': 'c2_020', 'title': 'Formative vs summative assessment',
         'body': 'Formative assessments provide ongoing feedback during instruction so teachers can adjust pacing, while summative assessments evaluate final learning outcomes.',
         'source': 'education/assessment-guide', 'date': '2026-03-20', 'category': 'education', 'author': 'Sam'},
        {'id': 'c2_021', 'title': 'Adaptive learning platforms',
         'body': 'Adaptive learning systems use student performance data to personalise lesson difficulty, pacing, and content recommendations in real time.',
         'source': 'education/learning-systems', 'date': '2026-03-21', 'category': 'education', 'author': 'Mina'},
        {'id': 'c2_022', 'title': 'Curriculum alignment',
         'body': 'Curriculum alignment ensures that learning objectives, instructional activities, and assessment tasks all address the same standards and competencies.',
         'source': 'education/curriculum-guide', 'date': '2026-03-22', 'category': 'education', 'author': 'Chen'},
        {'id': 'c2_023', 'title': 'Student engagement metrics',
         'body': 'Student engagement is measured through attendance rates, assignment submission frequency, discussion participation, and time-on-task analytics.',
         'source': 'education/assessment-guide', 'date': '2026-03-23', 'category': 'education', 'author': 'Sam'},
        # ── Healthcare ─────────────────────────────────────────────────────────────
        {'id': 'c2_024', 'title': 'Clinical decision support',
         'body': 'Clinical decision support systems retrieve patient history and evidence-based guidelines to recommend treatment options and alert clinicians to drug interactions.',
         'source': 'healthcare/clinical-systems',   'date': '2026-03-24', 'category': 'healthcare', 'author': 'Priya'},
        {'id': 'c2_025', 'title': 'Electronic health records',
         'body': 'Electronic health records store structured patient data including diagnoses, medications, lab results, allergies, and visit notes for longitudinal care.',
         'source': 'healthcare/records-management', 'date': '2026-03-25', 'category': 'healthcare', 'author': 'Bob'},
        # ── Legal / compliance ─────────────────────────────────────────────────────
        {'id': 'c2_026', 'title': 'Contract clause extraction',
         'body': 'RAG pipelines extract key clauses from legal contracts to surface obligations, deadlines, penalties, and indemnification terms for review teams.',
         'source': 'legal/contract-management', 'date': '2026-03-26', 'category': 'legal', 'author': 'Eli'},
        {'id': 'c2_027', 'title': 'Regulatory document search',
         'body': 'Compliance teams use semantic search across thousands of policy documents to find the most relevant regulatory paragraphs for a given business question.',
         'source': 'legal/compliance-guide',    'date': '2026-03-27', 'category': 'legal', 'author': 'Eli'},
        # ── E-commerce / analytics ─────────────────────────────────────────────────
        {'id': 'c2_028', 'title': 'Product recommendation engines',
         'body': 'Product recommendation engines combine collaborative filtering with dense embeddings to surface items aligned with a customer purchase and browsing history.',
         'source': 'ecommerce/recommendation-systems', 'date': '2026-03-28', 'category': 'ecommerce', 'author': 'Noa'},
        {'id': 'c2_029', 'title': 'Customer sentiment analysis',
         'body': 'Sentiment analysis classifies customer review text as positive, neutral, or negative using dense embeddings trained on labelled review datasets.',
         'source': 'ecommerce/analytics', 'date': '2026-03-29', 'category': 'ecommerce', 'author': 'Noa'},
    ]

    for row in corpus_docs:
        row['text'] = f"{row['title']}. {row['body']}"
    return corpus_docs


def build_eval_sets():
    """Returns (metadata_eval_set, labeled_eval_set) for the 29-document corpus."""
    metadata_eval_set = [
        {
            'query': 'concurrent API requests',
            'relevance': {'c2_001': 2, 'c2_002': 1},
            'metadata_filter': {'category': 'python', 'author': 'Alice'},
        },
        {
            'query': 'diversification and risk-adjusted investment returns',
            'relevance': {'c2_013': 2, 'c2_014': 1},
            'metadata_filter': {'category': 'finance', 'author': 'Riya'},
        },
        {
            'query': 'student assessment feedback and pacing',
            'relevance': {'c2_020': 2, 'c2_023': 1},
            'metadata_filter': {'category': 'education'},
        },
    ]

    # ── Labeled eval: cross-domain queries for recall/precision/MRR/NDCG ───────────
    labeled_eval_set = [
        # Technical / retrieval
        {'query': 'What stores embeddings locally on disk?',
         'relevance': {'c2_003': 2, 'c2_004': 1},           'metadata_filter': {'category': 'vector_db'}},
        {'query': 'Which method combines keyword and vector retrieval?',
         'relevance': {'c2_006': 2, 'c2_009': 1, 'c2_010': 1}, 'metadata_filter': {'category': 'retrieval'}},
        {'query': 'What scores a query and document together?',
         'relevance': {'c2_007': 2},                         'metadata_filter': {'category': 'retrieval'}},
        {'query': 'Which metric rewards the first relevant hit?',
         'relevance': {'c2_008': 2},                         'metadata_filter': {'category': 'evaluation'}},
        {'query': 'How do you improve concurrent API requests in Python?',
         'relevance': {'c2_001': 2, 'c2_002': 1},            'metadata_filter': {'category': 'python'}},
        {'query': 'How should I filter documents by source, date, category, and author?',
         'relevance': {'c2_005': 2, 'c2_012': 1},            'metadata_filter': {'category': 'retrieval'}},
        # Finance
        {'query': 'How does diversification reduce investment risk in a portfolio?',
         'relevance': {'c2_013': 2, 'c2_014': 1},            'metadata_filter': {'category': 'finance'}},
        {'query': 'What financial metric divides net income by outstanding shares?',
         'relevance': {'c2_015': 2},                         'metadata_filter': {'category': 'finance'}},
        {'query': 'How does inflation affect bond prices and interest rates?',
         'relevance': {'c2_016': 2, 'c2_013': 1},            'metadata_filter': {'category': 'finance'}},
        # Education
        {"query": "What are the six levels of Bloom's taxonomy for learning objectives?",
         'relevance': {'c2_019': 2, 'c2_022': 1},            'metadata_filter': {'category': 'education'}},
        {'query': 'How do adaptive learning platforms personalise lesson difficulty for students?',
         'relevance': {'c2_021': 2, 'c2_023': 1},            'metadata_filter': {'category': 'education'}},
        # Healthcare
        {'query': 'How do clinical decision support systems recommend patient treatments?',
         'relevance': {'c2_024': 2, 'c2_025': 1},            'metadata_filter': {'category': 'healthcare'}},
        # Legal
        {'query': 'What RAG techniques help extract obligations from legal contracts?',
         'relevance': {'c2_026': 2, 'c2_027': 1},            'metadata_filter': {'category': 'legal'}},
    ]

    return metadata_eval_set, labeled_eval_set


def persist_corpus(corpus_docs, metadata_eval_set, labeled_eval_set, corpus_dir):
    """
    Writes the corpus + eval sets to disk (docs.jsonl, docs.csv, metadata_eval_set.jsonl,
    labeled_eval_set.jsonl). Idempotent: skips if already persisted.
    """
    corpus_dir.mkdir(parents=True, exist_ok=True)
    if not (corpus_dir / 'docs.jsonl').exists():
        dump_jsonl(corpus_dir / 'docs.jsonl',              corpus_docs)
        dump_csv  (corpus_dir / 'docs.csv',                corpus_docs)
        dump_jsonl(corpus_dir / 'metadata_eval_set.jsonl', metadata_eval_set)
        dump_jsonl(corpus_dir / 'labeled_eval_set.jsonl',  labeled_eval_set)
        return True
    return False


def load_jsonl(path):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_csv_docs(path):
    rows = []
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


class SimpleBM25:
    def __init__(self, docs, text_key='text', k1=1.5, b=0.75):
        self.docs       = docs
        self.text_key   = text_key
        self.k1         = k1
        self.b          = b
        self.doc_tokens = [tokenize(doc[text_key]) for doc in docs]
        self.doc_lengths = [len(t) for t in self.doc_tokens]
        self.avgdl      = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.df         = Counter()
        for tokens in self.doc_tokens:
            for term in set(tokens):
                self.df[term] += 1
        self.n_docs = len(docs)

    def _idf(self, term):
        df = self.df.get(term, 0)
        return math.log(1 + ((self.n_docs - df + 0.5) / (df + 0.5)))

    def score(self, query):
        query_terms = tokenize(query)
        scores = []
        for tokens, doc_len in zip(self.doc_tokens, self.doc_lengths):
            tf    = Counter(tokens)
            score = 0.0
            for term in query_terms:
                if term not in tf:
                    continue
                tf_val = tf[term]
                idf    = self._idf(term)
                denom  = tf_val + self.k1 * (1 - self.b + self.b * (doc_len / max(self.avgdl, 1e-12)))
                score += idf * ((tf_val * (self.k1 + 1)) / denom)
            scores.append(score)
        return np.array(scores, dtype=np.float32)


def normalize_scores(scores):
    lo, hi = float(np.min(scores)), float(np.max(scores))
    if math.isclose(lo, hi):
        return np.zeros_like(scores, dtype=np.float32)
    return (scores - lo) / (hi - lo)


def rerank_like_cross_encoder(query, candidates):
    query_terms = set(tokenize(query))
    ranked = []
    for item in candidates:
        doc         = item['doc']
        doc_terms   = set(tokenize(doc['text']))
        title_terms = set(tokenize(doc['title']))
        score       = item['score']
        score += 0.6 * len(query_terms & doc_terms)  / max(len(query_terms), 1)
        score += 0.4 * len(query_terms & title_terms) / max(len(query_terms), 1)
        if query.lower() in doc['text'].lower():
            score += 1.0
        if doc['category'] in query.lower() or doc['category'] in doc['source']:
            score += 0.1
        ranked.append({'doc': doc, 'score': score})
    ranked.sort(key=lambda x: x['score'], reverse=True)
    return ranked


def recall_at_k(ranked_ids, relevance, k):
    relevant = {doc_id for doc_id, grade in relevance.items() if grade > 0}
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def precision_at_k(ranked_ids, relevance, k):
    if k == 0:
        return 0.0
    hits = sum(1 for doc_id in ranked_ids[:k] if relevance.get(doc_id, 0) > 0)
    return hits / k


def reciprocal_rank(ranked_ids, relevance):
    for idx, doc_id in enumerate(ranked_ids, start=1):
        if relevance.get(doc_id, 0) > 0:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(ranked_ids, relevance, k):
    def dcg(ids):
        return sum(
            (2 ** relevance.get(doc_id, 0) - 1) / math.log2(idx + 1)
            for idx, doc_id in enumerate(ids[:k], start=1)
        )
    ideal_gains = sorted(relevance.values(), reverse=True)
    ideal = sum(
        (2 ** g - 1) / math.log2(idx + 1)
        for idx, g in enumerate(ideal_gains[:k], start=1)
    )
    return dcg(ranked_ids) / ideal if ideal else 0.0


def evaluate_ranker(eval_rows, ranker, k=5):
    agg = defaultdict(float)
    for row in eval_rows:
        ranked_ids = [item['doc']['id'] for item in ranker(row['query'])]
        agg['recall']    += recall_at_k(ranked_ids,    row['relevance'], k)
        agg['precision'] += precision_at_k(ranked_ids, row['relevance'], k)
        agg['mrr']       += reciprocal_rank(ranked_ids, row['relevance'])
        agg['ndcg']      += ndcg_at_k(ranked_ids,      row['relevance'], k)
    n = max(len(eval_rows), 1)
    return {m: v / n for m, v in agg.items()}


def build_chroma_collection(chroma_dir, embedder, use_real_chromadb=False, collection_name='c2_2_demo'):
    if use_real_chromadb:
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(chroma_dir))
            try:
                client.delete_collection(collection_name)
            except Exception:
                pass
            collection = client.get_or_create_collection(name=collection_name)
            return 'chromadb', collection
        except Exception as exc:
            print(f'  chromadb unavailable ({type(exc).__name__}), falling back to mock')
    mock_client = SimplePersistentClient(str(chroma_dir))
    return 'mock:SimplePersistentClient', mock_client.get_or_create_collection(collection_name)


def build_pinecone_index(embedder, index_name='c2-2-demo'):
    api_key = os.getenv('PINECONE_API_KEY')
    try:
        from pinecone import Pinecone, ServerlessSpec
    except Exception:
        Pinecone = None
        ServerlessSpec = None
    if Pinecone is None or not api_key:
        return 'mock:no_api_key_or_package', MockPineconeIndex(index_name, embedder.dim)
    pc = Pinecone(api_key=api_key)
    try:
        pc.delete_index(index_name)
    except Exception:
        pass
    try:
        existing = pc.list_indexes().names()
    except Exception:
        existing = []
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=embedder.dim,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1'),
        )
    return 'pinecone', pc.Index(index_name)


def build_payloads(docs, embedder):
    doc_texts      = [d['text'] for d in docs]
    doc_embeddings = embedder.encode(doc_texts).tolist()
    doc_ids        = [d['id']   for d in docs]
    doc_metadatas  = [
        {'source': d['source'], 'date': d['date'],
         'category': d['category'], 'author': d['author']}
        for d in docs
    ]
    return doc_ids, doc_texts, doc_embeddings, doc_metadatas


def top1_hit(result_ids, expected_doc_id):
    return 1 if result_ids and result_ids[0] == expected_doc_id else 0


def rank_sparse(query, bm25, loaded_docs, top_k=5):
    scores = bm25.score(query)
    order  = np.argsort(scores)[::-1][:top_k]
    return [{'doc': loaded_docs[i], 'score': float(scores[i])} for i in order]


def rank_dense(query, embedder, dense_vectors, loaded_docs, top_k=5):
    qvec   = embedder.encode([query])[0]
    scores = np.array([cosine_similarity(qvec, dv) for dv in dense_vectors], dtype=np.float32)
    order  = np.argsort(scores)[::-1][:top_k]
    return [{'doc': loaded_docs[i], 'score': float(scores[i])} for i in order]


def rank_hybrid(query, bm25, embedder, dense_vectors, loaded_docs, alpha=0.55, top_k=5):
    sparse_s = normalize_scores(bm25.score(query))
    qvec     = embedder.encode([query])[0]
    dense_s  = normalize_scores(
        np.array([cosine_similarity(qvec, dv) for dv in dense_vectors], dtype=np.float32))
    combined = alpha * dense_s + (1 - alpha) * sparse_s
    order    = np.argsort(combined)[::-1][:top_k]
    return [
        {'doc': loaded_docs[i], 'score': float(combined[i]),
         'sparse': float(sparse_s[i]), 'dense': float(dense_s[i])}
        for i in order
    ]


def top1_accuracy_for_queries(eval_rows, ranker):
    hits = 0
    for row in eval_rows:
        ranked = ranker(row['query'])
        if ranked and ranked[0]['doc']['id'] in row['relevance']:
            hits += 1
    return hits / len(eval_rows)
