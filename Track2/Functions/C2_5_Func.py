import re
import json
import statistics
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
from typing import Optional, List

from pydantic import BaseModel, ValidationError


def content_filter(text: str) -> tuple:
    """
    Returns (allowed: bool, reason: str | None).
    Checks for harmful content and prompt injection attempts.
    """
    banned_patterns = [
        r'\b(malware|ransomware|exploit|hack.?password)\b',
        r'\b(bomb|weapon|explosive)\b',
        r'\b(suicide|self.harm)\b',
        r'ignore (all )?previous instructions',
        r'you are now (a )?different',
        r'disregard your (system )?prompt',
    ]
    text_lower = text.lower()
    for pattern in banned_patterns:
        if re.search(pattern, text_lower):
            return False, f'Blocked content pattern: `{pattern}`'
    return True, None


def calculate_cost(model: str, input_tokens: int, output_tokens: int, model_costs: dict) -> float:
    """Calculate USD cost for a single API call."""
    rates = model_costs.get(model, {'input': 0.80, 'output': 4.00})
    return (
        input_tokens  * rates['input']  / 1_000_000 +
        output_tokens * rates['output'] / 1_000_000
    )


@dataclass
class RequestRecord:
    request_id: str
    timestamp: str
    domain: str
    question_length: int
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str
    success: bool
    error_type: Optional[str] = None
    retrieval_hit: bool = True
    hallucination_flagged: bool = False


class MetricsCollector:
    """
    In-memory metrics store for development and testing.
    In production: replace storage with Prometheus / InfluxDB / CloudWatch.
    """

    def __init__(self):
        self.records: list[RequestRecord] = []
        self._domain_totals: dict = defaultdict(lambda: {'requests': 0, 'cost': 0.0, 'errors': 0})

    def record(self, rec: RequestRecord):
        self.records.append(rec)
        d = self._domain_totals[rec.domain]
        d['requests'] += 1
        d['cost'] += rec.cost_usd
        if not rec.success:
            d['errors'] += 1

    def latency_percentiles(self, domain: str = None) -> dict:
        data = [r.latency_ms for r in self.records
                if (domain is None or r.domain == domain) and r.success]
        if not data:
            return {}
        data.sort()
        n = len(data)
        return {
            'p50':   round(statistics.median(data), 1),
            'p95':   round(data[int(n * 0.95)], 1),
            'p99':   round(data[int(n * 0.99)], 1),
            'mean':  round(statistics.mean(data), 1),
            'count': n,
        }

    def cost_summary(self, domain: str = None) -> dict:
        data = [r for r in self.records if (domain is None or r.domain == domain)]
        if not data:
            return {}
        costs = [r.cost_usd for r in data]
        return {
            'total_usd':    round(sum(costs), 6),
            'avg_per_req':  round(statistics.mean(costs), 6),
            'max_req':      round(max(costs), 6),
            'requests':     len(data),
        }

    def failure_rate(self, domain: str = None) -> dict:
        data = [r for r in self.records if (domain is None or r.domain == domain)]
        if not data:
            return {}
        failures = [r for r in data if not r.success]
        rate = len(failures) / len(data) * 100
        error_types = defaultdict(int)
        for r in failures:
            error_types[r.error_type or 'unknown'] += 1
        return {
            'failure_rate_pct': round(rate, 2),
            'total_requests':   len(data),
            'failed_requests':  len(failures),
            'error_breakdown':  dict(error_types),
        }

    def retrieval_miss_rate(self) -> dict:
        data = self.records
        if not data:
            return {}
        misses = [r for r in data if not r.retrieval_hit]
        return {
            'miss_rate_pct':  round(len(misses) / len(data) * 100, 2),
            'total':          len(data),
            'misses':         len(misses),
        }

    def hallucination_rate(self) -> dict:
        flagged = [r for r in self.records if r.hallucination_flagged]
        total = len(self.records)
        return {
            'rate_pct': round(len(flagged) / total * 100, 2) if total else 0,
            'flagged':  len(flagged),
            'total':    total,
        }

    def print_dashboard(self):
        print(f'\n{"="*65}')
        print('  Production AI API — Monitoring Dashboard')
        print(f'{"="*65}')
        print(f'  Total requests : {len(self.records)}')
        print()

        print('  LATENCY (ms)')
        lp = self.latency_percentiles()
        if lp:
            print(f'    p50={lp["p50"]}ms  p95={lp["p95"]}ms  p99={lp["p99"]}ms  mean={lp["mean"]}ms')

        print()
        print('  COST')
        cs = self.cost_summary()
        if cs:
            print(f'    Total: ${cs["total_usd"]:.4f}  |  Avg/req: ${cs["avg_per_req"]:.6f}')

        print()
        print('  RELIABILITY')
        fr = self.failure_rate()
        if fr:
            print(f'    Failure rate: {fr["failure_rate_pct"]}%  '
                  f'({fr["failed_requests"]}/{fr["total_requests"]} requests)')
            if fr['error_breakdown']:
                print(f'    Error types: {fr["error_breakdown"]}')

        print()
        print('  RETRIEVAL')
        rm = self.retrieval_miss_rate()
        if rm:
            print(f'    Miss rate: {rm["miss_rate_pct"]}%  ({rm["misses"]}/{rm["total"]} queries)')

        print()
        print('  DOMAIN BREAKDOWN')
        for domain, totals in self._domain_totals.items():
            err_rate = totals['errors'] / totals['requests'] * 100 if totals['requests'] else 0
            print(f'    {domain:<12}: {totals["requests"]:>4} reqs  '
                  f'${totals["cost"]:.4f} cost  {err_rate:.1f}% errors')

        print(f'{"="*65}\n')


def create_log_record(
    request_id: str,
    domain: str,
    question: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    success: bool,
    model_costs: dict,
    error_type: str = None,
    retrieval_hit: bool = True,
    retrieval_chunks: int = 0,
    guardrail_triggered: str = None,
    hallucination_flagged: bool = False,
    user_id: str = None,
    environment: str = 'prod',
) -> dict:
    """Create a structured log record. Never log raw question text in production."""
    import hashlib
    return {
        'timestamp':              datetime.utcnow().isoformat() + 'Z',
        'request_id':             request_id,
        'service':                'ai-api',
        'version':                '1.0.0',
        'user_id':                hashlib.sha256((user_id or 'anon').encode()).hexdigest()[:16],
        'domain':                 domain,
        'question_hash':          hashlib.sha256(question.encode()).hexdigest()[:16],
        'question_length':        len(question),
        'model':                  model,
        'input_tokens':           input_tokens,
        'output_tokens':          output_tokens,
        'total_tokens':           input_tokens + output_tokens,
        'cost_usd':               calculate_cost(model, input_tokens, output_tokens, model_costs),
        'latency_ms':             round(latency_ms, 1),
        'success':                success,
        'error_type':             error_type,
        'retrieval_hit':          retrieval_hit,
        'retrieval_chunks':       retrieval_chunks,
        'guardrail_triggered':    guardrail_triggered,
        'hallucination_flagged':  hallucination_flagged,
        'environment':            environment,
    }


def detect_injection(text: str) -> tuple:
    """Return (is_injection: bool, matched_pattern: str | None)."""
    injection_patterns = [
        r'ignore (all )?previous instructions',
        r'disregard (your )?(system )?prompt',
        r'you are now (a |an )?(?!customer|student|patient)',   # role-override
        r'print (your )?(complete |full )?(system )?prompt',
        r'developer mode',
        r'jailbreak',
        r'act as (if you (are|were)|a|an)',
        r'forget (everything|all) (you|your)',
        r'new persona',
        r'simulate (being|a|an)',
    ]
    lower = text.lower()
    for pattern in injection_patterns:
        if re.search(pattern, lower):
            return True, pattern
    return False, None


def detect_leakage(text: str) -> tuple:
    """Return (leaks_data: bool, matched_pattern: str | None)."""
    leakage_patterns = [
        r'PREM-\d+\.\d+',           # internal rate codes
        r'internal policy (states|says|is)',
        r'my (system )?prompt (is|says|states)',
        r'i was (instructed|told) to',
        r'confidential|classified|internal only',
    ]
    for pattern in leakage_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True, pattern
    return False, None


@dataclass
class Document:
    doc_id: str
    content: str
    access_level: str    # 'public' | 'customer' | 'premium' | 'employee'
    domain: str


def get_accessible_docs(user_role: str, document_store: List[Document], domain: str = None) -> List[Document]:
    """
    Return only documents the user's role is authorised to see.
    Optionally filter by domain.
    """
    access_hierarchy = {
        'employee': ['public', 'customer', 'premium', 'employee'],
        'premium':  ['public', 'customer', 'premium'],
        'customer': ['public', 'customer'],
        'public':   ['public'],
    }
    allowed_levels = access_hierarchy.get(user_role, ['public'])
    return [
        doc for doc in document_store
        if doc.access_level in allowed_levels
        and (domain is None or doc.domain == domain)
    ]


class RagasEvaluator:
    """
    Production RAGAS evaluation harness.
    Evaluates any RAG pipeline across 4 quality dimensions.
    Works with finance, education, healthcare, or any domain.
    """

    def __init__(self, llm, metrics, metric_names):
        from ragas.llms import LangchainLLMWrapper
        self.evaluator_llm = LangchainLLMWrapper(llm)
        self.METRICS = metrics
        self.METRIC_NAMES = metric_names

    def evaluate(
        self,
        questions: list,
        answers: list,
        contexts: list,     # list of lists: one list of chunks per question
        ground_truths: list,
        domain_label: str = 'general',
    ):
        """
        Run all configured RAGAS metrics.

        Returns:
            DataFrame with one row per question and columns for each metric.
        """
        from datasets import Dataset
        from ragas import evaluate

        dataset = Dataset.from_dict({
            'question': questions,
            'answer': answers,
            'contexts': contexts,
            'ground_truth': ground_truths,
        })

        result = evaluate(
            dataset=dataset,
            metrics=self.METRICS,
            llm=self.evaluator_llm,
        )

        df = result.to_pandas()
        df.insert(0, 'domain', domain_label)
        return df

    def print_report(self, df):
        """Pretty-print evaluation scores with a visual bar."""
        import pandas as pd
        domain = df['domain'].iloc[0].upper()
        print(f'\n{"="*70}')
        print(f'  RAGAS Evaluation Report  |  Domain: {domain}')
        print(f'{"="*70}')
        print(f'{"Question":<42} {"Faith":>6} {"Relev":>6} {"Recall":>7} {"Prec":>6}')
        print('-' * 70)

        for _, row in df.iterrows():
            q = (str(row['question'])[:40] + '..') if len(str(row['question'])) > 42 else str(row['question'])
            scores = []
            for m in self.METRIC_NAMES:
                val = row.get(m, None)
                scores.append(f'{val:.2f}' if val is not None and not pd.isna(val) else ' N/A')
            print(f'{q:<42} {scores[0]:>6} {scores[1]:>6} {scores[2]:>7} {scores[3]:>6}')

        print('-' * 70)
        print('  Averages:')
        for m in self.METRIC_NAMES:
            if m in df.columns:
                avg = df[m].dropna().mean()
                bar = '#' * int(avg * 20)   # ASCII bar: 0.0 = '' | 1.0 = '####################'
                print(f'    {m:<26}: {avg:.3f}  [{bar:<20}]')
        print(f'{"="*70}\n')


class StructuredAnswer(BaseModel):
    answer: str
    confidence: str   # 'high' | 'medium' | 'low'
    disclaimer: Optional[str] = None


def schema_validate(data: dict) -> tuple:
    """
    Returns (valid: bool, error: str | None).
    Validates that the LLM output matches the expected schema.
    """
    try:
        StructuredAnswer(**data)
        return True, None
    except ValidationError as e:
        return False, str(e)


def citation_check(answer: str, source_docs: list) -> tuple:
    """
    Returns (passes: bool, reason: str | None).
    A basic check: at least one key phrase from sources appears in the answer.
    Production systems use embedding similarity for semantic matching.
    """
    answer_lower = answer.lower()
    for doc in source_docs:
        key_terms = [w.lower() for w in doc.split() if len(w) > 5]
        matches = sum(1 for t in key_terms if t in answer_lower)
        if matches >= 2:   # at least 2 key terms overlap
            return True, None
    return False, 'Answer could not be grounded in provided source documents'


REFUSAL_MESSAGES = {
    'content': "I'm unable to help with that request as it falls outside of what I can assist with.",
    'schema': 'The response was not in the expected format. Please try again.',
    'citation': 'I could not verify this response against our knowledge base. Please consult official documentation.',
    'default': "I'm sorry, I cannot process this request.",
}


def refusal_response(guardrail_type: str, detail: str = '') -> dict:
    msg = REFUSAL_MESSAGES.get(guardrail_type, REFUSAL_MESSAGES['default'])
    return {'answer': msg, 'safe': False, 'blocked_by': guardrail_type, 'detail': detail}


def generate_structured_response(question: str, context_docs: list, client_llm) -> dict:
    """Ask the LLM to return structured JSON with answer + confidence."""
    docs_text = '\n'.join(f'- {d}' for d in context_docs)
    prompt = f"""Answer the following question using ONLY the provided documents.
Return valid JSON with keys: answer (string), confidence (high/medium/low), disclaimer (string or null).

Documents:
{docs_text}

Question: {question}

JSON response:"""

    raw = client_llm.invoke(prompt).content.strip()
    raw = re.sub(r'^```json\s*|^```\s*|```$', '', raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {'answer': raw, 'confidence': 'low', 'disclaimer': None}


def guardrailed_chat(user_message: str, source_docs: list, client_llm) -> dict:
    """
    Full guardrail pipeline:
    1. Input content filter
    2. LLM call
    3. Output schema validation
    4. Output citation check
    5. Output content filter
    """
    allowed, reason = content_filter(user_message)
    if not allowed:
        return refusal_response('content', reason)

    try:
        llm_output = generate_structured_response(user_message, source_docs, client_llm)
    except Exception as e:
        return {'answer': 'Service error', 'safe': False, 'blocked_by': 'llm_error', 'detail': str(e)}

    valid, err = schema_validate(llm_output)
    if not valid:
        return refusal_response('schema', err)

    grounded, reason = citation_check(llm_output['answer'], source_docs)
    if not grounded:
        return refusal_response('citation', reason)

    allowed, reason = content_filter(llm_output['answer'])
    if not allowed:
        return refusal_response('content', f'Output blocked: {reason}')

    return {**llm_output, 'safe': True}


def unsecured_chatbot(user_input: str, secret_system_prompt: str) -> str:
    """
    VULNERABLE: passes user input directly to the LLM without sanitisation.
    An attacker can inject instructions to override the system prompt.
    """
    from langchain_anthropic import ChatAnthropic
    from langchain.schema import SystemMessage, HumanMessage

    llm = ChatAnthropic(model='claude-haiku-4-5-20251001', temperature=0)
    messages = [
        SystemMessage(content=secret_system_prompt),
        HumanMessage(content=user_input),  # raw user input — dangerous
    ]
    return llm.invoke(messages).content


def secured_chatbot(user_input: str) -> dict:
    """
    SECURED version with:
    1. Input injection detection
    2. Output leakage detection
    3. Structured refusals (no information about why exactly it was blocked)
    """
    is_injection, pattern = detect_injection(user_input)
    if is_injection:
        return {
            'response': "I'm here to help with banking questions. How can I assist you?",
            'blocked': True,
            'reason': 'input_injection',
        }

    # (In a real app, call the LLM here)
    simulated_responses = {
        'loan rate': 'Our standard personal loan rates range from 5.9% to 18.9% APR based on your credit profile.',
        'default': 'I can help you with questions about our loans, savings accounts, and investment products.',
    }
    key = 'loan rate' if 'rate' in user_input.lower() else 'default'
    llm_output = simulated_responses[key]

    leaks, pattern = detect_leakage(llm_output)
    if leaks:
        return {
            'response': 'I cannot share that information. Please contact our support team.',
            'blocked': True,
            'reason': 'output_leakage',
        }

    return {'response': llm_output, 'blocked': False}


def rbac_rag_query(user_role: str, question: str, document_store: List[Document], domain: str = 'finance') -> dict:
    """
    RBAC-aware RAG: only retrieve documents the user can access.
    """
    accessible = get_accessible_docs(user_role, document_store, domain)
    context_texts = [d.content for d in accessible]

    return {
        'user_role': user_role,
        'docs_available': len(accessible),
        'doc_ids': [d.doc_id for d in accessible],
        'context_preview': context_texts[:2],
        'note': f'User with role "{user_role}" can see {len(accessible)} of {len(document_store)} total docs',
    }
