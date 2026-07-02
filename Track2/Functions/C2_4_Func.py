from dataclasses import dataclass, field
from datetime import datetime


def build_c2_4_corpus():
    """Returns the 5-document finance + education corpus (text, domain metadata)."""
    finance_docs = [
        {'text': '''Annual Report 2023 — FinTech Corp.
Revenue grew 18% to $2.4B. Net income margin improved to 14%.
Digital payments segment contributed 60% of revenue.
Risk factors: regulatory scrutiny in EU, rising cost of capital.
Management targets 20% revenue growth in 2024 through SME expansion.''',
         'metadata': {'domain': 'finance', 'type': 'annual_report', 'company': 'FinTechCorp'}},

        {'text': '''Q3 2023 Earnings — EduLearn Inc.
Course enrollment rose 35% YoY to 1.2M active learners.
Revenue: $180M (+28% YoY). Gross margin: 72%.
B2B corporate training contracts now represent 40% of revenue.
Guidance: full-year revenue $720M, adjusted EBITDA margin 25%.
Key risk: teacher retention cost increased 15% due to talent competition.''',
         'metadata': {'domain': 'finance', 'type': 'earnings', 'company': 'EduLearnInc'}},

        {'text': '''Investment Thesis — Healthcare AI Sector 2024.
AI diagnostics market projected to reach $45B by 2027 (CAGR 48%).
Key players: imaging AI, clinical decision support, drug discovery AI.
Regulatory risk: FDA pathway for AI/ML-based software devices tightening.
Recommended allocation: 5–8% of tech portfolio for diversified exposure.''',
         'metadata': {'domain': 'finance', 'type': 'thesis', 'sector': 'HealthcareAI'}},
    ]

    education_docs = [
        {'text': '''Course Catalog 2024 — DataScience Track.
DS-101 Introduction to Python (no prerequisites, 6 weeks).
DS-201 Statistics for Data Science (requires DS-101, 8 weeks).
DS-301 Machine Learning Fundamentals (requires DS-201, 10 weeks).
DS-401 Deep Learning and Neural Networks (requires DS-301, 12 weeks).
Capstone project required for certificate completion.''',
         'metadata': {'domain': 'education', 'type': 'catalog', 'track': 'DataScience'}},

        {'text': '''Learning Outcomes Report 2023.
Completion rate for DS track: 68% (industry average 45%).
Top reason for dropout: time management (38%), difficulty (29%), career change (18%).
Interventions that improved completion: cohort learning (+12%), weekly check-ins (+9%).
NPS score: 72. Employer satisfaction with graduates: 4.4/5.''',
         'metadata': {'domain': 'education', 'type': 'outcomes_report'}},
    ]

    return finance_docs + education_docs


def persist_c2_4_corpus(documents, corpus_dir):
    """Idempotent: writes the corpus to docs.jsonl if not already persisted."""
    import json
    corpus_dir.mkdir(parents=True, exist_ok=True)
    path = corpus_dir / 'docs.jsonl'
    if not path.exists():
        with path.open('w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc) + '\n')
        return True
    return False


def load_c2_4_corpus(corpus_dir):
    """Reads the persisted corpus back from docs.jsonl."""
    import json
    docs = []
    with (corpus_dir / 'docs.jsonl').open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


@dataclass
class PromptVersion:
    name:       str
    version:    str
    template:   str
    description: str = ''
    author:     str = ''
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags:       list = field(default_factory=list)
    metrics:    dict = field(default_factory=dict)   # filled by A/B test


class PromptRegistry:
    """Central registry for versioned prompt templates."""

    def __init__(self):
        self._store: dict[str, dict[str, PromptVersion]] = {}

    def register(self, pv: PromptVersion):
        self._store.setdefault(pv.name, {})[pv.version] = pv
        print(f'Registered prompt  name={pv.name}  version={pv.version}')

    def get(self, name: str, version: str = 'latest') -> PromptVersion:
        if name not in self._store:
            raise KeyError(f'Prompt "{name}" not found')
        versions = self._store[name]
        if version == 'latest':
            version = sorted(versions.keys())[-1]
        if version not in versions:
            raise KeyError(f'Version "{version}" not found for "{name}"')
        return versions[version]

    def list_versions(self, name: str) -> list[str]:
        return sorted(self._store.get(name, {}).keys())

    def all_prompts(self) -> list[str]:
        return list(self._store.keys())

    def record_metric(self, name: str, version: str, metric: str, value: float):
        pv = self.get(name, version)
        pv.metrics.setdefault(metric, []).append(value)

    def compare_versions(self, name: str) -> dict:
        result = {}
        for ver, pv in self._store.get(name, {}).items():
            if pv.metrics:
                result[ver] = {
                    k: round(sum(v)/len(v), 3)
                    for k, v in pv.metrics.items()
                }
        return result


class ABTest:
    """Simple A/B test runner for prompt versions."""

    def __init__(self, client, registry: PromptRegistry, prompt_name: str,
                 version_a: str, version_b: str, traffic_split: float = 0.5,
                 model: str = 'claude-haiku-4-5-20251001'):
        self.client         = client
        self.model          = model
        self.registry       = registry
        self.prompt_name    = prompt_name
        self.version_a      = version_a
        self.version_b      = version_b
        self.traffic_split  = traffic_split  # fraction routed to A
        self.results: list[dict] = []

    def assign_version(self, request_id: str) -> str:
        """Deterministic assignment based on request_id hash."""
        h = hash(request_id) % 100
        return self.version_a if h < self.traffic_split * 100 else self.version_b

    def run(self, request_id: str, template_vars: dict, score_fn) -> dict:
        version = self.assign_version(request_id)
        pv      = self.registry.get(self.prompt_name, version)
        prompt  = pv.template.format(**template_vars)

        # Call the LLM
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{'role': 'user', 'content': prompt}],
        )
        output = response.content[0].text

        # Score the output
        score = score_fn(output)
        self.registry.record_metric(self.prompt_name, version, 'quality_score', score)

        result = {
            'request_id': request_id,
            'version':    version,
            'output':     output[:200],
            'score':      score,
        }
        self.results.append(result)
        return result

    def summary(self) -> dict:
        return self.registry.compare_versions(self.prompt_name)


def get_session_history(session_id: str, session_store: dict):
    from langchain_core.chat_history import InMemoryChatMessageHistory
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]


async def demo_mcp_client(server_script: str, python_executable: str):
    import json
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=python_executable, args=[server_script])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_resp = await session.list_tools()
            print('Available MCP tools:')
            for t in tools_resp.tools:
                print(f'  {t.name:25s} — {t.description}')

            result = await session.call_tool('get_stock_price', {'ticker': 'JPM'})
            data = json.loads(result.content[0].text)
            print(f'\nJPM stock data: {data}')

            result2 = await session.call_tool(
                'check_enrollment',
                {'course_id': 'DS-201', 'student_completed': ['DS-101']}
            )
            enroll = json.loads(result2.content[0].text)
            print(f'DS-201 enrollment check: {enroll}')

            result3 = await session.call_tool(
                'get_portfolio_summary',
                {'tickers': ['AAPL', 'JPM', 'UNH', 'EDU']}
            )
            portfolio = json.loads(result3.content[0].text)
            print(f'\nPortfolio sector weights: {portfolio["sector_weights"]}')


async def run_mcp_agent(user_query: str, api_key: str, server_script: str,
                         python_executable: str, model: str = 'claude-haiku-4-5-20251001'):
    """Run a Claude agent that uses MCP tools."""
    import anthropic
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    client = anthropic.Anthropic(api_key=api_key)
    params = StdioServerParameters(command=python_executable, args=[server_script])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools = await session.list_tools()
            anthropic_tools = [
                {
                    'name': t.name,
                    'description': t.description,
                    'input_schema': t.inputSchema,
                }
                for t in mcp_tools.tools
            ]

            messages = [{'role': 'user', 'content': user_query}]

            for _ in range(5):  # max 5 tool calls
                response = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    tools=anthropic_tools,
                    messages=messages,
                )

                if response.stop_reason == 'end_turn':
                    answer = ' '.join(
                        b.text for b in response.content if hasattr(b, 'text')
                    )
                    print(f'\nFinal answer:\n{answer}')
                    return answer

                tool_results = []
                for block in response.content:
                    if hasattr(block, 'type') and block.type == 'tool_use':
                        print(f'  Calling MCP tool: {block.name}({block.input})')
                        res = await session.call_tool(block.name, block.input)
                        tool_results.append({
                            'type': 'tool_result',
                            'tool_use_id': block.id,
                            'content': res.content[0].text,
                        })

                messages.append({'role': 'assistant', 'content': response.content})
                messages.append({'role': 'user',      'content': tool_results})

    return 'Max iterations reached'


def finance_specialist(query: str, context: dict, client, model: str = 'claude-haiku-4-5-20251001') -> dict:
    """Finance domain specialist — analyses financial queries."""
    import json
    system = (
        'You are a senior financial analyst. '
        'Answer the query using only the context provided. '
        'Return a JSON object with keys: answer (str), confidence (high/medium/low), '
        'data_used (list of strings from context).'
    )
    prompt = (
        f'Context provided by coordinator:\n{json.dumps(context, indent=2)}\n\n'
        f'Finance query: {query}'
    )
    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=system,
        messages=[{'role': 'user', 'content': prompt}],
    )
    text = response.content[0].text
    try:
        start = text.find('{')
        end   = text.rfind('}') + 1
        return json.loads(text[start:end]) if start != -1 else {'answer': text, 'confidence': 'medium', 'data_used': []}
    except Exception:
        return {'answer': text, 'confidence': 'medium', 'data_used': []}


def education_specialist(query: str, context: dict, client, model: str = 'claude-haiku-4-5-20251001') -> dict:
    """Education domain specialist — advises on learning paths."""
    import json
    system = (
        'You are an academic advisor specialising in data science and finance curricula. '
        'Answer using only the context provided. '
        'Return a JSON object with keys: recommendation (str), prerequisites_met (bool), '
        'next_steps (list of strings).'
    )
    prompt = (
        f'Context provided by coordinator:\n{json.dumps(context, indent=2)}\n\n'
        f'Education query: {query}'
    )
    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=system,
        messages=[{'role': 'user', 'content': prompt}],
    )
    text = response.content[0].text
    try:
        start = text.find('{')
        end   = text.rfind('}') + 1
        return json.loads(text[start:end]) if start != -1 else {'recommendation': text, 'prerequisites_met': None, 'next_steps': []}
    except Exception:
        return {'recommendation': text, 'prerequisites_met': None, 'next_steps': []}


def run_coordinator(user_query: str, shared_context: dict, client, tools: list,
                     system_prompt: str, finance_specialist_fn, education_specialist_fn,
                     model: str = 'claude-haiku-4-5-20251001'):
    """Coordinator that orchestrates finance and education specialists."""
    import json
    messages = [{
        'role': 'user',
        'content': f'User context: {json.dumps(shared_context)}\n\nUser query: {user_query}',
    }]

    coord_log = []

    for step in range(5):
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == 'end_turn':
            final = ' '.join(b.text for b in response.content if hasattr(b, 'text'))
            return final, coord_log

        tool_results = []
        for block in response.content:
            if not (hasattr(block, 'type') and block.type == 'tool_use'):
                continue

            tool_name = block.name
            sub_query = block.input.get('sub_query', '')
            ctx       = block.input.get('context', {})

            if tool_name == 'call_finance_specialist':
                specialist_result = finance_specialist_fn(sub_query, ctx)
                coord_log.append({'step': step, 'specialist': 'finance', 'query': sub_query})
            elif tool_name == 'call_education_specialist':
                specialist_result = education_specialist_fn(sub_query, ctx)
                coord_log.append({'step': step, 'specialist': 'education', 'query': sub_query})
            else:
                specialist_result = {'error': f'Unknown tool {tool_name}'}

            tool_results.append({
                'type': 'tool_result',
                'tool_use_id': block.id,
                'content': json.dumps(specialist_result),
            })

        messages.append({'role': 'assistant', 'content': response.content})
        messages.append({'role': 'user',      'content': tool_results})

    return 'Max steps reached', coord_log


def score_loan_response(output: str) -> float:
    """Heuristic quality score: higher for structured JSON with all required fields."""
    score = 0.5  # baseline
    if 'decision' in output.lower():  score += 0.15
    if 'risk_score' in output.lower(): score += 0.15
    if 'key_factors' in output.lower(): score += 0.1
    if 'conditions' in output.lower(): score += 0.1
    return round(score, 2)
