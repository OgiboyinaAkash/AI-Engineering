import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .helpers import parse_llm_output, build_system_prompt, compress_tool_result
from data.country_data import (
    COUNTRY_NAME_TO_ISO as _COUNTRY_NAME_TO_ISO,
    CITY_TO_COUNTRY as _CITY_TO_COUNTRY,
)

MAX_AGENT_STEPS = 20   # ReAct loop step ceiling (after pre-fetch)

logger = logging.getLogger(__name__)

# Tools that the agent MUST have called before a Final Answer is accepted.
_REQUIRED_TOOLS = frozenset({"destination_info", "travel_safety"})

# Topics pre-fetched in the parallel phase.
_PREFETCH_TOPICS = ["visa_requirements", "vaccination_rules", "travel_documents", "tourist_spots"]

# Purpose values accepted by language_phrases tool.
_PURPOSE_MAP = {
    "leisure": "leisure",
    "business": "business",
    "study abroad": "study",
    "medical travel": "medical",
    "family visit": "family",
    "medical": "medical",
    "study": "study",
    "family": "family",
}


def _resolve_country_code(val: str) -> str | None:
    """Resolve a country name or ISO code to a 2-letter ISO code. Returns None if unresolvable."""
    if not val:
        return None
    val_clean = val.strip()
    if len(val_clean) == 2 and val_clean.isalpha():
        return val_clean.upper()
    lower = val_clean.lower()
    return _COUNTRY_NAME_TO_ISO.get(lower)


def _dest_to_iso(destination: str) -> str | None:
    """Best-effort: city → country name → ISO code."""
    city_key = destination.lower().split(",")[0].strip()
    country = _CITY_TO_COUNTRY.get(city_key, city_key)
    iso = _COUNTRY_NAME_TO_ISO.get(country.lower())
    if iso:
        return iso
    return _resolve_country_code(destination)


# ── Tool execution ────────────────────────────────────────────────────────────

def _execute_tool(tools, tool_name, **kwargs):
    tool = tools.get(tool_name)
    if not tool:
        return f"Error: Tool '{tool_name}' not found. Available: {list(tools.keys())}"
    try:
        return tool.run(**kwargs)
    except TypeError as exc:
        logger.error("Wrong arguments for '%s': %s | kwargs=%s", tool_name, exc, kwargs)
        return f"Error: Wrong arguments for '{tool_name}'."
    except Exception as exc:
        logger.error("Tool '%s' raised an error: %s", tool_name, exc, exc_info=True)
        return f"Error executing '{tool_name}': {exc}"


def _is_rate_limit(exc):
    msg = str(exc).lower()
    return "rate" in msg or "429" in msg or "too many" in msg


# ── Parallel pre-fetch ────────────────────────────────────────────────────────

def _build_prefetch_tasks(tools: dict, travel_context: dict) -> list[tuple[str, dict]]:
    """Build (tool_name, kwargs) pairs for the parallel pre-fetch phase."""
    dest = (travel_context or {}).get("destination", "")
    if not dest:
        return []

    tasks = []

    if "destination_info" in tools:
        for topic in _PREFETCH_TOPICS:
            tasks.append(("destination_info", {"destination": dest, "topic": topic}))

    iso = _dest_to_iso(dest)

    if "travel_safety" in tools and iso:
        tasks.append(("travel_safety", {"country_code": iso}))

    if "hazard_news" in tools:
        tasks.append(("hazard_news", {"destination": dest}))

    if "public_holiday" in tools and iso:
        tasks.append(("public_holiday", {
            "country_code": iso,
            "year": int(travel_context.get("travel_year", 2026)),
        }))

    if "language_phrases" in tools and iso:
        raw_purpose = travel_context.get("purpose", "leisure")
        purpose = _PURPOSE_MAP.get(raw_purpose, "leisure")
        tasks.append(("language_phrases", {"country_code": iso, "purpose": purpose}))

    return tasks


def _run_prefetch(tools: dict, tasks: list[tuple[str, dict]]) -> list[tuple[str, dict, str]]:
    """Execute tasks concurrently. Returns (tool_name, kwargs, result) for each."""
    if not tasks:
        return []

    with ThreadPoolExecutor(max_workers=min(len(tasks), 8)) as pool:
        future_map = {
            pool.submit(_execute_tool, tools, name, **kwargs): (name, kwargs)
            for name, kwargs in tasks
        }
        results = []
        for future in as_completed(future_map):
            name, kwargs = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = f"Error: {exc}"
            results.append((name, kwargs, result))

    return results


# ── Main agent loop ───────────────────────────────────────────────────────────

def run_travel_agent(
    llm,
    tools,
    query,
    email_to,
    travel_context=None,
    max_steps=MAX_AGENT_STEPS,
    stream_callback=None,
    fallback_llm=None,
):
    """
    ReAct Travel Guide Agent with parallel pre-fetch, context compression,
    required-tool guard, and LLM fallback.

    stream_callback(step_type, content) for live UI updates.
    step_type: "step" | "action" | "observation" | "wait" | "error" | "final" | "security"
    """
    def notify(step_type, content):
        if stream_callback:
            stream_callback(step_type, content)

    # Compute prefetch tasks first so the system prompt knows what's pre-fetched
    prefetch_tasks = _build_prefetch_tasks(tools, travel_context or {})
    prefetched_names = {name for name, _ in prefetch_tasks}

    system_prompt = build_system_prompt(
        tools,
        travel_context=travel_context,
        prefetched=prefetched_names,
    )
    email_instruction = f"\n\nSend the briefing email to: {email_to}" if "email" in tools else ""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query + email_instruction),
    ]

    # ── Parallel pre-fetch phase ─────────────────────────────────────────────
    fired_tools: set[str] = set()
    if prefetch_tasks:
        notify("step", f"Pre-fetching {len(prefetch_tasks)} tools in parallel...")
        prefetch_results = _run_prefetch(tools, prefetch_tasks)

        for tool_name, kwargs, result in prefetch_results:
            notify("action", tool_name)
            notify("observation", f"{tool_name} → {str(result)[:200]}")
            compressed = compress_tool_result(result)
            topic = kwargs.get("topic", "")
            ai_content = (
                f"Thought: Gathering {tool_name}{' - ' + topic if topic else ''}.\n"
                f"Action: {tool_name}\n"
                f"Action Input: {json.dumps(kwargs)}"
            )
            messages.append(AIMessage(content=ai_content))
            messages.append(HumanMessage(content=f"Observation: {compressed}"))
            fired_tools.add(tool_name)

        messages.append(HumanMessage(content=(
            "Core research data has been gathered above. "
            "Review the observations and proceed with any remaining optional tools "
            "(flights, hotels, email) or write your Final Answer if you have sufficient information."
        )))

    # ── ReAct loop (handles remaining / optional tools + synthesis) ──────────
    for step in range(1, max_steps + 1):
        notify("step", f"Step {step}/{max_steps}")

        # LLM call with retry + fallback
        response = None
        for attempt in range(5):
            try:
                response = llm.invoke(messages, stop=["Observation:"])
                break
            except Exception as exc:
                if _is_rate_limit(exc) or "api" in str(exc).lower():
                    if fallback_llm and attempt == 0:
                        notify("wait", "Primary LLM error — switching to fallback provider")
                        try:
                            response = fallback_llm.invoke(messages, stop=["Observation:"])
                            break
                        except Exception:
                            pass
                    wait = 3 + attempt * 3
                    notify("wait", f"Rate limit — waiting {wait}s (attempt {attempt + 1}/5)")
                    time.sleep(wait)
                else:
                    return f"LLM error: {exc}"
        if response is None:
            return "Agent failed: repeated rate limit errors."

        llm_output = response.content.strip()
        parsed = parse_llm_output(llm_output)

        # ── Final Answer ─────────────────────────────────────────────────────
        if parsed["type"] == "final":
            # Guard: ensure required tools were called before accepting answer
            missing = _REQUIRED_TOOLS & tools.keys() - fired_tools
            if missing:
                correction = (
                    f"You must call these tools before writing the Final Answer: "
                    f"{sorted(missing)}. Please call them now."
                )
                notify("error", correction[:120])
                messages.append(AIMessage(content=llm_output))
                messages.append(HumanMessage(content=f"Observation: {correction}"))
                continue
            notify("final", parsed["content"])
            return parsed["content"]

        # ── Tool Action ──────────────────────────────────────────────────────
        elif parsed["type"] == "action":
            tool_name = parsed["tool"]
            tool_input = parsed["input"]

            if isinstance(tool_input, list):
                tool_input = tool_input[0] if tool_input and isinstance(tool_input[0], dict) else {}

            if not isinstance(tool_input, dict):
                feedback = f"Action Input must be a JSON object, got {type(tool_input).__name__}. Use {{...}} not [...] or a scalar."
                notify("error", feedback[:120])
                messages.append(AIMessage(content=llm_output))
                messages.append(HumanMessage(content=f"Observation: {feedback}"))
                continue

            if tool_name not in tools:
                feedback = f"Unknown tool '{tool_name}'. Use only: {list(tools.keys())}"
                notify("error", feedback[:120])
                messages.append(AIMessage(content=llm_output))
                messages.append(HumanMessage(content=f"Observation: {feedback}"))
                continue

            # Validate + resolve country_code fields
            country_code_invalid = False
            for field in ("country_code", "from_country", "to_country"):
                val = tool_input.get(field, "")
                if val:
                    resolved = _resolve_country_code(val)
                    if resolved:
                        tool_input[field] = resolved
                    else:
                        feedback = (
                            f"Invalid country code '{val}' for field '{field}'. "
                            "Use a 2-letter ISO country code (e.g. JP, US, IN, GB, DE)."
                        )
                        notify("error", feedback)
                        messages.append(AIMessage(content=llm_output))
                        messages.append(HumanMessage(content=f"Observation: {feedback}"))
                        country_code_invalid = True
                        break

            if country_code_invalid:
                continue

            # Clamp year field
            year_warning = ""
            if "year" in tool_input:
                try:
                    y = int(tool_input["year"])
                    clamped = max(2020, min(y, 2035))
                    if clamped != y:
                        year_warning = (
                            f"Note: year {y} is outside [2020–2035] and was adjusted to {clamped}. "
                        )
                    tool_input["year"] = clamped
                except (ValueError, TypeError):
                    year_warning = (
                        f"Note: year value '{tool_input['year']}' is invalid and was defaulted to 2026. "
                    )
                    tool_input["year"] = 2026

            if tool_name == "email":
                tool_input["to"] = email_to

            notify("action", tool_name)
            observation = _execute_tool(tools, tool_name, **tool_input)
            if year_warning:
                observation = year_warning + observation

            # Compress before adding to history to protect context window
            compressed_obs = compress_tool_result(observation)
            notify("observation", f"{tool_name} → {str(observation)[:200]}")
            fired_tools.add(tool_name)

            messages.append(AIMessage(content=llm_output))
            messages.append(HumanMessage(content=f"Observation: {compressed_obs}"))

        # ── Parse Error ──────────────────────────────────────────────────────
        else:
            feedback = (
                f"{parsed['content'][:200]} "
                "Use the exact format: Thought / Action / Action Input."
            )
            notify("error", feedback[:120])
            messages.append(AIMessage(content=llm_output))
            messages.append(HumanMessage(content=f"Observation: {feedback}"))

    return "Agent stopped: maximum steps reached without a Final Answer."
