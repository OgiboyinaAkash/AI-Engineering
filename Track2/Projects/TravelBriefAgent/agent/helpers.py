import json
import re
import logging

from tools import TOOL_SIGNATURES

logger = logging.getLogger(__name__)

_COMPRESS_BUDGET_WORDS = 300


def compress_tool_result(result: str, budget: int = _COMPRESS_BUDGET_WORDS) -> str:
    """Truncate a tool result to stay within the context budget (word-level)."""
    words = result.split()
    if len(words) <= budget:
        return result
    return " ".join(words[:budget]) + "\n[...truncated for context efficiency]"


# ── LLM output parser ─────────────────────────────────────────────────────────

def _repair_json(raw: str) -> dict | None:
    """Try progressively more forgiving JSON parsing strategies."""
    # 1. Direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 2. json-repair library (pip install json-repair)
    try:
        from json_repair import repair_json
        repaired = repair_json(raw)
        if repaired:
            return json.loads(repaired)
    except Exception:
        pass

    # 3. Manual fixes: escape unescaped newlines/carriage-returns inside strings,
    #    and replace curly/smart quotes before attempting another parse.
    try:
        fixed = raw
        fixed = fixed.replace("“", '"').replace("”", '"')
        fixed = fixed.replace("‘", "'").replace("’", "'")
        result = []
        in_string = False
        i = 0
        while i < len(fixed):
            ch = fixed[i]
            if ch == "\\" and i + 1 < len(fixed):
                result.append(ch)
                result.append(fixed[i + 1])
                i += 2
            elif ch == '"':
                in_string = not in_string
                result.append(ch)
                i += 1
            elif in_string and ch == "\n":
                result.append("\\n")
                i += 1
            elif in_string and ch == "\r":
                result.append("\\r")
                i += 1
            else:
                result.append(ch)
                i += 1
        return json.loads("".join(result))
    except Exception:
        pass

    return None


def extract_action_input(raw: str) -> dict | None:
    parsed = _repair_json(raw)
    if parsed is not None:
        return parsed

    # 4. Last-resort regex field extraction
    logger.debug("JSON repair failed; falling back to regex extraction: %s", raw[:200])
    result = {}
    str_fields = [
        "to", "subject", "city", "base_currency", "target_currency", "timezone",
        "country_code", "from_country", "to_country", "query", "topic",
        "origin_iata", "destination_iata", "departure_date", "origin", "destination",
        "check_in", "check_out", "purpose",
    ]
    for field in str_fields:
        m = re.search(rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
        if m:
            result[field] = m.group(1)
    for field in ["limit", "year", "adults"]:
        m = re.search(rf'"{field}"\s*:\s*(\d+)', raw)
        if m:
            result[field] = int(m.group(1))
    body_m = re.search(r'"body"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
    if body_m:
        result["body"] = body_m.group(1).replace("\\n", "\n").replace('\\"', '"')

    return result if result else None


def parse_llm_output(text: str) -> dict:
    """
    Position-aware parser.
    If Action appears before Final Answer → execute action first.
    """
    final_match  = re.search(r"Final Answer\s*:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    action_match = re.search(r"Action\s*:\s*(\w+)", text, re.IGNORECASE)
    input_match  = re.search(r"Action Input\s*:\s*(\{.*\})", text, re.DOTALL | re.IGNORECASE)

    final_pos  = final_match.start()  if final_match  else len(text)
    action_pos = action_match.start() if action_match else len(text)

    if action_match and input_match and action_pos < final_pos:
        raw_json = input_match.group(1).strip()
        action_input = extract_action_input(raw_json)
        if action_input is None:
            return {"type": "error", "content": "Could not parse Action Input JSON."}
        return {
            "type":  "action",
            "tool":  action_match.group(1).strip().lower(),
            "input": action_input,
        }

    if final_match:
        return {"type": "final", "content": final_match.group(1).strip()}

    return {"type": "error", "content": f"Could not parse output:\n{text[:300]}"}


# ── System prompt builder ─────────────────────────────────────────────────────

def build_system_prompt(
    tools,
    travel_context: dict | None = None,
    prefetched: set | None = None,
):
    """
    prefetched: set of tool names already run in the parallel pre-fetch phase.
    When provided, the prompt skips those tools from the recommended call order
    and tells the LLM to focus only on remaining optional tools.
    """
    prefetched = prefetched or set()

    tool_list = "\n".join(
        f"  - {name}: {tools[name].description}\n    Input: {sig}"
        for name, sig in TOOL_SIGNATURES.items()
        if name in tools
    )

    ctx_block = ""
    if travel_context:
        ctx_block = "\nTravel context extracted from the user's request:\n"
        for k, v in travel_context.items():
            if v:
                ctx_block += f"  {k}: {v}\n"
        hints = "\nUse these values when calling tools:\n"
        if "flight_search" in tools:
            hints += "  • flight_search — pass origin city, destination city, and departure_date (YYYY-MM-DD).\n"
        if "hotel_search" in tools:
            hints += "  • hotel_search  — pass destination city, check_in = departure_date, check_out = departure_date + duration_days.\n"
        ctx_block += hints

    has_email   = "email"         in tools
    has_flights = "flight_search" in tools and "hotel_search" in tools

    conclude = (
        "Then send the email and write your Final Answer.\n\n"
        "Thought: I have all the information and have sent the briefing."
        if has_email else
        "Then write your Final Answer directly — no email step needed.\n\n"
        "Thought: I have all the information."
    )

    flights_section = (
        "\n\n## ✈️ Flights\n"
        "[List flight options found — airline, stops, duration, price in USD. Highlight the cheapest.]\n\n"
        "## 🏨 Hotels\n"
        "[List hotel options found — name, stars, price per night, key amenities. Include estimated total cost for the stay.]\n\n"
        "## 💰 Estimated Trip Cost\n"
        "[Sum: cheapest flight + total hotel cost for duration + typical tourist entry fees = estimated total budget]"
        if has_flights else ""
    )

    final_answer_format = (
        "Final Answer — use EXACTLY these ## sections in this order. Every section must have real content:\n\n"
        "## ✈️ Trip Overview\n"
        "[Origin → destination, dates, duration, purpose, key highlights for this nationality/traveller]\n\n"
        "## 📋 Entry Requirements\n"
        "[Visa rules for the traveller's nationality, passport validity, required documents, vaccinations, customs]\n\n"
        "## 🛡️ Safety & Health\n"
        "[Risk level, recommended precautions, emergency numbers (police/ambulance/fire), health advisories]\n\n"
        "## 🗓️ When to Go\n"
        "[Public holidays during travel dates, seasonal weather and climate for this month, packing suggestions]\n\n"
        "## 🎌 Top Attractions\n"
        "[Top 5–7 must-see places or experiences with brief descriptions]\n\n"
        "## 🗣️ Language & Practical Tips\n"
        "[Essential local phrases, currency and tipping norms, transport options, SIM cards, cultural etiquette]"
        + flights_section +
        "\n\n## 📱 Pre-Departure Checklist\n"
        "[Documents, bookings, apps, travel insurance, health prep — formatted as a checklist]"
    )

    email_rule = (
        "\n- In email body JSON strings: escape double quotes as \\\" and newlines as \\n."
        if has_email else ""
    )

    intro = (
        "You are a Smart Travel Guide Agent. Given a travel query, use the available tools "
        "to gather relevant information, then compose a complete travel briefing"
        + (" and email it to the user." if has_email else ".")
    )

    # Build the tool call section based on what's already done vs still needed
    core_tools = {"destination_info", "travel_safety", "hazard_news", "public_holiday", "language_phrases"}
    already_done = prefetched & core_tools

    if already_done:
        done_list = ", ".join(sorted(already_done))
        prefetch_note = (
            f"\nNote: The following tools have already been called automatically and their "
            f"results appear in the conversation above: {done_list}.\n"
            "Do NOT call them again. Review their observations and proceed with any remaining tools below.\n"
        )
        step = 1
        remaining_steps = ""
        if has_flights:
            remaining_steps += f"{step}. flight_search  — available flights with prices\n"; step += 1
            remaining_steps += f"{step}. hotel_search   — available hotels with prices\n"; step += 1
        if has_email:
            remaining_steps += f"{step}. email          — send the briefing\n"

        tool_order_section = (
            prefetch_note
            + (f"\nRemaining tools to call (in order):\n{remaining_steps}" if remaining_steps else
               "\nAll data is gathered. Write your Final Answer now.\n")
        )
    else:
        step = 7
        optional_steps = ""
        if has_flights:
            optional_steps += f"{step}. flight_search     — available flights with prices\n"; step += 1
            optional_steps += f"{step}. hotel_search      — available hotels with prices\n"; step += 1
        optional_steps += f"{step}. public_holiday    — date planning\n"; step += 1
        optional_steps += f"{step}. language_phrases  — communication tips\n"; step += 1
        if has_email:
            optional_steps += f"{step}. email             — send the briefing\n"

        tool_order_section = (
            "Recommended tool call order:\n"
            "1. destination_info (visa_requirements)  — RAG lookup: official visa rules\n"
            "2. destination_info (vaccination_rules)  — RAG lookup: mandatory and recommended vaccines\n"
            "3. destination_info (travel_documents)   — RAG lookup: passport, permits, customs checklist\n"
            "4. destination_info (tourist_spots)      — RAG lookup: top attractions\n"
            "5. travel_safety     — live risk level, precautions, emergency numbers\n"
            "6. hazard_news       — recent safety alerts and hazards\n"
            + optional_steps
        )

    return f"""{intro}

Available tools:
{tool_list}
{ctx_block}
Follow this EXACT format for every reasoning step:

Thought: <reason about what you need to do next>
Action: <tool_name>
Action Input: <valid JSON matching the tool's input signature>

The system will reply with:
Observation: <tool result>

Repeat Thought → Action → Action Input until you have gathered enough information to write a complete travel briefing. {conclude}

{final_answer_format}

{tool_order_section}
Rules:
- Action Input MUST be valid JSON (double-quoted keys and string values).
- Use ONLY the exact tool names listed above.
- NEVER fabricate an Observation — wait for the system to provide it.
- NEVER call a tool that has already been called above.
- NEVER add, rename, or reorder the prescribed ## sections above.
- For seasonal weather and climate (## 🗓️ When to Go), use your training knowledge — no weather tool exists.
- Country codes: 2-letter ISO (JP, US, IN, GB, DE, FR, TH, AE).
- Dates: always use YYYY-MM-DD format for flight_search and hotel_search.
- Compute check_out by adding duration_days to the departure_date.{email_rule}
"""
