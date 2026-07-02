import os
import re
import time
import calendar
from datetime import date
from tools.base import get_secret

_RATE_LIMIT_SECONDS = 30

import streamlit as st

st.set_page_config(
    page_title="AI Travel Guide Agent",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Typography & layout ─────────────────────────────────────── */
html, body, [class*="css"] { font-family: "Inter", "Segoe UI", sans-serif; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* ── Capability badges ───────────────────────────────────────── */
.cap-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1d4ed8;
    font-size: 13px;
    font-weight: 500;
    margin: 3px 4px;
}

/* ── Service status rows (sidebar) ───────────────────────────── */
.service-row {
    display: flex;
    align-items: center;
    padding: 5px 0;
    font-size: 14px;
    color: #374151;
}
.dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 9px;
    flex-shrink: 0;
}
.dot-on  { background: #10b981; }
.dot-off { background: #d1d5db; }
.service-note { font-size: 12px; color: #9ca3af; margin-left: 18px; }

/* ── Progress log ────────────────────────────────────────────── */
.log-tool  { font-weight: 600; color: #1d4ed8; margin-top: 6px; }
.log-obs   { color: #4b5563; font-size: 13px; padding-left: 14px; }
.log-warn  { color: #d97706; font-size: 13px; }
.log-wait  { color: #7c3aed; font-size: 13px; }

/* ── Result section ──────────────────────────────────────────── */
.result-header {
    font-size: 20px;
    font-weight: 600;
    color: #111827;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

_EMAIL_RE = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

# Sync CHAT_GROQ_API_KEY → GROQ_API_KEY once at startup so langchain-groq can find it
_groq_key = get_secret("CHAT_GROQ_API_KEY")
if _groq_key:
    os.environ["GROQ_API_KEY"] = _groq_key


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_llm(provider, model_name, silent=False):
    """Build an LLM instance. Returns None (instead of stopping) when silent=True."""
    if provider == "Groq":
        from langchain_groq import ChatGroq
        key = get_secret("CHAT_GROQ_API_KEY")
        if not key:
            if not silent:
                st.error("CHAT_GROQ_API_KEY not found in environment variables.")
                st.stop()
            return None
        return ChatGroq(model=model_name, api_key=key, temperature=0)
    else:
        from langchain_anthropic import ChatAnthropic
        key = get_secret("ANTHROPIC_API_KEY")
        if not key:
            if not silent:
                st.error("ANTHROPIC_API_KEY not found in environment variables.")
                st.stop()
            return None
        return ChatAnthropic(model=model_name, api_key=key, temperature=0)


def _check_destination(dest: str) -> bool:
    """Return True if the destination city/country is in the known lookup tables."""
    from data.country_data import CITY_TO_COUNTRY, COUNTRY_NAME_TO_ISO
    key = dest.strip().lower().split(",")[0].strip()
    return key in CITY_TO_COUNTRY or key in COUNTRY_NAME_TO_ISO


def _parse_briefing_sections(text: str) -> dict[str, str]:
    """Split a markdown briefing on ## headings. Returns ordered dict of {section: content}."""
    sections: dict[str, str] = {}
    current_key = "Overview"
    current_lines: list[str] = []

    for line in text.split("\n"):
        if line.startswith("## ") or line.startswith("# "):
            content = "\n".join(current_lines).strip()
            if content:
                sections[current_key] = content
            current_key = line.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)

    content = "\n".join(current_lines).strip()
    if content:
        sections[current_key] = content

    return sections


TOOL_LABELS = {
    "destination_info": "Destination Info",
    "travel_safety":    "Travel Safety",
    "hazard_news":      "Hazard News",
    "flight_search":    "Flight Search",
    "hotel_search":     "Hotel Search",
    "public_holiday":   "Public Holidays",
    "language_phrases": "Language Phrases",
    "email":            "Email",
}

GROQ_MODELS   = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "meta-llama/llama-4-scout-17b-16e-instruct"]
CLAUDE_MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-8"]

MONTHS = list(calendar.month_name)[1:]
_TODAY        = date.today()
CURRENT_YEAR  = _TODAY.year
CURRENT_MONTH = _TODAY.month
CURRENT_DAY   = _TODAY.day


# ── Live rate-limit countdown (runs on every rerun while active) ──────────────
_now_top = time.time()
_elapsed_top = _now_top - st.session_state.get("last_run_time", 0)

if st.session_state.get("_rate_countdown") and _elapsed_top < _RATE_LIMIT_SECONDS:
    _remaining = int(_RATE_LIMIT_SECONDS - _elapsed_top)
    _pct = _elapsed_top / _RATE_LIMIT_SECONDS
    st.progress(_pct, text=f"⏳ Cooldown: {_remaining}s remaining — you can generate a new briefing soon.")
    time.sleep(1)
    st.rerun()
elif "_rate_countdown" in st.session_state and _elapsed_top >= _RATE_LIMIT_SECONDS:
    st.session_state.pop("_rate_countdown", None)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Model**")
    provider   = st.selectbox("Provider", ["Groq", "Claude (Anthropic)"], label_visibility="collapsed")
    models     = GROQ_MODELS if provider == "Groq" else CLAUDE_MODELS
    model_name = st.selectbox("Model", models, label_visibility="collapsed")

    st.divider()
    st.markdown("**Services**")

    services = [
        ("Travel Safety Advisories", "travel.state.gov", None,              None),
        ("Public Holidays",          "Nager.at",          None,              None),
        ("Language & Phrases",       "MyMemory",          None,              None),
        ("Travel Alerts & News",     "NewsAPI",           "NEWSAPI_API_KEY", "Manual news links"),
        ("Email Delivery",           "Resend",            "RESEND_API_KEY",  None),
        ("Flight & Hotel Search",    "SerpAPI",           "SERPAPI_API_KEY", "Booking links"),
    ]

    for name, endpoint, key, fallback in services:
        active = key is None or bool(get_secret(key))
        dot_cls = "dot-on" if active else "dot-off"
        st.markdown(
            f'<div class="service-row">'
            f'<span class="dot {dot_cls}"></span>'
            f'{name} <span style="color:#9ca3af;font-size:12px">({endpoint})</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if not active and fallback:
            st.markdown(
                f'<div class="service-row" style="padding-left:18px">'
                f'<span class="dot dot-on"></span>'
                f'<span style="font-size:12px;color:#6b7280">Fallback: {fallback}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif not active:
            st.markdown(
                '<div class="service-note" style="margin-left:18px">Not configured</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    rag_connected = bool(get_secret("PINECONE_API_KEY"))
    rag_dot = "dot-on" if rag_connected else "dot-off"
    st.markdown(
        f'<div class="service-row"><span class="dot {rag_dot}"></span>'
        f'<strong>Travel Knowledge Base</strong> '
        f'<span style="color:#9ca3af;font-size:12px">(Pinecone)</span></div>',
        unsafe_allow_html=True,
    )
    if not rag_connected:
        st.markdown(
            '<div class="service-row" style="padding-left:18px">'
            '<span class="dot dot-on"></span>'
            '<span style="font-size:12px;color:#6b7280">Fallback: Wikipedia</span></div>',
            unsafe_allow_html=True,
        )
    st.caption(
        "Curated visa rules, vaccination requirements, travel documents "
        "& top attractions — retrieved via semantic search."
    )

    if rag_connected:
        from tools.destination_info import get_index_count, clear_index
        st.metric("Indexed vectors", get_index_count())
        if st.button("Clear Knowledge Base", use_container_width=True, type="secondary"):
            if clear_index():
                st.warning("Knowledge base cleared.")
            else:
                st.error("Failed to clear index.")
            st.rerun()

    # ── Recent briefings history ───────────────────────────────────────────
    history = st.session_state.get("briefing_history", [])
    if history:
        st.divider()
        st.markdown("**Recent Briefings**")
        for i, h in enumerate(history):
            label = f"{h['destination']} — {h['month']} {h['year']}"
            with st.expander(label, expanded=False):
                preview = h["result"][:600] + ("..." if len(h["result"]) > 600 else "")
                st.markdown(preview)
                st.download_button(
                    "Download (.md)",
                    data=h["result"],
                    file_name=f"briefing_{h['destination'].replace(' ', '_').lower()}.md",
                    mime="text/markdown",
                    key=f"hist_dl_{i}",
                )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## AI Travel Guide Agent")
st.markdown("Plan smarter, travel safer — AI-powered briefings covering entry rules, safety, flights, hotels and local essentials.")

st.divider()


# ── Input panel ───────────────────────────────────────────────────────────────
st.markdown(
    '<p style="font-size:12px;color:#9ca3af;margin:0 0 10px 0">* Required field</p>',
    unsafe_allow_html=True,
)

_r1a, _r1b = st.columns(2)
with _r1a:
    origin = st.text_input("From", placeholder="e.g. Bangalore, India", key="inp_origin")
with _r1b:
    destination = st.text_input("Destination *", placeholder="e.g. Tokyo, Japan", key="inp_dest")

_dy1, _dy2, _dy3 = st.columns([1, 2, 1])

with _dy1:
    travel_year = st.selectbox(
        "Year",
        [CURRENT_YEAR, CURRENT_YEAR + 1, CURRENT_YEAR + 2],
        key="sel_year",
    )

_available_months = MONTHS[CURRENT_MONTH - 1:] if travel_year == CURRENT_YEAR else MONTHS

with _dy2:
    travel_month_name = st.selectbox("Month", _available_months, key="sel_month")

_travel_month_num = MONTHS.index(travel_month_name) + 1
_days_in_month    = calendar.monthrange(travel_year, _travel_month_num)[1]
_available_days   = (
    list(range(CURRENT_DAY, _days_in_month + 1))
    if travel_year == CURRENT_YEAR and _travel_month_num == CURRENT_MONTH
    else list(range(1, _days_in_month + 1))
)

with _dy3:
    travel_day = st.selectbox("Day", _available_days, key="sel_day")

_r3a, _r3b, _r3c = st.columns([1, 1, 2])
with _r3a:
    duration_val = st.number_input("Duration", min_value=1, max_value=180, value=7, step=1, key="inp_dur")
with _r3b:
    duration_unit = st.selectbox("Unit", ["days", "weeks", "months"], key="sel_unit")
with _r3c:
    travel_purpose = st.selectbox(
        "Purpose",
        ["Leisure / Tourism", "Business", "Study Abroad", "Medical Travel", "Family Visit"],
        key="sel_purpose",
    )

notes = st.text_area(
    "Additional notes (optional)",
    placeholder="e.g. Indian passport holder, vegetarian, first time visiting Japan...",
    height=70,
    key="inp_notes",
)

_chk1, _chk2 = st.columns(2)
with _chk1:
    send_email = st.checkbox("Send briefing by email", value=False, key="chk_email")
with _chk2:
    search_flights_hotels = st.checkbox("Search flights & hotels", value=False, key="chk_flights")

if send_email:
    email_to = st.text_input(
        "Email address *",
        placeholder="your@email.com",
        key="inp_email",
    )
else:
    email_to = ""

submitted = st.button(
    "Generate Travel Briefing",
    type="primary",
    key="btn_submit",
)


# ── Validation & rate limiting ────────────────────────────────────────────────
if submitted:
    st.session_state.pop("_result_data", None)

    missing = []
    if not destination.strip():
        missing.append("Destination")
    if missing:
        st.warning(f"Please enter the following before generating: {', '.join(missing)}.")
        st.stop()

    if send_email:
        email_stripped = email_to.strip()
        if not email_stripped:
            st.error("Enter an email address or uncheck 'Send briefing by email'.")
            st.stop()
        if not _EMAIL_RE.match(email_stripped):
            st.error(f"'{email_stripped}' is not a valid email address.")
            st.stop()

    # Destination normalization warning
    if not _check_destination(destination):
        st.info(
            f"'{destination}' is not in the known destinations list — "
            "the agent will still try its best, but results may be less accurate."
        )

    now = time.time()
    elapsed = now - st.session_state.get("last_run_time", 0)
    if elapsed < _RATE_LIMIT_SECONDS:
        remaining = int(_RATE_LIMIT_SECONDS - elapsed)
        st.session_state["_rate_countdown"] = True
        st.warning(f"Please wait {remaining}s before generating another briefing.")
        time.sleep(1)
        st.rerun()
    st.session_state["last_run_time"] = now

    # Build query
    travel_month_num = MONTHS.index(travel_month_name) + 1
    if duration_unit == "weeks":
        duration_days = duration_val * 7
    elif duration_unit == "months":
        duration_days = duration_val * 30
    else:
        duration_days = duration_val

    departure_date = date(travel_year, travel_month_num, travel_day)
    purpose_short  = travel_purpose.split("/")[0].strip().lower()

    parts = []
    if origin.strip():
        parts.append(f"I am travelling from {origin.strip()} to {destination.strip()}")
    else:
        parts.append(f"I am travelling to {destination.strip()}")
    parts.append(f"in {travel_month_name} {travel_year}")
    parts.append(f"for {duration_val} {duration_unit}")
    parts.append(f"purpose: {purpose_short}")
    if notes.strip():
        parts.append(notes.strip())
    query = ". ".join(parts) + "."

    travel_context = {
        "origin":            origin.strip() or "not specified",
        "destination":       destination.strip(),
        "travel_month":      travel_month_num,
        "travel_month_name": travel_month_name,
        "travel_year":       travel_year,
        "duration_days":     duration_days,
        "duration_label":    f"{duration_val} {duration_unit}",
        "purpose":           purpose_short,
        "departure_date":    departure_date.strftime("%Y-%m-%d"),
    }

    effective_email = email_to.strip() if send_email else "noemail@placeholder.skip"

    from tools import build_tools
    from agent import run_travel_agent

    tools = build_tools()
    if not search_flights_hotels:
        tools.pop("flight_search", None)
        tools.pop("hotel_search", None)
    if not send_email:
        tools.pop("email", None)

    # Primary LLM + optional fallback from the other provider
    llm = build_llm(provider, model_name)
    fallback_llm = None
    try:
        if provider == "Groq":
            fallback_llm = build_llm("Claude (Anthropic)", CLAUDE_MODELS[0], silent=True)
        else:
            fallback_llm = build_llm("Groq", GROQ_MODELS[0], silent=True)
    except Exception:
        pass

    # ── Live progress ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown("**Agent running...**")

    progress_bar   = st.progress(0, text="Initialising...")
    steps_expander = st.expander("Live steps", expanded=True)
    steps_md       = steps_expander.empty()

    steps_log: list[str] = []
    tool_call_count = [0]

    def stream_callback(step_type: str, content: str):
        if step_type == "action":
            label = TOOL_LABELS.get(content, content.replace("_", " ").title())
            steps_log.append(f"**{label}**")
            tool_call_count[0] += 1
            pct = min(tool_call_count[0] / max(len(tools), 1) * 0.9, 0.9)
            progress_bar.progress(pct, text=f"{label}...")
        elif step_type == "observation":
            preview = content[:200].replace("\n", " ")
            steps_log.append(f"  → {preview}")
        elif step_type == "wait":
            steps_log.append(f"  Waiting: {content}")
        elif step_type == "error":
            steps_log.append("  Parse issue — retrying...")
        steps_md.markdown("\n\n".join(steps_log[-30:]))

    with st.spinner(f"Researching {destination.strip()}...  This takes 60–90 seconds."):
        result = run_travel_agent(
            llm=llm,
            tools=tools,
            query=query,
            email_to=effective_email,
            travel_context=travel_context,
            stream_callback=stream_callback,
            fallback_llm=fallback_llm,
        )

    progress_bar.progress(1.0, text="Done")

    st.session_state["_result_data"] = {
        "result":      result,
        "destination": destination.strip(),
        "month":       travel_month_name,
        "year":        travel_year,
        "send_email":  send_email,
        "email_to":    email_to.strip(),
    }

    # Save to session history (latest first, max 5)
    history = st.session_state.get("briefing_history", [])
    history = [h for h in history if h.get("destination") != destination.strip()]
    history.insert(0, {
        "destination": destination.strip(),
        "month":       travel_month_name,
        "year":        travel_year,
        "result":      result,
    })
    st.session_state["briefing_history"] = history[:5]


# ── Results ───────────────────────────────────────────────────────────────────
if "_result_data" in st.session_state:
    _rd = st.session_state["_result_data"]
    result      = _rd["result"]
    _dest_label = _rd["destination"]
    _month_lbl  = _rd["month"]
    _year_lbl   = _rd["year"]

    st.divider()
    st.markdown(
        f'<div class="result-header">Travel Briefing — {_dest_label} &nbsp;|&nbsp; {_month_lbl} {_year_lbl}</div>',
        unsafe_allow_html=True,
    )

    blocked = result.startswith("Request blocked")
    failed  = result.startswith("Agent stopped") or result.startswith("Agent failed")

    if result and not failed and not blocked:
        # ── Tabbed display ────────────────────────────────────────────────────
        sections = _parse_briefing_sections(result)
        if len(sections) > 2:
            tab_names = list(sections.keys())
            tabs = st.tabs(tab_names + ["Full Briefing"])
            for tab_name, tab in zip(tab_names, tabs[:-1]):
                with tab:
                    st.markdown(sections[tab_name])
            with tabs[-1]:
                st.markdown(result)
        else:
            st.markdown(result)

        st.download_button(
            label="Download Briefing (.md)",
            data=result,
            file_name=f"travel_briefing_{_dest_label.replace(' ', '_').lower()}_{_month_lbl}_{_year_lbl}.md",
            mime="text/markdown",
        )

        # ── Email confirmation + preview ──────────────────────────────────────
        if _rd["send_email"] and _rd["email_to"]:
            st.success(f"Briefing emailed to {_rd['email_to']}")
            with st.expander("Email preview (what was sent)", expanded=False):
                st.caption(f"To: {_rd['email_to']}")
                preview_text = result[:1500] + ("...\n\n*(truncated — download for full briefing)*" if len(result) > 1500 else "")
                st.markdown(preview_text)
    else:
        st.error(f"Agent issue: {result}")
        if not blocked:
            st.info("Try again — the LLM occasionally needs a second attempt on complex routes.")
