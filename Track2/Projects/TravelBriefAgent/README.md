# AI Travel Guide Agent

A Streamlit app powered by a **RAG + ReAct** agent that researches your travel destination across 8 live data sources and delivers a complete, personalised travel briefing — optionally emailed to you.

---

## Features

- **Self-building RAG knowledge base** — visa rules, vaccination requirements, travel documents, and top attractions stored in Pinecone as paragraph-level chunks (200 words, 30-word overlap) with a 30-day TTL. On a cache miss or stale entry, the agent scrapes [Wikivoyage](https://en.wikivoyage.org) and [CDC Traveler's Health](https://wwwnc.cdc.gov/travel) live, embeds, and caches the result
- **Parallel tool pre-fetch** — destination info (×4 topics), travel safety, hazard news, public holidays, and language phrases run concurrently via `ThreadPoolExecutor` before the ReAct loop, cutting wall-clock time roughly in half
- **Live safety advisory** — real-time risk level from the US State Department (Level 1–4), precautions, and local emergency numbers
- **Travel alerts & news** — recent hazard alerts via NewsAPI; falls back to curated manual links when the key is absent
- **Flights & hotels** — live search via SerpAPI / Google Flights & Hotels; falls back to booking links when the key is absent
- **Public holidays** — date-aware planning via Nager.at with retry on transient failures
- **Language phrases** — essential phrases translated via MyMemory API with an in-memory cache and per-purpose phrase sets
- **Email delivery** — formatted HTML briefing sent via Resend (opt-in checkbox)
- **LLM provider fallback** — if the primary provider (Groq or Anthropic) hits a rate limit or API error, the agent silently retries with the other provider
- **Required-tool guard** — the agent cannot emit a Final Answer without having called `destination_info` and `travel_safety`; missing calls are injected as a correction
- **Context compression** — each tool result is word-budget trimmed (300 words) before appending to message history, preventing context-window overflow on multi-tool runs
- **Two LLM providers** — Groq (free, fast) or Claude (Anthropic), selectable per session in the sidebar
- **Selective tools** — unchecking "Search flights & hotels" removes those tools from the agent entirely; unchecking email removes the email tool — the LLM only knows about active tools
- **Structured briefing output** — the Final Answer always follows a fixed 7-section layout (Trip Overview → Entry Requirements → Safety & Health → When to Go → Top Attractions → Language & Practical Tips → Pre-Departure Checklist). When flights & hotels are enabled, three extra sections are appended: Flights, Hotels, and Estimated Trip Cost. Each section is displayed as a tab via `st.tabs()`; a Full Briefing tab and Markdown download are always present
- **Seasonal weather from LLM knowledge** — climate and packing advice is drawn from the model's training data (no weather API needed); the system prompt explicitly instructs this to prevent the agent from inventing a tool call
- **Session history** — last 5 briefings stored in session state, accessible from the sidebar with a per-briefing download button
- **Live rate-limit countdown** — a progress bar ticks down the 30-second cooldown second by second
- **Live agent reasoning** — every parallel pre-fetch and ReAct step streams into an expander in real time

---

## How the RAG layer works

```
User submits form (destination, dates, purpose)
  │
  ▼
Pinecone cache check  ──  stale if cached_at > 30 days
  ├─ HIT (fresh)  → return chunks instantly
  │
  └─ MISS / STALE → scrape live source
       ├─ vaccination_rules  →  CDC Traveler's Health  (Wikivoyage fallback)
       ├─ tourist_spots      →  Wikivoyage "See" section
       ├─ visa_requirements  →  Wikivoyage "Get in" section
       └─ travel_documents   →  Wikivoyage "Get in" section
            │
            ▼
       Split into 200-word chunks with 30-word overlap
            │
            ▼
       Embed (all-MiniLM-L6-v2) + upsert to Pinecone
       with destination, category, source, cached_at metadata
            │
            ▼
       Return to LLM  (tagged [Wikivoyage] or [CDC Traveler's Health])
```

No ChromeDriver or headless browser required — both sources are server-rendered HTML, scraped with `requests` + `BeautifulSoup`.

---

## Agent architecture

```
Form submit
  │
  ▼
Parallel pre-fetch  (ThreadPoolExecutor, up to 8 workers)
  ├─ destination_info × 4  (visa, vaccination, documents, tourist spots)
  ├─ travel_safety
  ├─ hazard_news
  ├─ public_holiday          (if in tools)
  └─ language_phrases        (if in tools)
       │
       ▼
  All results compressed + injected into message history
       │
       ▼
ReAct loop  (max 20 steps)
  ├─ Optional tools: flight_search, hotel_search, email
  ├─ Required-tool guard before Final Answer
  └─ LLM fallback on rate-limit / API error
       │
       ▼
Final Answer → parse into tabs → display + download
```

---

## Project Structure

```
TravelBriefAgent/
├── app.py                     # Streamlit UI — form, live progress, tabbed output,
│                              #   session history, rate-limit countdown
├── requirements.txt
├── README.md
│
├── .streamlit/
│   └── secrets.toml           # API keys — gitignored, never commit
│
├── agent/
│   ├── __init__.py            # exports run_travel_agent
│   ├── helpers.py             # LLM output parser, system prompt builder,
│   │                          #   compress_tool_result
│   └── loop.py                # parallel pre-fetch → ReAct loop → required-tool
│                              #   guard → LLM fallback
│
├── data/
│   ├── __init__.py
│   └── country_data.py        # ISO codes, IATA codes, emergency numbers,
│                              #   language map, fallback phrase sets
│
└── tools/
    ├── __init__.py            # tool registry + JSON signatures shown to LLM
    ├── base.py                # Tool base class, get_secret, sanitize_input,
    │                          #   retry decorator (exponential backoff)
    ├── destination_info.py    # RAG: Pinecone cache → chunked scrape → upsert
    ├── travel_safety.py       # US State Dept advisory + emergency numbers
    ├── hazard_news.py         # NewsAPI hazard alerts + manual link fallback
    ├── flights_hotels.py      # SerpAPI flights & hotels + booking link fallback
    ├── holidays.py            # Nager.at public holidays (retry decorator)
    ├── language.py            # MyMemory translation + in-memory phrase cache
    └── email.py               # Resend HTML email delivery
```

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd TravelBriefAgent
pip install -r requirements.txt
```

### 2. Create `.streamlit/secrets.toml`

> **Important:** No comments (`#`) in `secrets.toml` — Streamlit's TOML parser rejects them.

```toml
CHAT_GROQ_API_KEY     = "your_groq_key"
ANTHROPIC_API_KEY     = "your_anthropic_key"
NEWSAPI_API_KEY       = "your_newsapi_key"
SERPAPI_API_KEY       = "your_serpapi_key"
RESEND_API_KEY        = "your_resend_key"
RESEND_FROM_EMAIL     = "onboarding@resend.dev"
PINECONE_API_KEY      = "your_pinecone_key"
PINECONE_INDEX_NAME   = "travel-knowledge"
```

Only one LLM key is required (`CHAT_GROQ_API_KEY` or `ANTHROPIC_API_KEY`). All other keys are optional — the sidebar shows which services are active and what fallback applies.

### 3. Run

```bash
streamlit run app.py
```

The Pinecone index starts empty. The first query for a destination triggers a live scrape and caches the result. Subsequent queries are instant until the 30-day TTL expires.

---

## API Keys

| Key | Service | Free tier | Used for |
|---|---|---|---|
| `CHAT_GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | Yes | LLM (primary) |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) | Pay-as-you-go | LLM (primary or fallback) |
| `NEWSAPI_API_KEY` | [newsapi.org](https://newsapi.org) | 100 req/day | Hazard news alerts |
| `SERPAPI_API_KEY` | [serpapi.com](https://serpapi.com) | 100 searches/month | Flights & hotels |
| `RESEND_API_KEY` | [resend.com](https://resend.com) | 100 emails/day | Email delivery |
| `PINECONE_API_KEY` | [app.pinecone.io](https://app.pinecone.io) | Free tier | RAG vector store |

**No key needed:** travel safety (travel.state.gov), public holidays (Nager.at), language translation (MyMemory), destination scraping (Wikivoyage, CDC).

---

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select your repo and `app.py`
3. Under **Settings → Secrets**, paste your `secrets.toml` contents
4. Click **Deploy**
