import os
import time as _time
import logging
import requests
from .base import Tool, sanitize_input
from data.country_data import CITY_TO_COUNTRY as _CITY_TO_COUNTRY

_CACHE_TTL_SECONDS = 30 * 24 * 60 * 60  # 30 days

logger = logging.getLogger(__name__)

VALID_TOPICS = frozenset({
    "visa_requirements",
    "vaccination_rules",
    "travel_documents",
    "tourist_spots",
})


def _primary_city(query: str) -> str:
    return query.lower().split(",")[0].strip()


# ── Module-level Pinecone + embedding singletons ──────────────────────────────

_pinecone_index = None
_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    try:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Embedding model loaded — RAG enabled.")
    except ImportError:
        logger.warning(
            "sentence-transformers not installed — RAG disabled. "
            "Run: pip install sentence-transformers"
        )
    except Exception as exc:
        logger.warning("Failed to load embedding model: %s — RAG disabled.", exc)
    return _embed_model


def _get_index():
    global _pinecone_index
    if _pinecone_index is not None:
        return _pinecone_index
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        logger.warning(
            "PINECONE_API_KEY is not set — RAG disabled. "
            "Add it to .env or Streamlit secrets."
        )
        return None
    index_name = os.getenv("PINECONE_INDEX_NAME", "travel-knowledge")
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=api_key)
        _pinecone_index = pc.Index(index_name)
    except ImportError:
        logger.warning(
            "pinecone package not installed — RAG disabled. "
            "Run: pip install pinecone"
        )
    except Exception as exc:
        logger.warning(
            "Failed to connect to Pinecone index '%s': %s — RAG disabled.",
            index_name, exc,
        )
    return _pinecone_index


def get_index_count() -> int:
    index = _get_index()
    if not index:
        return 0
    try:
        return index.describe_index_stats().get("total_vector_count", 0)
    except Exception:
        return 0


def clear_index() -> bool:
    global _pinecone_index
    index = _get_index()
    if not index:
        return False
    try:
        index.delete(delete_all=True)
        _pinecone_index = None
        return True
    except Exception:
        return False


def _embed(text: str) -> list | None:
    model = _get_embed_model()
    if not model:
        return None
    return model.encode(text).tolist()


# ── Scrapers ──────────────────────────────────────────────────────────────────

_SCRAPE_HEADERS = {"User-Agent": "TravelBriefAgent/1.0 (educational travel assistant)"}
_MAX_CHARS = 2000

# Wikivoyage section names per topic
_WIKIVOYAGE_SECTIONS = {
    "visa_requirements": "Get in",
    "travel_documents":  "Get in",
    "tourist_spots":     "See",
    "vaccination_rules": "Stay healthy",
}


def _fetch_wikivoyage(destination: str, topic: str) -> str | None:
    """
    Fetches the relevant section from Wikivoyage using the MediaWiki API.
    API-based approach is reliable across all page structures — no DOM guessing.
    Wikivoyage is CC-BY-SA licensed and dedicated to travel content.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("beautifulsoup4 not installed — Wikivoyage scraping disabled. Run: pip install beautifulsoup4")
        return None

    import re as _re
    target_section = _WIKIVOYAGE_SECTIONS.get(topic, "Get in")
    slug = destination.strip().title().replace(" ", "_")
    api = "https://en.wikivoyage.org/w/api.php"

    # Step 1: get the sections list to find the index of the target section
    try:
        resp = requests.get(api, params={
            "action": "parse", "page": slug,
            "prop": "sections", "format": "json",
        }, headers=_SCRAPE_HEADERS, timeout=10)
        data = resp.json()
        if "error" in data:
            logger.warning("Wikivoyage: page '%s' not found", slug)
            return None
        sections = data["parse"]["sections"]
    except Exception as exc:
        logger.warning("Wikivoyage sections fetch failed for '%s': %s", slug, exc)
        return None

    section_index = None
    for s in sections:
        if target_section.lower() in s.get("line", "").lower():
            section_index = s["index"]
            break

    if section_index is None:
        logger.warning("Wikivoyage: section '%s' not found in '%s'", target_section, slug)
        return None

    # Step 2: fetch that section's rendered HTML
    try:
        resp = requests.get(api, params={
            "action": "parse", "page": slug,
            "prop": "text", "section": section_index, "format": "json",
        }, headers=_SCRAPE_HEADERS, timeout=10)
        html = resp.json()["parse"]["text"]["*"]
    except Exception as exc:
        logger.warning("Wikivoyage section fetch failed for '%s' §%s: %s", slug, section_index, exc)
        return None

    # Step 3: strip HTML, clean whitespace
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all(class_="mw-editsection"):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        text = _re.sub(r"\n{3,}", "\n\n", text)
        if len(text) < 100:
            logger.warning("Wikivoyage: section '%s' too short for '%s' (%d chars)", target_section, slug, len(text))
            return None
        logger.info("Wikivoyage: fetched '%s' for %s/%s (%d chars)", target_section, destination, topic, len(text))
        return text[:_MAX_CHARS]
    except Exception as exc:
        logger.warning("Wikivoyage parse error for '%s': %s", slug, exc)
        return None


def _fetch_cdc_vaccination(destination: str) -> str | None:
    """
    Scrapes vaccination recommendations from CDC Traveler's Health.
    URL pattern: wwwnc.cdc.gov/travel/destinations/traveler/none/{country-slug}
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return None

    slug = destination.strip().lower().replace(" ", "-")
    url = f"https://wwwnc.cdc.gov/travel/destinations/traveler/none/{slug}"

    try:
        resp = requests.get(url, timeout=10, headers=_SCRAPE_HEADERS)
        if resp.status_code != 200:
            logger.warning("CDC: HTTP %s for '%s'", resp.status_code, slug)
            return None
    except Exception as exc:
        logger.warning("CDC request failed for '%s': %s", slug, exc)
        return None

    try:
        soup = BeautifulSoup(resp.text, "html.parser")

        # Try known CDC section IDs first
        section = soup.find(id="vaccines-and-medicines") or soup.find(id="vaccines")

        if not section:
            # Fall back to finding by heading text
            for heading in soup.find_all(["h2", "h3"]):
                if "vaccine" in heading.get_text().lower():
                    chunks = []
                    for sibling in heading.find_next_siblings():
                        if sibling.name in ("h2", "h3"):
                            break
                        if sibling.name in ("p", "ul", "ol"):
                            chunks.append(sibling.get_text(separator=" ", strip=True))
                    text = "\n".join(filter(None, chunks))
                    if len(text) >= 100:
                        logger.info("CDC: fetched vaccination data for %s (%d chars)", destination, len(text))
                        return text[:_MAX_CHARS]
            logger.warning("CDC: vaccines section not found for '%s'", slug)
            return None

        text = section.get_text(separator="\n", strip=True)
        if len(text) < 100:
            return None
        logger.info("CDC: fetched vaccination data for %s (%d chars)", destination, len(text))
        return text[:_MAX_CHARS]
    except Exception as exc:
        logger.warning("CDC parse failed for '%s': %s", slug, exc)
        return None


def _scrape(destination: str, topic: str) -> tuple[str | None, str]:
    """
    Dispatch to the best scraper for the topic.
    Returns (text, source_label) or (None, '').
    CDC is tried first for vaccination — it's the authoritative source.
    Wikivoyage covers all four topics as primary or fallback.
    """
    if topic == "vaccination_rules":
        text = _fetch_cdc_vaccination(destination)
        if text:
            return text, "CDC Traveler's Health"

    text = _fetch_wikivoyage(destination, topic)
    if text:
        return text, "Wikivoyage"

    return None, ""


# ── Pinecone upsert ───────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = 200, overlap: int = 30) -> list[str]:
    """Split text into overlapping word-level chunks for finer-grained RAG retrieval."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        if i + chunk_size >= len(words):
            break
        i += chunk_size - overlap
    return chunks


def _upsert_to_index(text: str, destination: str, category: str, source: str = "scraped") -> bool:
    """
    Chunk text, embed each chunk, and upsert into Pinecone with a cached_at timestamp.
    Silently no-ops when the index or embedding model is unavailable.
    """
    index = _get_index()
    if not index or not _get_embed_model():
        return False

    chunks = _chunk_text(text)
    cached_at = int(_time.time())
    base_id = f"{source.split()[0].lower()}_{destination}_{category}"

    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = _embed(chunk)
        if not embedding:
            continue
        doc_id = f"{base_id}_c{i}" if len(chunks) > 1 else base_id
        vectors.append({
            "id": doc_id,
            "values": embedding,
            "metadata": {
                "destination": destination,
                "category": category,
                "text": chunk,
                "source": source,
                "cached_at": cached_at,
            },
        })

    if not vectors:
        return False

    try:
        index.upsert(vectors=vectors)
        logger.info(
            "Cached %d chunk(s) to Pinecone: %s/%s (source=%s)",
            len(vectors), destination, category, source,
        )
        return True
    except Exception as exc:
        logger.warning("Failed to upsert '%s' to Pinecone: %s", base_id, exc)
        return False


# ── Main tool ─────────────────────────────────────────────────────────────────

class DestinationInfoTool(Tool):
    """
    RAG-backed lookup for travel information. On a Pinecone miss, scrapes
    Wikivoyage (tourist spots, travel docs, visa basics) or CDC Traveler's
    Health (vaccination rules), then caches the result for future queries.
    """

    name = "destination_info"
    description = (
        "Look up authoritative travel information for a destination. "
        "Uses a cached knowledge base (Pinecone) first; falls back to live scraping "
        "from Wikivoyage or CDC when the destination isn't cached yet. "
        "topic must be one of: visa_requirements, vaccination_rules, travel_documents, tourist_spots."
    )

    def _rag_query(self, destination: str, topic: str) -> str | None:
        index = _get_index()
        if not index:
            return None

        city_key = _primary_city(destination)
        country_key = _CITY_TO_COUNTRY.get(city_key, city_key)
        query_text = f"{destination} {topic.replace('_', ' ')}"
        embedding = _embed(query_text)
        if not embedding:
            return None

        now = int(_time.time())

        def _query(filter_dict, top_k=3):
            try:
                results = index.query(
                    vector=embedding,
                    top_k=top_k,
                    filter=filter_dict,
                    include_metadata=True,
                )
                fresh = []
                for m in results.matches:
                    if not (m.metadata and m.metadata.get("text")):
                        continue
                    cached_at = m.metadata.get("cached_at", 0)
                    if now - cached_at > _CACHE_TTL_SECONDS:
                        logger.info(
                            "Stale cache entry for %s/%s (%d days old) — will re-scrape",
                            destination, topic, (now - cached_at) // 86400,
                        )
                        continue
                    fresh.append(m.metadata["text"])
                return fresh
            except Exception:
                return []

        # Always filter by category — destination-only or unfiltered fallbacks
        # cause cross-topic contamination (e.g. CDC vaccination doc appearing
        # in tourist_spots results).
        # tourist_spots are indexed at city level; policy topics at country level.
        if topic == "tourist_spots":
            docs = _query({"destination": {"$eq": city_key}, "category": {"$eq": topic}})
            if not docs and city_key != country_key:
                docs = _query({"destination": {"$eq": country_key}, "category": {"$eq": topic}})
        else:
            docs = _query({"destination": {"$eq": country_key}, "category": {"$eq": topic}})
            if not docs and city_key != country_key:
                docs = _query({"destination": {"$eq": city_key}, "category": {"$eq": topic}})

        if not docs:
            return None

        header = (
            f"[Knowledge Base] {destination.title()} — "
            f"{topic.replace('_', ' ').title()}:\n\n"
        )
        return header + "\n\n---\n\n".join(docs)

    def run(self, destination: str = None, topic: str = "visa_requirements", query: str = None) -> str:
        query = sanitize_input(destination or query or "")
        topic = sanitize_input(topic).lower().replace(" ", "_")
        if not query:
            return "Error: 'destination' parameter is required."

        if topic not in VALID_TOPICS:
            return (
                f"Invalid topic '{topic}'. Valid topics: "
                + ", ".join(sorted(VALID_TOPICS))
                + ". Please re-query with a valid topic."
            )

        # 1. Try Pinecone cache
        rag_result = self._rag_query(query, topic)
        if rag_result:
            return rag_result

        # 2. Scrape live data → cache in Pinecone → return
        dest_key = _primary_city(query)
        country_key = _CITY_TO_COUNTRY.get(dest_key, dest_key)
        # Policy topics (visa, vaccination, documents) live on the country page;
        # tourist_spots are city-level.
        scrape_key = dest_key if topic == "tourist_spots" else country_key
        scraped_text, source_label = _scrape(scrape_key, topic)
        if scraped_text:
            saved = _upsert_to_index(scraped_text, dest_key, topic, source=source_label)
            cache_note = " · saved to cache for future lookups" if saved else ""
            header = (
                f"[{source_label}{cache_note}] {query.title()} — "
                f"{topic.replace('_', ' ').title()}:\n\n"
            )
            return header + scraped_text

        # 3. Nothing found
        return (
            f"No data found for '{query}' ({topic}). Verify directly with:\n"
            "- Destination country's embassy or official immigration website\n"
            "- IATA Travel Centre: iatatravelcentre.com\n"
            "- US State Dept: travel.state.gov | UK FCDO: gov.uk/foreign-travel-advice\n"
            "- CDC Traveler's Health: wwwnc.cdc.gov/travel\n"
            "- WHO travel health: who.int/travel-advice"
        )
