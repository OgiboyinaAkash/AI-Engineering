import logging
import requests
from .base import Tool, sanitize_input

logger = logging.getLogger(__name__)


def _deduplicate(articles):
    """Remove articles whose titles share >60% word overlap with an already-seen title."""
    seen_word_sets = []
    unique = []
    for article in articles:
        title_words = set(article.get("title", "").lower().split())
        if not title_words:
            continue
        is_dup = any(
            len(title_words & seen) / max(len(title_words | seen), 1) > 0.6
            for seen in seen_word_sets
        )
        if not is_dup:
            unique.append(article)
            seen_word_sets.append(title_words)
    return unique


def _relevance_score(article, destination):
    """Score articles by destination mention and hazard keyword density."""
    dest_lower = destination.lower()
    text = (
        article.get("title", "") + " " + (article.get("description") or "")
    ).lower()
    score = 0
    if dest_lower in text:
        score += 2
    hazard_words = [
        "outbreak", "virus", "epidemic", "protest", "unrest", "strike",
        "disaster", "flood", "earthquake", "warning", "emergency", "attack",
        "explosion", "cyclone", "hurricane", "typhoon", "landslide", "tsunami",
    ]
    score += sum(1 for w in hazard_words if w in text)
    return score


class HazardNewsTool(Tool):
    name = "hazard_news"
    description = (
        "Search for recent safety hazards at the destination: health outbreaks, "
        "virus alerts, political unrest, protests, natural disasters, or travel warnings. "
        "Pass the destination city or country name."
    )

    def __init__(self, api_key):
        self.api_key = api_key
        if not api_key:
            logger.warning(
                "NEWSAPI_API_KEY is not set — hazard_news tool will return an error at query time. "
                "Get a free key at newsapi.org."
            )

    @staticmethod
    def _manual_links(destination: str) -> str:
        return (
            f"Live hazard news unavailable for '{destination}'. "
            "Check these sources before travel:\n"
            f"  • Google News: news.google.com (search '{destination} travel warning')\n"
            "  • ReliefWeb: reliefweb.int (disasters & humanitarian alerts)\n"
            "  • WHO Outbreaks: who.int/disease-outbreak-news\n"
            "  • GDACS Alerts: gdacs.org (earthquakes, floods, cyclones)\n"
            "  • OSAC: osac.gov (US overseas security advisories)"
        )

    def run(self, destination, limit=5):
        destination = sanitize_input(destination)
        limit = max(1, min(int(limit), 5))

        if not self.api_key:
            return self._manual_links(destination)

        hazard_query = (
            f"{destination} "
            "(outbreak OR virus OR epidemic OR protest OR unrest OR "
            "strike OR disaster OR flood OR earthquake OR warning OR emergency OR attack)"
        )
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": hazard_query,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": 10,  # fetch more, then dedup down to limit
            "apiKey": self.api_key,
        }
        try:
            response = requests.get(url, params=params, timeout=self.TIMEOUT)
        except requests.exceptions.Timeout:
            return self._manual_links(destination)
        except Exception:
            return self._manual_links(destination)

        if response.status_code != 200:
            return self._manual_links(destination)
        data = response.json()
        if data.get("status") != "ok":
            return self._manual_links(destination)

        articles = data.get("articles", [])
        if not articles:
            return f"No hazard alerts found for '{destination}'. Conditions appear normal."

        # Deduplicate then sort by relevance
        articles = _deduplicate(articles)
        articles.sort(key=lambda a: _relevance_score(a, destination), reverse=True)
        articles = articles[:limit]

        lines = [f"Safety & Hazard News for {destination}:\n"]
        for i, article in enumerate(articles, start=1):
            lines.append(
                f"{i}. {article['title']}\n"
                f"   Source: {article['source']['name']}\n"
                f"   Published: {article['publishedAt'][:10]}\n"
            )
        return "\n".join(lines)
