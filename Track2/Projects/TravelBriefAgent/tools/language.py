import time
import requests
from .base import Tool, sanitize_input
from data.country_data import COUNTRY_LANG_MAP, PHRASE_SETS_FALLBACK as _PHRASE_SETS_FALLBACK

PURPOSE_MAP = {
    "leisure": "leisure", "tourism": "leisure",
    "business": "business",
    "medical": "medical",
    "study": "study", "education": "study",
    "family": "family",
}

# In-memory cache: (phrase, lang_code) → translated string
_TRANSLATION_CACHE: dict = {}

_TRANSLATE_DELAY = 0.35  # seconds between uncached MyMemory requests


def _translate(phrase: str, lang_code: str) -> str | None:
    cache_key = (phrase, lang_code)
    if cache_key in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[cache_key]

    time.sleep(_TRANSLATE_DELAY)
    try:
        resp = requests.get(
            "https://api.mymemory.translated.net/get",
            params={"q": phrase, "langpair": f"en|{lang_code}"},
            timeout=6,
        )
        if resp.status_code == 200:
            translated = resp.json().get("responseData", {}).get("translatedText")
            if translated:
                _TRANSLATION_CACHE[cache_key] = translated
                return translated
    except Exception:
        pass
    return None


class LanguageTool(Tool):
    name = "language_phrases"
    description = (
        "Get essential travel phrases translated to the destination country's language. "
        "Accepts a 2-letter ISO country code and an optional purpose "
        "(leisure, business, medical, study, family)."
    )

    def run(self, country_code, purpose="general"):
        country_code = sanitize_input(country_code).upper()[:2]
        purpose = sanitize_input(purpose).lower()

        lang = COUNTRY_LANG_MAP.get(country_code)
        if not lang or lang == "en":
            return (
                f"The primary language in {country_code} is English — no translation needed.\n"
                "Tip: Locals always appreciate a polite greeting in the local dialect."
            )

        purpose_key = PURPOSE_MAP.get(purpose.split("/")[0].strip(), "general")

        phrases = _PHRASE_SETS_FALLBACK[purpose_key]

        header = (
            f"Essential {purpose_key.title()} phrases for {country_code} "
            f"(language: {lang}):\n"
        )
        lines = [header]
        translated_count = 0
        for phrase in phrases:
            translated = _translate(phrase, lang)
            if translated:
                lines.append(f'  "{phrase}"  →  "{translated}"')
                translated_count += 1
            else:
                lines.append(f'  "{phrase}"  →  [see translation app]')

        if translated_count == 0:
            lines.append(
                f"\nNote: Live translation is currently unavailable (MyMemory rate limit or network issue). "
                f"Use Google Translate or DeepL for {lang} phrases before departure."
            )
        elif translated_count < len(phrases):
            lines.append(
                f"\nNote: {len(phrases) - translated_count} phrase(s) could not be translated "
                "(MyMemory rate limit may apply — try again in a moment)."
            )
        lines.append("Tip: Download an offline translation app for this language before you fly.")
        return "\n".join(lines)
