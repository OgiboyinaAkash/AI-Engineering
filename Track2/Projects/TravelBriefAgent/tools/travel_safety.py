import re
import requests
from .base import Tool, sanitize_input
from data.country_data import (
    PRECAUTIONS_FALLBACK as _PRECAUTIONS_FALLBACK,
    EMERGENCY_FALLBACK as _EMERGENCY_FALLBACK,
    ISO_TO_STATE_DEPT_SLUG as _SLUG_MAP,
)

_STATE_DEPT_BASE = (
    "https://travel.state.gov/content/travel/en/traveladvisories/traveladvisories"
)
_LEVEL_MAP = {
    "1": ("🟢", "Low Risk — Exercise Normal Precautions",       "low"),
    "2": ("🟡", "Moderate Risk — Exercise Increased Caution",   "moderate"),
    "3": ("🟠", "High Risk — Reconsider Travel",                "high"),
    "4": ("🔴", "Very High Risk — Do Not Travel",               "very_high"),
}


class TravelSafetyTool(Tool):
    """
    Combines SafetyAdvisoryTool and LocalEmergencyTool into a single call.
    Returns risk level + precautions (from RAG) + emergency phone numbers (from RAG).
    """

    name = "travel_safety"
    description = (
        "Get the safety advisory risk level, recommended precautions, and local emergency "
        "phone numbers (police, ambulance, fire) for a destination country. "
        "Pass the 2-letter ISO country code (e.g. JP, TH, FR, IN)."
    )

    # ── Advisory ──────────────────────────────────────────────────────────────

    def _fetch_usgov_advisory(self, country_code: str):
        """Fetch advisory level from US State Dept HTML. Returns (level_str, label) or None."""
        slug = _SLUG_MAP.get(country_code)
        if not slug:
            return None
        url = f"{_STATE_DEPT_BASE}/{slug}-travel-advisory.html"
        try:
            resp = requests.get(
                url, timeout=self.TIMEOUT,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            if resp.status_code != 200:
                return None
            # Match "Level X:" or "Level X -" or "Level X –" (colon and dash both used)
            m = re.search(
                r'Level\s+([1-4])\s*[-–:]\s*([A-Za-z][^\n<]{3,60})',
                resp.text,
            )
            if m:
                return m.group(1), m.group(2).strip()
        except Exception:
            pass
        return None

    def _advisory(self, country_code: str) -> str:
        live_section = ""
        precaution_key = "moderate"

        result = self._fetch_usgov_advisory(country_code)
        if result:
            level_str, label = result
            icon, _, precaution_key = _LEVEL_MAP.get(level_str, _LEVEL_MAP["2"])
            live_section = (
                f"Safety Advisory — {country_code} (US State Dept):\n"
                f"Risk Level: {icon} Level {level_str} — {label}\n"
                f"Source: travel.state.gov\n"
            )

        if not live_section:
            live_section = (
                f"Safety Advisory — {country_code}:\n"
                "Live risk score unavailable — check your government's official advisory:\n"
                "  US:  https://travel.state.gov/content/travel/en/traveladvisories\n"
                "  UK:  https://www.gov.uk/foreign-travel-advice\n"
                "  AU:  https://www.smartraveller.gov.au\n"
            )

        precautions = "\n".join(f"  • {p}" for p in _PRECAUTIONS_FALLBACK[precaution_key])

        return live_section + f"\nRecommended Precautions:\n{precautions}"

    # ── Emergency numbers ─────────────────────────────────────────────────────

    def _emergency(self, country_code: str) -> str:
        numbers = _EMERGENCY_FALLBACK.get(country_code)
        if not numbers:
            return (
                f"Specific emergency numbers for '{country_code}' are not in the database.\n"
                "Universal fallback: 112 works in most countries worldwide."
            )
        lines = [f"Emergency Numbers — {country_code}:"]
        for service, number in numbers.items():
            lines.append(f"  {service}: {number}")
        lines.append("  Universal (EU/international): 112")
        lines.append("Tip: Save these in your phone before departure.")
        return "\n".join(lines)

    # ── Combined ──────────────────────────────────────────────────────────────

    def run(self, country_code: str) -> str:
        code = sanitize_input(country_code).upper()[:2]
        advisory_text  = self._advisory(code)
        emergency_text = self._emergency(code)
        return advisory_text + "\n\n" + emergency_text
