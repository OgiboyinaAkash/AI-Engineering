import requests
from .base import Tool, sanitize_input, retry


class PublicHolidayTool(Tool):
    name = "public_holiday"
    description = "Check upcoming public holidays for a country using its 2-letter ISO code (e.g. JP, US, IN)."

    @retry(max_attempts=3, backoff=2.0, exceptions=(requests.RequestException,))
    def _fetch(self, url: str):
        return requests.get(url, timeout=self.TIMEOUT)

    @staticmethod
    def _fallback_links(country_code: str, year: int) -> str:
        return (
            f"Public holiday data unavailable for '{country_code}' ({year}) — "
            "nager.at could not be reached after retries.\n"
            "Check official sources:\n"
            f"  • Time and Date: timeanddate.com/holidays/{country_code.lower()}/{year}\n"
            f"  • Officeholidays: officeholidays.com/countries/{country_code.lower()}/{year}\n"
            "  • Your destination country's government website"
        )

    def run(self, country_code, year=2026):
        country_code = sanitize_input(country_code).upper()[:2]
        try:
            year = max(2020, min(int(year), 2035))
        except (ValueError, TypeError):
            year = 2026

        url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}"
        try:
            response = self._fetch(url)
        except requests.exceptions.Timeout:
            return self._fallback_links(country_code, year)
        except Exception:
            return self._fallback_links(country_code, year)

        if response.status_code == 404:
            return (
                f"No holiday data found for '{country_code}' — "
                "this country may not be supported by nager.at.\n"
                f"Check: timeanddate.com/holidays/{country_code.lower()}/{year}"
            )
        if response.status_code != 200:
            return self._fallback_links(country_code, year)

        try:
            holidays = response.json()
        except Exception:
            return self._fallback_links(country_code, year)

        if not holidays:
            return f"No public holidays found for {country_code} in {year}."

        lines = [f"Public Holidays in {country_code} ({year}):"]
        for h in holidays[:10]:
            date_str = h.get("date", "N/A")
            local_name = h.get("localName", "")
            name = h.get("name", "")
            lines.append(f"  {date_str}: {local_name} ({name})")
        if len(holidays) > 10:
            lines.append(f"  ... and {len(holidays) - 10} more.")
        return "\n".join(lines)
