"""
Flight search via Serpapi (Google Flights). Hotel tool returns curated booking
links — no free hotel search API with open registration exists.
City→IATA data lives in data.country_data.
"""
import logging
import requests
from .base import Tool, sanitize_input
from data.country_data import CITY_IATA as _CITY_IATA, SORTED_IATA_KEYS as _SORTED_KEYS

logger = logging.getLogger(__name__)

_SERPAPI_URL = "https://serpapi.com/search.json"


def _city_to_iata(city: str) -> str | None:
    city_lower = city.lower().strip()
    if city_lower in _CITY_IATA:
        return _CITY_IATA[city_lower]
    if len(city_lower) == 3 and city_lower.isalpha():
        return city_lower.upper()
    for key in _SORTED_KEYS:
        if len(key) >= 4 and city_lower.startswith(key):
            return _CITY_IATA[key]
    return None


def _fmt_duration(minutes: int) -> str:
    return f"{minutes // 60}h {minutes % 60}m"


# ── Flight Search ──────────────────────────────────────────────────────────────

class FlightSearchTool(Tool):
    name = "flight_search"
    description = (
        "Search for available flights between two cities on a given date using Google Flights via Serpapi. "
        "Returns up to 5 options sorted by price with airline, stops, duration, and USD price. "
        "Requires SERPAPI_KEY; falls back to booking links without it."
    )
    TIMEOUT = 20

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or ""
        if not self.api_key:
            logger.warning("SERPAPI_KEY not set — flight_search will return booking links only.")

    def run(self, origin: str, destination: str, departure_date: str, adults: int = 1) -> str:
        origin         = sanitize_input(origin)
        destination    = sanitize_input(destination)
        departure_date = sanitize_input(departure_date)
        try:
            adults = max(1, min(int(adults), 9))
        except (ValueError, TypeError):
            adults = 1

        origin_code = _city_to_iata(origin)
        dest_code   = _city_to_iata(destination)

        if not origin_code:
            return f"Airport not found for '{origin}'. Use a major city name like 'Mumbai' or 'London'."
        if not dest_code:
            return f"Airport not found for '{destination}'."

        if not self.api_key:
            return self._fallback(origin, origin_code, destination, dest_code, departure_date)

        try:
            resp = requests.get(
                _SERPAPI_URL,
                params={
                    "engine":        "google_flights",
                    "departure_id":  origin_code,
                    "arrival_id":    dest_code,
                    "outbound_date": departure_date,
                    "type":          "2",           # one-way
                    "adults":        adults,
                    "currency":      "USD",
                    "hl":            "en",
                    "api_key":       self.api_key,
                },
                timeout=self.TIMEOUT,
            )
        except requests.exceptions.Timeout:
            return "Flight search timed out — please try again."
        except Exception as exc:
            return f"Flight search error: {exc}"

        if resp.status_code == 401:
            return "SERPAPI_KEY is invalid. Check your key at serpapi.com/manage-api-key."
        if resp.status_code != 200:
            logger.warning("Serpapi returned HTTP %s", resp.status_code)
            return self._fallback(origin, origin_code, destination, dest_code, departure_date)

        body = resp.json()

        # Serpapi returns best_flights (top picks) and other_flights
        all_flights = body.get("best_flights", []) + body.get("other_flights", [])
        if not all_flights:
            error_msg = body.get("error", "")
            if error_msg:
                return f"Google Flights error: {error_msg}"
            return (
                f"No flights found from {origin_code} to {dest_code} on {departure_date}. "
                "Try an adjacent date or check google.com/flights directly."
            )

        all_flights.sort(key=lambda f: f.get("price", 99999))

        lines = [
            f"Flights: {origin.title()} ({origin_code}) → {destination.title()} ({dest_code})",
            f"Date: {departure_date} | Passengers: {adults}",
            f"Source: Google Flights\n",
        ]

        for i, itinerary in enumerate(all_flights[:5], 1):
            price      = itinerary.get("price", "N/A")
            segments   = itinerary.get("flights", [])
            stops      = max(len(segments) - 1, 0)
            total_min  = itinerary.get("total_duration", 0)
            stop_label = "Non-stop" if stops == 0 else f"{stops} stop(s)"

            airlines = ", ".join(
                dict.fromkeys(s.get("airline", "") for s in segments if s.get("airline"))
            )

            dep_time = arr_time = ""
            if segments:
                dep_time = segments[0].get("departure_airport", {}).get("time", "")
                arr_time = segments[-1].get("arrival_airport", {}).get("time", "")

            lines.append(
                f"{i}. {airlines} | {stop_label} | {_fmt_duration(total_min)} "
                f"| {dep_time} → {arr_time} | ${price} USD"
            )

        price_insight = body.get("price_insights", {})
        if price_insight.get("typical_price_range"):
            low, high = price_insight["typical_price_range"]
            lines.append(f"\nTypical price range for this route: ${low}–${high} USD")

        lines.append("Book at: google.com/flights · kiwi.com · skyscanner.com")
        return "\n".join(lines)

    def _fallback(self, origin, origin_code, destination, dest_code, departure_date):
        return (
            f"Flights: {origin.title()} ({origin_code}) → {destination.title()} ({dest_code}) "
            f"| {departure_date}\n"
            f"Live flight data requires SERPAPI_KEY (100 free searches/month at serpapi.com).\n\n"
            f"Search on:\n"
            f"  • google.com/flights\n"
            f"  • skyscanner.com\n"
            f"  • kayak.com\n"
            f"  • kiwi.com\n"
            f"  • makemytrip.com (India routes)\n\n"
            f"Tip: Book 6–8 weeks ahead for the best international fares."
        )


# ── Hotel Search ───────────────────────────────────────────────────────────────

class HotelSearchTool(Tool):
    name = "hotel_search"
    description = (
        "Search for hotels in a city for given check-in and check-out dates using Google Hotels via Serpapi. "
        "Returns up to 5 options with name, rating, reviews, and price per night. "
        "Requires SERPAPI_KEY; falls back to booking links without it."
    )
    TIMEOUT = 20

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or ""
        if not self.api_key:
            logger.warning("SERPAPI_KEY not set — hotel_search will return booking links only.")

    def run(self, city: str, check_in: str, check_out: str, adults: int = 1) -> str:
        city      = sanitize_input(city)
        check_in  = sanitize_input(check_in)
        check_out = sanitize_input(check_out)
        try:
            adults = max(1, min(int(adults), 9))
        except (ValueError, TypeError):
            adults = 1

        if not self.api_key:
            return self._fallback(city, check_in, check_out)

        try:
            resp = requests.get(
                _SERPAPI_URL,
                params={
                    "engine":      "google_hotels",
                    "q":           f"hotels in {city}",
                    "check_in_date":  check_in,
                    "check_out_date": check_out,
                    "adults":      adults,
                    "currency":    "USD",
                    "hl":          "en",
                    "gl":          "us",
                    "api_key":     self.api_key,
                },
                timeout=self.TIMEOUT,
            )
        except requests.exceptions.Timeout:
            return "Hotel search timed out — please try again."
        except Exception as exc:
            return f"Hotel search error: {exc}"

        if resp.status_code == 401:
            return "SERPAPI_KEY is invalid. Check your key at serpapi.com/manage-api-key."
        if resp.status_code != 200:
            logger.warning("Serpapi hotels returned HTTP %s", resp.status_code)
            return self._fallback(city, check_in, check_out)

        body = resp.json()
        hotels = body.get("properties", [])

        if not hotels:
            error_msg = body.get("error", "")
            if error_msg:
                return f"Google Hotels error: {error_msg}"
            return self._fallback(city, check_in, check_out)

        lines = [
            f"Hotels in {city.title()} | Check-in: {check_in} | Check-out: {check_out}",
            f"Source: Google Hotels\n",
        ]

        for i, hotel in enumerate(hotels[:5], 1):
            name        = hotel.get("name", "Unknown Hotel")
            rating      = hotel.get("overall_rating", "")
            reviews     = hotel.get("reviews", "")
            price       = hotel.get("rate_per_night", {}).get("lowest", "N/A")
            hotel_class = hotel.get("hotel_class", "")
            amenities   = hotel.get("amenities", [])[:3]

            rating_str  = f"{rating}/5" if rating else "No rating"
            reviews_str = f"({reviews} reviews)" if reviews else ""
            stars_str   = f" | {hotel_class}" if hotel_class else ""
            amenity_str = f" | {', '.join(amenities)}" if amenities else ""

            lines.append(
                f"{i}. {name}{stars_str}\n"
                f"   Rating: {rating_str} {reviews_str}\n"
                f"   Price: {price}/night{amenity_str}"
            )

        lines.append("\nBook at: booking.com · hotels.com · agoda.com")
        return "\n".join(lines)

    def _fallback(self, city, check_in, check_out):
        return (
            f"Hotels in {city.title()} | {check_in} → {check_out}\n"
            f"Live hotel data requires SERPAPI_KEY (free at serpapi.com).\n\n"
            f"Compare prices on:\n"
            f"  • booking.com — widest selection, free cancellation filters\n"
            f"  • agoda.com — best for Asia, frequent flash deals\n"
            f"  • hotels.com — loyalty rewards (10 nights = 1 free)\n"
            f"  • airbnb.com — apartments and long stays\n\n"
            f"Tip: Book 4–6 weeks ahead for best availability and rates."
        )
