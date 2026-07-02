from .base import get_secret as _get_secret

# ── Registered tools ──────────────────────────────────────────────────────────
from .destination_info import DestinationInfoTool
from .travel_safety import TravelSafetyTool
from .hazard_news import HazardNewsTool
from .holidays import PublicHolidayTool
from .language import LanguageTool
from .flights_hotels import FlightSearchTool, HotelSearchTool
from .email import EmailTool


# Signatures shown to the LLM — order matches recommended call sequence
TOOL_SIGNATURES = {
    "destination_info": '{"destination": "<city or country name e.g. Tokyo>", "topic": "<visa_requirements|vaccination_rules|travel_documents|tourist_spots>"}',
    "travel_safety":    '{"country_code": "<2-letter ISO e.g. JP>"}',
    "hazard_news":      '{"destination": "<city or country name>", "limit": <1-5>}',
    "flight_search":    '{"origin": "<city name e.g. Bangalore>", "destination": "<city name e.g. Tokyo>", "departure_date": "<YYYY-MM-DD>", "adults": <1-9>}',
    "hotel_search":     '{"city": "<city name e.g. Tokyo>", "check_in": "<YYYY-MM-DD>", "check_out": "<YYYY-MM-DD>", "adults": <1-9>}',
    "public_holiday":   '{"country_code": "<2-letter ISO e.g. JP>", "year": <4-digit year>}',
    "language_phrases": '{"country_code": "<2-letter ISO e.g. JP>", "purpose": "<leisure|business|medical|study|family>"}',
    "email":            '{"to": "<email>", "subject": "<subject>", "body": "<full briefing>"}',
}


def build_tools():
    return {
        "destination_info": DestinationInfoTool(),
        "travel_safety":    TravelSafetyTool(),
        "hazard_news":      HazardNewsTool(_get_secret("NEWSAPI_API_KEY")),
        "flight_search":    FlightSearchTool(_get_secret("SERPAPI_API_KEY")),
        "hotel_search":     HotelSearchTool(_get_secret("SERPAPI_API_KEY")),
        "public_holiday":   PublicHolidayTool(),
        "language_phrases": LanguageTool(),
        "email":            EmailTool(_get_secret("RESEND_API_KEY")),
    }
