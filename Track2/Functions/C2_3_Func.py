from abc import ABC, abstractmethod

import requests


class Tool(ABC):
    name = ""
    description = ""

    @abstractmethod
    def run(self, **kwargs):
        raise NotImplementedError


class WeatherTool(Tool):
    name = "weather"
    description = (
        "Get current weather information for a city."
    )

    def __init__(self, api_key):
        self.api_key = api_key

    def run(self, city):
        if not self.api_key:
            return {
                "status": "success",
                "source": "local_mock",
                "city": city,
                "temperature_c": 24,
                "condition": "clear sky",
                "humidity": 48,
            }

        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": self.api_key,
            "units": "metric"
        }
        try:
            response = requests.get(url, params=params, timeout=5)
            if response.status_code != 200:
                return {
                    "status": "error",
                    "error_type": "unexpected_output",
                    "message": response.json().get("message", "weather lookup failed"),
                }
            data = response.json()
            return {
                "status": "success",
                "source": "api",
                "city": city,
                "temperature_c": data["main"]["temp"],
                "condition": data["weather"][0]["description"],
                "humidity": data["main"]["humidity"],
            }
        except Exception as exc:
            return {
                "status": "error",
                "error_type": "timeout",
                "message": str(exc),
            }


class ExchangeRateTool(Tool):
    name = "exchange_rate"
    description = (
        "Get exchange rate between two currencies."
    )
    def __init__(self, api_key):
        self.api_key = api_key
    def run(self, base_currency, target_currency):
        url = (
            f"https://v6.exchangerate-api.com/v6/"
            f"{self.api_key}/latest/{base_currency.upper()}"
        )
        response = requests.get(url)
        if response.status_code != 200:
            return "Failed to fetch exchange rates."
        data = response.json()
        if data["result"] != "success":
            return data.get("error-type", "Unknown error")
        rates = data["conversion_rates"]
        if target_currency.upper() not in rates:
            return f"Currency '{target_currency}' not found."
        rate = rates[target_currency.upper()]
        return (
            f"Exchange Rate:\n"
            f"1 {base_currency.upper()} = "
            f"{rate} {target_currency.upper()}"
        )


class NewsTool(Tool):
    name = "news"
    description = (
        "Get latest news headlines for a topic."
    )
    def __init__(self, api_key):
        self.api_key = api_key
    def run(self, topic="technology", limit=5):
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": topic,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": limit,
            "apiKey": self.api_key
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return "Failed to fetch news."
        data = response.json()
        if data["status"] != "ok":
            return data.get("message", "Unknown error")
        articles = data["articles"]
        if not articles:
            return f"No news found for '{topic}'."
        result = [f"Latest news about {topic}:\n"]
        for i, article in enumerate(articles, start=1):
            result.append(
                f"{i}. {article['title']}\n"
                f"   Source: {article['source']['name']}\n"
                f"   URL: {article['url']}\n"
            )
        return "\n".join(result)


class TimezoneTool(Tool):
    name = "timezone"
    description = (
        "Get timezone information from a timezone name."
    )
    def __init__(self, api_key):
        self.api_key = api_key
    def run(self, timezone):
        url = "https://api.api-ninjas.com/v1/timezone"
        headers = {
            "X-Api-Key": self.api_key
        }
        params = {
            "timezone": timezone
        }
        response = requests.get(
            url,
            headers=headers,
            params=params
        )
        if response.status_code != 200:
            return f"Error: {response.text}"
        data = response.json()
        return (
            f"Timezone: {data['timezone']}\n"
            f"Local Time: {data['local_time']}\n"
            f"UTC Offset: {data['utc_offset']} seconds"
        )


class EmailTool(Tool):
    name = "email"
    description = "Send an email using Resend."
    def __init__(self, api_key):
        self.api_key = api_key
    def run(self, to, subject, body):
        url = "https://api.resend.com/emails"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "from": "onboarding@resend.dev",
            "to": [to],
            "subject": subject,
            "html": f"<p>{body}</p>"
        }
        response = requests.post(
            url,
            headers=headers,
            json=payload
        )
        if response.status_code not in [200, 201]:
            return f"Error: {response.text}"
        data = response.json()
        return (
            "Email sent successfully!\n"
            f"Email ID: {data.get('id')}"
        )


def execute_tool(tool_name, tools, **kwargs):
    tool = tools.get(tool_name)
    if not tool:
        return "Tool not found"
    return tool.run(**kwargs)
