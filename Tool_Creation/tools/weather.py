"""
Weather tool for LLM function calling.

Uses the Open-Meteo API (free, no API key required) for real weather data.
- Geocoding: https://geocoding-api.open-meteo.com
- Weather:   https://api.open-meteo.com
"""

import json
import requests

# ---------------------------------------------------------------------------
# Tool schema (Ollama / OpenAI-compatible format)
# ---------------------------------------------------------------------------

WEATHER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather for a given location using the Open-Meteo API. Returns temperature, condition, humidity, wind speed, and more.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. 'San Diego, CA' or 'Tokyo, Japan'",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit. Defaults to fahrenheit.",
                },
            },
            "required": ["location"],
        },
    },
}

# ---------------------------------------------------------------------------
# WMO weather code → human-readable condition
# https://open-meteo.com/en/docs#weathervariables
# ---------------------------------------------------------------------------
_WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snowfall", 73: "Moderate snowfall", 75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _geocode(location: str) -> dict | None:
    """Resolve a city name to lat/lon via Open-Meteo Geocoding API."""
    r = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location, "count": 1, "language": "en", "format": "json"},
        timeout=10,
    )
    r.raise_for_status()
    results = r.json().get("results")
    if not results:
        return None
    hit = results[0]
    return {
        "name": hit.get("name"),
        "country": hit.get("country", ""),
        "lat": hit["latitude"],
        "lon": hit["longitude"],
    }


def _fetch_current_weather(lat: float, lon: float, unit: str) -> dict:
    """Fetch current weather from Open-Meteo."""
    temp_unit = "fahrenheit" if unit == "fahrenheit" else "celsius"
    wind_unit = "mph" if unit == "fahrenheit" else "kmh"
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                       "weather_code,wind_speed_10m,wind_direction_10m",
            "temperature_unit": temp_unit,
            "wind_speed_unit": wind_unit,
            "timezone": "auto",
        },
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------

def get_current_weather(location: str, unit: str = "fahrenheit") -> str:
    """
    Return current weather as a JSON string using the Open-Meteo API.
    No API key required.
    """
    geo = _geocode(location)
    if geo is None:
        return json.dumps({"error": f"Could not find location: {location}"})

    try:
        data = _fetch_current_weather(geo["lat"], geo["lon"], unit)
    except Exception as e:
        return json.dumps({"error": f"Weather API error: {str(e)}"})

    current = data.get("current", {})
    weather_code = current.get("weather_code", -1)

    result = {
        "location": f"{geo['name']}, {geo['country']}",
        "coordinates": {"lat": geo["lat"], "lon": geo["lon"]},
        "temperature": current.get("temperature_2m"),
        "feels_like": current.get("apparent_temperature"),
        "unit": unit,
        "condition": _WMO_CODES.get(weather_code, f"Unknown ({weather_code})"),
        "humidity": current.get("relative_humidity_2m"),
        "wind_speed": current.get("wind_speed_10m"),
        "wind_direction": current.get("wind_direction_10m"),
        "timezone": data.get("timezone", ""),
    }
    return json.dumps(result)


# Quick self-test
if __name__ == "__main__":
    print("=== San Diego ===")
    print(get_current_weather("San Diego, CA"))
    print("\n=== Tokyo (celsius) ===")
    print(get_current_weather("Tokyo", unit="celsius"))
    print("\n=== Paris ===")
    print(get_current_weather("Paris", unit="celsius"))
