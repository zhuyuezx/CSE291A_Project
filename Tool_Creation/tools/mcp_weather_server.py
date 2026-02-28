#!/usr/bin/env python3
"""
MCP Weather Server — exposes weather tools via the Model Context Protocol.
Uses the Open-Meteo API (free, no API key required) for real weather data.

Run standalone:
    python mcp_weather_server.py

Or use as a stdio server (for MCP clients):
    The server communicates over stdin/stdout using the MCP protocol.
"""

import json
import requests
from mcp.server.fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP("weather")

# ---------------------------------------------------------------------------
# WMO weather code → human-readable condition
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


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_current_weather(location: str, unit: str = "fahrenheit") -> str:
    """Get the current weather for a given location using the Open-Meteo API.

    Args:
        location: City name, e.g. 'San Diego, CA' or 'Tokyo, Japan'
        unit: Temperature unit - 'celsius' or 'fahrenheit' (default: fahrenheit)

    Returns:
        JSON string with temperature, condition, humidity, wind speed, and more
    """
    geo = _geocode(location)
    if geo is None:
        return json.dumps({"error": f"Could not find location: {location}"})

    try:
        temp_unit = "fahrenheit" if unit == "fahrenheit" else "celsius"
        wind_unit = "mph" if unit == "fahrenheit" else "kmh"
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": geo["lat"], "longitude": geo["lon"],
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                           "weather_code,wind_speed_10m,wind_direction_10m",
                "temperature_unit": temp_unit,
                "wind_speed_unit": wind_unit,
                "timezone": "auto",
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return json.dumps({"error": f"Weather API error: {str(e)}"})

    current = data.get("current", {})
    weather_code = current.get("weather_code", -1)

    return json.dumps({
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
    })


@mcp.tool()
def get_weather_forecast(location: str, days: int = 3, unit: str = "fahrenheit") -> str:
    """Get a weather forecast for the next N days using the Open-Meteo API.

    Args:
        location: City name, e.g. 'San Diego, CA'
        days: Number of days to forecast (1-7, default: 3)
        unit: Temperature unit - 'celsius' or 'fahrenheit' (default: fahrenheit)

    Returns:
        JSON string with daily forecast including high/low temps and conditions
    """
    geo = _geocode(location)
    if geo is None:
        return json.dumps({"error": f"Could not find location: {location}"})

    days = max(1, min(days, 7))
    try:
        temp_unit = "fahrenheit" if unit == "fahrenheit" else "celsius"
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": geo["lat"], "longitude": geo["lon"],
                "daily": "temperature_2m_max,temperature_2m_min,weather_code,"
                         "precipitation_probability_max,wind_speed_10m_max",
                "temperature_unit": temp_unit,
                "wind_speed_unit": "mph" if unit == "fahrenheit" else "kmh",
                "timezone": "auto",
                "forecast_days": days,
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return json.dumps({"error": f"Forecast API error: {str(e)}"})

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    forecast = []
    for i in range(len(dates)):
        wcode = daily.get("weather_code", [0])[i] if i < len(daily.get("weather_code", [])) else 0
        forecast.append({
            "date": dates[i],
            "temp_high": daily.get("temperature_2m_max", [None])[i],
            "temp_low": daily.get("temperature_2m_min", [None])[i],
            "unit": unit,
            "condition": _WMO_CODES.get(wcode, f"Unknown ({wcode})"),
            "precipitation_chance": daily.get("precipitation_probability_max", [None])[i],
            "wind_speed_max": daily.get("wind_speed_10m_max", [None])[i],
        })

    return json.dumps({
        "location": f"{geo['name']}, {geo['country']}",
        "forecast": forecast,
    })


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
