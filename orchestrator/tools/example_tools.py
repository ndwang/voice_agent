"""
Example Tools

Example tool implementations for testing tool calling functionality.
"""
from datetime import datetime
from typing import Dict, Any
from orchestrator.tools.registry import tool


@tool(
    name="get_current_time",
    description="Get the current time in a specified timezone",
    schema={
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "IANA timezone name (e.g., 'America/New_York', 'Asia/Tokyo', 'Europe/London', 'UTC')"
            }
        },
        "required": ["timezone"]
    }
)
async def get_current_time(params: Dict[str, Any]) -> str:
    """
    Get the current time in a specified timezone.

    Args:
        params: Dict with "timezone" key (IANA timezone name)

    Returns:
        Current time string in the specified timezone
    """
    timezone = params.get("timezone", "UTC")

    try:
        import pytz
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz)
        return current_time.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception as e:
        return f"Error: Invalid timezone '{timezone}'. {str(e)}"


@tool(
    name="calculate",
    description="Perform a simple mathematical calculation (addition, subtraction, multiplication, division)",
    schema={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5', '(100 - 20) / 4')"
            }
        },
        "required": ["expression"]
    }
)
async def calculate(params: Dict[str, Any]) -> str:
    """
    Perform a simple mathematical calculation.

    Args:
        params: Dict with "expression" key (string expression to evaluate)

    Returns:
        Result of the calculation as a string
    """
    expression = params.get("expression", "")

    if not expression:
        return "Error: No expression provided"

    try:
        # Safely evaluate simple math expressions
        # Note: This is a simplified example. In production, use a safer parser.
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters"

        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: Failed to calculate '{expression}'. {str(e)}"


@tool(
    name="get_weather",
    description="Get current weather information for a location (mock data for testing)",
    schema={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City or location name (e.g., 'New York', 'Tokyo', 'London')"
            }
        },
        "required": ["location"]
    }
)
async def get_weather(params: Dict[str, Any]) -> str:
    """
    Get mock weather information for a location.

    Note: This is a mock tool for testing. In production, integrate with a real weather API.

    Args:
        params: Dict with "location" key (city name)

    Returns:
        Mock weather information
    """
    location = params.get("location", "Unknown")

    # Mock weather data
    import random
    temperatures = [15, 18, 22, 25, 28, 30]
    conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Windy"]

    temp = random.choice(temperatures)
    condition = random.choice(conditions)

    return f"Weather in {location}: {temp}°C, {condition}"
