# tools.py
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any
import datetime as dt

# --- 3.1 Tool argument schemas (validated) ---
class WeatherArgs(BaseModel):
    city: str = Field(..., description="City name, e.g., 'Bangalore'")
    units: Literal["metric", "imperial"] = "metric"

class CreateTicketArgs(BaseModel):
    title: str
    priority: Literal["low", "medium", "high"] = "medium"
    description: Optional[str] = None

# --- 3.2 Tool implementations ---
def get_weather(args: WeatherArgs) -> Dict[str, Any]:
    # TODO: call your weather API here; demo returns a stub.
    return {
        "city": args.city,
        "units": args.units,
        "summary": "Partly cloudy",
        "temp": 28.4,
        "retrieved_at": dt.datetime.now().isoformat(timespec="seconds"),
    }

def create_ticket(args: CreateTicketArgs) -> Dict[str, Any]:
    # TODO: write into your DB / Jira / ServiceNow, etc.
    return {
        "id": "TKT-"+dt.datetime.now().strftime("%Y%m%d%H%M%S"),
        "title": args.title,
        "priority": args.priority,
        "status": "open"
    }

# --- 3.3 Tool registry (name -> (schema, fn)) ---
TOOL_REGISTRY = {
    "get_weather": (WeatherArgs, get_weather),
    "create_ticket": (CreateTicketArgs, create_ticket),
}

# --- 3.4 JSON schemas for the LLM tool-calling API ---
OPENAI_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": WeatherArgs.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "Create a support ticket with a title, priority and optional description.",
            "parameters": CreateTicketArgs.model_json_schema()
        }
    }
]
