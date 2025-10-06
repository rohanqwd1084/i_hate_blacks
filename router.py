# router.py
import re
from typing import Optional, Tuple, Dict, Any
from tools import TOOL_REGISTRY

# Simple keyword/regex rules (customize!)
RULES = [
    {
        "pattern": r"\bweather in (?P<city>[A-Za-z\s]+)\b",
        "tool": "get_weather",
        "argmap": lambda m: {"city": m.group("city").strip(), "units": "metric"},
    },
    {
        "pattern": r"\bcreate (?:a )?ticket\b.*?\btitle[:\- ](?P<title>[^;|]+)(?:;|$)",
        "tool": "create_ticket",
        "argmap": lambda m: {"title": m.group("title").strip()},
    },
]

def rule_route(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    for r in RULES:
        m = re.search(r["pattern"], text, flags=re.IGNORECASE)
        if m:
            return r["tool"], r["argmap"](m)
    return None
