# llm_tools.py
import json
from typing import Any, Dict, Tuple
from openai import OpenAI
from tools import OPENAI_TOOL_SCHEMAS

client = OpenAI()

SYSTEM_PROMPT = (
    "You are a voice assistant that decides when to call functions. "
    "Only call a function when it helps fulfill the user's request. "
    "If asking for weather, use get_weather; for creating tickets, use create_ticket. "
    "Otherwise, answer briefly."
)

def llm_decide_and_call(user_text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Returns one of:
      ("tool_call", {"name": <toolname>, "arguments": <dict>})
      ("text", {"content": <assistant_reply>})
    """
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        tools=OPENAI_TOOL_SCHEMAS,
        tool_choice="auto"
    )

    # The Responses API returns a structured object; we normalize:
    out = resp.output if hasattr(resp, "output") else resp  # SDKs differ slightly
    # Walk through tool calls if present:
    for item in getattr(out, "content", []):
        if item.type == "tool_use":
            return ("tool_call", {"name": item.name, "arguments": item.input})

    # No tool chosen → extract text
    text_parts = []
    for item in getattr(out, "content", []):
        if item.type == "output_text":
            text_parts.append(item.text)
    return ("text", {"content": " ".join(text_parts).strip() or "(no response)"})
