# app.py
import os, json
from dotenv import load_dotenv
from tools import TOOL_REGISTRY
from router import rule_route
from llm_tools import llm_decide_and_call
from asr import transcribe

load_dotenv()  # loads OPENAI_API_KEY

def safe_execute(tool_name: str, raw_args: dict):
    # Enforce allowlist + argument validation
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Tool not allowed: {tool_name}")
    Schema, fn = TOOL_REGISTRY[tool_name]
    args = Schema(**raw_args)  # pydantic validation
    return fn(args)

def handle_text(text: str) -> str:
    # 1) Rules first
    routed = rule_route(text)
    if routed:
        tool, args = routed
        result = safe_execute(tool, args)
        return f"[{tool}] → {json.dumps(result, ensure_ascii=False)}"

    # 2) LLM tool-calling fallback
    mode, payload = llm_decide_and_call(text)
    if mode == "tool_call":
        tool, args = payload["name"], payload["arguments"]
        result = safe_execute(tool, args)
        return f"[{tool}] → {json.dumps(result, ensure_ascii=False)}"
    else:
        return payload["content"]

if __name__ == "__main__":
    # Demo: transcribe an audio file and route it
    audio_path = "sample.wav"  # put a short WAV/MP3 here
    text = transcribe(audio_path, engine="openai")  # or engine="local"
    print("User (transcribed):", text)
    answer = handle_text(text)
    print("Bot:", answer)
