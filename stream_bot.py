import os, sys, time, wave, queue, json, datetime as dt, threading, tempfile, signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

# ========== ENV ==========
from dotenv import load_dotenv
load_dotenv()

# ========== DEPENDENCIES ==========
import numpy as np
import sounddevice as sd
import webrtcvad
from pydantic import BaseModel, Field
from typing import Literal as Lit

# Optional: OpenAI client (STT + LLM tool-calling)
from openai import OpenAI
openai_client = OpenAI()

# =========================================
# 1) TOOLING: define your callable functions
# =========================================

class WeatherArgs(BaseModel):
    city: str = Field(..., description="City name, e.g., 'Bangalore'")
    units: Lit["metric", "imperial"] = "metric"

class CreateTicketArgs(BaseModel):
    title: str
    priority: Lit["low", "medium", "high"] = "medium"
    description: Optional[str] = None

def get_weather(args: WeatherArgs) -> Dict[str, Any]:
    # TODO: call a real API; demo returns stub
    return {
        "city": args.city,
        "units": args.units,
        "summary": "Partly cloudy",
        "temp": 28.4,
        "retrieved_at": dt.datetime.now().isoformat(timespec="seconds"),
    }

def create_ticket(args: CreateTicketArgs) -> Dict[str, Any]:
    # TODO: write into DB/Jira/etc.
    return {
        "id": "TKT-" + dt.datetime.now().strftime("%Y%m%d%H%M%S"),
        "title": args.title,
        "priority": args.priority,
        "status": "open"
    }

TOOL_REGISTRY: Dict[str, Tuple[Any, Any]] = {
    "get_weather": (WeatherArgs, get_weather),
    "create_ticket": (CreateTicketArgs, create_ticket),
}

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

# ===================================
# 2) RULES (deterministic, regex-based)
# ===================================
import re

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

# =================================
# 3) LLM tool-calling (OpenAI)
# =================================

SYSTEM_PROMPT = (
    "You are a voice assistant that decides when to call functions. "
    "Only call a function when it helps fulfill the user's request. "
    "If asking for weather, use get_weather; for creating tickets, use create_ticket. "
    "Ask a brief clarifying question only if absolutely necessary. Otherwise answer briefly."
)

def llm_decide_and_call(user_text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Returns:
      ("tool_call", {"name": <toolname>, "arguments": <dict>})
      ("text", {"content": <assistant_reply>})
    """
    resp = openai_client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        tools=OPENAI_TOOL_SCHEMAS,
        tool_choice="auto",
    )

    # Normalize to tool use vs output text
    content_items = getattr(resp, "output", None)
    if content_items is None:
        # Some SDKs expose resp.output directly as a list-like; fallback to resp if needed
        content_items = resp

    # Try to find tool_use items
    for item in getattr(content_items, "content", []):
        if getattr(item, "type", None) == "tool_use":
            return ("tool_call", {"name": item.name, "arguments": item.input})

    # Otherwise collect text
    texts = []
    for item in getattr(content_items, "content", []):
        if getattr(item, "type", None) == "output_text":
            texts.append(item.text)
    return ("text", {"content": " ".join(texts).strip() or "(no response)"})

def safe_execute(tool_name: str, raw_args: dict):
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Tool not allowed: {tool_name}")
    Schema, fn = TOOL_REGISTRY[tool_name]
    args = Schema(**raw_args)  # validate
    return fn(args)

def handle_text(text: str) -> str:
    # 1) Rules first
    routed = rule_route(text)
    if routed:
        tool, args = routed
        result = safe_execute(tool, args)
        return f"[{tool}] → {json.dumps(result, ensure_ascii=False)}"

    # 2) LLM fallback
    mode, payload = llm_decide_and_call(text)
    if mode == "tool_call":
        tool, args = payload["name"], payload["arguments"]
        result = safe_execute(tool, args)
        return f"[{tool}] → {json.dumps(result, ensure_ascii=False)}"
    else:
        return payload["content"]

# =================================
# 4) ASR (OpenAI or local faster-whisper)
# =================================

def transcribe_openai(filepath: str, model: str = "gpt-4o-mini-transcribe") -> str:
    """
    Uses OpenAI STT; model name may vary by account (e.g., 'gpt-4o-mini-transcribe' or 'whisper-1').
    """
    with open(filepath, "rb") as f:
        # SDK method names may vary slightly depending on version
        result = openai_client.audio.transcriptions.create(file=f, model=model)
    return getattr(result, "text", str(result))

def transcribe_local(filepath: str, size: Lit["tiny","base","small","medium","large"]="small") -> str:
    from faster_whisper import WhisperModel
    model = WhisperModel(size, compute_type="int8_float16")
    segments, info = model.transcribe(filepath, beam_size=5, vad_filter=True)
    return " ".join(seg.text.strip() for seg in segments)

def transcribe(filepath: str, engine: Lit["openai","local"]="openai") -> str:
    if engine == "openai":
        return transcribe_openai(filepath)
    return transcribe_local(filepath)

# =================================
# 5) AUDIO STREAMING + VAD CHUNKING
# =================================

@dataclass
class AudioConfig:
    samplerate: int = 16000          # required for webrtcvad
    channels: int = 1
    dtype: str = "int16"
    frame_ms: int = 30               # 10, 20, or 30 ms only for VAD
    silence_flush_ms: int = 800      # flush chunk after this silence
    max_chunk_seconds: int = 5       # safety cap to avoid huge chunks
    engine: Lit["openai","local"] = "openai"  # STT engine

class StreamBot:
    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self.blocksize = int(self.cfg.samplerate * self.cfg.frame_ms / 1000)
        self.audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
        self.vad = webrtcvad.Vad(2)  # 0-3 aggressiveness
        self.shutdown = threading.Event()
        self._last_voiced_ts = None
        self._chunk_frames: list[bytes] = []
        self._chunk_samples = 0

    def _callback(self, indata, frames, time_info, status):
        if status:
            # Non-fatal warnings from PortAudio show here
            pass
        self.audio_q.put(indata.copy())

    def _write_wav(self, pcm_bytes: bytes) -> str:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(self.cfg.channels)
            wf.setsampwidth(2)  # int16 -> 2 bytes
            wf.setframerate(self.cfg.samplerate)
            wf.writeframes(pcm_bytes)
        return tmp.name

    def _flush_chunk_if_ready(self, force: bool = False):
        if not self._chunk_frames:
            return
        # silence condition
        silence_elapsed = 0
        if self._last_voiced_ts is not None:
            silence_elapsed = (time.time() - self._last_voiced_ts) * 1000

        too_long = (self._chunk_samples / self.cfg.samplerate) >= self.cfg.max_chunk_seconds

        if force or (silence_elapsed >= self.cfg.silence_flush_ms) or too_long:
            pcm = b"".join(self._chunk_frames)
            self._chunk_frames.clear()
            self._chunk_samples = 0
            self._last_voiced_ts = None

            # Write to temp wav
            wav_path = self._write_wav(pcm)
            try:
                text = transcribe(wav_path, engine=self.cfg.engine).strip()
            finally:
                # Clean up temp
                try: os.remove(wav_path)
                except: pass

            if text:
                print(f"\nUser (transcribed): {text}")
                try:
                    reply = handle_text(text)
                except Exception as e:
                    reply = f"(tool error) {e}"
                print(f"Bot: {reply}\n> Speak…", flush=True)

    def run(self):
        print("Starting mic stream… Press Ctrl+C to stop.")
        sd.default.samplerate = self.cfg.samplerate
        sd.default.channels = self.cfg.channels
        dtype = self.cfg.dtype

        with sd.InputStream(channels=self.cfg.channels,
                            samplerate=self.cfg.samplerate,
                            dtype=dtype,
                            blocksize=self.blocksize,
                            callback=self._callback):
            print("> Speak…", flush=True)
            while not self.shutdown.is_set():
                try:
                    data = self.audio_q.get(timeout=0.1)  # np.ndarray shape: (blocksize, 1)
                except queue.Empty:
                    # check if pending chunk must be flushed due to silence
                    self._flush_chunk_if_ready(force=False)
                    continue

                # Flatten to bytes (int16 PCM mono)
                pcm_bytes = data.astype(np.int16).tobytes()

                # VAD expects frames of exactly 10/20/30ms; we configured blocksize accordingly
                is_voiced = self.vad.is_speech(pcm_bytes, self.cfg.samplerate)

                if is_voiced:
                    self._chunk_frames.append(pcm_bytes)
                    self._chunk_samples += len(pcm_bytes) // 2
                    self._last_voiced_ts = time.time()
                else:
                    # No voice in this frame; maybe flush based on accumulated silence
                    self._flush_chunk_if_ready(force=False)

        # Final flush on shutdown
        self._flush_chunk_if_ready(force=True)

def _handle_sigint(bot: StreamBot):
    def handler(sig, frame):
        print("\nStopping…")
        bot.shutdown.set()
    return handler

if __name__ == "__main__":
    cfg = AudioConfig(
        samplerate=16000,
        channels=1,
        dtype="int16",
        frame_ms=30,           # keep 10/20/30 for VAD
        silence_flush_ms=800,  # chunk closes after ~0.8s pause
        max_chunk_seconds=5,
        engine="openai",       # or "local" (requires model download)
    )
    bot = StreamBot(cfg)
    signal.signal(signal.SIGINT, _handle_sigint(bot))
    bot.run()
