# asr.py
from pathlib import Path
from typing import Literal, Optional

# Option A: OpenAI managed STT (simple)
def transcribe_openai(filepath: str, model: str = "gpt-4o-mini-transcribe") -> str:
    """
    Uses OpenAI's STT (model name may vary; 'gpt-4o-mini-transcribe' or similar).
    """
    from openai import OpenAI
    client = OpenAI()
    with open(filepath, "rb") as f:
        # API shape may vary; this is a commonly used pattern:
        result = client.audio.transcriptions.create(
            file=f, model=model  # alternative: model="whisper-1" if available in your account
        )
    # result.text is typical; adjust to actual field if your SDK returns .text or .transcript
    return getattr(result, "text", str(result))

# Option B: Local faster-whisper (offline)
def transcribe_local(filepath: str, size: Literal["tiny","base","small","medium","large"]="small") -> str:
    from faster_whisper import WhisperModel
    model = WhisperModel(size, compute_type="int8_float16")  # good perf on CPU/GPU
    segments, info = model.transcribe(filepath, beam_size=5, vad_filter=True)
    return " ".join(seg.text.strip() for seg in segments)

def transcribe(filepath: str, engine: Literal["openai","local"]="openai") -> str:
    if engine == "openai":
        return transcribe_openai(filepath)
    return transcribe_local(filepath)
