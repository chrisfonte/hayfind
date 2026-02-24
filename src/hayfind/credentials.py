from __future__ import annotations

import os
from pathlib import Path


def load_gemini_api_key() -> str:
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.getenv(key)
        if value:
            return value.strip()

    cred_path = Path("~/.credentials/gemini/api_key").expanduser()
    if cred_path.exists():
        value = cred_path.read_text(encoding="utf-8").strip()
        if value:
            return value

    raise RuntimeError(
        "Gemini API key not found. Set GEMINI_API_KEY/GOOGLE_API_KEY "
        "or create ~/.credentials/gemini/api_key"
    )


def load_openai_api_key() -> str:
    value = (os.getenv("OPENAI_API_KEY") or "").strip()
    if value:
        return value
    raise RuntimeError("OpenAI API key not found. Set OPENAI_API_KEY")
