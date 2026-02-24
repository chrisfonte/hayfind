from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from typing import Any, TypeVar

try:
    from google import genai
    from google.genai.errors import ClientError
except ImportError:  # pragma: no cover - exercised only when dependency is missing.
    genai = None
    ClientError = None  # type: ignore[assignment]

from hayfind.credentials import load_gemini_api_key

MODEL_NAME = os.getenv("HAYFIND_GEMINI_EMBED_MODEL", "gemini-embedding-001")

# Gemini batch embeddings endpoint: max 100 items per request.
DEFAULT_BATCH_SIZE = 50
DEFAULT_SLEEP_S = 0.05

T = TypeVar("T")


def _env_int(name: str, default: int) -> int:
    value = (os.getenv(name) or "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = (os.getenv(name) or "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_retry_delay_seconds(message: str) -> float | None:
    # Examples observed in the wild:
    # - "Please retry in 54.073497947s."
    # - "retryDelay': '54s'"
    m = re.search(r"retry in ([0-9.]+)s", message, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"retryDelay[^0-9]*([0-9]+)s", message, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _call_with_retry(fn: Callable[[], T], *, max_attempts: int = 5) -> T:
    """Retry on Gemini rate limits (429) using server-supplied retry delay when available."""
    attempt = 0
    backoff = 1.0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            # Rate limit / quota exhausted.
            if ClientError is not None and isinstance(exc, ClientError):
                if getattr(exc, "status_code", None) == 429:
                    msg = str(exc)
                    delay = _parse_retry_delay_seconds(msg) or backoff
                    time.sleep(delay + 0.25)
                    backoff = min(backoff * 2, 60.0)
                    if attempt < max_attempts:
                        continue

            if attempt >= max_attempts:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


class GeminiEmbedder:
    def __init__(self, client: Any | None = None) -> None:
        if client is not None:
            self._client = client
            return

        if genai is None:
            raise RuntimeError(
                "Missing dependency: install `google-genai` to use Gemini embeddings"
            )

        env_keys = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
        has_env_key = any((os.getenv(key) or "").strip() for key in env_keys)
        if has_env_key:
            self._client = genai.Client()
            return

        self._client = genai.Client(api_key=load_gemini_api_key())

    def _extract_values(self, payload: Any) -> list[float]:
        if isinstance(payload, dict) and "values" in payload:
            return [float(v) for v in payload["values"]]
        if hasattr(payload, "values"):
            return [float(v) for v in payload.values]
        if isinstance(payload, list):
            return [float(v) for v in payload]
        if isinstance(payload, tuple):
            return [float(v) for v in payload]
        if payload is None:
            return []
        if hasattr(payload, "__iter__") and not isinstance(payload, (str, bytes)):
            return [float(v) for v in payload]
        raise RuntimeError("Unexpected embedding vector format from Gemini API")

    def _extract_embeddings(self, payload: Any) -> list[list[float]]:
        if isinstance(payload, dict):
            if "embeddings" in payload:
                return [self._extract_values(item) for item in payload["embeddings"]]
            if "embedding" in payload:
                return [self._extract_values(payload["embedding"])]
        if hasattr(payload, "embeddings"):
            return [self._extract_values(item) for item in payload.embeddings]
        if hasattr(payload, "embedding"):
            return [self._extract_values(payload.embedding)]
        raise RuntimeError("Unexpected embedding response format from Gemini API")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document chunks.

        Gemini API constraints:
        - Max 100 items per batch request
        - Rate limits can trigger 429s with a suggested retry delay

        Tunables:
        - HAYFIND_EMBED_BATCH_SIZE (default 50; hard-capped at 100)
        - HAYFIND_EMBED_SLEEP_S (default 0.05) sleep between batches
        """
        if not texts:
            return []

        batch_size = min(_env_int("HAYFIND_EMBED_BATCH_SIZE", DEFAULT_BATCH_SIZE), 100)
        sleep_s = _env_float("HAYFIND_EMBED_SLEEP_S", DEFAULT_SLEEP_S)

        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            def _do() -> Any:
                return self._client.models.embed_content(
                    model=MODEL_NAME,
                    contents=batch,
                )

            response = _call_with_retry(_do)
            all_vectors.extend(self._extract_embeddings(response))
            if sleep_s:
                time.sleep(sleep_s)

        return all_vectors

    def embed_query(self, query: str) -> list[float]:
        response = _call_with_retry(
            lambda: self._client.models.embed_content(model=MODEL_NAME, contents=query)
        )
        vectors = self._extract_embeddings(response)
        if not vectors:
            raise RuntimeError("Empty embedding response from Gemini API")
        return vectors[0]
