from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable
from typing import Any, TypeVar

import httpx

try:
    from google import genai
    from google.genai.errors import ClientError
except ImportError:  # pragma: no cover - exercised only when dependency is missing.
    genai = None
    ClientError = None  # type: ignore[assignment]

try:
    from openai import APIStatusError, OpenAI, RateLimitError
except ImportError:  # pragma: no cover - exercised only when dependency is missing.
    OpenAI = None
    APIStatusError = None  # type: ignore[assignment]
    RateLimitError = None  # type: ignore[assignment]

from hayfind.credentials import load_gemini_api_key, load_openai_api_key

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("HAYFIND_GEMINI_EMBED_MODEL", "gemini-embedding-001")
OPENAI_MODEL_NAME = os.getenv("HAYFIND_OPENAI_EMBED_MODEL", "text-embedding-3-small")
LOCAL_EMBED_URL = os.getenv("HAYFIND_LOCAL_EMBED_URL", "http://127.0.0.1:11434/api/embeddings")
LOCAL_MODEL_NAME = os.getenv("HAYFIND_LOCAL_EMBED_MODEL", "nomic-embed-text")

# Gemini batch embeddings endpoint: max 100 items per request.
DEFAULT_BATCH_SIZE = 20  # was 50 — safe for Gemini free tier (~2 API calls/sec at 0.5s sleep)
DEFAULT_SLEEP_S = 0.5  # was 0.05 — keeps throughput within 60 req/min free-tier limit

PROVIDER_BATCH_CAPS: dict[str, int] = {
    "gemini": 100,
    "openai": 2048,
    "local": 512,
}

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


def _is_rate_or_quota_error(exc: Exception) -> bool:
    if ClientError is not None and isinstance(exc, ClientError):
        if getattr(exc, "status_code", None) == 429:
            return True
    if RateLimitError is not None and isinstance(exc, RateLimitError):
        return True
    if APIStatusError is not None and isinstance(exc, APIStatusError):
        if getattr(exc, "status_code", None) == 429:
            return True
    message = str(exc).lower()
    return "429" in message or "rate limit" in message or "quota" in message


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

    def _extract_embeddings(self, payload: Any) -> list[list[float]]:
        # The new SDK returns a response object with an 'embeddings' attribute
        # which is a list of objects that have a 'values' attribute.
        if hasattr(payload, "embeddings"):
            return [[float(v) for v in item.values] for item in payload.embeddings]
        if hasattr(payload, "embedding"):
            return [[float(v) for v in payload.embedding.values]]

        # Fallback for dict-like access if needed
        if isinstance(payload, dict):
            if "embeddings" in payload:
                return [[float(v) for v in item["values"]] for item in payload["embeddings"]]
            if "embedding" in payload:
                return [[float(v) for v in payload["embedding"]["values"]]]

        raise RuntimeError(f"Unexpected embedding response format from Gemini API: {type(payload)}")

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

        batch_size = min(
            _env_int("HAYFIND_EMBED_BATCH_SIZE", DEFAULT_BATCH_SIZE),
            PROVIDER_BATCH_CAPS["gemini"],
        )
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


class OpenAIEmbedder:
    def __init__(self, client: Any | None = None) -> None:
        if client is not None:
            self._client = client
            return
        if OpenAI is None:
            raise RuntimeError("Missing dependency: install `openai` to use OpenAI embeddings")
        self._client = OpenAI(api_key=load_openai_api_key())

    def _extract_embeddings(self, payload: Any) -> list[list[float]]:
        if isinstance(payload, dict):
            data = payload.get("data", [])
        elif hasattr(payload, "data"):
            data = payload.data
        else:
            raise RuntimeError("Unexpected embedding response format from OpenAI API")

        vectors: list[list[float]] = []
        for item in data:
            if isinstance(item, dict):
                embedding = item.get("embedding")
            else:
                embedding = getattr(item, "embedding", None)
            if embedding is None:
                raise RuntimeError("Unexpected embedding vector format from OpenAI API")
            vectors.append([float(v) for v in embedding])
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        batch_size = _env_int("HAYFIND_EMBED_BATCH_SIZE", DEFAULT_BATCH_SIZE)
        sleep_s = _env_float("HAYFIND_EMBED_SLEEP_S", DEFAULT_SLEEP_S)

        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            def _do() -> Any:
                return self._client.embeddings.create(model=OPENAI_MODEL_NAME, input=batch)

            response = _call_with_retry(_do)
            all_vectors.extend(self._extract_embeddings(response))
            if sleep_s:
                time.sleep(sleep_s)

        return all_vectors

    def embed_query(self, query: str) -> list[float]:
        response = _call_with_retry(
            lambda: self._client.embeddings.create(model=OPENAI_MODEL_NAME, input=[query])
        )
        vectors = self._extract_embeddings(response)
        if not vectors:
            raise RuntimeError("Empty embedding response from OpenAI API")
        return vectors[0]


class LocalEmbedder:
    def __init__(
        self,
        *,
        client: Any | None = None,
        url: str | None = None,
        model: str | None = None,
    ) -> None:
        self._client = client or httpx.Client(timeout=30.0)
        self._url = url or LOCAL_EMBED_URL
        self._model = model or LOCAL_MODEL_NAME

    def _extract_embedding(self, payload: Any) -> list[float]:
        if not isinstance(payload, dict):
            raise RuntimeError("Unexpected embedding response format from local provider")
        embedding = payload.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError("Unexpected embedding vector format from local provider")
        return [float(v) for v in embedding]

    def _embed_one(self, text: str) -> list[float]:
        response = self._client.post(
            self._url,
            json={"model": self._model, "prompt": text},
        )
        response.raise_for_status()
        return self._extract_embedding(response.json())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        batch_size = _env_int("HAYFIND_EMBED_BATCH_SIZE", DEFAULT_BATCH_SIZE)
        sleep_s = _env_float("HAYFIND_EMBED_SLEEP_S", DEFAULT_SLEEP_S)
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            for text in texts[i : i + batch_size]:
                all_vectors.append(self._embed_one(text))
            if sleep_s:
                time.sleep(sleep_s)
        return all_vectors

    def embed_query(self, query: str) -> list[float]:
        return self._embed_one(query)


class AdaptiveThrottle:
    """Tracks 429 rate and auto-adjusts sleep interval.

    Only activated when ``HAYFIND_ADAPTIVE_THROTTLE=1``.
    Doubles sleep on repeated rate limits (cap 10s); halves on sustained success (floor base).
    """

    def __init__(self, base_sleep: float) -> None:
        self._sleep = base_sleep
        self._base = base_sleep
        self._recent_limits: list[float] = []
        self._success_count: int = 0

    def record_rate_limit(self) -> None:
        now = time.time()
        self._recent_limits = [t for t in self._recent_limits if now - t < 60]
        self._recent_limits.append(now)
        self._success_count = 0
        if len(self._recent_limits) >= 2:
            self._sleep = min(self._sleep * 2, 10.0)
            logger.warning("Adaptive throttle: backing off to %.1fs sleep", self._sleep)

    def record_success(self) -> None:
        self._success_count += 1
        if self._success_count >= 20 and not self._recent_limits:
            self._sleep = max(self._sleep / 2, self._base)
            self._success_count = 0

    def sleep(self) -> None:
        if self._sleep:
            time.sleep(self._sleep)


class FallbackEmbedder:
    def __init__(
        self,
        *,
        primary: Any,
        primary_provider: str,
        fallback: Any,
        fallback_provider: str,
    ) -> None:
        self._primary = primary
        self._primary_provider = primary_provider
        self._fallback = fallback
        self._fallback_provider = fallback_provider
        self._switched = False

    def _switch(self, *, error: Exception) -> None:
        if self._switched:
            return
        self._switched = True
        logger.warning(
            "Embedding provider switch: %s -> %s after rate/quota failure: %s",
            self._primary_provider,
            self._fallback_provider,
            error,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        batch_size = min(_env_int("HAYFIND_EMBED_BATCH_SIZE", DEFAULT_BATCH_SIZE), 100)
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            if self._switched:
                all_vectors.extend(self._fallback.embed_documents(batch))
                continue
            try:
                all_vectors.extend(self._primary.embed_documents(batch))
            except Exception as exc:  # noqa: BLE001
                if not _is_rate_or_quota_error(exc):
                    raise
                self._switch(error=exc)
                all_vectors.extend(self._fallback.embed_documents(batch))

        return all_vectors

    def embed_query(self, query: str) -> list[float]:
        if self._switched:
            return self._fallback.embed_query(query)
        try:
            return self._primary.embed_query(query)
        except Exception as exc:  # noqa: BLE001
            if not _is_rate_or_quota_error(exc):
                raise
            self._switch(error=exc)
            return self._fallback.embed_query(query)


def _normalize_provider(name: str | None) -> str:
    provider = (name or "gemini").strip().lower()
    if provider not in {"gemini", "openai", "local"}:
        raise RuntimeError("Unsupported HAYFIND_EMBED_PROVIDER. Use one of: gemini, openai, local")
    return provider


def _normalize_fallback_provider(name: str | None) -> str | None:
    if name is None:
        return None
    provider = name.strip().lower()
    if not provider:
        return None
    if provider not in {"gemini", "openai"}:
        raise RuntimeError(
            "Unsupported HAYFIND_EMBED_FALLBACK_PROVIDER. Use one of: gemini, openai"
        )
    return provider


def _build_provider(provider: str) -> Any:
    if provider == "gemini":
        return GeminiEmbedder()
    if provider == "openai":
        return OpenAIEmbedder()
    if provider == "local":
        return LocalEmbedder()
    raise RuntimeError(f"Unsupported embedding provider: {provider}")


def get_embedder(*, for_indexing: bool = False) -> Any:
    provider = _normalize_provider(os.getenv("HAYFIND_EMBED_PROVIDER"))
    fallback_provider = _normalize_fallback_provider(os.getenv("HAYFIND_EMBED_FALLBACK_PROVIDER"))

    primary = _build_provider(provider)
    if not for_indexing or not fallback_provider or fallback_provider == provider:
        return primary
    fallback = _build_provider(fallback_provider)
    return FallbackEmbedder(
        primary=primary,
        primary_provider=provider,
        fallback=fallback,
        fallback_provider=fallback_provider,
    )
