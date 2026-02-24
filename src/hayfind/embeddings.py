from __future__ import annotations

import os
from typing import Any

try:
    from google import genai
except ImportError:  # pragma: no cover - exercised only when dependency is missing.
    genai = None

from hayfind.credentials import load_gemini_api_key

MODEL_NAME = os.getenv("HAYFIND_GEMINI_EMBED_MODEL", "gemini-embedding-001")


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

        Note: the Gemini API batch embeddings endpoint limits batches to 100
        items per request. We chunk requests to stay under that cap.
        """
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.models.embed_content(
                model=MODEL_NAME,
                contents=batch,
            )
            all_vectors.extend(self._extract_embeddings(response))

        return all_vectors

    def embed_query(self, query: str) -> list[float]:
        response = self._client.models.embed_content(model=MODEL_NAME, contents=query)
        vectors = self._extract_embeddings(response)
        if not vectors:
            raise RuntimeError("Empty embedding response from Gemini API")
        return vectors[0]
