from __future__ import annotations

from typing import Any

import google.generativeai as genai

from hayfind.credentials import load_gemini_api_key

MODEL_NAME = "models/text-embedding-004"


class GeminiEmbedder:
    def __init__(self) -> None:
        genai.configure(api_key=load_gemini_api_key())

    def _extract_embedding(self, payload: Any) -> list[float]:
        if isinstance(payload, dict):
            if "embedding" in payload:
                emb = payload["embedding"]
                if isinstance(emb, dict) and "values" in emb:
                    return emb["values"]
                return emb
            if "values" in payload:
                return payload["values"]
        if hasattr(payload, "embedding"):
            emb = payload.embedding
            if hasattr(emb, "values"):
                return list(emb.values)
            return list(emb)
        if hasattr(payload, "values"):
            return list(payload.values)
        raise RuntimeError("Unexpected embedding response format from Gemini API")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            response = genai.embed_content(
                model=MODEL_NAME,
                content=text,
                task_type="retrieval_document",
            )
            embeddings.append(self._extract_embedding(response))
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        response = genai.embed_content(
            model=MODEL_NAME,
            content=query,
            task_type="retrieval_query",
        )
        return self._extract_embedding(response)
