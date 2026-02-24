from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

from hayfind.config import data_dir

COLLECTION_NAME = "documents"


def _patch_chroma_telemetry() -> None:
    """Patch Chroma telemetry batching bug (KeyError in Posthog.capture).

    Some Chroma versions can raise KeyError when deleting a batch_key under
    concurrent access. This patch makes the deletion idempotent.

    This is a pragmatic MVP fix; upstream should fix the race.
    """

    try:
        from chromadb.telemetry.product.posthog import Posthog
    except Exception:
        return

    if getattr(Posthog.capture, "__hayfind_patched__", False):
        return

    original = Posthog.capture

    def capture(self, event):  # type: ignore[no-untyped-def]
        try:
            return original(self, event)
        except KeyError:
            # Ignore telemetry batching race.
            return None

    capture.__hayfind_patched__ = True  # type: ignore[attr-defined]
    Posthog.capture = capture  # type: ignore[assignment]


def chroma_client() -> chromadb.ClientAPI:
    """Return a Chroma client."""

    _patch_chroma_telemetry()

    settings = Settings(anonymized_telemetry=False)

    host = os.getenv("CHROMA_HOST")
    port = os.getenv("CHROMA_PORT")
    if host and port:
        return chromadb.HttpClient(host=host, port=int(port), settings=settings)

    persist_path = Path(os.getenv("HAYFIND_CHROMA_PATH", data_dir() / "chroma"))
    persist_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_path), settings=settings)


def get_collection() -> Collection:
    client = chroma_client()
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def safe_where(
    repo: str | None = None,
    path_prefix: str | None = None,
    file_id: str | None = None,
) -> dict[str, Any] | None:
    clauses: list[dict[str, Any]] = []
    if repo:
        clauses.append({"repo": repo})
    if path_prefix:
        clauses.append({"path": {"$contains": path_prefix}})
    if file_id:
        clauses.append({"file_id": file_id})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
