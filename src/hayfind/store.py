from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

from hayfind.config import data_dir

COLLECTION_NAME = "documents"


def chroma_client() -> chromadb.ClientAPI:
    """Return a Chroma client.

    We explicitly disable anonymized telemetry by default because some Chroma
    versions have been observed to throw internal KeyErrors in telemetry batching
    under concurrent use.
    """

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
