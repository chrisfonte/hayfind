from __future__ import annotations

from hayfind.embeddings import get_embedder
from hayfind.indexer import get_index_state
from hayfind.models import SearchHit, SearchResponse, StatusResponse
from hayfind.store import COLLECTION_NAME, get_collection, safe_where


def search(
    query: str,
    top_k: int = 5,
    repo: str | None = None,
    path_prefix: str | None = None,
) -> SearchResponse:
    collection = get_collection()
    embedder = get_embedder()
    query_embedding = embedder.embed_query(query)

    where = safe_where(repo=repo)
    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["metadatas", "documents", "distances"],
    )

    hits: list[SearchHit] = []
    docs = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    for doc, meta, distance in zip(docs, metadatas, distances, strict=False):
        if not meta:
            continue
        path = meta.get("path", "")
        if path_prefix and not path.startswith(path_prefix):
            continue
        snippet = (doc or meta.get("chunk_text") or "").strip().replace("\n", " ")
        hits.append(
            SearchHit(
                repo=meta.get("repo", ""),
                path=path,
                score=1.0 - float(distance),
                snippet=snippet[:240],
                chunk_index=int(meta.get("chunk_index", 0)),
            )
        )

    return SearchResponse(query=query, hits=hits)


def status() -> StatusResponse:
    collection = get_collection()
    doc_count = collection.count()
    state = get_index_state()
    repos = sorted(state.get("repos", {}).keys())
    return StatusResponse(
        collection=COLLECTION_NAME,
        doc_count=doc_count,
        repos=repos,
        last_indexed_at=state.get("last_indexed_at"),
    )
