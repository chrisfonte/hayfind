from __future__ import annotations

from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    path: str
    force: bool = False


class IndexResponse(BaseModel):
    repo: str
    indexed_files: int
    skipped_files: int
    removed_files: int
    chunk_count: int


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    repo: str | None = None
    path_prefix: str | None = None


class SearchHit(BaseModel):
    repo: str
    path: str
    score: float
    snippet: str
    chunk_index: int


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit] = Field(default_factory=list)


class StatusResponse(BaseModel):
    collection: str
    doc_count: int
    repos: list[str]
    last_indexed_at: str | None = None
