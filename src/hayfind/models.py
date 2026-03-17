from __future__ import annotations

from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    path: str
    force: bool = False
    batch_offset: int = 0
    batch_files: int | None = None


class IndexResponse(BaseModel):
    repo: str
    indexed_files: int
    skipped_files: int
    removed_files: int
    chunk_count: int
    total_files: int | None = None
    processed_files: int | None = None
    next_offset: int | None = None
    done: bool = True
    error_count: int = 0
    error_files: list[str] = Field(default_factory=list)


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


class RepoCheckpoint(BaseModel):
    offset: int
    total_files: int
    started_at: str
    done: bool = False


class StatusResponse(BaseModel):
    collection: str
    doc_count: int
    repos: list[str]
    last_indexed_at: str | None = None
    checkpoints: dict[str, RepoCheckpoint] = Field(default_factory=dict)
