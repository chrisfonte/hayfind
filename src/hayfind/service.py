from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from hayfind.config import DEFAULT_CONFIG_PATH, load_config
from hayfind.indexer import index_repo
from hayfind.models import (
    IndexRequest,
    IndexResponse,
    SearchRequest,
    SearchResponse,
    StatusResponse,
)
from hayfind.search import search as run_search
from hayfind.search import status as run_status

app = FastAPI(title="hayfind", version="0.1.0")


@app.get("/status", response_model=StatusResponse)
def status_endpoint() -> StatusResponse:
    return run_status()


@app.post("/search", response_model=SearchResponse)
def search_endpoint(req: SearchRequest) -> SearchResponse:
    return run_search(
        req.query,
        top_k=req.top_k,
        repo=req.repo,
        path_prefix=req.path_prefix,
    )


@app.post("/index", response_model=IndexResponse)
def index_endpoint(req: IndexRequest) -> IndexResponse:
    repo_path = Path(req.path).expanduser().resolve()
    if not repo_path.exists() or not repo_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Path not found or not a directory: {repo_path}",
        )
    if req.batch_offset < 0:
        raise HTTPException(status_code=400, detail="batch_offset must be >= 0")
    if req.batch_files is not None and req.batch_files <= 0:
        raise HTTPException(status_code=400, detail="batch_files must be > 0")

    cfg = load_config(DEFAULT_CONFIG_PATH)
    repo_cfg = next((r for r in cfg.repos if r.path == repo_path), None)
    include = repo_cfg.include if repo_cfg else ["**/*"]
    exclude = (
        repo_cfg.exclude
        if repo_cfg
        else [".git/**", "**/.git/**", "**/node_modules/**", "**/.venv/**", "**/__pycache__/**"]
    )
    stats = index_repo(
        repo_path,
        include=include,
        exclude=exclude,
        force=req.force,
        batch_offset=req.batch_offset,
        batch_files=req.batch_files,
    )
    return IndexResponse(
        repo=stats.repo,
        indexed_files=stats.indexed_files,
        skipped_files=stats.skipped_files,
        removed_files=stats.removed_files,
        chunk_count=stats.chunk_count,
        total_files=stats.total_files,
        processed_files=stats.processed_files,
        next_offset=stats.next_offset,
        done=stats.done,
        error_count=stats.error_count,
        error_files=stats.error_files,
    )
