from __future__ import annotations

import hashlib
import json
import mimetypes
from dataclasses import dataclass
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path

from hayfind.config import RepoConfig, data_dir
from hayfind.embeddings import get_embedder
from hayfind.store import get_collection, safe_where

STATE_PATH = data_dir() / "state.json"


@dataclass(slots=True)
class IndexStats:
    repo: str
    indexed_files: int = 0
    skipped_files: int = 0
    removed_files: int = 0
    chunk_count: int = 0
    total_files: int = 0
    processed_files: int = 0
    next_offset: int | None = None
    done: bool = True
    error_count: int = 0
    error_files: list[str] | None = None


def is_binary(path: Path) -> bool:
    mime, _ = mimetypes.guess_type(path.name)
    if mime and not mime.startswith("text/"):
        # Allow common text-like sources with non-text mime guesses.
        allowed = {"application/json", "application/xml", "application/javascript"}
        if mime not in allowed:
            return True

    try:
        with path.open("rb") as f:
            block = f.read(8192)
        return b"\x00" in block
    except OSError:
        return True


def chunk_text(content: str, max_chars: int = 1200) -> list[str]:
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = para if not current else f"{current}\n\n{para}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(para) <= max_chars:
            current = para
        else:
            # Hard-split a very large paragraph.
            for i in range(0, len(para), max_chars):
                piece = para[i : i + max_chars]
                if len(piece) == max_chars:
                    chunks.append(piece)
                else:
                    current = piece
    if current:
        chunks.append(current)
    return chunks


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def iter_files(repo: RepoConfig):
    seen: set[Path] = set()
    for pattern in repo.include:
        for path in repo.path.glob(pattern):
            if path in seen or not path.is_file():
                continue
            rel = path.relative_to(repo.path)
            rel_posix = rel.as_posix()
            if any(fnmatch(rel_posix, ex) for ex in repo.exclude):
                continue
            seen.add(path)
            yield path, rel


def collect_repo_files(repo: RepoConfig) -> list[tuple[Path, Path]]:
    files = list(iter_files(repo))
    files.sort(key=lambda item: item[1].as_posix())
    return files


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)  # atomic rename — prevents corruption on crash mid-write


def update_last_indexed(repo: str) -> None:
    state = _load_state()
    state.setdefault("repos", {})[repo] = datetime.now(UTC).isoformat()
    state["last_indexed_at"] = datetime.now(UTC).isoformat()
    _save_state(state)


def get_last_indexed() -> str | None:
    return _load_state().get("last_indexed_at")


def get_index_state() -> dict:
    return _load_state()


def write_checkpoint(repo: str, offset: int, total_files: int, force: bool) -> None:
    """Persist in-progress batch offset after each successful batch.

    Allows CLI to resume from this offset on restart rather than starting from 0.
    """
    state = _load_state()
    state.setdefault("checkpoints", {})[repo] = {
        "started_at": datetime.now(UTC).isoformat(),
        "offset": offset,
        "total_files": total_files,
        "force": force,
        "done": False,
    }
    _save_state(state)


def clear_checkpoint(repo: str) -> None:
    """Remove checkpoint on successful run completion (done=True)."""
    state = _load_state()
    state.get("checkpoints", {}).pop(repo, None)
    _save_state(state)


def get_checkpoint(repo: str) -> dict | None:
    """Return in-progress checkpoint for repo, or None if absent."""
    return _load_state().get("checkpoints", {}).get(repo)


def index_repo(
    repo_path: Path,
    include: list[str],
    exclude: list[str],
    force: bool = False,
    batch_offset: int = 0,
    batch_files: int | None = None,
) -> IndexStats:
    repo_cfg = RepoConfig(path=repo_path.resolve(), include=include, exclude=exclude)
    repo_name = repo_cfg.path.name
    stats = IndexStats(repo=repo_name, error_files=[])

    collection = get_collection()
    embedder = get_embedder(for_indexing=True)
    all_files = collect_repo_files(repo_cfg)
    stats.total_files = len(all_files)

    start = max(0, batch_offset)
    if batch_files is None:
        end = stats.total_files
    else:
        end = min(stats.total_files, start + batch_files)
    selected_files = all_files[start:end]
    stats.processed_files = end
    stats.done = end >= stats.total_files
    stats.next_offset = None if stats.done else end

    active_file_ids: set[str] = {f"{repo_name}:{rel.as_posix()}" for _, rel in all_files}

    for path, rel in selected_files:
        if is_binary(path):
            stats.skipped_files += 1
            continue

        try:
            raw = path.read_bytes()
        except OSError:
            stats.skipped_files += 1
            stats.error_count += 1
            if stats.error_files is not None and len(stats.error_files) < 10:
                stats.error_files.append(rel.as_posix())
            continue

        digest = sha256_bytes(raw)
        text = raw.decode("utf-8", errors="ignore")
        if not text.strip():
            stats.skipped_files += 1
            continue

        file_id = f"{repo_name}:{rel.as_posix()}"
        active_file_ids.add(file_id)
        existing = collection.get(where=safe_where(file_id=file_id), include=["metadatas"])
        existing_metas = existing.get("metadatas", []) if existing else []

        if existing_metas and not force:
            existing_sha = existing_metas[0].get("sha256")
            if existing_sha == digest:
                stats.skipped_files += 1
                continue

        if existing and existing.get("ids"):
            collection.delete(ids=existing["ids"])

        chunks = chunk_text(text)
        if not chunks:
            stats.skipped_files += 1
            continue

        try:
            embeddings = embedder.embed_documents(chunks)
        except Exception:
            stats.error_count += 1
            if stats.error_files is not None and len(stats.error_files) < 10:
                stats.error_files.append(rel.as_posix())
            continue
        ids: list[str] = []
        metadatas: list[dict] = []
        documents: list[str] = []

        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            _ = embedding
            ids.append(f"{file_id}:{digest}:{idx}")
            documents.append(chunk)
            metadatas.append(
                {
                    "repo": repo_name,
                    "path": rel.as_posix(),
                    "file_id": file_id,
                    "sha256": digest,
                    "mtime": path.stat().st_mtime,
                    "size_bytes": path.stat().st_size,
                    "chunk_index": idx,
                    "chunk_text": chunk[:9000],
                }
            )

        collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        stats.indexed_files += 1
        stats.chunk_count += len(chunks)

    # Write checkpoint after each batch so CLI can resume on restart.
    # Only checkpoint in batched mode (batch_files is not None) — single-shot runs don't need it.
    if batch_files is not None and not stats.done:
        write_checkpoint(repo_name, end, stats.total_files, force)

    if stats.done:
        # Chroma always returns ids; include only controls additional fields.
        # Batch retrieval of existing repo documents to avoid "too many SQL variables" errors.
        limit = 5000
        offset = 0
        stale_ids: list[str] = []
        removed_file_ids: set[str] = set()

        while True:
            existing_repo = collection.get(
                where=safe_where(repo=repo_name),
                include=["metadatas"],
                limit=limit,
                offset=offset,
            )
            if not existing_repo or not existing_repo.get("ids"):
                break

            for idx, doc_id in enumerate(existing_repo["ids"]):
                meta = existing_repo.get("metadatas", [])[idx]
                if meta and meta.get("file_id") not in active_file_ids:
                    stale_ids.append(doc_id)
                    removed_file_ids.add(meta.get("file_id", ""))

            if len(existing_repo["ids"]) < limit:
                break
            offset += limit

        if stale_ids:
            # Batch deletion to avoid similar SQL variable limits.
            for i in range(0, len(stale_ids), limit):
                collection.delete(ids=stale_ids[i : i + limit])
            stats.removed_files = len([x for x in removed_file_ids if x])

        update_last_indexed(repo_name)
        clear_checkpoint(repo_name)  # clean completion — remove in-progress marker
    return stats
