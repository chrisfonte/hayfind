# Codex build prompt — hayfind (2026-02-24)

You are building a new public repo: `chrisfonte/hayfind`.

Goal: ship a working MVP of **hayfind** — a local semantic search service + CLI for text-based files across repos.

## Hard requirements
- Language: Python 3.11+
- Provide BOTH:
  1) HTTP service (local) exposing endpoints: `POST /index`, `POST /search`, `GET /status`
  2) CLI `hayfind` wrapping the HTTP service
- Vector store: ChromaDB with persistent storage.
  - Prefer local persistent storage by default.
  - Also support connecting to a running ChromaDB server via env vars (host/port) if present.
- Embeddings: **Gemini** embedding model `text-embedding-004`.
  - API key read from (first found):
    1) env `GEMINI_API_KEY` or `GOOGLE_API_KEY`
    2) file `~/.credentials/gemini/api_key`
  - Never commit secrets.
- Config file:
  - Default path: `~/.config/hayfind/config.yaml`
  - Contains a list of repos to index + include/exclude globs.
- Indexing:
  - `hayfind index <path>` indexes a directory/repo.
  - Must skip binaries automatically (best-effort: detect via `mimetypes` and/or null bytes).
  - Store per-document metadata: `repo`, `path`, `sha256`, `mtime`, `size_bytes`.
  - Implement incremental indexing: re-embed only changed/new files (compare sha256).
- Search:
  - `hayfind search "query"` returns top matches with file path + snippet.
  - Support filters: `--repo`, `--path-prefix`.
  - Support `--json` output.
- Status:
  - `hayfind status` shows what’s indexed: doc count, repos, last indexing time.
- Reindex:
  - `hayfind reindex` forces full reindex of configured repos.
- Git hook installer:
  - `hayfind install-hooks` installs a `post-merge` hook in configured repos that triggers incremental indexing.
  - Hook should call `hayfind index <repoPath>`.

## Repo/packaging requirements
- Include: `README.md`, `LICENSE` (MIT), `pyproject.toml`, `src/` layout, and minimal tests.
- CLI should be installable via `pipx install .` and runnable as `hayfind`.
- Provide a `Makefile` or `justfile` with common commands (lint/test/serve).
- Provide `examples/` with:
  - sample `config.yaml`
  - example commands
- Add `prompts/` folder (this file already exists) and reference it from README.

## Suggested architecture
- `hayfindd` service: FastAPI + uvicorn.
- CLI: Typer.
- HTTP client from CLI to service: `httpx`.
- Service uses Chroma collection `documents`.
- Text chunking: simple line-based or paragraph-based chunker (keep it simple). Store `chunk_index` and `chunk_text` metadata.

## Acceptance checks (document in README)
- `python -m hayfindd --help` or `hayfind serve --help`
- `hayfind index ~/projects/some-repo`
- `hayfind search "foo" --repo some-repo`
- `hayfind status`

## Non-goals (for MVP)
- No file watching daemon.
- No auth.
- No hosted deployment.

Implement the MVP end-to-end and keep code clean and well-documented.
