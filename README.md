# hayfind

`hayfind` is a local semantic code/text search MVP with:

- A local HTTP service (`hayfindd`) exposing:
  - `POST /index`
  - `POST /search`
  - `GET /status`
- A CLI (`hayfind`) that wraps the service.

This repository includes the original build prompt in [`prompts/`](prompts/).

## Features

- Python 3.11–3.13 (ChromaDB currently breaks on Python 3.14)
- ChromaDB-backed vector index (`documents` collection)
  - Persistent local storage by default
  - Optional remote Chroma server via `CHROMA_HOST` + `CHROMA_PORT`
- Pluggable embeddings providers (`gemini`, `openai`, or local Ollama) with optional fallback
- Incremental indexing by file hash (`sha256`)
- Binary-file skipping (MIME + null-byte checks)
- Search filters: `--repo`, `--path-prefix`
- JSON output mode for search/status/index/reindex
- Git hook installer (`post-merge`) for configured repos
- Batched indexing over `/index` for large repositories (progress per batch)

## Install

```bash
python -m pip install -e .
# or
pipx install .
```

## Config

Default config path:

- `~/.config/hayfind/config.yaml`
- Data dir (state + local Chroma path base): `~/.local/share/hayfind`
- Override data dir with `HAYFIND_DATA_DIR=/absolute/path`

Generate a starter config:

```bash
hayfind init-config
```

Example config is in [`examples/config.yaml`](examples/config.yaml).

## Embeddings provider and env vars

Provider selection:

- `HAYFIND_EMBED_PROVIDER=gemini|openai|local` (default: `gemini`)
- `HAYFIND_EMBED_FALLBACK_PROVIDER=openai|gemini` (optional, indexing only)

Gemini:

- `HAYFIND_GEMINI_EMBED_MODEL` (default: `gemini-embedding-001`)
- API key lookup order:
1. `GEMINI_API_KEY`
2. `GOOGLE_API_KEY`
3. `~/.credentials/gemini/api_key`

OpenAI:

- `HAYFIND_OPENAI_EMBED_MODEL` (default: `text-embedding-3-small`)
- `OPENAI_API_KEY` (required for `openai` provider or fallback)

Local Ollama:

- `HAYFIND_LOCAL_EMBED_URL` (default: `http://127.0.0.1:11434/api/embeddings`)
- `HAYFIND_LOCAL_EMBED_MODEL` (default: `nomic-embed-text`)

Quick setup:

```bash
ollama serve
ollama pull nomic-embed-text
```

Migration note:

- A ChatGPT subscription does not include API access by itself. You need an OpenAI API key (`OPENAI_API_KEY`) from the OpenAI API platform.

Do not commit secrets.

## Run service

```bash
hayfind serve --help
python -m hayfindd --help
hayfind serve
```

Default service URL used by CLI: `http://127.0.0.1:8765`

## CLI usage

```bash
export HAYFIND_EMBED_PROVIDER=gemini
export HAYFIND_EMBED_FALLBACK_PROVIDER=openai
export OPENAI_API_KEY=...
hayfind index ~/projects/some-repo
hayfind index ~/projects/some-repo --batch-files 200
hayfind search "foo" --repo some-repo
hayfind search "foo" --path-prefix src/ --json
hayfind status
hayfind reindex
hayfind install-hooks
```

Local-only embedding example:

```bash
export HAYFIND_EMBED_PROVIDER=local
export HAYFIND_LOCAL_EMBED_URL=http://127.0.0.1:11434/api/embeddings
export HAYFIND_LOCAL_EMBED_MODEL=nomic-embed-text
hayfind index ~/projects/some-repo
hayfind search "foo" --repo some-repo
```

## HTTP API

### `POST /index`

Request:

```json
{
  "path": "/absolute/path/to/repo",
  "force": false,
  "batch_offset": 0,
  "batch_files": 200
}
```

`batch_offset` and `batch_files` are optional. CLI defaults to batched calls for reliability on large repos. Override default batch size with:

- `--batch-files <N>`
- `HAYFIND_INDEX_BATCH_FILES` (default `200`)

### `POST /search`

Request:

```json
{
  "query": "fastapi",
  "top_k": 5,
  "repo": "my-repo",
  "path_prefix": "src/"
}
```

### `GET /status`

Returns collection name, total indexed chunk count, repos, and last indexing time.

## Git hook installer

`hayfind install-hooks` installs a `post-merge` hook in each configured git repo:

```sh
hayfind index <repoPath>
```

## Development

```bash
make install-dev
make lint
make test
make serve
```

## Acceptance checks

```bash
python -m hayfindd --help
hayfind serve --help
hayfind index ~/projects/some-repo
hayfind search "foo" --repo some-repo
hayfind status
```

## Non-goals (MVP)

- File watcher daemon
- Authentication
- Hosted deployment
