from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import typer
import uvicorn

from hayfind.config import DEFAULT_CONFIG_PATH, AppConfig, ensure_config_path, load_config

app = typer.Typer(help="hayfind CLI")


class ServiceClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or os.getenv("HAYFIND_URL", "http://127.0.0.1:8765")

    def post(self, path: str, payload: dict) -> dict:
        try:
            with httpx.Client(base_url=self.base_url, timeout=120.0) as client:
                response = client.post(path, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            # Surface server-side errors (e.g. embedding quota) rather than
            # always claiming the service isn't running.
            body = exc.response.text.strip()
            msg = (
                f"Service error ({path}) against {self.base_url} (HTTP {exc.response.status_code})."
            )
            if body:
                msg += f"\n{body}"
            raise typer.BadParameter(msg) from exc
        except httpx.HTTPError as exc:
            msg = (
                f"Service request failed ({path}) against {self.base_url}. "
                "Start it with: hayfind serve"
            )
            raise typer.BadParameter(msg) from exc

    def get(self, path: str) -> dict:
        try:
            with httpx.Client(base_url=self.base_url, timeout=30.0) as client:
                response = client.get(path)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as exc:
            msg = (
                f"Service request failed ({path}) against {self.base_url}. "
                "Start it with: hayfind serve"
            )
            raise typer.BadParameter(msg) from exc


def _print_json(data: dict) -> None:
    typer.echo(json.dumps(data, indent=2))


def _index_repo_in_batches(
    client: ServiceClient,
    repo_path: Path,
    *,
    force: bool,
    batch_files: int,
    show_progress: bool,
    safe_mode: bool = False,
    force_restart: bool = False,
) -> dict:
    if safe_mode:
        os.environ["HAYFIND_EMBED_BATCH_SIZE"] = "10"
        os.environ["HAYFIND_EMBED_SLEEP_S"] = "2.0"
        batch_files = min(batch_files, 50)
        if show_progress:
            typer.echo("Safe mode: embed_batch_size=10, sleep=2.0s, cli_batch=50")

    # Check for an in-progress checkpoint and resume unless --force-restart
    start_offset = 0
    if not force_restart:
        try:
            status_data = client.get("/status")
            checkpoints = status_data.get("checkpoints", {})
            cp = checkpoints.get(repo_path.name)
            if cp and not cp.get("done", True):
                start_offset = int(cp.get("offset", 0))
                force = cp.get("force", force)
                if show_progress:
                    typer.echo(
                        f"Resuming {repo_path.name} from offset "
                        f"{start_offset}/{cp.get('total_files', '?')}"
                    )
        except Exception:  # noqa: BLE001
            pass  # service may not be running yet; proceed from 0

    payload = {
        "path": str(repo_path.expanduser().resolve()),
        "force": force,
        "batch_offset": start_offset,
        "batch_files": batch_files,
    }
    aggregate = {
        "repo": repo_path.name,
        "indexed_files": 0,
        "skipped_files": 0,
        "removed_files": 0,
        "chunk_count": 0,
        "total_files": 0,
        "processed_files": 0,
        "done": False,
        "error_count": 0,
        "error_files": [],
    }
    batch_number = 0
    while True:
        result = client.post("/index", payload)
        batch_number += 1

        aggregate["repo"] = result.get("repo", aggregate["repo"])
        aggregate["indexed_files"] += int(result.get("indexed_files", 0))
        aggregate["skipped_files"] += int(result.get("skipped_files", 0))
        aggregate["chunk_count"] += int(result.get("chunk_count", 0))
        aggregate["error_count"] += int(result.get("error_count", 0))
        aggregate["total_files"] = int(result.get("total_files") or aggregate["total_files"])
        aggregate["processed_files"] = int(
            result.get("processed_files") or aggregate["processed_files"]
        )
        for path in result.get("error_files", []):
            if path not in aggregate["error_files"] and len(aggregate["error_files"]) < 10:
                aggregate["error_files"].append(path)

        if show_progress:
            typer.echo(
                (
                    "Batch {batch}: {repo} processed {processed}/{total} files "
                    "(indexed={indexed}, skipped={skipped}, chunks={chunks}, errors={errors})"
                ).format(
                    batch=batch_number,
                    repo=aggregate["repo"],
                    processed=aggregate["processed_files"],
                    total=aggregate["total_files"],
                    indexed=result.get("indexed_files", 0),
                    skipped=result.get("skipped_files", 0),
                    chunks=result.get("chunk_count", 0),
                    errors=result.get("error_count", 0),
                )
            )

        if result.get("done", True):
            aggregate["done"] = True
            aggregate["removed_files"] = int(result.get("removed_files", 0))
            break

        next_offset = result.get("next_offset")
        if next_offset is None:
            raise typer.BadParameter("Indexing response missing next_offset while done=false")
        payload["batch_offset"] = int(next_offset)
    return aggregate


def _load_or_init_config() -> AppConfig:
    ensure_config_path(DEFAULT_CONFIG_PATH)
    return load_config(DEFAULT_CONFIG_PATH)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8765, reload: bool = False) -> None:
    """Run the local hayfind service."""
    uvicorn.run("hayfind.service:app", host=host, port=port, reload=reload)


@app.command()
def init_config() -> None:
    """Create the default config file if it does not exist."""
    path = ensure_config_path(DEFAULT_CONFIG_PATH)
    typer.echo(f"Config ready at {path}")


@app.command()
def index(
    path: str,
    force: bool = False,
    batch_files: int = typer.Option(
        int(os.getenv("HAYFIND_INDEX_BATCH_FILES", "200")),
        "--batch-files",
        min=1,
        help="Number of files to process per /index request.",
    ),
    safe_mode: bool = typer.Option(
        False,
        "--safe-mode",
        help="Ultra-conservative rate limits (batch_size=10, sleep=2.0s).",
    ),
    force_restart: bool = typer.Option(
        False,
        "--force-restart",
        help="Ignore existing checkpoint and start from offset 0.",
    ),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Index one repository path via service."""
    client = ServiceClient()
    result = _index_repo_in_batches(
        client,
        Path(path),
        force=force,
        batch_files=batch_files,
        show_progress=not as_json,
        safe_mode=safe_mode,
        force_restart=force_restart,
    )
    if as_json:
        _print_json(result)
        return
    msg = (
        "Indexed {repo}: files={indexed_files}, skipped={skipped_files}, "
        "removed={removed_files}, chunks={chunk_count}"
    ).format(**result)
    typer.echo(msg)


@app.command()
def search(
    query: str,
    repo: str | None = typer.Option(None, "--repo"),
    path_prefix: str | None = typer.Option(None, "--path-prefix"),
    top_k: int = typer.Option(5, "--top-k"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Search indexed files."""
    payload = {
        "query": query,
        "repo": repo,
        "path_prefix": path_prefix,
        "top_k": top_k,
    }
    result = ServiceClient().post("/search", payload)
    if as_json:
        _print_json(result)
        return

    hits = result.get("hits", [])
    if not hits:
        typer.echo("No matches found.")
        return

    for idx, hit in enumerate(hits, start=1):
        typer.echo(f"{idx}. [{hit['repo']}] {hit['path']} (score={hit['score']:.4f})")
        typer.echo(f"   {hit['snippet']}")


@app.command()
def status(as_json: bool = typer.Option(False, "--json")) -> None:
    """Show index status."""
    result = ServiceClient().get("/status")
    if as_json:
        _print_json(result)
        return

    typer.echo(f"Collection: {result['collection']}")
    typer.echo(f"Documents: {result['doc_count']}")
    typer.echo(f"Repos: {', '.join(result['repos']) if result['repos'] else '(none)'}")
    typer.echo(f"Last indexed: {result.get('last_indexed_at') or '(never)'}")
    checkpoints = result.get("checkpoints", {})
    if checkpoints:
        typer.echo("In-progress checkpoints:")
        for repo, cp in checkpoints.items():
            typer.echo(
                f"  {repo}: offset={cp['offset']}/{cp['total_files']} (started {cp['started_at']})"
            )


@app.command()
def reindex(
    safe_mode: bool = typer.Option(
        False,
        "--safe-mode",
        help="Ultra-conservative rate limits (batch_size=10, sleep=2.0s).",
    ),
    force_restart: bool = typer.Option(
        False,
        "--force-restart",
        help="Ignore existing checkpoint and start from offset 0.",
    ),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Force reindex for all repos configured in config.yaml."""
    config = _load_or_init_config()
    if not config.repos:
        raise typer.BadParameter(
            f"No repos configured in {DEFAULT_CONFIG_PATH}. Add repos then rerun."
        )

    client = ServiceClient()
    results: list[dict] = []
    for repo in config.repos:
        results.append(
            _index_repo_in_batches(
                client,
                repo.path,
                force=True,
                batch_files=int(os.getenv("HAYFIND_INDEX_BATCH_FILES", "200")),
                show_progress=not as_json,
                safe_mode=safe_mode,
                force_restart=force_restart,
            )
        )

    if as_json:
        _print_json({"results": results})
        return

    for result in results:
        msg = (
            "Reindexed {repo}: files={indexed_files}, skipped={skipped_files}, "
            "removed={removed_files}, chunks={chunk_count}"
        ).format(**result)
        typer.echo(msg)


def _write_hook(repo_path: Path) -> Path:
    hooks_dir = repo_path / ".git" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook_path = hooks_dir / "post-merge"
    hook_path.write_text(
        f"#!/usr/bin/env sh\n# Generated by hayfind install-hooks\nhayfind index {repo_path}\n",
        encoding="utf-8",
    )
    hook_path.chmod(0o755)
    return hook_path


@app.command("install-hooks")
def install_hooks() -> None:
    """Install post-merge git hooks for configured repos."""
    config = _load_or_init_config()
    if not config.repos:
        raise typer.BadParameter(f"No repos configured in {DEFAULT_CONFIG_PATH}")

    installed = 0
    for repo in config.repos:
        git_dir = repo.path / ".git"
        if not git_dir.exists():
            typer.echo(f"Skipping {repo.path} (not a git repo)")
            continue
        path = _write_hook(repo.path)
        typer.echo(f"Installed hook: {path}")
        installed += 1

    typer.echo(f"Hooks installed: {installed}")


if __name__ == "__main__":
    app()
