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
def index(path: str, force: bool = False, as_json: bool = typer.Option(False, "--json")) -> None:
    """Index one repository path via service."""
    payload = {"path": str(Path(path).expanduser().resolve()), "force": force}
    result = ServiceClient().post("/index", payload)
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


@app.command()
def reindex(as_json: bool = typer.Option(False, "--json")) -> None:
    """Force reindex for all repos configured in config.yaml."""
    config = _load_or_init_config()
    if not config.repos:
        raise typer.BadParameter(
            f"No repos configured in {DEFAULT_CONFIG_PATH}. Add repos then rerun."
        )

    client = ServiceClient()
    results: list[dict] = []
    for repo in config.repos:
        payload = {"path": str(repo.path), "force": True}
        results.append(client.post("/index", payload))

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
