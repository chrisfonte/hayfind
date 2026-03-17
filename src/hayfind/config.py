from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path("~/.config/hayfind/config.yaml").expanduser()
DEFAULT_DATA_DIR = Path("~/.local/share/hayfind").expanduser()


@dataclass(slots=True)
class RepoConfig:
    path: Path
    include: list[str] = field(default_factory=lambda: ["**/*"])
    exclude: list[str] = field(
        default_factory=lambda: [
            ".git/**",
            "**/.git/**",
            "**/node_modules/**",
            "**/.venv/**",
            "**/__pycache__/**",
        ]
    )


@dataclass(slots=True)
class AppConfig:
    repos: list[RepoConfig] = field(default_factory=list)


def ensure_config_path(path: Path = DEFAULT_CONFIG_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        default = {
            "repos": [
                {
                    "path": str(Path.cwd()),
                    "include": ["**/*"],
                    "exclude": [
                        ".git/**",
                        "**/.git/**",
                        "**/node_modules/**",
                        "**/.venv/**",
                        "**/__pycache__/**",
                    ],
                }
            ]
        }
        path.write_text(yaml.safe_dump(default, sort_keys=False), encoding="utf-8")
    return path


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    if not path.exists():
        return AppConfig()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    repos: list[RepoConfig] = []
    for entry in raw.get("repos", []):
        repo_path = Path(entry["path"]).expanduser().resolve()
        repos.append(
            RepoConfig(
                path=repo_path,
                include=list(entry.get("include", ["**/*"])),
                exclude=list(
                    entry.get(
                        "exclude",
                        [
                            ".git/**",
                            "**/.git/**",
                            "**/node_modules/**",
                            "**/.venv/**",
                            "**/__pycache__/**",
                        ],
                    )
                ),
            )
        )
    return AppConfig(repos=repos)


def data_dir() -> Path:
    path = Path(os.getenv("HAYFIND_DATA_DIR", str(DEFAULT_DATA_DIR))).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path
