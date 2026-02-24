from pathlib import Path

from hayfind.config import AppConfig, ensure_config_path, load_config


def test_ensure_and_load_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    ensure_config_path(cfg_path)
    assert cfg_path.exists()

    cfg = load_config(cfg_path)
    assert isinstance(cfg, AppConfig)
    assert len(cfg.repos) == 1
