from pathlib import Path

from typer.testing import CliRunner

from hayfind import cli

runner = CliRunner()


def test_install_hooks(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".git" / "hooks").mkdir(parents=True)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"repos:\n  - path: {repo}\n", encoding="utf-8")

    monkeypatch.setattr(cli, "DEFAULT_CONFIG_PATH", cfg)
    result = runner.invoke(cli.app, ["install-hooks"])

    assert result.exit_code == 0
    hook = repo / ".git" / "hooks" / "post-merge"
    assert hook.exists()
    assert "hayfind index" in hook.read_text(encoding="utf-8")
