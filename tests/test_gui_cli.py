"""Tests for the `interpkit gui` CLI command."""

from __future__ import annotations

import builtins

from typer.testing import CliRunner

from interpkit.cli.main import app

runner = CliRunner()


def test_gui_command_exists():
    result = runner.invoke(app, ["gui", "--help"])
    assert result.exit_code == 0
    assert "--host" in result.output
    assert "--port" in result.output
    assert "--no-browser" in result.output


def test_gui_missing_extra_prints_hint(monkeypatch):
    """When fastapi/uvicorn are absent, the command prints an install hint
    and exits non-zero instead of a raw ImportError traceback."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("interpkit.gui") or name in ("fastapi", "uvicorn"):
            raise ImportError("simulated missing extra")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = runner.invoke(app, ["gui"])
    assert result.exit_code == 1
    assert "gui" in result.output.lower()
    assert "pip install" in result.output
