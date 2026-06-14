"""Tests for the `interpkit gui` CLI command."""

from __future__ import annotations

import builtins
import re

from typer.testing import CliRunner

from interpkit.cli.main import app

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Strip ANSI color codes; Rich splits e.g. ``--host`` across them under color."""
    return _ANSI.sub("", text)


def test_gui_command_exists():
    result = runner.invoke(app, ["gui", "--help"])
    assert result.exit_code == 0
    output = _plain(result.output)
    assert "--host" in output
    assert "--port" in output
    assert "--no-browser" in output


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
    output = _plain(result.output)
    assert "gui" in output.lower()
    assert "pip install" in output
