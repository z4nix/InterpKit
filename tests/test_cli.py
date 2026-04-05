"""Tests for the interpkit CLI — argument parsing, output formatting, flags, errors."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch

import pytest
import torch
from typer.testing import CliRunner

from interpkit.cli.main import _json_dump, app

runner = CliRunner()


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_output_format():
    """Reset CLI output format between tests."""
    import interpkit.cli.main as cli_mod

    cli_mod._output_format = "rich"
    yield
    cli_mod._output_format = "rich"


@pytest.fixture()
def mock_model(gpt2_model):
    """Patch _load_model to return the session-scoped gpt2_model."""
    with patch("interpkit.cli.main._load_model", return_value=gpt2_model) as m:
        yield m


@pytest.fixture()
def mock_model_callable(gpt2_model):
    """Patch _load_model and capture call kwargs."""
    calls = []

    def _fake_load(model_name, device=None, dtype=None, device_map=None):
        calls.append(
            {"model_name": model_name, "device": device, "dtype": dtype, "device_map": device_map}
        )
        return gpt2_model

    with patch("interpkit.cli.main._load_model", side_effect=_fake_load):
        yield calls


# ══════════════════════════════════════════════════════════════════
# Help / no-args
# ══════════════════════════════════════════════════════════════════


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "InterpKit" in result.stdout or "interpkit" in result.stdout.lower()


def test_no_args_contains_commands():
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "scan" in result.stdout
    assert "inspect" in result.stdout
    assert "trace" in result.stdout


# ══════════════════════════════════════════════════════════════════
# Simple commands (positional args only)
# ══════════════════════════════════════════════════════════════════


def test_inspect(mock_model):
    result = runner.invoke(app, ["inspect", "gpt2"])
    assert result.exit_code == 0


def test_scan(mock_model):
    result = runner.invoke(app, ["scan", "gpt2", "The capital of France is"])
    assert result.exit_code == 0


def test_lens(mock_model):
    result = runner.invoke(app, ["lens", "gpt2", "The capital of France is"])
    assert result.exit_code == 0


def test_attribute(mock_model):
    result = runner.invoke(app, ["attribute", "gpt2", "The capital of France is"])
    assert result.exit_code == 0


def test_decompose(mock_model):
    result = runner.invoke(app, ["decompose", "gpt2", "The capital of France is"])
    assert result.exit_code == 0


def test_dla(mock_model):
    result = runner.invoke(app, ["dla", "gpt2", "The capital of France is"])
    assert result.exit_code == 0


# ══════════════════════════════════════════════════════════════════
# Commands with required --option flags
# ══════════════════════════════════════════════════════════════════


def test_trace(mock_model):
    result = runner.invoke(app, [
        "trace", "gpt2",
        "--clean", "The capital of France is",
        "--corrupted", "The capital of Germany is",
    ])
    assert result.exit_code == 0


def test_patch(mock_model):
    result = runner.invoke(app, [
        "patch", "gpt2",
        "--clean", "The capital of France is",
        "--corrupted", "The capital of Germany is",
        "--at", "transformer.h.8.mlp",
    ])
    assert result.exit_code == 0


def test_activations(mock_model):
    result = runner.invoke(app, [
        "activations", "gpt2", "hello world",
        "--at", "transformer.h.8",
    ])
    assert result.exit_code == 0


def test_ablate(mock_model):
    result = runner.invoke(app, [
        "ablate", "gpt2", "hello world",
        "--at", "transformer.h.8.mlp",
    ])
    assert result.exit_code == 0


def test_attention(mock_model):
    result = runner.invoke(app, [
        "attention", "gpt2", "hello world",
        "--layer", "0", "--head", "0",
    ])
    assert result.exit_code == 0


def test_steer(mock_model):
    result = runner.invoke(app, [
        "steer", "gpt2", "The weather is",
        "--positive", "Love",
        "--negative", "Hate",
        "--at", "transformer.h.8",
    ])
    assert result.exit_code == 0


def test_find_circuit(mock_model):
    result = runner.invoke(app, [
        "find-circuit", "gpt2",
        "--clean", "The capital of France is",
        "--corrupted", "The capital of Germany is",
    ])
    assert result.exit_code == 0


# ══════════════════════════════════════════════════════════════════
# Commands requiring external resources
# ══════════════════════════════════════════════════════════════════


def test_probe_with_json_file(mock_model):
    probe_data = {
        "texts": ["I love this", "I hate this", "Great movie", "Bad movie"] * 3,
        "labels": [1, 0, 1, 0] * 3,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(probe_data, f)
        f.flush()
        path = f.name

    try:
        result = runner.invoke(app, [
            "probe", "gpt2",
            "--at", "transformer.h.8",
            "--data", path,
        ])
        assert result.exit_code == 0
    finally:
        os.unlink(path)


def test_diff(gpt2_model):
    with patch("interpkit.cli.main._load_model", return_value=gpt2_model):
        result = runner.invoke(app, [
            "diff", "gpt2", "gpt2", "hello world",
        ])
        assert result.exit_code == 0


def test_report(mock_model):
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = f.name
    try:
        result = runner.invoke(app, [
            "report", "gpt2", "The capital of France is",
            "--save", path,
        ])
        assert result.exit_code == 0
        assert os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ══════════════════════════════════════════════════════════════════
# --format json
# ══════════════════════════════════════════════════════════════════


def _extract_json(stdout: str) -> dict | list:
    """Extract the JSON object from mixed Rich + JSON CLI output.

    The CLI prints Rich tables first, then appends the JSON via print().
    We find the outermost JSON object/array from the output.
    """
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = stdout.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(stdout)):
            if stdout[i] == start_char:
                depth += 1
            elif stdout[i] == end_char:
                depth -= 1
            if depth == 0:
                return json.loads(stdout[start : i + 1])
    raise ValueError(f"No JSON found in output:\n{stdout[:500]}")


def test_format_json_lens(mock_model):
    result = runner.invoke(app, [
        "--format", "json",
        "lens", "gpt2", "The capital of France is",
    ])
    assert result.exit_code == 0
    parsed = _extract_json(result.stdout)
    assert isinstance(parsed, (dict, list))


def test_format_json_dla(mock_model):
    result = runner.invoke(app, [
        "--format", "json",
        "dla", "gpt2", "The capital of France is",
    ])
    assert result.exit_code == 0
    parsed = _extract_json(result.stdout)
    assert isinstance(parsed, dict)


def test_format_json_scan(mock_model):
    result = runner.invoke(app, [
        "--format", "json",
        "scan", "gpt2", "The capital of France is",
    ])
    assert result.exit_code == 0
    parsed = _extract_json(result.stdout)
    assert isinstance(parsed, dict)


def test_format_json_decompose(mock_model):
    result = runner.invoke(app, [
        "--format", "json",
        "decompose", "gpt2", "The capital of France is",
    ])
    assert result.exit_code == 0
    parsed = _extract_json(result.stdout)
    assert isinstance(parsed, dict)


def test_format_json_trace(mock_model):
    result = runner.invoke(app, [
        "--format", "json",
        "trace", "gpt2",
        "--clean", "The capital of France is",
        "--corrupted", "The capital of Germany is",
    ])
    assert result.exit_code == 0
    parsed = _extract_json(result.stdout)
    assert isinstance(parsed, (dict, list))


def test_format_json_attention(mock_model):
    result = runner.invoke(app, [
        "--format", "json",
        "attention", "gpt2", "hello world",
        "--layer", "0", "--head", "0",
    ])
    assert result.exit_code == 0
    parsed = _extract_json(result.stdout)
    assert isinstance(parsed, dict)


def test_format_json_ablate(mock_model):
    result = runner.invoke(app, [
        "--format", "json",
        "ablate", "gpt2", "hello world",
        "--at", "transformer.h.8.mlp",
    ])
    assert result.exit_code == 0
    parsed = _extract_json(result.stdout)
    assert isinstance(parsed, dict)


def test_format_json_patch(mock_model):
    result = runner.invoke(app, [
        "--format", "json",
        "patch", "gpt2",
        "--clean", "hello world",
        "--corrupted", "goodbye world",
        "--at", "transformer.h.8.mlp",
    ])
    assert result.exit_code == 0
    parsed = _extract_json(result.stdout)
    assert isinstance(parsed, dict)


def test_format_json_attribute(mock_model):
    result = runner.invoke(app, [
        "--format", "json",
        "attribute", "gpt2", "The capital of France is",
    ])
    assert result.exit_code == 0
    parsed = _extract_json(result.stdout)
    assert isinstance(parsed, dict)


# ══════════════════════════════════════════════════════════════════
# --save / --html flag handling
# ══════════════════════════════════════════════════════════════════


def test_save_lens(mock_model):
    path = os.path.join(tempfile.gettempdir(), "cli_lens_test.png")
    try:
        result = runner.invoke(app, [
            "lens", "gpt2", "The capital of France is",
            "--save", path,
        ])
        assert result.exit_code == 0
        assert os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_save_trace(mock_model):
    path = os.path.join(tempfile.gettempdir(), "cli_trace_test.png")
    try:
        result = runner.invoke(app, [
            "trace", "gpt2",
            "--clean", "The capital of France is",
            "--corrupted", "The capital of Germany is",
            "--save", path,
        ])
        assert result.exit_code == 0
        assert os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_save_attention(mock_model):
    path = os.path.join(tempfile.gettempdir(), "cli_attn_test.png")
    try:
        result = runner.invoke(app, [
            "attention", "gpt2", "hello world",
            "--layer", "0", "--head", "0",
            "--save", path,
        ])
        assert result.exit_code == 0
        assert os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_save_attribute(mock_model):
    path = os.path.join(tempfile.gettempdir(), "cli_attr_test.png")
    try:
        result = runner.invoke(app, [
            "attribute", "gpt2", "The capital of France is",
            "--save", path,
        ])
        assert result.exit_code == 0
        assert os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_save_dla(mock_model):
    path = os.path.join(tempfile.gettempdir(), "cli_dla_test.png")
    try:
        result = runner.invoke(app, [
            "dla", "gpt2", "The capital of France is",
            "--save", path,
        ])
        assert result.exit_code == 0
        assert os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_html_lens(mock_model):
    path = os.path.join(tempfile.gettempdir(), "cli_lens_test.html")
    try:
        result = runner.invoke(app, [
            "lens", "gpt2", "The capital of France is",
            "--html", path,
        ])
        assert result.exit_code == 0
        assert os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_html_attention(mock_model):
    path = os.path.join(tempfile.gettempdir(), "cli_attn_test.html")
    try:
        result = runner.invoke(app, [
            "attention", "gpt2", "hello world",
            "--html", path,
        ])
        assert result.exit_code == 0
        assert os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_html_attribute(mock_model):
    path = os.path.join(tempfile.gettempdir(), "cli_attr_test.html")
    try:
        result = runner.invoke(app, [
            "attribute", "gpt2", "The capital of France is",
            "--html", path,
        ])
        assert result.exit_code == 0
        assert os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_html_dla(mock_model):
    path = os.path.join(tempfile.gettempdir(), "cli_dla_test.html")
    try:
        result = runner.invoke(app, [
            "dla", "gpt2", "The capital of France is",
            "--html", path,
        ])
        assert result.exit_code == 0
        assert os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ══════════════════════════════════════════════════════════════════
# Flag forwarding to _load_model
# ══════════════════════════════════════════════════════════════════


def test_device_flag_forwarded(mock_model_callable):
    runner.invoke(app, ["inspect", "gpt2", "--device", "cpu"])
    assert mock_model_callable[-1]["device"] == "cpu"


def test_dtype_flag_forwarded(mock_model_callable):
    runner.invoke(app, ["inspect", "gpt2", "--dtype", "float16"])
    assert mock_model_callable[-1]["dtype"] == "float16"


def test_device_map_flag_forwarded(mock_model_callable):
    runner.invoke(app, ["inspect", "gpt2", "--device-map", "auto"])
    assert mock_model_callable[-1]["device_map"] == "auto"


def test_all_model_flags_forwarded(mock_model_callable):
    runner.invoke(app, [
        "lens", "gpt2", "hello",
        "--device", "cpu",
        "--dtype", "bfloat16",
        "--device-map", "auto",
    ])
    call = mock_model_callable[-1]
    assert call["device"] == "cpu"
    assert call["dtype"] == "bfloat16"
    assert call["device_map"] == "auto"


# ══════════════════════════════════════════════════════════════════
# CLI-specific argument parsing
# ══════════════════════════════════════════════════════════════════


def test_trace_top_k_zero_means_scan_all(mock_model):
    """--top-k 0 on trace should set effective_top_k = None."""
    with patch.object(mock_model.return_value, "trace", wraps=mock_model.return_value.trace) as spy:
        result = runner.invoke(app, [
            "trace", "gpt2",
            "--clean", "hello", "--corrupted", "goodbye",
            "--top-k", "0",
        ])
        assert result.exit_code == 0
        spy.assert_called_once()
        _, kwargs = spy.call_args
        assert kwargs["top_k"] is None


def test_dla_token_as_integer(mock_model):
    """--token 42 should be parsed as int."""
    with patch.object(mock_model.return_value, "dla", wraps=mock_model.return_value.dla) as spy:
        result = runner.invoke(app, [
            "dla", "gpt2", "hello",
            "--token", "42",
        ])
        assert result.exit_code == 0
        spy.assert_called_once()
        _, kwargs = spy.call_args
        assert kwargs["token"] == 42
        assert isinstance(kwargs["token"], int)


def test_dla_token_as_string(mock_model):
    """--token Paris should remain a string."""
    with patch.object(mock_model.return_value, "dla", wraps=mock_model.return_value.dla) as spy:
        result = runner.invoke(app, [
            "dla", "gpt2", "hello",
            "--token", "Paris",
        ])
        assert result.exit_code == 0
        spy.assert_called_once()
        _, kwargs = spy.call_args
        assert kwargs["token"] == "Paris"
        assert isinstance(kwargs["token"], str)


def test_activations_comma_separated_modules(mock_model):
    """--at with commas should split into multiple modules."""
    with patch.object(
        mock_model.return_value, "activations",
        wraps=mock_model.return_value.activations,
    ) as spy:
        result = runner.invoke(app, [
            "activations", "gpt2", "hello",
            "--at", "transformer.h.0,transformer.h.1",
        ])
        assert result.exit_code == 0
        spy.assert_called_once()
        _, kwargs = spy.call_args
        assert kwargs["at"] == ["transformer.h.0", "transformer.h.1"]


# ══════════════════════════════════════════════════════════════════
# Error / edge-case tests
# ══════════════════════════════════════════════════════════════════


def test_missing_required_positional_arg():
    result = runner.invoke(app, ["inspect"])
    assert result.exit_code != 0


def test_trace_missing_clean_flag():
    with patch("interpkit.cli.main._load_model"):
        result = runner.invoke(app, ["trace", "gpt2"])
        assert result.exit_code != 0


def test_patch_missing_at_flag():
    with patch("interpkit.cli.main._load_model"):
        result = runner.invoke(app, [
            "patch", "gpt2",
            "--clean", "hello",
            "--corrupted", "goodbye",
        ])
        assert result.exit_code != 0


def test_ablate_missing_at_flag():
    with patch("interpkit.cli.main._load_model"):
        result = runner.invoke(app, [
            "ablate", "gpt2", "hello",
        ])
        assert result.exit_code != 0


def test_steer_missing_positive_flag():
    with patch("interpkit.cli.main._load_model"):
        result = runner.invoke(app, [
            "steer", "gpt2", "hello",
            "--negative", "bad",
            "--at", "transformer.h.8",
        ])
        assert result.exit_code != 0


# ══════════════════════════════════════════════════════════════════
# _json_dump unit tests
# ══════════════════════════════════════════════════════════════════


def test_json_dump_tensor(capsys):
    _json_dump({"x": torch.tensor([1.0, 2.0, 3.0])})
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["x"] == [1.0, 2.0, 3.0]


def test_json_dump_scalar_tensor(capsys):
    _json_dump({"x": torch.tensor(42.0)})
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["x"] == 42.0


def test_json_dump_float_castable(capsys):
    import numpy as np

    _json_dump({"x": np.float32(3.14)})
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert abs(parsed["x"] - 3.14) < 0.01


def test_json_dump_non_serializable_fallback(capsys):
    _json_dump({"x": object()})
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert isinstance(parsed["x"], str)


def test_json_dump_nested_mixed(capsys):
    _json_dump({
        "tensor": torch.tensor([1, 2]),
        "string": "hello",
        "number": 42,
        "nested": {"t": torch.tensor(0.5)},
    })
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["tensor"] == [1, 2]
    assert parsed["string"] == "hello"
    assert parsed["number"] == 42
    assert parsed["nested"]["t"] == 0.5


def test_json_dump_empty_tensor(capsys):
    _json_dump({"x": torch.tensor([])})
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["x"] == []


def test_json_dump_multidim_tensor(capsys):
    _json_dump({"x": torch.ones(2, 3)})
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["x"] == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
