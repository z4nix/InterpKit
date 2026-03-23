"""Tests for interactive HTML visualization generation."""

from __future__ import annotations

import os
import tempfile

from interpkit.core.html import html_attention, html_attribution, html_trace, save_html


def test_html_attention_generates_valid_html():
    attention_data = [
        {
            "layer": 0,
            "head": 0,
            "weights": [[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]],
            "entropy": 1.05,
            "top_pairs": [(0, 0, 0.7), (1, 1, 0.6)],
        },
        {
            "layer": 0,
            "head": 1,
            "weights": [[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.1, 0.4, 0.5]],
            "entropy": 1.10,
            "top_pairs": [(0, 0, 0.5), (2, 2, 0.5)],
        },
    ]
    tokens = ["The", "capital", "is"]

    html_str = html_attention(attention_data, tokens)

    assert "<!DOCTYPE html>" in html_str
    assert "Attention" in html_str
    assert "The" in html_str
    assert "capital" in html_str
    # Heads are rendered via JS; verify data is embedded
    assert '"head": 0' in html_str
    assert '"head": 1' in html_str


def test_html_attention_empty():
    html_str = html_attention([], None)
    assert "No attention data" in html_str


def test_html_trace_generates_valid_html():
    results = [
        {"module": "transformer.h.8.mlp", "effect": 0.42, "role": "mlp"},
        {"module": "transformer.h.5.attn", "effect": 0.35, "role": "attention"},
        {"module": "transformer.h.2.mlp", "effect": 0.12, "role": "mlp"},
    ]

    html_str = html_trace(results)

    assert "<!DOCTYPE html>" in html_str
    assert "Causal Trace" in html_str
    assert "transformer.h.8.mlp" in html_str
    assert "0.42" in html_str


def test_html_trace_empty():
    html_str = html_trace([])
    assert "No results" in html_str


def test_html_attribution_generates_valid_html():
    tokens = ["The", "capital", "of", "France", "is"]
    scores = [0.1, 0.3, 0.05, 0.8, 0.2]

    html_str = html_attribution(tokens, scores)

    assert "<!DOCTYPE html>" in html_str
    assert "Attribution" in html_str
    assert "France" in html_str
    assert "Threshold" in html_str


def test_html_attribution_empty():
    html_str = html_attribution([], [])
    assert "No data" in html_str


def test_save_html_writes_file():
    content = "<html><body>test</body></html>"

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = f.name

    try:
        save_html(content, path)
        assert os.path.exists(path)

        with open(path) as f:
            saved = f.read()
        assert saved == content
    finally:
        os.unlink(path)


def test_html_attention_saves_via_op(gpt2_model):
    """Test that attention op correctly generates HTML via html= parameter."""
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = f.name

    try:
        gpt2_model.attention("Hello world", layer=0, html=path)
        assert os.path.exists(path)

        with open(path) as f:
            content = f.read()
        assert "<!DOCTYPE html>" in content
        assert "Attention" in content
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_html_attribution_saves_via_op(gpt2_model):
    """Test that attribute op correctly generates HTML via html= parameter."""
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = f.name

    try:
        gpt2_model.attribute("Hello world", html=path)
        assert os.path.exists(path)

        with open(path) as f:
            content = f.read()
        assert "<!DOCTYPE html>" in content
        assert "Attribution" in content
    finally:
        if os.path.exists(path):
            os.unlink(path)
