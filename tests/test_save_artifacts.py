"""Save/output artifact validation across architectures.

Existing tests in test_plots.py and test_html.py only use GPT-2 and only
check file existence.  These tests validate file content (PNG magic bytes,
HTML markers) across diverse architectures.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

TEXT = "The capital of France is"
PAIR_CLEAN = "The Eiffel Tower is in Paris"
PAIR_CORRUPT = "The Eiffel Tower is in Rome"

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _first_layer(model):
    layers = model.arch_info.layer_names
    if not layers:
        pytest.skip("No layers detected")
    return layers[0]


def _assert_valid_png(path: str) -> None:
    assert os.path.exists(path), f"File not created: {path}"
    with open(path, "rb") as f:
        header = f.read(8)
    assert header == PNG_MAGIC, f"Not a valid PNG: header={header!r}"


def _assert_valid_html(path: str) -> None:
    assert os.path.exists(path), f"File not created: {path}"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read(2000)
    assert "<!DOCTYPE" in content or "<html" in content, (
        f"File does not contain HTML markers: {path}"
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Attention save / html
# ═══════════════════════════════════════════════════════════════════════════


def test_attention_save_valid_image_bart(bart_model):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "attn.png")
        bart_model.attention(TEXT, layer=0, head=0, save=path)
        _assert_valid_png(path)


def test_attention_html_valid_smollm(smollm_model):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "attn.html")
        smollm_model.attention(TEXT, layer=0, html=path)
        _assert_valid_html(path)


# ═══════════════════════════════════════════════════════════════════════════
#  Lens save / html
# ═══════════════════════════════════════════════════════════════════════════


def test_lens_save_image_pythia(pythia_model):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "lens.png")
        pythia_model.lens(TEXT, save=path)
        _assert_valid_png(path)


def test_lens_html_valid_gpt_neo(gpt_neo_model):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "lens.html")
        gpt_neo_model.lens(TEXT, html=path)
        _assert_valid_html(path)


# ═══════════════════════════════════════════════════════════════════════════
#  Attribute save / html
# ═══════════════════════════════════════════════════════════════════════════


def test_attribute_save_image_electra(electra_model):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "attr.png")
        electra_model.attribute(TEXT, save=path)
        _assert_valid_png(path)


def test_attribute_html_valid_opt(opt_model):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "attr.html")
        opt_model.attribute(TEXT, html=path)
        _assert_valid_html(path)


# ═══════════════════════════════════════════════════════════════════════════
#  DLA save / html
# ═══════════════════════════════════════════════════════════════════════════


def test_dla_save_image_gpt2(gpt2_model):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dla.png")
        gpt2_model.dla(TEXT, save=path)
        _assert_valid_png(path)


def test_dla_html_valid_smollm(smollm_model):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dla.html")
        smollm_model.dla(TEXT, html=path)
        _assert_valid_html(path)


# ═══════════════════════════════════════════════════════════════════════════
#  Scan save
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.timeout(120)
def test_scan_save_prefix_gpt2(gpt2_model):
    with tempfile.TemporaryDirectory() as d:
        prefix = os.path.join(d, "scan_out")
        gpt2_model.scan(TEXT, save=prefix)
        files = os.listdir(d)
        assert len(files) > 0, "scan(save=prefix) should create at least one file"


# ═══════════════════════════════════════════════════════════════════════════
#  Trace save
# ═══════════════════════════════════════════════════════════════════════════


def test_trace_save_image_albert(albert_model):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trace.png")
        albert_model.trace(PAIR_CLEAN, PAIR_CORRUPT, top_k=3, save=path)
        _assert_valid_png(path)


# ═══════════════════════════════════════════════════════════════════════════
#  Steer save
# ═══════════════════════════════════════════════════════════════════════════


def test_steer_save_image_pythia(pythia_model):
    layer = _first_layer(pythia_model)
    vec = pythia_model.steer_vector("happy", "sad", at=layer)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "steer.png")
        pythia_model.steer(TEXT, vector=vec, at=layer, save=path)
        _assert_valid_png(path)


# ═══════════════════════════════════════════════════════════════════════════
#  Report HTML
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.timeout(120)
def test_report_valid_html_smollm(smollm_model):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "report.html")
        smollm_model.report(TEXT, save=path)
        _assert_valid_html(path)
