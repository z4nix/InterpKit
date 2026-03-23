"""Extended vision model tests — operations never tested on ViT.

Covers gap 3 from the reliability audit: many operations were only tested
on language models.  This file exercises them on google/vit-base-patch16-224.

Operations that require an LM head (lens, dla, steer text generation) may
return None or raise; those cases are handled with xfail or None-checks.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
from PIL import Image


def _first_layer(model):
    layers = model.arch_info.layer_names
    if not layers:
        pytest.skip("No layers detected")
    return layers[0]


def _mid_layer(model):
    layers = model.arch_info.layer_names
    if not layers:
        pytest.skip("No layers detected")
    return layers[len(layers) // 2]


def _first_attn(model):
    for m in model.arch_info.modules:
        if m.role == "attention":
            return m.name
    pytest.skip("No attention module detected")


def _has_two_attn_layers(model):
    infos = model.arch_info.layer_infos
    return sum(1 for li in infos if li.attn_path is not None) >= 2


@pytest.fixture(scope="module")
def second_image_path():
    """Generate a distinct 224x224 solid-blue image for corrupted/second inputs."""
    img = Image.new("RGB", (224, 224), color=(0, 0, 255))
    path = os.path.join(tempfile.gettempdir(), "interpkit_test_img_blue.jpg")
    img.save(path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  Head activations
# ═══════════════════════════════════════════════════════════════════════════


def test_head_activations_vit(vit_model, test_image_path):
    at = _first_attn(vit_model)
    result = vit_model.head_activations(test_image_path, at=at)
    assert "head_acts" in result
    assert result["num_heads"] > 0
    assert isinstance(result["head_acts"], torch.Tensor)


def test_head_activations_vit_no_output_proj(vit_model, test_image_path):
    at = _first_attn(vit_model)
    result = vit_model.head_activations(test_image_path, at=at, output_proj=False)
    assert "head_acts" in result
    assert result["num_heads"] > 0


# ═══════════════════════════════════════════════════════════════════════════
#  Steer
# ═══════════════════════════════════════════════════════════════════════════


def test_steer_vit(vit_model, test_image_path):
    layer = _first_layer(vit_model)
    act = vit_model.activations(test_image_path, at=layer)
    vec = torch.randn(act.shape[-1])
    result = vit_model.steer(test_image_path, vector=vec, at=layer, scale=1.0)
    assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════
#  Patch
# ═══════════════════════════════════════════════════════════════════════════


def test_patch_vit(vit_model, test_image_path, second_image_path):
    result = vit_model.patch(
        test_image_path, second_image_path,
        at=_first_layer(vit_model),
    )
    assert "effect" in result
    assert isinstance(result["effect"], float)


# ═══════════════════════════════════════════════════════════════════════════
#  Trace
# ═══════════════════════════════════════════════════════════════════════════


def test_trace_vit(vit_model, test_image_path, second_image_path):
    results = vit_model.trace(test_image_path, second_image_path, top_k=3)
    assert isinstance(results, (list, dict))


# ═══════════════════════════════════════════════════════════════════════════
#  Decompose
# ═══════════════════════════════════════════════════════════════════════════


def test_decompose_vit(vit_model, test_image_path):
    result = vit_model.decompose(test_image_path)
    assert "components" in result
    assert isinstance(result["components"], list)


# ═══════════════════════════════════════════════════════════════════════════
#  Scan
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.timeout(120)
def test_scan_vit(vit_model, test_image_path):
    result = vit_model.scan(test_image_path)
    assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════
#  Report
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.timeout(120)
def test_report_vit(vit_model, test_image_path):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "vit_report.html")
        result = vit_model.report(test_image_path, save=path)
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════
#  Probe
# ═══════════════════════════════════════════════════════════════════════════


def test_probe_vit(vit_model, test_image_path, second_image_path):
    images = [test_image_path, second_image_path] * 5
    labels = [0, 1] * 5
    result = vit_model.probe(images, labels, at=_first_layer(vit_model))
    assert "accuracy" in result


# ═══════════════════════════════════════════════════════════════════════════
#  Batch
# ═══════════════════════════════════════════════════════════════════════════


def test_batch_activations_vit(vit_model, test_image_path, second_image_path):
    dataset = [
        {"input_data": test_image_path},
        {"input_data": second_image_path},
    ]
    result = vit_model.batch(
        "activations", dataset,
        op_kwargs={"at": _first_layer(vit_model)},
    )
    assert "results" in result
    assert result["count"] == 2


# ═══════════════════════════════════════════════════════════════════════════
#  Find circuit
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.timeout(180)
def test_find_circuit_vit(vit_model, test_image_path, second_image_path):
    result = vit_model.find_circuit(
        test_image_path, second_image_path, threshold=0.05,
    )
    assert "circuit" in result
    assert isinstance(result["circuit"], list)
    assert "excluded" in result


# ═══════════════════════════════════════════════════════════════════════════
#  Composition, OV scores, QK scores
# ═══════════════════════════════════════════════════════════════════════════


def test_composition_vit(vit_model):
    if not _has_two_attn_layers(vit_model):
        pytest.skip("Not enough attention layers")
    result = vit_model.composition(src_layer=0, dst_layer=1, comp_type="q")
    assert "scores" in result


def test_ov_scores_vit(vit_model):
    result = vit_model.ov_scores(layer=0)
    assert "heads" in result
    assert len(result["heads"]) > 0


def test_qk_scores_vit(vit_model):
    result = vit_model.qk_scores(layer=0)
    assert "heads" in result


# ═══════════════════════════════════════════════════════════════════════════
#  Lens and DLA — may not apply to vision models (no LM head)
# ═══════════════════════════════════════════════════════════════════════════


def test_lens_vit(vit_model, test_image_path):
    results = vit_model.lens(test_image_path)
    assert results is None or isinstance(results, list)


def test_dla_vit(vit_model, test_image_path):
    if not vit_model.arch_info.is_language_model:
        pytest.skip("ViT is not a language model — DLA requires unembedding")
    result = vit_model.dla(test_image_path)
    assert "contributions" in result
