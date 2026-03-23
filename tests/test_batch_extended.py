"""Extended batch API and SAE feature tests.

Covers gaps 5 and 6 from the reliability audit:
  - batch() with operations beyond "attention": ablate, lens, attribute, dla, activations
  - aggregate=False
  - op_kwargs passthrough
  - trace_batch with mode="position"
  - dla_batch on different models
  - SAE features with attribute=True
  - SAE on non-GPT-2 architectures
"""

from __future__ import annotations

import pytest
import torch

from interpkit.ops.sae import load_sae_from_tensors

TEXT = "The capital of France is"
TEXT_B = "The capital of Germany is"
PAIR_CLEAN = "The Eiffel Tower is in Paris"
PAIR_CORRUPT = "The Eiffel Tower is in Rome"


def _first_layer(model):
    layers = model.arch_info.layer_names
    if not layers:
        pytest.skip("No layers detected")
    return layers[0]


def _first_attn(model):
    for m in model.arch_info.modules:
        if m.role == "attention":
            return m.name
    pytest.skip("No attention module detected")


def _first_mlp(model):
    for m in model.arch_info.modules:
        if m.role == "mlp":
            return m.name
    pytest.skip("No MLP module detected")


def _make_sae_for_model(model, d_sae: int = 64):
    """Build a synthetic SAE matching the model's hidden dimension and device."""
    d_in = model.arch_info.hidden_size
    if d_in is None:
        pytest.skip("hidden_size not detected")
    device = model._device
    return load_sae_from_tensors(
        W_enc=torch.randn(d_in, d_sae, device=device) * 0.01,
        W_dec=torch.randn(d_sae, d_in, device=device) * 0.01,
        b_enc=torch.zeros(d_sae, device=device),
        b_dec=torch.zeros(d_in, device=device),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Batch API depth — different operations
# ═══════════════════════════════════════════════════════════════════════════


def test_batch_ablate_gpt_neo(gpt_neo_model):
    dataset = [
        {"input_data": TEXT},
        {"input_data": TEXT_B},
    ]
    result = gpt_neo_model.batch(
        "ablate", dataset,
        op_kwargs={"at": _first_layer(gpt_neo_model), "method": "zero"},
    )
    assert "results" in result
    assert result["count"] == 2
    assert "summary" in result
    assert "mean_effect" in result["summary"]


@pytest.mark.timeout(120)
def test_batch_lens_smollm(smollm_model):
    dataset = [
        {"text": TEXT},
        {"text": TEXT_B},
    ]
    result = smollm_model.batch("lens", dataset)
    assert "results" in result
    assert result["count"] == 2


@pytest.mark.timeout(120)
def test_batch_attribute_bart(bart_model):
    dataset = [
        {"input_data": TEXT},
        {"input_data": TEXT_B},
    ]
    result = bart_model.batch(
        "attribute", dataset,
        op_kwargs={"method": "gradient"},
    )
    assert "results" in result
    assert result["count"] == 2


def test_batch_dla_pythia(pythia_model):
    dataset = [
        {"input_data": TEXT},
        {"input_data": TEXT_B},
    ]
    result = pythia_model.batch("dla", dataset)
    assert "results" in result
    assert result["count"] == 2


def test_batch_activations_electra(electra_model):
    dataset = [
        {"input_data": TEXT},
        {"input_data": TEXT_B},
    ]
    result = electra_model.batch(
        "activations", dataset,
        op_kwargs={"at": _first_layer(electra_model)},
    )
    assert "results" in result
    assert result["count"] == 2


# ═══════════════════════════════════════════════════════════════════════════
#  Batch API — aggregate=False and op_kwargs
# ═══════════════════════════════════════════════════════════════════════════


def test_batch_aggregate_false_gpt2(gpt2_model):
    dataset = [
        {"input_data": "Hello world"},
        {"input_data": "The sky is blue"},
    ]
    result = gpt2_model.batch("attention", dataset, aggregate=False)
    assert "results" in result
    assert result["count"] == 2
    assert "summary" not in result


def test_batch_op_kwargs_gpt2(gpt2_model):
    dataset = [
        {"input_data": TEXT},
        {"input_data": TEXT_B},
    ]
    result = gpt2_model.batch(
        "ablate", dataset,
        op_kwargs={"at": _first_layer(gpt2_model), "method": "mean"},
    )
    assert "results" in result
    assert result["count"] == 2
    for r in result["results"]:
        assert r["method"] == "mean"


# ═══════════════════════════════════════════════════════════════════════════
#  trace_batch — mode="position" and new models
# ═══════════════════════════════════════════════════════════════════════════


def test_trace_batch_mode_position_smollm(smollm_model):
    dataset = [
        {"clean": PAIR_CLEAN, "corrupted": PAIR_CORRUPT},
        {"clean": "Paris is in France", "corrupted": "Paris is in Germany"},
    ]
    result = smollm_model.trace_batch(dataset, top_k=3, mode="position")
    assert "results" in result
    assert result["count"] == 2


def test_trace_batch_albert(albert_model):
    dataset = [
        {"clean": PAIR_CLEAN, "corrupted": PAIR_CORRUPT},
        {"clean": "Berlin is in Germany", "corrupted": "Berlin is in France"},
    ]
    result = albert_model.trace_batch(dataset, top_k=3)
    assert "results" in result
    assert result["count"] == 2


# ═══════════════════════════════════════════════════════════════════════════
#  dla_batch — new model
# ═══════════════════════════════════════════════════════════════════════════


def test_dla_batch_opt(opt_model):
    texts = [TEXT, TEXT_B]
    result = opt_model.dla_batch(texts, top_k=3)
    assert "results" in result
    assert result["count"] == 2


# ═══════════════════════════════════════════════════════════════════════════
#  SAE / Features — attribute=True
# ═══════════════════════════════════════════════════════════════════════════


def test_features_attribute_true_gpt2(gpt2_model):
    sae = _make_sae_for_model(gpt2_model, d_sae=64)
    result = gpt2_model.features(
        TEXT,
        at="transformer.h.8.mlp",
        sae=sae,
        top_k=5,
        attribute=True,
    )
    assert "top_features" in result
    assert "feature_attributions" in result
    assert isinstance(result["feature_attributions"], list)
    assert len(result["feature_attributions"]) > 0

    entry = result["feature_attributions"][0]
    assert "feature_idx" in entry
    assert "activation" in entry


# ═══════════════════════════════════════════════════════════════════════════
#  SAE / Features — non-GPT-2 architectures
# ═══════════════════════════════════════════════════════════════════════════


def test_features_non_gpt2_pythia(pythia_model):
    sae = _make_sae_for_model(pythia_model, d_sae=64)
    mlp = _first_mlp(pythia_model)
    result = pythia_model.features(
        TEXT, at=mlp, sae=sae, top_k=5,
    )
    assert "top_features" in result
    assert "reconstruction_error" in result
    assert "sparsity" in result
    assert result["total_features"] == 64


def test_features_non_gpt2_smollm(smollm_model):
    sae = _make_sae_for_model(smollm_model, d_sae=64)
    mlp = _first_mlp(smollm_model)
    result = smollm_model.features(
        TEXT, at=mlp, sae=sae, top_k=5,
    )
    assert "top_features" in result
    assert result["total_features"] == 64


def test_features_attribute_pythia(pythia_model):
    """SAE with attribute=True on a non-GPT-2 model."""
    sae = _make_sae_for_model(pythia_model, d_sae=64)
    mlp = _first_mlp(pythia_model)
    result = pythia_model.features(
        TEXT, at=mlp, sae=sae, top_k=5, attribute=True,
    )
    assert "top_features" in result
    assert "feature_attributions" in result
