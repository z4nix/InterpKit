"""Tests for Edge Attribution Patching (model.eap) and find_circuit(method="eap")."""

from __future__ import annotations

import math

import pytest

CLEAN = "The capital of France is"
CORRUPTED = "The capital of Germany is"


def test_eap_schema_and_edge_ordering(gpt2_model):
    result = gpt2_model.eap(CLEAN, CORRUPTED, top_k_edges=15)
    edges = result["edges"]
    assert len(edges) == 15
    for i, e in enumerate(edges):
        assert set(e) >= {"src", "dst", "src_layer", "dst_layer", "score", "rank"}
        assert e["rank"] == i
        # Edges only flow forward: a component can only feed residual
        # layers at or after its own layer.
        assert e["src_layer"] <= e["dst_layer"]
        assert math.isfinite(e["score"])
    scores = [abs(e["score"]) for e in edges]
    assert scores == sorted(scores, reverse=True)

    nodes = result["nodes"]
    kinds = {n["type"] for n in nodes}
    assert {"attn", "mlp", "embed"} <= kinds
    assert result["meta"]["method"] == "eap"
    assert result["meta"]["n_forward_passes"] == 2


def test_eap_deterministic(gpt2_model):
    a = gpt2_model.eap(CLEAN, CORRUPTED, top_k_edges=5)
    b = gpt2_model.eap(CLEAN, CORRUPTED, top_k_edges=5)
    assert [(e["src"], e["dst"], e["score"]) for e in a["edges"]] == [
        (e["src"], e["dst"], e["score"]) for e in b["edges"]
    ]


def test_eap_node_score_matches_atp_at_injection_layer(gpt2_model):
    """In a pre-LN block the residual add gives grad(resid_l) == grad(mlp_out),
    so the EAP node score for L{l}.mlp must equal the AtP score computed at
    the same module — a cross-implementation correctness check.

    Uses ``compute_atp_scores`` directly because ``model.atp``'s candidate
    list filters to param-bearing modules, which excludes the ``.mlp``
    container (its params live in the c_fc/c_proj children).
    """
    from interpkit.ops.atp import compute_atp_scores

    eap_nodes = {
        n["node"]: n for n in gpt2_model.eap(CLEAN, CORRUPTED)["nodes"]
    }
    paths = ["transformer.h.4.mlp", "transformer.h.10.mlp"]
    ci, ri = gpt2_model._prepare_pair(CLEAN, CORRUPTED)
    atp_scores = compute_atp_scores(gpt2_model, ci, ri, paths)

    checked = 0
    for layer in (4, 10):
        node = eap_nodes.get(f"L{layer}.mlp")
        atp = atp_scores.get(f"transformer.h.{layer}.mlp")
        if node is None or atp is None or math.isnan(atp):
            continue
        assert node["score"] == pytest.approx(atp, rel=1e-3, abs=1e-4)
        checked += 1
    assert checked > 0


def test_eap_ig_consistent_with_plain_eap(gpt2_model):
    plain = gpt2_model.eap(CLEAN, CORRUPTED, top_k_edges=None)
    ig = gpt2_model.eap(CLEAN, CORRUPTED, ig_steps=2, top_k_edges=None)
    assert ig["meta"]["method"] == "eap-ig"
    assert ig["meta"]["n_backward_passes"] == 2
    assert all(math.isfinite(e["score"]) for e in ig["edges"])
    # IG refines magnitudes but the dominant node should stay near the top.
    plain_top3 = {n["node"] for n in plain["nodes"][:3]}
    assert ig["nodes"][0]["node"] in plain_top3


def test_eap_rejects_misaligned_pair(gpt2_model):
    with pytest.raises(ValueError, match="token-aligned"):
        gpt2_model.eap("Hi", "The quick brown fox jumps over the lazy dog")


def test_eap_rejects_bad_args(gpt2_model):
    with pytest.raises(ValueError, match="Unknown metric"):
        gpt2_model.eap(CLEAN, CORRUPTED, metric="kl_div")
    with pytest.raises(ValueError, match="ig_steps"):
        gpt2_model.eap(CLEAN, CORRUPTED, ig_steps=-1)


def test_eap_seq2seq_gated(t5_model):
    from interpkit import OperationNotSupportedForArchitecture

    with pytest.raises(OperationNotSupportedForArchitecture):
        t5_model.eap("translate: hello", "translate: goodbye")


# ── find_circuit integration ──────────────────────────────────────


def test_find_circuit_eap_keeps_legacy_schema(gpt2_model):
    result = gpt2_model.find_circuit(CLEAN, CORRUPTED, method="eap", threshold=0.2)
    # Legacy keys unchanged
    assert set(result) >= {
        "circuit", "excluded", "verification", "threshold",
        "total_components", "num_pairs",
    }
    assert isinstance(result["verification"]["faithfulness"], float)
    assert math.isfinite(result["verification"]["faithfulness"])
    # Additive keys
    assert result["meta"]["method"] == "eap"
    assert result["meta"]["selection"] == "eap"
    assert result["edges"]
    # EAP-selected components carry their raw score and normalised effect
    for c in result["circuit"]:
        assert 0.0 <= c["effect"] <= 1.0
        assert "eap_score" in c
    # Normalisation contract: the top component has effect exactly 1.0
    top_effect = max(c["effect"] for c in result["circuit"])
    assert top_effect == pytest.approx(1.0)


def test_find_circuit_eap_agrees_with_node_ranking(gpt2_model):
    eap_top = {n["node"] for n in gpt2_model.eap(CLEAN, CORRUPTED)["nodes"][:5]}
    circuit = gpt2_model.find_circuit(CLEAN, CORRUPTED, method="eap", threshold=0.5)
    circuit_names = {c["component"] for c in circuit["circuit"]}
    assert circuit_names <= eap_top | {
        n["node"] for n in gpt2_model.eap(CLEAN, CORRUPTED)["nodes"]
    }
    assert len(circuit_names & eap_top) >= 1


def test_find_circuit_invalid_method_rejected(gpt2_model):
    with pytest.raises(ValueError, match="Unknown method"):
        gpt2_model.find_circuit(CLEAN, CORRUPTED, method="vibes")
