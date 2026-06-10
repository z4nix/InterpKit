"""Tests for model.generate / model.intervene — interventions during decoding."""

from __future__ import annotations

import pytest
import torch

from interpkit import (
    AblateIntervention,
    FnIntervention,
    SteerIntervention,
)

PROMPT = "The capital of France is"
N_NEW = 4
HIDDEN = 768  # gpt2 hidden size


def _steer_direction() -> torch.Tensor:
    """Deterministic non-uniform direction.

    A uniform vector (``torch.ones``) is exactly cancelled by the next
    LayerNorm's mean-centering, so it would steer nothing.
    """
    g = torch.Generator().manual_seed(0)
    return torch.randn(HIDDEN, generator=g)


def _baseline(gpt2_model, **kwargs):
    return gpt2_model.generate(PROMPT, max_new_tokens=N_NEW, **kwargs)


def test_generate_matches_raw_hf_greedy(gpt2_model):
    result = _baseline(gpt2_model)
    enc = gpt2_model._prepare(PROMPT)
    with torch.no_grad():
        raw = gpt2_model._model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
            max_new_tokens=N_NEW,
            do_sample=False,
            pad_token_id=gpt2_model._tokenizer.eos_token_id,
        )
    assert torch.equal(result["output_ids"], raw)
    assert result["response"]
    assert result["interventions"] == []


def test_generate_steering_changes_response(gpt2_model):
    baseline = _baseline(gpt2_model)
    steered = _baseline(
        gpt2_model,
        interventions=[
            SteerIntervention(
                "transformer.h.6", vector=_steer_direction(), scale=20.0,
            )
        ],
    )
    assert steered["response"] != baseline["response"]
    assert steered["interventions"][0]["type"] == "steer"
    assert steered["interventions"][0]["at"] == "transformer.h.6"


def test_generate_hooks_removed_after_run(gpt2_model):
    baseline = _baseline(gpt2_model)
    _baseline(
        gpt2_model,
        interventions=[
            SteerIntervention("transformer.h.6", vector=_steer_direction(), scale=20.0)
        ],
    )
    again = _baseline(gpt2_model)
    assert again["response"] == baseline["response"]


def test_generate_capture_lens_shape(gpt2_model):
    result = _baseline(gpt2_model, capture="lens")
    n_generated = result["output_ids"].shape[-1] - result["input_ids"].shape[-1]
    steps = result["steps"]
    assert len(steps) == n_generated
    n_blocks = len(gpt2_model.arch_info.blocks)
    for i, step in enumerate(steps):
        assert step["step"] == i
        assert isinstance(step["token"], str)
        assert len(step["lens"]) == n_blocks
        for entry in step["lens"]:
            assert 0.0 <= entry["top1_prob"] <= 1.0
    # Final-block lens of each step must agree with the actually-generated
    # token (greedy decoding + validated head pipeline).
    for step in steps:
        assert step["lens"][-1]["top1_id"] == step["token_id"]


def test_generate_capture_logits(gpt2_model):
    result = _baseline(gpt2_model, capture="logits")
    steps = result["steps"]
    assert len(steps) > 0
    for step in steps:
        assert step["logits"].shape[-1] == gpt2_model.arch_info.vocab_size
        assert 0.0 <= step["prob"] <= 1.0


def test_generate_positional_intervention_kv_cache(gpt2_model):
    """An intervention pinned to absolute position prompt_len+2 must leave
    decode steps 0-1 byte-identical to baseline (KV-cache window mapping)."""
    baseline = _baseline(gpt2_model)
    prompt_len = int(baseline["input_ids"].shape[-1])
    intervened = _baseline(
        gpt2_model,
        interventions=[
            SteerIntervention(
                "transformer.h.6",
                vector=_steer_direction(),
                scale=40.0,
                positions=(prompt_len + 2,),
            )
        ],
    )
    base_new = baseline["output_ids"][0, prompt_len:].tolist()
    int_new = intervened["output_ids"][0, prompt_len:].tolist()
    # Steps 0, 1, 2 are produced before/at the intervened position's output
    # enters the context — the intervention modifies the block output for
    # token at position prompt_len+2 (generated token 2), which first
    # influences the *prediction of token 3*.
    assert int_new[:3] == base_new[:3]
    assert int_new != base_new


def test_generate_fn_intervention_runs(gpt2_model):
    result = _baseline(
        gpt2_model,
        interventions=[FnIntervention("transformer.h.4", fn=lambda t, _ctx: t * 0.0)],
    )
    assert isinstance(result["response"], str)


def test_generate_invalid_capture_rejected(gpt2_model):
    with pytest.raises(ValueError, match="Unknown capture"):
        gpt2_model.generate(PROMPT, max_new_tokens=2, capture="telepathy")


def test_generate_typo_module_path_rejected(gpt2_model):
    with pytest.raises(KeyError, match="not found"):
        gpt2_model.generate(
            PROMPT,
            max_new_tokens=2,
            interventions=[
                SteerIntervention("transformr.h.6", vector=torch.ones(HIDDEN))
            ],
        )


def test_intervene_context_manager_composes_with_ops(gpt2_model):
    enc = gpt2_model._prepare(PROMPT)
    clean = gpt2_model._forward(enc)
    with gpt2_model.intervene(
        AblateIntervention("transformer.h.4.mlp", method="zero")
    ):
        ablated = gpt2_model._forward(enc)
    after = gpt2_model._forward(enc)
    assert not torch.allclose(clean, ablated)
    assert torch.allclose(clean, after)


def test_intervene_removes_hooks_on_exception(gpt2_model):
    enc = gpt2_model._prepare(PROMPT)
    clean = gpt2_model._forward(enc)
    with pytest.raises(RuntimeError, match="boom"), gpt2_model.intervene(
        AblateIntervention("transformer.h.4.mlp", method="zero")
    ):
        raise RuntimeError("boom")
    after = gpt2_model._forward(enc)
    assert torch.allclose(clean, after)


def test_generate_seq2seq_interventions_gated(t5_model):
    from interpkit import OperationNotSupportedForArchitecture

    with pytest.raises(OperationNotSupportedForArchitecture, match="encoder-decoder"):
        t5_model.generate(
            "translate English to German: hello",
            max_new_tokens=2,
            interventions=[
                SteerIntervention("decoder.block.0", vector=torch.ones(512))
            ],
        )
