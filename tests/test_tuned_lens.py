"""Tests for the tuned lens (ops/tuned_lens.py + lens(kind="tuned"))."""

from __future__ import annotations

import pytest
import torch

from interpkit.core.exceptions import InterpkitError
from interpkit.ops.tuned_lens import (
    TunedLens,
    load_tuned_lens,
    save_tuned_lens,
    train_tuned_lens,
    tuned_lens_loss,
)

TEXT = "The capital of France is"
CORPUS = [
    "The capital of France is Paris.",
    "Water boils at one hundred degrees.",
    "The sun rises in the east.",
    "Cats are small domesticated animals.",
    "Rome is the capital of Italy.",
    "Snow is cold and white in winter.",
]


def _fresh_lens(model) -> TunedLens:
    from interpkit.core.support_matrix import lens_blocks

    paths = [b.path for b in lens_blocks(model.arch_info)]
    return TunedLens(model.arch_info.hidden_size, paths)


@pytest.fixture(scope="module")
def trained_lens(gpt2_model):
    """One short shared training run for the read-only tests."""
    return train_tuned_lens(
        gpt2_model, CORPUS, steps=10, batch_size=3, max_length=24,
        seed=0, progress=False,
    )


def test_identity_init_reproduces_logit_lens(gpt2_model):
    """Untrained translators are exact identity maps — tuned output must
    match the raw logit lens (the strongest guard on the integration)."""
    base = gpt2_model.lens(TEXT)
    tuned = gpt2_model.lens(TEXT, kind="tuned", tuned_lens=_fresh_lens(gpt2_model))
    assert len(base) == len(tuned)
    for a, b in zip(base, tuned):
        assert a["layer_name"] == b["layer_name"]
        assert a["top1_token"] == b["top1_token"]
        assert a["top1_prob"] == pytest.approx(b["top1_prob"], abs=1e-6)
        assert b["lens_kind"] == "tuned"


def test_training_reduces_kl(gpt2_model, trained_lens):
    identity_loss = tuned_lens_loss(gpt2_model, _fresh_lens(gpt2_model), CORPUS)
    trained_loss = tuned_lens_loss(gpt2_model, trained_lens, CORPUS)
    assert trained_loss < identity_loss
    assert trained_lens.meta["trained"] is True
    cfg = trained_lens.meta["train_config"]
    assert cfg["final_loss"] < cfg["first_loss"]


def test_training_deterministic(gpt2_model):
    a = train_tuned_lens(
        gpt2_model, CORPUS, steps=4, batch_size=2, max_length=16,
        seed=0, progress=False,
    )
    b = train_tuned_lens(
        gpt2_model, CORPUS, steps=4, batch_size=2, max_length=16,
        seed=0, progress=False,
    )
    for ta, tb in zip(a.state_dict().values(), b.state_dict().values()):
        assert torch.allclose(ta, tb)


def test_save_load_roundtrip(gpt2_model, trained_lens, tmp_path):
    save_tuned_lens(trained_lens, tmp_path)
    loaded = load_tuned_lens(tmp_path, model=gpt2_model)
    for ta, tb in zip(
        trained_lens.state_dict().values(), loaded.state_dict().values(),
    ):
        assert torch.equal(ta.cpu(), tb.cpu())
    assert loaded.meta["hidden_size"] == trained_lens.meta["hidden_size"]
    result = gpt2_model.lens(TEXT, kind="tuned", tuned_lens=str(tmp_path))
    assert result[0]["lens_kind"] == "tuned"


def test_load_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_tuned_lens"):
        load_tuned_lens(tmp_path / "nope")


def test_validate_against_mismatches(gpt2_model):
    from interpkit.core.support_matrix import lens_blocks

    paths = [b.path for b in lens_blocks(gpt2_model.arch_info)]
    wrong_hidden = TunedLens(16, paths)
    with pytest.raises(InterpkitError, match="hidden_size"):
        wrong_hidden.validate_against(gpt2_model.arch_info, paths)

    wrong_count = TunedLens(gpt2_model.arch_info.hidden_size, paths[:3])
    with pytest.raises(InterpkitError, match="translators"):
        wrong_count.validate_against(gpt2_model.arch_info, paths)

    wrong_paths = TunedLens(
        gpt2_model.arch_info.hidden_size, ["a.b"] * len(paths),
    )
    with pytest.raises(InterpkitError, match="block paths"):
        wrong_paths.validate_against(gpt2_model.arch_info, paths)


def test_lens_kind_validation(gpt2_model, trained_lens):
    with pytest.raises(ValueError, match="Unknown kind"):
        gpt2_model.lens(TEXT, kind="psychic")
    with pytest.raises(ValueError, match="needs a tuned lens"):
        gpt2_model.lens(TEXT, kind="tuned")
    with pytest.raises(ValueError, match="kind='tuned'"):
        gpt2_model.lens(TEXT, tuned_lens=trained_lens)


def test_train_rejects_bad_args(gpt2_model):
    with pytest.raises(ValueError, match="non-empty list"):
        train_tuned_lens(gpt2_model, [], progress=False)
    with pytest.raises(ValueError, match="steps"):
        train_tuned_lens(gpt2_model, CORPUS, steps=0, progress=False)


def test_default_logit_lens_unchanged(gpt2_model):
    """The default path must not regress: no kind kwarg → logit lens with
    the legacy entry schema plus the additive lens_kind field."""
    result = gpt2_model.lens(TEXT)
    assert result
    for entry in result:
        assert entry["lens_kind"] == "logit"
        assert set(entry) >= {
            "layer_name", "top1_token", "top1_prob", "top5_tokens", "top5_probs",
        }
