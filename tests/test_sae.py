"""Tests for the SAE feature decomposition (no HF downloads — uses synthetic SAEs)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from interpkit.ops.sae import SAE, load_sae, load_sae_from_path, load_sae_from_tensors, run_features


def _make_synthetic_sae(d_in: int = 768, d_sae: int = 128) -> SAE:
    """Create a tiny synthetic SAE for testing."""
    return load_sae_from_tensors(
        W_enc=torch.randn(d_in, d_sae) * 0.01,
        W_dec=torch.randn(d_sae, d_in) * 0.01,
        b_enc=torch.zeros(d_sae),
        b_dec=torch.zeros(d_in),
    )


def test_sae_encode_decode():
    sae = _make_synthetic_sae(64, 32)
    x = torch.randn(4, 64)

    features = sae.encode(x)
    assert features.shape == (4, 32)
    assert (features >= 0).all(), "ReLU should produce non-negative features"

    x_hat = sae.decode(features)
    assert x_hat.shape == (4, 64)


def test_sae_forward():
    sae = _make_synthetic_sae(64, 32)
    x = torch.randn(4, 64)

    features, x_hat = sae.forward(x)
    assert features.shape == (4, 32)
    assert x_hat.shape == (4, 64)


def test_sae_from_tensors_metadata():
    sae = load_sae_from_tensors(
        W_enc=torch.randn(64, 32),
        W_dec=torch.randn(32, 64),
        b_enc=torch.zeros(32),
        b_dec=torch.zeros(64),
        metadata={"name": "test"},
    )
    assert sae.d_in == 64
    assert sae.d_sae == 32
    assert sae.metadata["name"] == "test"


def test_run_features_with_gpt2(gpt2_model):
    sae = _make_synthetic_sae(d_in=768, d_sae=128)

    result = run_features(
        gpt2_model,
        "The capital of France is",
        at="transformer.h.8.mlp",
        sae=sae,
        top_k=10,
        print_results=False,
    )

    assert "top_features" in result
    assert "reconstruction_error" in result
    assert "sparsity" in result
    assert "num_active_features" in result
    assert "total_features" in result
    assert result["total_features"] == 128
    assert len(result["top_features"]) <= 10

    for idx, val in result["top_features"]:
        assert isinstance(idx, int)
        assert isinstance(val, float)


def test_run_features_dimension_mismatch(gpt2_model):
    sae = _make_synthetic_sae(d_in=64, d_sae=32)

    try:
        run_features(
            gpt2_model,
            "Hello",
            at="transformer.h.0.mlp",
            sae=sae,
            print_results=False,
        )
        raise AssertionError("Should have raised ValueError for dimension mismatch")
    except ValueError as e:
        assert "does not match" in str(e)


def test_model_features_method(gpt2_model):
    sae = _make_synthetic_sae(d_in=768, d_sae=128)

    result = gpt2_model.features(
        "The capital of France is",
        at="transformer.h.8.mlp",
        sae=sae,
        top_k=5,
    )

    assert isinstance(result, dict)
    assert len(result["top_features"]) <= 5


def test_sae_sparsity():
    sae = _make_synthetic_sae(64, 32)
    x = torch.randn(10, 64)
    features = sae.encode(x)

    sparsity = (features == 0).float().mean().item()
    assert 0.0 <= sparsity <= 1.0


# ── local file loading ──────────────────────────────────────────────────


def _save_sae_pt(directory: Path, d_in: int = 64, d_sae: int = 32) -> Path:
    """Save synthetic SAE weights as a .pt file and return the path."""
    weights = {
        "W_enc": torch.randn(d_in, d_sae) * 0.01,
        "W_dec": torch.randn(d_sae, d_in) * 0.01,
        "b_enc": torch.zeros(d_sae),
        "b_dec": torch.zeros(d_in),
    }
    filepath = directory / "sae_weights.pt"
    torch.save(weights, filepath)
    return filepath


def _save_sae_safetensors(directory: Path, d_in: int = 64, d_sae: int = 32) -> Path:
    """Save synthetic SAE weights as a .safetensors file and return the path."""
    from safetensors.torch import save_file

    weights = {
        "W_enc": torch.randn(d_in, d_sae) * 0.01,
        "W_dec": torch.randn(d_sae, d_in) * 0.01,
        "b_enc": torch.zeros(d_sae),
        "b_dec": torch.zeros(d_in),
    }
    filepath = directory / "sae_weights.safetensors"
    save_file(weights, str(filepath))
    return filepath


def test_load_sae_from_path_pt():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = _save_sae_pt(Path(tmpdir))
        sae = load_sae_from_path(filepath)
        assert sae.d_in == 64
        assert sae.d_sae == 32
        assert sae.W_enc.shape == (64, 32)


def test_load_sae_from_path_safetensors():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = _save_sae_safetensors(Path(tmpdir))
        sae = load_sae_from_path(filepath)
        assert sae.d_in == 64
        assert sae.d_sae == 32


def test_load_sae_from_path_with_config():
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = _save_sae_pt(Path(tmpdir))
        cfg_path = Path(tmpdir) / "cfg.json"
        cfg_path.write_text(json.dumps({"model": "gpt2", "layer": 8}))

        sae = load_sae_from_path(filepath)
        assert sae.metadata["model"] == "gpt2"
        assert sae.metadata["layer"] == 8


def test_load_sae_from_path_missing_file():
    with pytest.raises(FileNotFoundError):
        load_sae_from_path("/nonexistent/path/sae_weights.pt")


def test_load_sae_from_path_missing_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "bad_sae.pt"
        torch.save({"W_enc": torch.randn(64, 32)}, filepath)
        with pytest.raises(KeyError, match="missing keys"):
            load_sae_from_path(filepath)


def test_load_sae_from_path_bad_extension():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "weights.npz"
        filepath.write_text("not real")
        with pytest.raises(ValueError, match="Unsupported"):
            load_sae_from_path(filepath)


def test_load_sae_auto_detects_local_pt():
    """load_sae() should auto-detect a local .pt path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = _save_sae_pt(Path(tmpdir))
        sae = load_sae(str(filepath))
        assert sae.d_in == 64
        assert sae.d_sae == 32


def test_load_sae_auto_detects_local_safetensors():
    """load_sae() should auto-detect a local .safetensors path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = _save_sae_safetensors(Path(tmpdir))
        sae = load_sae(str(filepath))
        assert sae.d_in == 64
        assert sae.d_sae == 32


# ── subfolder shorthand parsing ──────────────────────────────────────────


def test_split_repo_and_subfolder_basic():
    from interpkit.ops.sae import _split_repo_and_subfolder

    repo, sub = _split_repo_and_subfolder("jbloom/GPT2-SAEs/blocks.8.hook_resid_pre")
    assert repo == "jbloom/GPT2-SAEs"
    assert sub == "blocks.8.hook_resid_pre"


def test_split_repo_and_subfolder_nested_subfolder():
    from interpkit.ops.sae import _split_repo_and_subfolder

    repo, sub = _split_repo_and_subfolder("org/name/dir1/dir2")
    assert repo == "org/name"
    assert sub == "dir1/dir2"


def test_split_repo_and_subfolder_two_segments_unchanged():
    from interpkit.ops.sae import _split_repo_and_subfolder

    repo, sub = _split_repo_and_subfolder("gpt2/sae")
    assert repo == "gpt2/sae"
    assert sub is None


def test_split_repo_and_subfolder_local_relative_path_unchanged():
    from interpkit.ops.sae import _split_repo_and_subfolder

    repo, sub = _split_repo_and_subfolder("./local/dir/file")
    assert repo == "./local/dir/file"
    assert sub is None


def test_split_repo_and_subfolder_absolute_path_unchanged():
    from interpkit.ops.sae import _split_repo_and_subfolder

    repo, sub = _split_repo_and_subfolder("/abs/path/to/file")
    assert repo == "/abs/path/to/file"
    assert sub is None


def test_split_repo_and_subfolder_existing_local_unchanged():
    from interpkit.ops.sae import _split_repo_and_subfolder

    with tempfile.TemporaryDirectory() as tmpdir:
        nested = Path(tmpdir) / "a" / "b"
        nested.mkdir(parents=True)
        target = nested / "c.txt"
        target.write_text("data")
        s = str(target)
        repo, sub = _split_repo_and_subfolder(s)
        assert repo == s
        assert sub is None


def test_load_sae_explicit_subfolder_kw_forwarded():
    """When subfolder= is passed, _download_weights must receive it."""
    import interpkit.ops.sae as sae_mod

    captured: dict[str, str | None] = {}

    def fake_download(hf_id: str, *, subfolder: str | None = None):
        captured["hf_id"] = hf_id
        captured["subfolder"] = subfolder
        return {
            "W_enc": torch.randn(64, 32),
            "W_dec": torch.randn(32, 64),
            "b_enc": torch.zeros(32),
            "b_dec": torch.zeros(64),
        }

    def fake_config(hf_id: str, *, subfolder: str | None = None):
        return {}

    orig_w = sae_mod._download_weights
    orig_c = sae_mod._download_config
    sae_mod._download_weights = fake_download
    sae_mod._download_config = fake_config
    try:
        load_sae("org/name", subfolder="blocks.8.hook_resid_pre")
    finally:
        sae_mod._download_weights = orig_w
        sae_mod._download_config = orig_c

    assert captured == {"hf_id": "org/name", "subfolder": "blocks.8.hook_resid_pre"}


def test_load_sae_shorthand_routes_to_subfolder():
    """The 'org/name/sub' shorthand must call into _download_weights with subfolder set."""
    import interpkit.ops.sae as sae_mod

    captured: dict[str, str | None] = {}

    def fake_download(hf_id: str, *, subfolder: str | None = None):
        captured["hf_id"] = hf_id
        captured["subfolder"] = subfolder
        return {
            "W_enc": torch.randn(64, 32),
            "W_dec": torch.randn(32, 64),
            "b_enc": torch.zeros(32),
            "b_dec": torch.zeros(64),
        }

    def fake_config(hf_id: str, *, subfolder: str | None = None):
        return {}

    orig_w = sae_mod._download_weights
    orig_c = sae_mod._download_config
    sae_mod._download_weights = fake_download
    sae_mod._download_config = fake_config
    try:
        load_sae("jbloom/GPT2-Small-SAEs-Reformatted/blocks.8.hook_resid_pre")
    finally:
        sae_mod._download_weights = orig_w
        sae_mod._download_config = orig_c

    assert captured["hf_id"] == "jbloom/GPT2-Small-SAEs-Reformatted"
    assert captured["subfolder"] == "blocks.8.hook_resid_pre"


def test_load_sae_conflicting_shorthand_and_kw_raises():
    with pytest.raises(ValueError, match="Conflicting subfolder"):
        load_sae("org/name/sub_a", subfolder="sub_b")


def test_load_sae_matching_shorthand_and_kw_ok():
    """Passing the same subfolder via both syntaxes should not error."""
    import interpkit.ops.sae as sae_mod

    def fake_download(hf_id: str, *, subfolder: str | None = None):
        return {
            "W_enc": torch.randn(64, 32),
            "W_dec": torch.randn(32, 64),
            "b_enc": torch.zeros(32),
            "b_dec": torch.zeros(64),
        }

    def fake_config(hf_id: str, *, subfolder: str | None = None):
        return {}

    orig_w = sae_mod._download_weights
    orig_c = sae_mod._download_config
    sae_mod._download_weights = fake_download
    sae_mod._download_config = fake_config
    try:
        sae = load_sae("org/name/sub", subfolder="sub")
    finally:
        sae_mod._download_weights = orig_w
        sae_mod._download_config = orig_c

    assert sae.d_in == 64
