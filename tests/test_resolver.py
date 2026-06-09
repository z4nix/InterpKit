"""Unit tests for the three-layer architecture resolver (Phase 0).

Covers:
- Family classification across all five ArchFamily values
- Block discovery for flat (LM/ViT) and hierarchical (CNN) topologies
- Runtime-hook detection of residuals and pre-head module
- Convention scans for HF and timm attribute names
- Override application via the with_overrides() escape hatch
- Validation: paths must resolve to real modules
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from interpkit.core.arch import (
    ArchFamily,
    ArchInfo,
    BlockSpec,
    resolve_arch,
)
from interpkit.core.exceptions import ArchitectureSpecMismatch

# ---------------------------------------------------------------------------
# Synthetic architectures for testing resolver paths in isolation
# ---------------------------------------------------------------------------


class _SimpleLM(nn.Module):
    """Minimal LM-shaped module: embed → layers → norm → head.

    Takes sequence-shaped input (B, seq) and produces token logits
    (B, seq, vocab) so the family classifier picks CAUSAL_LM.
    """

    def __init__(self, n_layers: int = 4):
        super().__init__()
        self.embed = nn.Embedding(100, 32)
        self.layers = nn.ModuleList([
            _ResidualBlock(32) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(32)
        self.head = nn.Linear(32, 100)

    def forward(self, x):
        x = self.embed(x)  # (B, seq, hidden)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)  # (B, seq, vocab)


class _ResidualBlock(nn.Module):
    """Residual block: out = x + transform(x). Has a fake attention attribute
    so the resolver classifies the block as has_attention=True."""

    def __init__(self, dim: int):
        super().__init__()
        self.attn = _FakeAttention(dim)
        self.transform = nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.transform(x) * 0.01


class _FakeAttention(nn.Module):
    """Fake attention with q_proj/k_proj/v_proj children to satisfy
    the runtime-attention-detection heuristic."""

    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)


class _NonResidualMLP(nn.Module):
    """Non-residual stack: out = transform(x). For testing the
    has_residual=False classification path."""

    def __init__(self, n_layers: int = 3):
        super().__init__()
        self.embed = nn.Embedding(100, 32)
        self.layers = nn.ModuleList([nn.Linear(32, 32) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(32)
        self.head = nn.Linear(32, 100)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = self.norm(x)
        return self.head(x)


class _SimpleCNN(nn.Module):
    """Minimal CNN: stem → layers → pool → fc."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1)
        self.layers = nn.Sequential(
            nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU()),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Resolver tests
# ---------------------------------------------------------------------------


class TestSyntheticLM:
    """Resolver runs on a hand-built LM-shaped nn.Module."""

    def setup_method(self):
        self.model = _SimpleLM(n_layers=4)
        self.sample = torch.zeros(1, 5, dtype=torch.long)

    def test_family_classified_as_causal_lm(self):
        arch = resolve_arch(self.model, self.sample)
        assert arch.family == ArchFamily.CAUSAL_LM

    def test_blocks_discovered(self):
        arch = resolve_arch(self.model, self.sample)
        assert len(arch.blocks) == 4

    def test_block_residuals_detected(self):
        arch = resolve_arch(self.model, self.sample)
        # All four blocks are residual via the runtime-hook detector.
        assert all(b.has_residual for b in arch.blocks)

    def test_block_attention_detected(self):
        arch = resolve_arch(self.model, self.sample)
        assert all(b.has_attention for b in arch.blocks)

    def test_pre_head_module_resolved(self):
        arch = resolve_arch(self.model, self.sample)
        assert arch.pre_head_module is not None
        assert arch.pre_head_path == "norm"

    def test_head_module_resolved(self):
        arch = resolve_arch(self.model, self.sample)
        assert arch.head_module is self.model.head
        assert arch.head_path == "head"

    def test_layer_of_resolves_block_index(self):
        arch = resolve_arch(self.model, self.sample)
        assert arch.layer_of("layers.2") == 2
        assert arch.layer_of("layers.2.attn.q_proj") == 2
        # Non-block path returns None
        assert arch.layer_of("head") is None


class TestSyntheticCNN:
    """Resolver runs on a hand-built CNN."""

    def setup_method(self):
        self.model = _SimpleCNN()
        self.sample = torch.zeros(1, 3, 32, 32)

    def test_family_classified_as_cnn(self):
        arch = resolve_arch(self.model, self.sample)
        assert arch.family in (ArchFamily.CNN_RESIDUAL, ArchFamily.CNN_PLAIN)

    def test_spatial_flag_true(self):
        arch = resolve_arch(self.model, self.sample)
        assert arch.spatial is True

    def test_head_resolved_via_fc_attribute(self):
        """The convention scan finds model.fc (timm convention)."""
        arch = resolve_arch(self.model, self.sample)
        assert arch.head_module is self.model.fc


class TestNonResidualStack:
    """Non-residual MLP stack should classify has_residual=False."""

    def test_blocks_have_no_residual(self):
        model = _NonResidualMLP(n_layers=3)
        sample = torch.zeros(1, 5, dtype=torch.long)
        arch = resolve_arch(model, sample)
        # Pure linear stack with ReLU is not residual.
        assert not any(b.has_residual for b in arch.blocks)


# ---------------------------------------------------------------------------
# Override / escape hatch
# ---------------------------------------------------------------------------


class TestOverrides:
    """User-supplied arch_override always wins."""

    def test_override_head_path_works(self):
        model = _SimpleLM()
        sample = torch.zeros(1, 5, dtype=torch.long)
        # Override head to point at the embedding (silly but valid)
        arch = resolve_arch(
            model, sample,
            arch_override={"head_path": "embed"},
        )
        assert arch.head_path == "embed"
        assert arch.head_module is model.embed

    def test_override_family_works(self):
        model = _SimpleLM()
        sample = torch.zeros(1, 5, dtype=torch.long)
        arch = resolve_arch(
            model, sample,
            arch_override={"family": "vision_transformer"},
        )
        assert arch.family == ArchFamily.VISION_TRANSFORMER

    def test_override_unknown_key_raises(self):
        model = _SimpleLM()
        sample = torch.zeros(1, 5, dtype=torch.long)
        with pytest.raises(ArchitectureSpecMismatch, match="Unknown arch_override key"):
            resolve_arch(
                model, sample,
                arch_override={"this_key_does_not_exist": "foo"},
            )

    def test_override_invalid_path_raises(self):
        model = _SimpleLM()
        sample = torch.zeros(1, 5, dtype=torch.long)
        with pytest.raises((ArchitectureSpecMismatch, AttributeError)):
            resolve_arch(
                model, sample,
                arch_override={"head_path": "nonexistent.path"},
            )


# ---------------------------------------------------------------------------
# ArchInfo all_paths / layer_of
# ---------------------------------------------------------------------------


class TestArchInfoMethods:
    """Public methods on ArchInfo."""

    def test_all_paths_returns_known_modules(self):
        model = _SimpleLM()
        sample = torch.zeros(1, 5, dtype=torch.long)
        arch = resolve_arch(model, sample)
        paths = arch.all_paths()
        assert "head" in paths
        assert "embed" in paths
        assert "layers.0" in paths

    def test_layer_of_returns_int_or_none(self):
        arch = ArchInfo(
            blocks=[BlockSpec(path=f"transformer.h.{i}") for i in range(5)],
        )
        assert arch.layer_of("transformer.h.3") == 3
        assert arch.layer_of("transformer.h.3.attn.c_attn") == 3
        assert arch.layer_of("lm_head") is None
