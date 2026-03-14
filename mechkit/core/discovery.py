"""Auto-discover model structure from HF config, module name heuristics, and forward pass."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Heuristic patterns for semantic module role detection
# ---------------------------------------------------------------------------

_ATTENTION_PATTERNS = re.compile(
    r"(^|\.)(self_attn|attn|attention|mha|multi_head_attention)(\.|\b)", re.IGNORECASE
)
_MLP_PATTERNS = re.compile(
    r"(^|\.)(mlp|ffn|feed_forward|dense|fc[_\d]|intermediate)(\.|\b)", re.IGNORECASE
)
_HEAD_PATTERNS = re.compile(
    r"(^|\.)(lm_head|head|classifier|output_projection|qa_outputs)(\.|\b)", re.IGNORECASE
)
_NORM_PATTERNS = re.compile(
    r"(^|\.)(layer_?norm|rms_?norm|norm|ln_f|ln_\d)(\.|\b)", re.IGNORECASE
)
_EMBED_PATTERNS = re.compile(
    r"(^|\.)(embed|wte|wpe|embedding|token_embedding|position_embedding)(\.|\b)",
    re.IGNORECASE,
)


@dataclass
class ModuleInfo:
    """Discovered information about a single named module."""

    name: str
    type_name: str
    param_count: int
    output_shape: tuple[int, ...] | None = None
    role: str | None = None  # "attention", "mlp", "head", "norm", "embed", or None


@dataclass
class ModelArchInfo:
    """Aggregated architecture information for a model."""

    arch_family: str | None = None  # e.g. "GPT2LMHeadModel", "MambaForCausalLM"
    num_layers: int | None = None
    hidden_size: int | None = None
    num_attention_heads: int | None = None
    vocab_size: int | None = None
    has_lm_head: bool = False
    output_head_name: str | None = None
    unembedding_name: str | None = None
    modules: list[ModuleInfo] = field(default_factory=list)
    layer_names: list[str] = field(default_factory=list)
    is_tl_model: bool = False

    @property
    def is_language_model(self) -> bool:
        return self.has_lm_head and self.unembedding_name is not None


def _classify_role(name: str) -> str | None:
    if _HEAD_PATTERNS.search(name):
        return "head"
    if _ATTENTION_PATTERNS.search(name):
        return "attention"
    if _MLP_PATTERNS.search(name):
        return "mlp"
    if _NORM_PATTERNS.search(name):
        return "norm"
    if _EMBED_PATTERNS.search(name):
        return "embed"
    return None


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters(recurse=False))


def _parse_hf_config(model: nn.Module) -> dict[str, Any]:
    """Extract architecture metadata from an HF model's config, if present."""
    config = getattr(model, "config", None)
    if config is None:
        return {}
    info: dict[str, Any] = {}
    info["arch_family"] = type(model).__name__

    for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
        val = getattr(config, attr, None)
        if val is not None:
            info["num_layers"] = val
            break

    for attr in ("hidden_size", "n_embd", "d_model"):
        val = getattr(config, attr, None)
        if val is not None:
            info["hidden_size"] = val
            break

    for attr in ("num_attention_heads", "n_head", "num_heads"):
        val = getattr(config, attr, None)
        if val is not None:
            info["num_attention_heads"] = val
            break

    info["vocab_size"] = getattr(config, "vocab_size", None)
    return info


def _find_unembedding(model: nn.Module) -> str | None:
    """Try to find the unembedding / LM head weight matrix."""
    for name, module in model.named_modules():
        if _HEAD_PATTERNS.search(name) and hasattr(module, "weight"):
            return name
    return None


def _detect_layers(modules: list[ModuleInfo]) -> list[str]:
    """Identify repeated structural blocks that look like transformer/SSM layers.

    Strategy: find modules whose names follow a pattern like ``something.N``
    where N is a sequential integer, and whose siblings have identical structure.
    We pick the longest such group.
    """
    pattern = re.compile(r"^(.+)\.(\d+)$")
    groups: dict[str, list[str]] = {}
    for m in modules:
        match = pattern.match(m.name)
        if match:
            prefix = match.group(1)
            groups.setdefault(prefix, []).append(m.name)

    if not groups:
        return []

    best_prefix = max(groups, key=lambda k: len(groups[k]))
    layers = sorted(groups[best_prefix], key=lambda n: int(n.rsplit(".", 1)[-1]))
    return layers


def discover(
    model: nn.Module,
    dummy_input: Any | None = None,
) -> ModelArchInfo:
    """Run full auto-discovery on a model.

    Parameters
    ----------
    model:
        Any ``nn.Module``, optionally with an HF ``.config`` attribute.
    dummy_input:
        If provided, used for a forward pass to capture output shapes.
        Can be a tensor, dict of tensors, or tuple of tensors.
    """
    hf_meta = _parse_hf_config(model)

    # Enumerate all named modules
    module_infos: list[ModuleInfo] = []
    for name, mod in model.named_modules():
        if name == "":
            continue
        info = ModuleInfo(
            name=name,
            type_name=type(mod).__name__,
            param_count=_count_params(mod),
            role=_classify_role(name),
        )
        module_infos.append(info)

    # Output shape enumeration via hooks
    if dummy_input is not None:
        shapes: dict[str, tuple[int, ...]] = {}
        hooks = []

        def _make_hook(mod_name: str):
            def hook_fn(_mod: nn.Module, _inp: Any, output: Any) -> None:
                if isinstance(output, torch.Tensor):
                    shapes[mod_name] = tuple(output.shape)
                elif isinstance(output, (tuple, list)) and len(output) > 0:
                    first = output[0]
                    if isinstance(first, torch.Tensor):
                        shapes[mod_name] = tuple(first.shape)
            return hook_fn

        for name, mod in model.named_modules():
            if name == "":
                continue
            hooks.append(mod.register_forward_hook(_make_hook(name)))

        try:
            with torch.no_grad():
                if isinstance(dummy_input, dict):
                    model(**dummy_input)
                elif isinstance(dummy_input, (tuple, list)):
                    model(*dummy_input)
                else:
                    model(dummy_input)
        finally:
            for h in hooks:
                h.remove()

        for info in module_infos:
            info.output_shape = shapes.get(info.name)

    # Find unembedding
    unembed_name = _find_unembedding(model)
    has_lm_head = unembed_name is not None

    # Detect layer names
    layer_names = _detect_layers(module_infos)

    return ModelArchInfo(
        arch_family=hf_meta.get("arch_family"),
        num_layers=hf_meta.get("num_layers"),
        hidden_size=hf_meta.get("hidden_size"),
        num_attention_heads=hf_meta.get("num_attention_heads"),
        vocab_size=hf_meta.get("vocab_size"),
        has_lm_head=has_lm_head,
        output_head_name=unembed_name,
        unembedding_name=unembed_name,
        modules=module_infos,
        layer_names=layer_names,
    )
