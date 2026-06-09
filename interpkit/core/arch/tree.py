"""Module-tree primitives and weight extraction.

Low-level, dependency-free helpers shared by every other resolver
submodule (and by ops that read weights directly): dotted-path
resolution, canonical weight extraction, fused-QKV splitting, and the
tolerant forward used by the runtime probes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from interpkit.core.arch.types import LayerInfo


def module_at_path(model: nn.Module, path: str) -> nn.Module:
    """Resolve a dotted path to a submodule, raising AttributeError if missing.

    ``ModuleList`` / ``Sequential`` numeric indices are integer-coerced so
    paths like ``"transformer.h.4.attn"`` resolve correctly. An empty path
    returns the model itself.
    """
    if not path:
        return model
    obj: Any = model
    for part in path.split("."):
        if part.isdigit() and isinstance(obj, (nn.ModuleList, nn.Sequential)):
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def path_of(model: nn.Module, target: nn.Module) -> str | None:
    """Reverse-lookup: return the module's name, or None if not found."""
    for name, mod in model.named_modules():
        if mod is target:
            return name
    return None


def get_weight(mod: nn.Module) -> torch.Tensor:
    """Extract ``mod.weight`` with a runtime assertion for mypy."""
    w = mod.weight
    assert isinstance(w, torch.Tensor)
    return w


def canonical_linear_weight(module: nn.Module) -> torch.Tensor:
    """Return *module*'s weight matrix in canonical ``(out_features, in_features)`` shape.

    ``transformers.pytorch_utils.Conv1D`` (used internally by GPT-2 /
    DistilGPT2) stores its weight transposed compared with
    ``nn.Linear`` — shape ``(in_features, out_features)``. Five+ op
    files used to duplicate the same brittle ``type(mod).__name__ ==
    "Conv1D"; w.T if is_conv1d else w`` idiom; this helper centralises
    that logic with a proper ``isinstance`` check that survives if
    transformers ever moves or renames the class.

    Returns the weight tensor as-is for ``nn.Linear`` and transposed
    for ``Conv1D``. Always ``Tensor.float()`` -callable downstream.
    """
    weight = getattr(module, "weight", None)
    if weight is None:
        raise ValueError(
            f"canonical_linear_weight: module {type(module).__name__} has "
            "no .weight attribute."
        )
    try:
        from transformers.pytorch_utils import Conv1D
    except ImportError:
        Conv1D = None  # type: ignore[assignment]
    if Conv1D is not None and isinstance(module, Conv1D):
        return weight.T
    # Fallback: string-name check for environments where transformers
    # pytorch_utils.Conv1D import path changed.
    if type(module).__name__ == "Conv1D":
        return weight.T
    return weight


def _split_fused_weight(
    w: torch.Tensor,
    proj_type: str,
    num_heads: int,
    num_kv_heads: int | None,
    *,
    is_conv1d: bool,
    interleaved: bool,
) -> torch.Tensor:
    """Split a fused QKV weight and return one of Q / K / V.

    Returns ``(proj_dim, d_model)``.
    """
    num_kv_heads = num_kv_heads or num_heads
    idx = {"q": 0, "k": 1, "v": 2}[proj_type]

    if is_conv1d:
        total = w.shape[1]
        head_dim = total // (num_heads + 2 * num_kv_heads)
        sizes = [num_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim]
        start = sum(sizes[:idx])
        return w[:, start : start + sizes[idx]].T

    total = w.shape[0]
    d_model = w.shape[1]
    head_dim = total // (num_heads + 2 * num_kv_heads)

    if interleaved and num_kv_heads == num_heads:
        hd = total // (3 * num_heads)
        if total == 3 * num_heads * hd:
            grouped = w.view(num_heads, 3, hd, d_model)
            return grouped[:, idx, :, :].reshape(-1, d_model)

    sizes = [num_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim]
    start = sum(sizes[:idx])
    return w[start : start + sizes[idx], :]


def extract_proj_weight(
    model: nn.Module,
    layer_info: LayerInfo,
    proj_type: str,
    num_heads: int,
    num_kv_heads: int | None = None,
) -> torch.Tensor | None:
    """Return the Q, K, or V weight for *proj_type* ``("q"|"k"|"v")``.

    Shape of the returned tensor: ``(proj_dim, d_model)``.
    """
    if layer_info.qkv_style == "separate":
        path = {"q": layer_info.q_proj_path,
                "k": layer_info.k_proj_path,
                "v": layer_info.v_proj_path}.get(proj_type)
        if path is None:
            return None
        mod = module_at_path(model, path)
        w = get_weight(mod)
        return w.T if type(mod).__name__ == "Conv1D" else w

    if layer_info.qkv_style == "fused" and layer_info.qkv_proj_path is not None:
        mod = module_at_path(model, layer_info.qkv_proj_path)
        return _split_fused_weight(
            get_weight(mod), proj_type, num_heads, num_kv_heads,
            is_conv1d=(type(mod).__name__ == "Conv1D"),
            interleaved=(layer_info.qkv_layout == "interleaved"),
        )

    return None


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters(recurse=False))


def _is_norm_module(mod: nn.Module) -> bool:
    """Check if a module is a normalization layer by its actual type."""
    if isinstance(mod, (nn.LayerNorm, nn.GroupNorm,
                        nn.BatchNorm1d, nn.BatchNorm2d,
                        nn.InstanceNorm1d, nn.InstanceNorm2d)):
        return True
    return "norm" in type(mod).__name__.lower()


def _safe_forward(model: nn.Module, sample_input: Any) -> Any:
    """Run a forward pass for resolution, swallowing recoverable errors.

    Used by hook-based detection helpers. The resolver tolerates partial
    forward failures (e.g. shape mismatches on probe inputs) but propagates
    fatal errors so the caller can raise a clear ``ArchitectureNotSupported``.
    """
    with torch.no_grad():
        if isinstance(sample_input, dict):
            return model(**sample_input)
        if isinstance(sample_input, (tuple, list)):
            return model(*sample_input)
        return model(sample_input)


def _is_pretrained_model(module: nn.Module) -> bool:
    """Check if module is a HF PreTrainedModel without forcing import at top level."""
    try:
        from transformers import PreTrainedModel
    except ImportError:
        return False
    return isinstance(module, PreTrainedModel)
