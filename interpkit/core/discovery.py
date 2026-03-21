"""Auto-discover model structure from HF config, module name heuristics, and forward pass."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from collections import deque

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Heuristic patterns for semantic module role detection
# ---------------------------------------------------------------------------

_ATTENTION_PATTERNS = re.compile(
    r"(^|\.)(self_attn|self_attention|attn|attention|mha|multi_head_attention)(\.|\b)",
    re.IGNORECASE,
)
_MLP_PATTERNS = re.compile(
    r"(^|\.)(mlp|ffn|feed_forward|fc_?\d*)(\.|\b)", re.IGNORECASE
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

_FUSED_QKV_NAMES = frozenset({"c_attn", "qkv", "query_key_value"})
_Q_PROJ_NAMES = frozenset({"q_proj", "query", "q_lin"})
_K_PROJ_NAMES = frozenset({"k_proj", "key", "k_lin"})
_V_PROJ_NAMES = frozenset({"v_proj", "value", "v_lin"})
_O_PROJ_NAMES = frozenset({"c_proj", "out_proj", "o_proj", "dense", "out_lin", "o"})
_INTERLEAVED_QKV_CLASSES = frozenset({"GPTNeoXAttention"})


@dataclass
class ModuleInfo:
    """Discovered information about a single named module."""

    name: str
    type_name: str
    param_count: int
    output_shape: tuple[int, ...] | None = None
    role: str | None = None  # "attention", "mlp", "head", "norm", "embed", or None


@dataclass
class LayerInfo:
    """Resolved structural details for a single transformer layer."""

    name: str
    index: int
    attn_path: str | None = None
    mlp_path: str | None = None
    o_proj_path: str | None = None
    q_proj_path: str | None = None
    k_proj_path: str | None = None
    v_proj_path: str | None = None
    qkv_proj_path: str | None = None
    qkv_style: str = "unknown"
    qkv_layout: str = "concatenated"


@dataclass
class ModelArchInfo:
    """Aggregated architecture information for a model."""

    arch_family: str | None = None  # e.g. "GPT2LMHeadModel", "MambaForCausalLM"
    num_layers: int | None = None
    hidden_size: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    vocab_size: int | None = None
    has_lm_head: bool = False
    output_head_name: str | None = None
    unembedding_name: str | None = None
    modules: list[ModuleInfo] = field(default_factory=list)
    layer_names: list[str] = field(default_factory=list)
    layer_infos: list[LayerInfo] = field(default_factory=list)
    is_tl_model: bool = False
    is_encoder_decoder: bool = False
    project_out_path: str | None = None

    @property
    def is_language_model(self) -> bool:
        return self.has_lm_head and self.unembedding_name is not None


def _classify_role(name: str, type_name: str = "") -> str | None:
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

    # Fallback: check the module class name for models that use non-standard
    # path names but descriptive class names (e.g. BloomAttention,
    # RecurrentGemmaSdpaAttention).
    if type_name:
        tn = type_name.lower()
        if "attention" in tn or "selfattn" in tn:
            return "attention"
        if "mlp" in tn or "feedforward" in tn:
            return "mlp"

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

    for attr in ("num_key_value_heads", "num_kv_heads"):
        val = getattr(config, attr, None)
        if val is not None:
            info["num_key_value_heads"] = val
            break

    info["vocab_size"] = getattr(config, "vocab_size", None)
    return info


_LM_HEAD_PATTERNS = re.compile(
    r"(^|\.)(lm_head|output_projection|embed_out)(\.|\b)", re.IGNORECASE
)


def _find_unembedding(model: nn.Module) -> str | None:
    """Try to find the unembedding / LM head weight matrix.

    Only matches names that are unambiguously language-model heads
    (``lm_head``, ``output_projection``). Generic names like ``head``,
    ``classifier``, and ``qa_outputs`` are excluded to avoid false
    positives on vision and QA models.  If the model has a
    ``config.vocab_size``, a broader search is attempted with a
    shape check as a safety net.
    """
    # Strict pass: unambiguous LM head names
    for name, module in model.named_modules():
        if _LM_HEAD_PATTERNS.search(name) and hasattr(module, "weight"):
            return name

    # Relaxed pass: allow generic head names only when the output
    # dimension matches vocab_size from the config.
    vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    if vocab_size is not None:
        for name, module in model.named_modules():
            if _HEAD_PATTERNS.search(name) and hasattr(module, "weight"):
                out_features = getattr(module, "out_features", None)
                if out_features == vocab_size:
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


# ---------------------------------------------------------------------------
# Per-layer structure resolution
# ---------------------------------------------------------------------------

_ATTN_RE = re.compile(
    r"^(self_attn|self_attention|attn|attention|mha|multi_head_attention)$",
    re.IGNORECASE,
)
_MLP_RE = re.compile(r"^(mlp|ffn|feed_forward)$", re.IGNORECASE)


def _get_mod_by_path(model: nn.Module, path: str) -> nn.Module:
    mod = model
    for part in path.split("."):
        mod = getattr(mod, part)
    return mod


def _find_submodule_recursive(
    parent: nn.Module,
    parent_path: str,
    pattern: re.Pattern[str],
) -> tuple[str, nn.Module] | None:
    """BFS for the shallowest submodule whose base name matches *pattern*."""
    queue: deque[tuple[str, nn.Module]] = deque()
    for name, mod in parent.named_children():
        queue.append((name, mod))
    while queue:
        rel_name, mod = queue.popleft()
        base = rel_name.split(".")[-1]
        if pattern.search(base):
            return f"{parent_path}.{rel_name}", mod
        for child_name, child_mod in mod.named_children():
            queue.append((f"{rel_name}.{child_name}", child_mod))
    return None


def _resolve_projections(
    attn_mod: nn.Module,
    attn_path: str,
    info: LayerInfo,
) -> None:
    """Locate Q/K/V projections inside *attn_mod* and set fields on *info*."""
    for child_name, child_mod in attn_mod.named_modules():
        if not child_name or not hasattr(child_mod, "weight"):
            continue
        base = child_name.split(".")[-1]
        full = f"{attn_path}.{child_name}"
        if base in _FUSED_QKV_NAMES:
            info.qkv_style = "fused"
            info.qkv_proj_path = full
            info.qkv_layout = (
                "interleaved"
                if type(attn_mod).__name__ in _INTERLEAVED_QKV_CLASSES
                else "concatenated"
            )
        elif base in _Q_PROJ_NAMES:
            info.q_proj_path = full
            if info.qkv_style == "unknown":
                info.qkv_style = "separate"
        elif base in _K_PROJ_NAMES:
            info.k_proj_path = full
        elif base in _V_PROJ_NAMES:
            info.v_proj_path = full


def _resolve_output_proj(
    attn_mod: nn.Module | None,
    attn_path: str | None,
    layer_mod: nn.Module,
    layer_path: str,
    info: LayerInfo,
) -> None:
    """Locate the output projection, skipping modules already tagged as Q/K/V."""
    skip = {info.qkv_proj_path, info.q_proj_path, info.k_proj_path, info.v_proj_path}
    skip.discard(None)
    if attn_mod is not None and attn_path is not None:
        for name, mod in attn_mod.named_modules():
            if not name or not hasattr(mod, "weight"):
                continue
            full = f"{attn_path}.{name}"
            if full in skip:
                continue
            base = name.split(".")[-1]
            if base in _O_PROJ_NAMES:
                info.o_proj_path = full
                return
    for name, mod in layer_mod.named_modules():
        if not name or not hasattr(mod, "weight"):
            continue
        full = f"{layer_path}.{name}"
        if full in skip:
            continue
        base = name.split(".")[-1]
        if base in _O_PROJ_NAMES:
            info.o_proj_path = full
            return


def _resolve_layer_info(
    model: nn.Module,
    layer_name: str,
    layer_idx: int,
) -> LayerInfo:
    """Build a fully resolved :class:`LayerInfo` for one transformer layer."""
    info = LayerInfo(name=layer_name, index=layer_idx)
    try:
        layer_mod = _get_mod_by_path(model, layer_name)
    except AttributeError:
        return info

    attn_result = _find_submodule_recursive(layer_mod, layer_name, _ATTN_RE)
    if attn_result is not None:
        info.attn_path = attn_result[0]
        _resolve_projections(attn_result[1], attn_result[0], info)

    mlp_result = _find_submodule_recursive(layer_mod, layer_name, _MLP_RE)
    if mlp_result is not None:
        info.mlp_path = mlp_result[0]

    _resolve_output_proj(
        attn_result[1] if attn_result else None,
        attn_result[0] if attn_result else None,
        layer_mod,
        layer_name,
        info,
    )
    return info


def _detect_project_out(model: nn.Module) -> str | None:
    """Find a ``project_out`` layer (OPT-style embed_dim != hidden_size)."""
    for name, mod in model.named_modules():
        if "project_out" in name and hasattr(mod, "weight"):
            return name
    return None


# ---------------------------------------------------------------------------
# Centralised weight extraction (used by ops/circuits and ops/heads)
# ---------------------------------------------------------------------------


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
        mod = _get_mod_by_path(model, path)
        w = mod.weight
        return w.T if type(mod).__name__ == "Conv1D" else w

    if layer_info.qkv_style == "fused" and layer_info.qkv_proj_path is not None:
        mod = _get_mod_by_path(model, layer_info.qkv_proj_path)
        return _split_fused_weight(
            mod.weight, proj_type, num_heads, num_kv_heads,
            is_conv1d=(type(mod).__name__ == "Conv1D"),
            interleaved=(layer_info.qkv_layout == "interleaved"),
        )

    return None


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
        mod_type_name = type(mod).__name__
        info = ModuleInfo(
            name=name,
            type_name=mod_type_name,
            param_count=_count_params(mod),
            role=_classify_role(name, mod_type_name),
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

    # Resolve per-layer structural details
    layer_infos = [
        _resolve_layer_info(model, ln, idx) for idx, ln in enumerate(layer_names)
    ]

    # Detect project_out for models with embed_dim != hidden_size
    project_out_path = _detect_project_out(model)

    # Check encoder-decoder
    config = getattr(model, "config", None)
    is_enc_dec = getattr(config, "is_encoder_decoder", False)

    return ModelArchInfo(
        arch_family=hf_meta.get("arch_family"),
        num_layers=hf_meta.get("num_layers"),
        hidden_size=hf_meta.get("hidden_size"),
        num_attention_heads=hf_meta.get("num_attention_heads"),
        num_key_value_heads=hf_meta.get("num_key_value_heads"),
        vocab_size=hf_meta.get("vocab_size"),
        has_lm_head=has_lm_head,
        output_head_name=unembed_name,
        unembedding_name=unembed_name,
        modules=module_infos,
        layer_names=layer_names,
        layer_infos=layer_infos,
        is_encoder_decoder=is_enc_dec,
        project_out_path=project_out_path,
    )
