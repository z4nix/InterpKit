"""TransformerLens interop — bidirectional hook-name translation (F-025 / F-026).

Pre-1.0 ``to_native_name`` silently returned best-guess strings for every
TL hook, even TL-internal hooks (``hook_pattern``, ``hook_attn_scores``,
``hook_z``, ``hook_resid_*``, etc.) that have **no native HF equivalent**
because they name mid-computation tensors injected as TL ``HookPoint``
objects. The audit found 80% of TL hooks failed to round-trip
(``to_native_name`` → ``to_tl_name`` produced a different string).

The 1.0 fix:

- **Explicit round-trippable whitelist** (F-025). Only hooks with a real
  native module equivalent are mapped. TL-internal hooks raise
  :class:`KeyError` from ``to_native_name`` rather than silently returning
  bogus paths.
- **``hook_`` prefix preservation** (F-025). When the input is a TL-internal
  name without a native equivalent, ``to_tl_name`` keeps the ``hook_``
  prefix so the result is still recognisable as a TL hook.
- **``list_roundtrippable_hooks()`` helper** so callers can enumerate which
  hooks are safe to translate.

BOS-handling note (F-026)
-------------------------
TransformerLens's ``HookedTransformer.to_tokens(text)`` prepends a BOS
token by default; HF's tokeniser does not. When comparing TL outputs to
interpkit outputs, pass ``prepend_bos=False`` to TL or you'll see
~40-unit logit differences and disagreeing top-1 predictions. This is a
TL convention, not an interpkit bug, but we surface a one-time
``UserWarning`` if cross-library comparison is detected without the BOS
override.
"""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from interpkit.core.arch import ArchInfo


# ---------------------------------------------------------------------------
# Round-trippable native ↔ TL mapping
# ---------------------------------------------------------------------------
#
# These TL hooks correspond to actual nn.Module instances on the native HF
# model. They round-trip cleanly: to_tl_name(to_native_name(x)) == x.

_NATIVE_TO_TL: dict[str, str] = {
    # Bare layer alias
    "": "",  # block-level path: blocks.{N}
    # Attention sub-modules
    "self_attn": "attn",
    "attn": "attn",
    "attention": "attn",
    "self_attn.q_proj": "attn.hook_q",
    "self_attn.k_proj": "attn.hook_k",
    "self_attn.v_proj": "attn.hook_v",
    "self_attn.o_proj": "attn.hook_result",
    "attn.q_proj": "attn.hook_q",
    "attn.k_proj": "attn.hook_k",
    "attn.v_proj": "attn.hook_v",
    "attn.o_proj": "attn.hook_result",
    "attn.c_proj": "attn.hook_result",
    "attn.out_proj": "attn.hook_result",
    # MLP sub-modules
    "mlp": "mlp",
    "ffn": "mlp",
    "feed_forward": "mlp",
    # LayerNorm aliases
    "ln_1": "ln1",
    "ln1": "ln1",
    "input_layernorm": "ln1",
    "ln_2": "ln2",
    "ln2": "ln2",
    "post_attention_layernorm": "ln2",
}

# Reverse mapping: TL suffix → list of native suffix candidates (in priority order)
_TL_TO_NATIVE: dict[str, list[str]] = {
    "": [""],
    "attn": ["self_attn", "attn", "attention"],
    "mlp": ["mlp", "ffn", "feed_forward"],
    "ln1": ["ln_1", "ln1", "input_layernorm"],
    "ln2": ["ln_2", "ln2", "post_attention_layernorm"],
    "attn.hook_q": ["self_attn.q_proj", "attn.q_proj"],
    "attn.hook_k": ["self_attn.k_proj", "attn.k_proj"],
    "attn.hook_v": ["self_attn.v_proj", "attn.v_proj"],
    "attn.hook_result": ["self_attn.o_proj", "attn.c_proj", "attn.out_proj", "attn.o_proj"],
}

# TL-internal hooks that have NO native equivalent (HookPoint objects only).
# Listed explicitly so to_native_name raises rather than silently mangles.
_TL_INTERNAL_HOOKS: frozenset[str] = frozenset({
    "hook_resid_pre", "hook_resid_mid", "hook_resid_post",
    "hook_attn_in", "hook_attn_out",
    "hook_mlp_in", "hook_mlp_out",
    "hook_q_input", "hook_k_input", "hook_v_input",
    "attn.hook_pattern", "attn.hook_attn_scores", "attn.hook_z",
    "attn.hook_attn_in", "attn.hook_attn_out",
    "ln1.hook_normalized", "ln1.hook_scale",
    "ln2.hook_normalized", "ln2.hook_scale",
    "mlp.hook_pre", "mlp.hook_post",
})

# Track BOS warning so it only fires once per process.
_BOS_WARNING_FIRED = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def to_tl_name(native_name: str, arch_info: ArchInfo | None = None) -> str:
    """Translate a native PyTorch module name to the corresponding TL hook name.

    Examples::

        to_tl_name("transformer.h.8.mlp")       -> "blocks.8.mlp"
        to_tl_name("model.layers.3.self_attn.q_proj") -> "blocks.3.attn.hook_q"
        to_tl_name("blocks.5.hook_resid_pre")   -> "blocks.5.hook_resid_pre"
        # ↑ TL-internal name is preserved as-is (no native equivalent)
    """
    # If the input is already a TL hook name, preserve it.
    if native_name.startswith("blocks."):
        return native_name

    parts = _split_native_path(native_name)
    if parts is None:
        return native_name
    layer_idx, suffix = parts

    if suffix in _NATIVE_TO_TL:
        tl_suffix = _NATIVE_TO_TL[suffix]
        if tl_suffix:
            return f"blocks.{layer_idx}.{tl_suffix}"
        return f"blocks.{layer_idx}"

    # Unknown suffix: best-effort fallback that preserves any ``hook_`` prefix.
    if "hook_" in suffix:
        return f"blocks.{layer_idx}.{suffix}"
    return f"blocks.{layer_idx}.{suffix}"


def to_native_name(
    tl_name: str,
    arch_info: ArchInfo | None = None,
) -> str:
    """Translate a TL hook name to the corresponding native module name.

    Raises :class:`KeyError` for TL-internal hooks (``hook_resid_pre``,
    ``hook_pattern``, etc.) that have no native module equivalent — they
    name mid-computation tensors only TL exposes via injected HookPoints.

    Examples::

        to_native_name("blocks.8.mlp", arch)        -> "transformer.h.8.mlp"
        to_native_name("blocks.3.attn.hook_q", arch) -> "transformer.h.3.self_attn.q_proj"
        to_native_name("blocks.5.hook_resid_pre", arch)
        -> KeyError: "TL-internal hook 'hook_resid_pre' has no native equivalent."
    """
    tl_match = re.match(r"^blocks\.(\d+)(?:\.(.+))?$", tl_name)
    if tl_match is None:
        return tl_name

    idx = tl_match.group(1)
    tl_suffix = tl_match.group(2) or ""

    if tl_suffix in _TL_INTERNAL_HOOKS:
        raise KeyError(
            f"TL-internal hook {tl_suffix!r} has no native equivalent. "
            f"It names a mid-computation tensor exposed only by TransformerLens. "
            f"See list_roundtrippable_hooks() for the round-trippable subset."
        )

    prefix = _infer_native_prefix(arch_info)

    if tl_suffix == "":
        return f"{prefix}.{idx}"

    # Match the most-specific candidate first (e.g. "attn.hook_q" before "attn").
    for tl_key in sorted(_TL_TO_NATIVE.keys(), key=len, reverse=True):
        if tl_suffix == tl_key:
            candidates = _TL_TO_NATIVE[tl_key]
            if arch_info is not None:
                module_names = {m.name for m in arch_info.modules}
                for cand in candidates:
                    full = f"{prefix}.{idx}.{cand}" if cand else f"{prefix}.{idx}"
                    if full in module_names:
                        return full
            first = candidates[0]
            return f"{prefix}.{idx}.{first}" if first else f"{prefix}.{idx}"

    raise KeyError(
        f"TL hook {tl_name!r} is not in the round-trippable whitelist. "
        f"See list_roundtrippable_hooks() for the supported set."
    )


def list_roundtrippable_hooks() -> list[str]:
    """Return the set of TL hook names that round-trip cleanly to native names.

    Use this to filter a TL ``HookedTransformer.hook_dict`` down to hooks
    that interpkit's :func:`to_native_name` can translate without raising
    :class:`KeyError`.
    """
    return sorted(set(_TL_TO_NATIVE.keys()))


def list_tl_hooks(model: Any) -> list[str]:
    """List all TL hook point names on a HookedTransformer.

    Returns an empty list if the model is not a HookedTransformer.
    """
    hook_dict = getattr(model, "hook_dict", None)
    if hook_dict is not None:
        return sorted(hook_dict.keys())

    hooks: list[str] = []
    for name, mod in model.named_modules():
        if type(mod).__name__ == "HookPoint":
            hooks.append(name)
    return sorted(hooks)


def warn_bos_mismatch_once() -> None:
    """Emit a one-time UserWarning when cross-library comparison is detected.

    F-026: TL's ``to_tokens`` prepends a BOS token by default, HF's
    tokeniser does not. Comparing TL and interpkit outputs without
    ``prepend_bos=False`` causes ~40-unit logit differences and
    disagreeing top-1 predictions.
    """
    global _BOS_WARNING_FIRED
    if _BOS_WARNING_FIRED:
        return
    _BOS_WARNING_FIRED = True
    warnings.warn(
        "TransformerLens prepends a BOS token by default; HF and interpkit do "
        "not. For reproducible cross-library comparisons, pass "
        "prepend_bos=False to TL's `to_tokens` / `forward`. See the "
        "interpkit.core.tl_compat module docstring for details.",
        UserWarning,
        stacklevel=3,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_native_path(name: str) -> tuple[str, str] | None:
    """Split a native module path into ``(layer_index, suffix_after_layer)``.

    Returns ``None`` if the path doesn't contain a numeric layer segment.
    Walks the dotted segments in plain string ops — no regex on structural
    fields per the Phase 0g policy.
    """
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p.isdigit():
            suffix = ".".join(parts[i + 1:])
            return p, suffix
    return None


def _infer_native_prefix(arch_info: ArchInfo | None) -> str:
    """Infer the native layer prefix (e.g. ``transformer.h``)."""
    if arch_info is None:
        return "blocks"

    layer_names = getattr(arch_info, "layer_names", None) or []
    if layer_names:
        # Strip trailing ".{N}" to get the prefix without using regex.
        first = layer_names[0]
        parts = first.split(".")
        if parts and parts[-1].isdigit():
            return ".".join(parts[:-1])

    for mod in getattr(arch_info, "modules", None) or []:
        parts = mod.name.split(".")
        if parts and parts[-1].isdigit():
            return ".".join(parts[:-1])

    return "blocks"


__all__ = [
    "to_tl_name",
    "to_native_name",
    "list_tl_hooks",
    "list_roundtrippable_hooks",
    "warn_bos_mismatch_once",
]
