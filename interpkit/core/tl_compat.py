"""TransformerLens interop — bidirectional name translation between native and TL hook names."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from interpkit.core.discovery import ModelArchInfo

# ── TL canonical hook names ──────────────────────────────────────
#
# TL standardizes all transformer architectures to:
#   blocks.{N}.hook_resid_pre
#   blocks.{N}.hook_resid_post
#   blocks.{N}.attn.hook_q / hook_k / hook_v / hook_z / hook_result
#   blocks.{N}.attn.hook_pattern
#   blocks.{N}.hook_attn_out
#   blocks.{N}.hook_mlp_out
#   blocks.{N}.mlp.hook_pre / hook_post
#   blocks.{N}.ln1 / ln2
#
# These map to native names like:
#   transformer.h.{N}           -> blocks.{N}
#   transformer.h.{N}.attn      -> blocks.{N}.attn
#   transformer.h.{N}.mlp       -> blocks.{N}.mlp
#   model.layers.{N}.self_attn  -> blocks.{N}.attn
#   model.layers.{N}.mlp        -> blocks.{N}.mlp

# Patterns for extracting layer index and component from native names
_NATIVE_LAYER_RE = re.compile(
    r"^(?P<prefix>.+?)[.\[](?P<idx>\d+)[.\]]*(?P<suffix>.*)$"
)

# Common native -> TL component mappings
_COMPONENT_TO_TL: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\.self_attn$|\.attn$|\.attention$", re.I), ".attn"),
    (re.compile(r"\.mlp$|\.ffn$|\.feed_forward$", re.I), ".mlp"),
    (re.compile(r"\.ln_?1$|\.input_layernorm$", re.I), ".ln1"),
    (re.compile(r"\.ln_?2$|\.post_attention_layernorm$", re.I), ".ln2"),
    (re.compile(r"\.self_attn\.q_proj$|\.attn\.q_proj$", re.I), ".attn.hook_q"),
    (re.compile(r"\.self_attn\.k_proj$|\.attn\.k_proj$", re.I), ".attn.hook_k"),
    (re.compile(r"\.self_attn\.v_proj$|\.attn\.v_proj$", re.I), ".attn.hook_v"),
    (re.compile(r"\.self_attn\.o_proj$|\.attn\.c_proj$|\.attn\.out_proj$", re.I), ".attn.hook_result"),
]

# TL hook -> native component suffix patterns
_TL_TO_COMPONENT: list[tuple[str, list[str]]] = [
    (".attn", ["attn", "self_attn", "attention"]),
    (".mlp", ["mlp", "ffn", "feed_forward"]),
    (".ln1", ["ln_1", "ln1", "input_layernorm"]),
    (".ln2", ["ln_2", "ln2", "post_attention_layernorm"]),
    (".attn.hook_q", ["attn.q_proj", "self_attn.q_proj"]),
    (".attn.hook_k", ["attn.k_proj", "self_attn.k_proj"]),
    (".attn.hook_v", ["attn.v_proj", "self_attn.v_proj"]),
    (".attn.hook_result", ["attn.c_proj", "attn.out_proj", "self_attn.o_proj"]),
]


def to_tl_name(native_name: str, arch_info: "ModelArchInfo | None" = None) -> str:
    """Translate a native PyTorch module name to the corresponding TL hook name.

    Examples::

        to_tl_name("transformer.h.8.mlp")   -> "blocks.8.mlp"
        to_tl_name("transformer.h.8.attn")  -> "blocks.8.attn"
        to_tl_name("model.layers.3.self_attn.q_proj") -> "blocks.3.attn.hook_q"
    """
    m = _NATIVE_LAYER_RE.match(native_name)
    if m is None:
        return native_name

    idx = m.group("idx")
    raw_suffix = m.group("suffix")

    # Normalize suffix to always start with "." for pattern matching
    clean_suffix = raw_suffix.lstrip(".")
    dotted_suffix = f".{clean_suffix}" if clean_suffix else ""

    # Try specific component mappings first
    for pattern, tl_suffix in _COMPONENT_TO_TL:
        if dotted_suffix and pattern.search(dotted_suffix):
            return f"blocks.{idx}{tl_suffix}"

    # Bare layer reference (e.g. "transformer.h.8")
    if not clean_suffix:
        return f"blocks.{idx}"

    # Fallback: preserve suffix as-is under blocks.{N}
    return f"blocks.{idx}.{clean_suffix}"


def to_native_name(
    tl_name: str,
    arch_info: "ModelArchInfo | None" = None,
) -> str:
    """Translate a TL hook name back to the most likely native module name.

    Requires ``arch_info`` from a loaded model to resolve the native layer prefix
    (e.g. ``transformer.h`` vs ``model.layers``). Without it, returns a best-guess.

    Examples::

        to_native_name("blocks.8.mlp", arch_info) -> "transformer.h.8.mlp"
        to_native_name("blocks.3.attn.hook_q", arch_info) -> "transformer.h.3.attn.q_proj"
    """
    # Parse TL name: blocks.{N}.{rest}
    tl_match = re.match(r"^blocks\.(\d+)(?:\.(.+))?$", tl_name)
    if tl_match is None:
        return tl_name

    idx = tl_match.group(1)
    tl_suffix = tl_match.group(2) or ""

    # Determine native layer prefix from arch_info
    prefix = _infer_native_prefix(arch_info)

    if not tl_suffix:
        return f"{prefix}.{idx}"

    # Try specific TL -> native mappings
    for tl_component, native_candidates in _TL_TO_COMPONENT:
        tl_component_clean = tl_component.lstrip(".")
        if tl_suffix == tl_component_clean:
            # Pick the first candidate that exists in the module tree, or fall back to first
            if arch_info is not None:
                module_names = {m.name for m in arch_info.modules}
                for candidate in native_candidates:
                    full = f"{prefix}.{idx}.{candidate}"
                    if full in module_names:
                        return full
            return f"{prefix}.{idx}.{native_candidates[0]}"

    # Strip TL-specific "hook_" prefixes for unknown suffixes
    clean = re.sub(r"hook_", "", tl_suffix)
    return f"{prefix}.{idx}.{clean}"


def list_tl_hooks(model: Any) -> list[str]:
    """List all TL hook point names on a HookedTransformer.

    Returns an empty list if the model is not a HookedTransformer.
    """
    hook_dict = getattr(model, "hook_dict", None)
    if hook_dict is not None:
        return sorted(hook_dict.keys())

    # Fallback: look for HookPoint modules
    hooks = []
    for name, mod in model.named_modules():
        if type(mod).__name__ == "HookPoint":
            hooks.append(name)
    return sorted(hooks)


def _infer_native_prefix(arch_info: "ModelArchInfo | None") -> str:
    """Infer the native layer name prefix (e.g. 'transformer.h', 'model.layers')."""
    if arch_info is None:
        return "blocks"

    if arch_info.layer_names:
        first = arch_info.layer_names[0]
        # Strip trailing .{digit} to get prefix
        m = re.match(r"^(.+?)\.\d+$", first)
        if m:
            return m.group(1)

    # Scan modules for repeating indexed patterns
    for mod in arch_info.modules:
        m = re.match(r"^(.+?)\.\d+$", mod.name)
        if m:
            return m.group(1)

    return "blocks"
