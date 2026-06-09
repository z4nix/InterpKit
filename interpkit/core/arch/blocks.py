"""Block discovery.

Finds the model's flat block list (transformer layers or flattened CNN
stages-of-blocks), the decoder block list for seq2seq models, and the
repeated-layer name groups used for legacy per-layer resolution.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import torch.nn as nn

from interpkit.core.arch.probe import _detect_block_metadata
from interpkit.core.arch.types import BlockSpec

if TYPE_CHECKING:
    from interpkit.core.arch.types import ModuleInfo


def _find_blocks(
    model: nn.Module,
    n_layers_hint: int | None,
    sample_input: Any,
) -> list[BlockSpec]:
    """Discover the model's block list. Handles flat and hierarchical structures.

    Pass 1 (flat): find a ``ModuleList`` of identical block classes whose
    length matches ``n_layers_hint``. Covers GPT-2, Llama, OPT, T5,
    ViT — the typical transformer.

    Pass 2 (flat without hint): same as pass 1 but takes the longest
    ``ModuleList`` of identical-class children. Used when no config
    provides ``num_hidden_layers`` (timm models).

    Pass 3 (hierarchical): find a ``ModuleList`` whose children each
    contain an inner ``ModuleList`` (CNN stages-of-blocks). Flatten into
    one block list with stage metadata.

    Pass 4 (sequential): timm CNNs sometimes use ``nn.Sequential`` for
    block lists; same selection rules apply.

    After candidate blocks are enumerated, ``_detect_block_metadata``
    runs once on the full model to fill in ``has_residual`` and
    ``has_attention`` per block.
    """
    blocks: list[BlockSpec] = []

    # Pass 1: ModuleList of n_layers identical blocks
    if n_layers_hint is not None and n_layers_hint > 0:
        for name, mod in model.named_modules():
            if (isinstance(mod, nn.ModuleList) and len(mod) == n_layers_hint
                    and _all_same_class(mod)):
                blocks = _flat_blocks(name, mod)
                break

    # Pass 2: longest ModuleList of identical-class children (no hint)
    if not blocks:
        best_flat: tuple[str, nn.ModuleList] | None = None
        for name, mod in model.named_modules():
            if (isinstance(mod, nn.ModuleList) and len(mod) >= 2 and _all_same_class(mod)
                    and (best_flat is None or len(mod) > len(best_flat[1]))
                    and not _looks_like_stage_list(mod)):
                best_flat = (name, mod)
        if best_flat is not None:
            blocks = _flat_blocks(best_flat[0], best_flat[1])

    # Pass 3: hierarchical (stages-of-blocks for CNN)
    if not blocks:
        for name, mod in model.named_modules():
            if isinstance(mod, nn.ModuleList) and _looks_like_stage_list(mod):
                blocks = _hierarchical_blocks(name, mod)
                if blocks:
                    break

    # Pass 4: nn.Sequential block list (timm CNN style)
    if not blocks:
        best_seq: tuple[str, nn.Sequential] | None = None
        for name, mod in model.named_modules():
            if (isinstance(mod, nn.Sequential) and len(mod) >= 2 and _all_same_class(mod)
                    and (best_seq is None or len(mod) > len(best_seq[1]))):
                best_seq = (name, mod)
        if best_seq is not None:
            blocks = _flat_blocks(best_seq[0], best_seq[1])

    # Single full-model forward to fill in per-block metadata
    if blocks:
        metadata = _detect_block_metadata(model, [b.path for b in blocks], sample_input)
        for b in blocks:
            meta = metadata.get(b.path, {})
            b.has_attention = meta.get("has_attention", False)
            b.has_residual = meta.get("has_residual", False)
            b.mechanism = meta.get("mechanism", "unknown")

    return blocks


def _all_same_class(container: nn.Module) -> bool:
    """All direct children share the same Python class."""
    children = list(container.children())
    if not children:
        return False
    first_class = type(children[0])
    return all(type(c) is first_class for c in children)


def _looks_like_stage_list(mod: nn.ModuleList) -> bool:
    """A ModuleList of stages each containing an inner ModuleList of blocks.

    Distinguishes CNN ``stages`` from a flat block list. We don't require
    all inner ModuleLists to have the same length (ResNet has [3,4,6,3]).
    """
    if len(mod) < 2:
        return False
    return all(_find_inner_modulelist(stage) is not None for stage in mod)


def _find_inner_modulelist(stage: nn.Module) -> tuple[str, nn.Module] | None:
    """Find a child ``ModuleList`` within a stage. Returns ``(attr_name, list)``."""
    for name, child in stage.named_children():
        if isinstance(child, (nn.ModuleList, nn.Sequential)) and len(child) >= 1:
            return name, child
    return None


def _flat_blocks(container_path: str, container: nn.Module) -> list[BlockSpec]:
    """Build BlockSpec list from a flat ``ModuleList`` / ``Sequential``.

    Block metadata (``has_attention`` / ``has_residual``) is filled in by
    :func:`interpkit.core.arch.probe._detect_block_metadata` after all
    blocks are enumerated.
    """
    return [
        BlockSpec(path=f"{container_path}.{i}", stage=None)
        for i in range(len(container))
    ]


def _hierarchical_blocks(container_path: str, container: nn.ModuleList) -> list[BlockSpec]:
    """Flatten a stages-of-blocks CNN into a single BlockSpec list."""
    out: list[BlockSpec] = []
    for stage_idx, stage in enumerate(container):
        inner = _find_inner_modulelist(stage)
        if inner is None:
            continue
        inner_attr, inner_list = inner
        for block_idx in range(len(inner_list)):
            out.append(BlockSpec(
                path=f"{container_path}.{stage_idx}.{inner_attr}.{block_idx}",
                stage=stage_idx,
            ))
    return out


def _find_decoder_blocks(
    model: nn.Module,
    n_layers_hint: int | None,
    sample_input: Any,
) -> list[BlockSpec]:
    """Locate the decoder block list on a seq2seq model.

    For T5/MT5/Marian/Pegasus: ``model.decoder.block`` is a ``ModuleList``
    of ``T5Block``s.

    For BART/MBart/Plbart: ``model.model.decoder.layers`` is a
    ``ModuleList`` of ``BartDecoderLayer``s.

    Falls back to a recursive search for any ``ModuleList`` whose dotted
    path contains ``decoder`` and whose length matches the layer hint.
    """
    candidates: list[tuple[str, nn.ModuleList]] = []
    for name, mod in model.named_modules():
        if (isinstance(mod, nn.ModuleList) and "decoder" in name and _all_same_class(mod)
                and (n_layers_hint is None or len(mod) == n_layers_hint)):
            candidates.append((name, mod))

    if not candidates:
        return []

    # Prefer the most specific (deepest) match.
    candidates.sort(key=lambda c: -len(c[0].split(".")))
    name, mod = candidates[0]
    decoder_blocks = _flat_blocks(name, mod)

    # Detect per-block metadata (has_attention / has_residual) on the
    # decoder forward path. Use the standard helper for consistency.
    metadata = _detect_block_metadata(model, [b.path for b in decoder_blocks], sample_input)
    for b in decoder_blocks:
        meta = metadata.get(b.path, {})
        b.has_attention = meta.get("has_attention", True)
        b.has_residual = meta.get("has_residual", True)
        b.mechanism = meta.get("mechanism", "unknown")
    return decoder_blocks


def _likely_block_prefixes(model: nn.Module) -> set[str]:
    """Approximate set of module-path prefixes that contain block stacks.

    Used by ``_find_pre_head_module`` to avoid picking norms that live
    inside transformer blocks.
    """
    out: set[str] = set()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 2 and _all_same_class(mod):
            out.add(name)
    return out


def _layer_names_from_blocks(blocks: list[BlockSpec] | None) -> list[str] | None:
    """Derive the legacy ``layer_names`` from an already-discovered block stack.

    Returns the block paths when *blocks* is a non-empty **flat** stack
    (every ``BlockSpec.stage`` is ``None``) so the legacy per-layer view
    (``layer_names`` / ``layer_infos``) and the :class:`BlockSpec` API share
    a single source of truth — eliminating the drift risk of two independent
    discoveries (``_find_blocks`` vs the ``_detect_layers`` regex grouping).

    Returns ``None`` for an empty or hierarchical (stage-tagged) stack,
    signalling the caller to fall back to :func:`_detect_layers`. This
    deliberately preserves the existing behaviour for two cases:

    - **Shared-layer models** (ALBERT): ``_find_blocks`` returns ``[]`` (the
      single physical block is wrapped in length-1 ``ModuleList``s), so the
      regex grouping is what locates the physical block path that the
      resolver's shared-layer synthesis later reuses.
    - **Hierarchical CNNs**: blocks carry ``stage`` metadata and the legacy
      per-stage ``layer_names`` view is consumed as-is by CNN-supporting ops.
    """
    if not blocks:
        return None
    if any(b.stage is not None for b in blocks):
        return None
    return [b.path for b in blocks]


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
