"""Architecture resolution orchestrator.

``resolve_arch`` runs three layers per model — overrides > conventions >
walker+hooks — and assembles a single :class:`ArchInfo`. Each layer fills
only the fields the previous layer could not; user overrides are applied
last so a faulty resolver never beats a manual hint.

- **Layer 1 (overrides)** — user-provided ``arch_override=`` wins
  unconditionally (applied last).
- **Layer 2 (conventions)** — HuggingFace ``PreTrainedModel`` accessors
  and timm attribute conventions.
- **Layer 3 (walker + runtime hooks)** — pure structural detection that
  works on any ``nn.Module`` that runs.

``discover`` is the structural-discovery entry point (module/layer/role
enumeration); ``resolve_arch`` calls it to populate the per-layer fields.

Validation contract: after resolution, ops with full-pipeline correctness
requirements (lens, dla) call
``interpkit.core.support_matrix.validate_lens_pipeline`` on first use per
``Model``. If the resolver picked wrong paths, that assertion fires loudly
with a diagnostic + ``arch_override`` workaround.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import torch
import torch.nn as nn

from interpkit.core.arch.blocks import (
    _detect_layers,
    _find_blocks,
    _find_decoder_blocks,
    _layer_names_from_blocks,
)
from interpkit.core.arch.family import (
    _classify_family,
    _detect_cls_token,
    _detect_disentangled_attention,
    _detect_shared_layers,
    _hf_meta,
    _parse_hf_config,
    _resolve_topology,
)
from interpkit.core.arch.heads import (
    _convention_find_embedding,
    _convention_find_head,
    _convention_find_num_classes,
    _detect_project_out,
    _find_distilbert_mlm_components,
    _find_intermediate_linear,
    _find_mlm_head_module,
    _find_mlm_project_out,
    _find_pre_head_module,
    _find_unembedding,
    _hf_find_classifier_head,
)
from interpkit.core.arch.layers import _assign_roles, _resolve_layer_info
from interpkit.core.arch.probe import (
    _classify_block_mechanism,
    _probe_output_shape,
    _walker_find_input_consumer,
    _walker_find_output_producer,
)
from interpkit.core.arch.tree import (
    _count_params,
    _is_pretrained_model,
    module_at_path,
    path_of,
)
from interpkit.core.arch.types import ArchFamily, ArchInfo, BlockSpec, ModuleInfo
from interpkit.core.exceptions import ArchitectureSpecMismatch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-family override dict for irreducible quirks
# ---------------------------------------------------------------------------

ARCH_OVERRIDES: dict[str, dict[str, Any]] = {
    "opt": {"may_have_project_out": True},
    "t5": {
        "relpos_bias_path": "encoder.block.0.layer.0.SelfAttention.relative_attention_bias",
        "decoder_relpos_path": "decoder.block.0.layer.0.SelfAttention.relative_attention_bias",
    },
    "deberta": {"disentangled_position": True},
    "swin": {"window_attention": True, "shifted": True},
    # Add per-family entries only when the test suite catches a quirk
    # that the conventions + walker cannot handle.
}


# ---------------------------------------------------------------------------
# Structural discovery (module / layer / role enumeration)
# ---------------------------------------------------------------------------


def discover(
    model: nn.Module,
    dummy_input: Any | None = None,
    *,
    blocks: list[BlockSpec] | None = None,
) -> ArchInfo:
    """Run full structural auto-discovery on a model.

    Enumerates named modules, detects the repeated-block layer stack,
    resolves per-layer attn/mlp/QKV structure, assigns module roles, and
    finds the unembedding / project_out. Returns an :class:`ArchInfo`
    populated with only the structural fields (family classification and
    the resolver-driven head/block fields are filled in by
    :func:`resolve_arch`).

    Parameters
    ----------
    model:
        Any ``nn.Module``, optionally with an HF ``.config`` attribute.
    dummy_input:
        If provided, used for a forward pass to capture output shapes.
        Can be a tensor, dict of tensors, or tuple of tensors.
    blocks:
        The block stack already discovered by :func:`_find_blocks`. When a
        non-empty **flat** stack is supplied, ``layer_names`` is derived
        from it so the legacy per-layer view and the :class:`BlockSpec` API
        describe the same stack by construction. When ``None`` / empty /
        hierarchical, falls back to the standalone :func:`_detect_layers`
        regex grouping (preserving shared-layer and CNN behaviour).
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
        except Exception:  # noqa: BLE001
            # Output shapes are best-effort (used only by inspect / scan).
            # A forward failure here — e.g. a seq2seq model fed a dummy input
            # without proper decoder_input_ids (MarianMT, T5, BART) — must NOT
            # abort discovery; layer detection and role assignment below do
            # not depend on captured shapes.
            pass
        finally:
            for h in hooks:
                h.remove()

        for info in module_infos:
            info.output_shape = shapes.get(info.name)

    # Find unembedding
    unembed_name = _find_unembedding(model)
    has_lm_head = unembed_name is not None

    # Detect layer names. Prefer the already-discovered block stack so the
    # legacy per-layer view (layer_names / layer_infos) and the BlockSpec
    # API share a single source of truth; fall back to the standalone regex
    # grouping for empty / hierarchical stacks (see _layer_names_from_blocks).
    layer_names = _layer_names_from_blocks(blocks)
    if layer_names is None:
        layer_names = _detect_layers(module_infos)

    if not layer_names:
        warnings.warn(
            "Could not auto-detect layer structure for this model. "
            "If this model has repeated blocks (transformer layers, SSM layers, etc.), "
            "use interpkit.register() to manually specify architecture components. "
            "See: https://github.com/z4nix/interpkit#register",
            stacklevel=2,
        )

    # Resolve per-layer structural details
    block_types = hf_meta.get("block_types")
    hidden_size_hint = hf_meta.get("hidden_size")
    layer_infos = [
        _resolve_layer_info(
            model, ln, idx, block_types=block_types, hidden_size=hidden_size_hint,
        )
        for idx, ln in enumerate(layer_names)
    ]

    if layer_names and all(li.layer_type == "recurrent" for li in layer_infos):
        warnings.warn(
            "All layers classified as recurrent — attention and MLP submodules "
            "were not found. If this model has standard transformer blocks, "
            "use interpkit.register() to manually specify attention and MLP paths.",
            stacklevel=2,
        )

    # Assign semantic roles using resolved structure
    _assign_roles(module_infos, model, layer_infos, layer_names, unembed_name)

    # Detect project_out for models with embed_dim != hidden_size
    project_out_path = _detect_project_out(model)

    # Check encoder-decoder
    config = getattr(model, "config", None)
    is_enc_dec = getattr(config, "is_encoder_decoder", False)

    return ArchInfo(
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


def _populate_legacy_fields(
    module: nn.Module,
    sample_input: Any | None,
    blocks: list[BlockSpec] | None = None,
) -> ArchInfo:
    """Run :func:`discover` to fill in the per-layer / module fields.

    *blocks* is the stack already found by :func:`_find_blocks`; it is passed
    through so ``discover`` can derive ``layer_names`` from it (single source
    of truth) instead of re-deriving the stack independently.

    Returns an empty ``ArchInfo`` on any failure so resolution completes
    even when discovery can't run (e.g. exotic models the resolver
    handles via the walker but discovery can't).
    """
    try:
        return discover(module, dummy_input=sample_input, blocks=blocks)
    except Exception:  # noqa: BLE001
        return ArchInfo()


# ---------------------------------------------------------------------------
# Override application
# ---------------------------------------------------------------------------


def apply_overrides(
    model: nn.Module, arch: ArchInfo, overrides: dict[str, Any],
) -> ArchInfo:
    """Apply user overrides to *arch*, returning the (mutated) instance.

    Each override key is a public field name on :class:`ArchInfo` (for
    path/scalar fields) or one of the special-cased aliases:

    - ``head_path`` / ``embed_path`` / ``pre_head_path`` /
      ``project_out_path`` / ``blocks_path`` — string paths re-resolved
      via :func:`module_at_path` rooted at *model*.
    - ``family`` — string or :class:`ArchFamily` enum.
    - ``residual_topology`` — string in
      ``{"pre_ln", "post_ln", "seq2seq", "parallel"}``; bypasses the
      ``_resolve_topology`` heuristic.
    - ``lm_blocks`` — either a single dotted path to a ``nn.ModuleList``
      container (re-resolved like ``blocks_path``) or a list of explicit
      block paths.

    Unknown keys raise :class:`ArchitectureSpecMismatch`. Legacy
    field names (``unembedding_name``, ``output_head_name``) are rejected
    with a migration message pointing at the canonical replacement.
    """
    if not overrides:
        return arch

    _legacy_renames = {
        "unembedding_name": "head_path",
        "output_head_name": "head_path",
    }

    valid_topology = {"pre_ln", "post_ln", "seq2seq", "parallel"}

    for key, value in overrides.items():
        if key in _legacy_renames:
            raise ArchitectureSpecMismatch(
                f"arch_override[{key!r}] was renamed; pass "
                f"{_legacy_renames[key]!r} instead."
            )
        if key == "head_path":
            arch.head_path = value
            arch.head_module = module_at_path(model, value)
        elif key == "embed_path":
            arch.embed_path = value
            arch.embed_module = module_at_path(model, value)
        elif key == "pre_head_path":
            arch.pre_head_path = value
            arch.pre_head_module = module_at_path(model, value)
        elif key == "project_out_path":
            arch.project_out_path = value
            arch.project_out_module = module_at_path(model, value)
        elif key == "blocks_path":
            container = module_at_path(model, value)
            if not isinstance(container, nn.ModuleList):
                raise ArchitectureSpecMismatch(
                    f"arch_override['blocks_path']={value!r} resolves to "
                    f"{type(container).__name__}, expected nn.ModuleList."
                )
            arch.blocks = [
                BlockSpec(path=f"{value}.{i}", stage=None)
                for i in range(len(container))
            ]
        elif key == "lm_blocks":
            if isinstance(value, str):
                container = module_at_path(model, value)
                if not isinstance(container, nn.ModuleList):
                    raise ArchitectureSpecMismatch(
                        f"arch_override['lm_blocks']={value!r} resolves to "
                        f"{type(container).__name__}, expected nn.ModuleList."
                    )
                # Stored on decoder_blocks so the lm_blocks property
                # dispatches consistently for seq2seq users; for non-
                # seq2seq the property falls back to arch.blocks anyway.
                arch.decoder_blocks = [
                    BlockSpec(path=f"{value}.{i}", stage=None)
                    for i in range(len(container))
                ]
            elif isinstance(value, list) and all(isinstance(p, str) for p in value):
                arch.decoder_blocks = [
                    BlockSpec(path=p, stage=None) for p in value
                ]
            else:
                raise ArchitectureSpecMismatch(
                    "arch_override['lm_blocks'] must be a dotted path str "
                    "or a list[str] of explicit block paths."
                )
        elif key == "family":
            arch.family = ArchFamily(value) if isinstance(value, str) else value
        elif key == "residual_topology":
            if value not in valid_topology:
                raise ArchitectureSpecMismatch(
                    f"arch_override['residual_topology']={value!r} not in "
                    f"{sorted(valid_topology)}."
                )
            arch.residual_topology = value
        elif key in arch.__dataclass_fields__:
            setattr(arch, key, value)
        else:
            raise ArchitectureSpecMismatch(
                f"Unknown arch_override key: {key!r}. "
                f"Valid keys: head_path, embed_path, pre_head_path, "
                f"project_out_path, blocks_path, lm_blocks, family, "
                f"residual_topology, plus public ArchInfo fields."
            )

    arch.overrides_used = dict(overrides)
    return arch


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def resolve_arch(
    module: nn.Module,
    sample_input: Any | None = None,
    *,
    arch_override: dict[str, Any] | None = None,
) -> ArchInfo:
    """Resolve a model's architecture using HF accessors > conventions > walker.

    Parameters
    ----------
    module:
        The PyTorch ``nn.Module``. May or may not be a HF PreTrainedModel.
    sample_input:
        A small input the model can run (text, image tensor, dict for HF).
        Required for runtime-hook detection. If ``None``, the resolver
        will still complete but spatial / residual / pre-head detection
        is skipped (paths-only resolution, suitable for inspection).
    arch_override:
        Optional dict of explicit overrides — applied last so the user
        always wins over detection. Valid keys: ``head_path``,
        ``embed_path``, ``pre_head_path``, ``project_out_path``,
        ``blocks_path``, ``family``, plus public ``ArchInfo`` fields.

    Returns
    -------
    ArchInfo
        Populated with both the resolver fields (``family``, ``blocks``,
        ``head_path``, etc.) and the per-layer structural fields used by
        existing ops.
    """
    arch_override = arch_override or {}

    found: dict[str, Any] = {}
    hf_meta: dict[str, Any] = {}

    # Layer 2a: HuggingFace accessors
    if _is_pretrained_model(module):
        try:
            embed = module.get_input_embeddings()
            if embed is not None:
                found.setdefault("embed_module", embed)
        except (AttributeError, NotImplementedError):
            pass
        try:
            head = module.get_output_embeddings()
        except (AttributeError, NotImplementedError):
            head = None
        if head is None:
            head = _hf_find_classifier_head(module)
        if head is not None:
            found.setdefault("head_module", head)
        hf_meta = _hf_meta(module)

    # Layer 2b: timm + common conventions
    if "embed_module" not in found:
        embed = _convention_find_embedding(module)
        if embed is not None:
            found["embed_module"] = embed
    if "head_module" not in found:
        num_classes = _convention_find_num_classes(module)
        head = _convention_find_head(module, num_classes=num_classes)
        if head is not None:
            found["head_module"] = head

    # Layer 3: walker fallback for fields still missing
    if sample_input is not None:
        if "embed_module" not in found:
            embed = _walker_find_input_consumer(module, sample_input)
            if embed is not None:
                found["embed_module"] = embed
        if "head_module" not in found:
            num_classes = _convention_find_num_classes(module)
            head = _walker_find_output_producer(module, sample_input, num_classes=num_classes)
            if head is not None:
                found["head_module"] = head

    # Block discovery (Phase 0b) — always via walker
    n_layers_hint = hf_meta.get("num_layers")
    blocks: list[BlockSpec] = []
    if sample_input is not None:
        blocks = _find_blocks(module, n_layers_hint, sample_input)

    # Pre-head + project_out (Phase 0c).
    # For encoder-decoder models, bias the structural fallback toward the
    # decoder side — that's where lens/dla logically live for seq2seq.
    pre_head_module: nn.Module | None = None
    pre_head_path: str | None = None
    project_out_module: nn.Module | None = None
    project_out_path: str | None = None
    if "head_module" in found and sample_input is not None:
        prefer_prefix: str | None = None
        if hf_meta.get("is_encoder_decoder"):
            for candidate in ("decoder", "model.decoder"):
                try:
                    module_at_path(module, candidate)
                    prefer_prefix = candidate
                    break
                except (AttributeError, IndexError, KeyError, TypeError):
                    continue
        pre_head_module, pre_head_path = _find_pre_head_module(
            module, found["head_module"], sample_input, prefer_prefix=prefer_prefix,
        )
        project_out_module, project_out_path = _find_intermediate_linear(
            module, pre_head_module, found["head_module"],
            hidden_size=hf_meta.get("hidden_size"),
        )

    # Family classification — uses model output shape to distinguish
    # token-logit LMs from class-logit classifiers, regardless of how
    # ``get_output_embeddings`` is wired.
    is_enc_dec = bool(hf_meta.get("is_encoder_decoder", False))
    is_classification = False
    has_lm_head = False
    if sample_input is not None and "head_module" in found:
        out_shape = _probe_output_shape(module, sample_input)
        if out_shape is not None:
            # Output shape (B,) or (B, num_classes) → classification.
            # Output shape (B, seq, vocab) → LM.
            if len(out_shape) <= 2:
                is_classification = True
            else:
                vocab = hf_meta.get("vocab_size")
                if vocab is not None and out_shape[-1] == vocab:
                    has_lm_head = True
                elif len(out_shape) == 2:
                    is_classification = True
                else:
                    has_lm_head = True

    if sample_input is not None:
        family, spatial = _classify_family(
            module, blocks, sample_input,
            is_encoder_decoder=is_enc_dec,
            has_lm_head=has_lm_head,
            is_classification=is_classification,
        )
    else:
        family, spatial = ArchFamily.UNKNOWN, False

    # Build ArchInfo (also populate legacy fields from hf_meta + walker)
    embed_path = path_of(module, found["embed_module"]) if "embed_module" in found else None
    head_path = path_of(module, found["head_module"]) if "head_module" in found else None

    # Populate the per-layer structural fields (layer_names, layer_infos,
    # modules) via the discovery machinery so every op keeps working.
    legacy = _populate_legacy_fields(module, sample_input, blocks)

    # N-002: resolve MLM head pipeline + decoder blocks for seq2seq.
    mlm_head_module: nn.Module | None = None
    mlm_head_path: str | None = None
    distilbert_components: tuple[nn.Module | None, nn.Module | None, nn.Module | None] = (None, None, None)
    if family == ArchFamily.MLM:
        mlm_head_module, mlm_head_path = _find_mlm_head_module(module)
        distilbert_components = _find_distilbert_mlm_components(module)

        # N-006: when the MLM model has hidden_size != embedding_size
        # (ELECTRA, ALBERT), the head includes a hidden→embedding dense
        # Linear before the final decoder. Surface that Linear as
        # ``project_out`` so DLA / lens can project the residual stream
        # through it, identical to how OPT's ``project_out`` is used.
        if project_out_module is None:
            extra_proj_out, extra_proj_out_path = _find_mlm_project_out(
                module, mlm_head_module, distilbert_components,
            )
            if extra_proj_out is not None:
                project_out_module = extra_proj_out
                project_out_path = extra_proj_out_path

    decoder_blocks: list[BlockSpec] = []
    if family == ArchFamily.SEQ2SEQ_LM and sample_input is not None:
        decoder_blocks = _find_decoder_blocks(module, n_layers_hint, sample_input)

    # N-005: ALBERT-style shared-layer detection.
    is_shared_layers = _detect_shared_layers(module)

    # N-002 (shared-layer lens): ALBERT's single physical block is wrapped in
    # length-1 ModuleLists that block discovery skips, leaving ``blocks=[]`` —
    # which made lens/dla return a soft "no block detected" None. Synthesise N
    # logical blocks all pointing at the one physical block path so every op
    # that consumes ``arch.blocks`` / ``arch.lm_blocks`` works on shared-weight
    # models, instead of each op re-deriving the workaround.
    if not blocks and is_shared_layers and legacy.layer_infos:
        n_shared = hf_meta.get("num_layers") or legacy.num_layers
        physical = legacy.layer_infos[0]
        physical_path = physical.name
        if n_shared and physical_path:
            # The synthesised logical blocks must describe the one physical
            # block faithfully so capability/mechanism detection sees the same
            # structure a non-shared model would expose (has_residual defaults
            # True; carry has_attention + mechanism from the physical block).
            try:
                mech = _classify_block_mechanism(module_at_path(module, physical_path))
            except (AttributeError, IndexError, KeyError, TypeError):
                mech = "unknown"
            has_attn = physical.attn_path is not None
            blocks = [
                BlockSpec(path=physical_path, has_attention=has_attn, mechanism=mech)
                for _ in range(int(n_shared))
            ]

    # N-004: DeBERTa-v3 DisentangledSelfAttention detection.
    has_disentangled_attention = _detect_disentangled_attention(module)

    arch = ArchInfo(
        family=family,
        spatial=spatial,
        embed_module=found.get("embed_module"),
        embed_path=embed_path,
        head_module=found.get("head_module"),
        head_path=head_path,
        blocks=blocks,
        decoder_blocks=decoder_blocks,
        pre_head_module=pre_head_module,
        pre_head_path=pre_head_path,
        project_out_module=project_out_module,
        project_out_path=project_out_path or legacy.project_out_path,
        mlm_head_module=mlm_head_module,
        mlm_head_path=mlm_head_path,
        distilbert_vocab_transform=distilbert_components[0],
        distilbert_vocab_layer_norm=distilbert_components[1],
        distilbert_vocab_projector=distilbert_components[2],
        is_shared_layers=is_shared_layers,
        has_disentangled_attention=has_disentangled_attention,
        has_cls_token=_detect_cls_token(module),
        residual_topology=_resolve_topology(getattr(module, "config", None)),
        # Scalar metadata populated from both hf_meta and discovery
        arch_family=hf_meta.get("arch_family") or legacy.arch_family,
        num_layers=hf_meta.get("num_layers") or legacy.num_layers,
        hidden_size=hf_meta.get("hidden_size") or legacy.hidden_size,
        num_attention_heads=hf_meta.get("num_attention_heads") or legacy.num_attention_heads,
        num_key_value_heads=hf_meta.get("num_key_value_heads") or legacy.num_key_value_heads,
        vocab_size=hf_meta.get("vocab_size") or legacy.vocab_size,
        has_lm_head=("head_module" in found),
        output_head_name=head_path or legacy.output_head_name,
        unembedding_name=head_path or legacy.unembedding_name,
        is_encoder_decoder=is_enc_dec,
        # Per-layer structural details — needed by ops not on the blocks API
        modules=legacy.modules,
        layer_names=legacy.layer_names,
        layer_infos=legacy.layer_infos,
    )
    arch._root = module

    # Layer 1: apply user overrides (wins over everything)
    if arch_override:
        arch = arch.with_overrides(arch_override)

    # Validate that resolved paths exist on the module
    _validate_resolved_paths(module, arch)

    # Contract (N-002): a shared-layer model with a known layer structure must
    # always expose synthesised logical blocks, so every op that consumes
    # ``arch.blocks`` / ``arch.lm_blocks`` works on shared-weight models. This
    # replaces the old ``residual._synth_shared_lm_blocks`` fallback — the
    # synthesis now happens once, here, and is asserted instead of re-derived.
    if arch.is_shared_layers and arch.layer_infos and (arch.num_layers or 0) > 0:
        assert arch.blocks, (
            "shared-layer model resolved with empty blocks; N-002 block "
            "synthesis failed (expected non-empty arch.blocks)"
        )

    return arch


def _validate_resolved_paths(module: nn.Module, arch: ArchInfo) -> None:
    """Sanity-check that every resolved path exists on the module.

    Catches override typos and resolver bugs at construction time.
    """
    for label, path in [
        ("embed_path", arch.embed_path),
        ("head_path", arch.head_path),
        ("pre_head_path", arch.pre_head_path),
        ("project_out_path", arch.project_out_path),
    ]:
        if path is None:
            continue
        try:
            module_at_path(module, path)
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise ArchitectureSpecMismatch(
                f"Resolved {label}={path!r} but module not found on model: {exc}"
            ) from None
