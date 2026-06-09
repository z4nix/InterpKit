"""Family classification, residual-stream topology, and config parsing.

Determines the :class:`ArchFamily` (and spatial flag) from HF config
metadata + structure, the residual topology, and the various
architecture flags (shared layers, disentangled attention, CLS token).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import torch.nn as nn

from interpkit.core.arch.types import ArchFamily

if TYPE_CHECKING:
    from interpkit.core.arch.types import BlockSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HF config metadata
# ---------------------------------------------------------------------------


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

    for attr in ("block_types", "layers_block_type"):
        val = getattr(config, attr, None)
        if val is not None:
            info["block_types"] = list(val)
            break

    return info


def _hf_meta(module: nn.Module) -> dict[str, Any]:
    """Extract HF config metadata for the ArchInfo scalar fields."""
    config = getattr(module, "config", None)
    if config is None:
        return {}
    info: dict[str, Any] = {"arch_family": type(module).__name__}
    for src, dst in [
        (("num_hidden_layers", "n_layer", "num_layers", "n_layers"), "num_layers"),
        (("hidden_size", "n_embd", "d_model"), "hidden_size"),
        (("num_attention_heads", "n_head", "num_heads"), "num_attention_heads"),
        (("num_key_value_heads", "num_kv_heads"), "num_key_value_heads"),
        (("vocab_size",), "vocab_size"),
    ]:
        for attr in src:
            val = getattr(config, attr, None)
            if val is not None:
                info[dst] = val
                break
    info["is_encoder_decoder"] = getattr(config, "is_encoder_decoder", False)
    info["model_type"] = getattr(config, "model_type", None)
    return info


# ---------------------------------------------------------------------------
# Family classification
# ---------------------------------------------------------------------------


# Precision-exception overrides applied BEFORE config.architectures
# suffix matching. New entries belong here only when an audit-caught
# regression demonstrates that suffix matching produces the wrong family
# for a specific class — keep small.
#
# ElectraForPreTraining: binary-discriminator head, no vocab-sized
# projection. Must classify as ENCODER_ONLY so lens fails loudly with
# OperationNotSupportedForArchitecture instead of silently picking a
# stray vocab-sized Linear and emitting garbage tokens.
#
# ElectraForCausalLM: uses a multi-step MLM-style head pipeline
# (generator_predictions wrapping dense + activation + LayerNorm +
# generator_lm_head) even though the class name suggests causal LM.
# lens / dla need the MLM head path to reproduce HF's logits exactly.
_TYPE_NAME_FAMILY_OVERRIDES: dict[str, ArchFamily] = {
    "ElectraForPreTraining": ArchFamily.ENCODER_ONLY,
    "ElectraForCausalLM": ArchFamily.MLM,
}


# config.architectures suffix priority order. Scoped to the LM families
# present in ArchFamily today. Non-LM task suffixes
# (ForQuestionAnswering, ForTokenClassification, ForSequenceClassification,
# ForImageClassification, ForMultipleChoice) fall through to Layer 3's
# is_classification branch in _family_from_type_name.
_SUFFIX_PRIORITY: list[tuple[str, ArchFamily]] = [
    ("ForMaskedLM",              ArchFamily.MLM),
    ("ForCausalLM",              ArchFamily.CAUSAL_LM),
    ("ForConditionalGeneration", ArchFamily.SEQ2SEQ_LM),
]


# Dedup set for the disagreement warning emitted when config.architectures
# and the type-name fallback would produce different families. Keyed by
# (type(model).__name__, tuple(architectures)) so the message fires at
# most once per (class, metadata) combination per process.
_FAMILY_DISAGREEMENT_WARNED: set[tuple[str, tuple[str, ...]]] = set()


# Topology-level post-LN model_types.
_POST_LN_MODEL_TYPES_FOR_TOPOLOGY = frozenset({
    "bert", "roberta", "distilbert", "electra", "albert",
    "deberta", "deberta-v2", "deberta-v3",
    "xlm-roberta", "camembert", "mobilebert", "convbert",
    "bigbird", "ernie", "luke", "rembert",
})


def _family_from_arch_suffixes(suffixes: list[str]) -> ArchFamily | None:
    """Match the first config.architectures entry whose suffix is in _SUFFIX_PRIORITY.

    Returns the highest-priority family that matches any architecture
    in *suffixes*, or None if no suffix matched. Multiple architectures
    may list conflicting suffixes — first match in priority order wins.
    """
    for suffix, family in _SUFFIX_PRIORITY:
        for arch in suffixes:
            if arch.endswith(suffix):
                return family
    return None


def _family_from_type_name(
    model: nn.Module,
    blocks: list[BlockSpec],
    *,
    is_encoder_decoder: bool,
    has_lm_head: bool,
    is_classification: bool,
) -> ArchFamily | None:
    """Layer-3 type-name fallback: classify from type(model).__name__ + structure.

    Used when config.architectures is empty or didn't match a known suffix
    (custom modules built via load_module, exotic checkpoints). Retains the
    full structural classification — encoder-decoder, MLM-vs-CausalLM,
    vision-classifier-vs-CNN — that the pre-3-layer ``_classify_family``
    body used to perform.
    """
    if is_encoder_decoder:
        return ArchFamily.SEQ2SEQ_LM

    if has_lm_head and not is_classification:
        class_name = type(model).__name__
        if class_name.endswith("ForMaskedLM"):
            return ArchFamily.MLM
        return ArchFamily.CAUSAL_LM

    if is_classification:
        has_attention_blocks = any(b.has_attention for b in blocks)
        has_conv = _has_conv_layer(model)
        if has_attention_blocks:
            # Hybrids (e.g. CoAtNet) are still treated as ViT for op support.
            return ArchFamily.VISION_TRANSFORMER
        if has_conv:
            has_residual_blocks = any(b.has_residual for b in blocks) if blocks else False
            return (ArchFamily.CNN_RESIDUAL if has_residual_blocks
                    else ArchFamily.CNN_PLAIN)
        return ArchFamily.UNKNOWN

    if any(b.has_attention for b in blocks):
        # Encoder transformer with no detectable LM head — typically
        # AutoModel-loaded base classes. Surface as ENCODER_ONLY so ops
        # that need a head fail with OperationNotSupportedForArchitecture.
        if _is_encoder_only(model):
            return ArchFamily.ENCODER_ONLY
        return ArchFamily.CAUSAL_LM

    return ArchFamily.UNKNOWN


def _classify_family(
    model: nn.Module,
    blocks: list[BlockSpec],
    sample_input: Any,
    *,
    is_encoder_decoder: bool,
    has_lm_head: bool,
    is_classification: bool,
) -> tuple[ArchFamily, bool]:
    """Classify model family + spatial-flag via a 3-layer order.

    Layer 1: ``_TYPE_NAME_FAMILY_OVERRIDES`` (precision exceptions).
    Layer 2: ``config.architectures`` suffix match (canonical HF metadata).
    Layer 3: type-name + structure fallback (handles encoder-decoder,
             encoder-only, classification, vision branches).

    When Layer 2 and Layer 3 produce different LM families on the same
    model, a ``logger.warning`` fires once per (class, suffixes) pair —
    the user can override with ``arch_override={"family": "..."}``.

    Returns ``(family, spatial)``:
    - ``family`` is the architectural class for op-support checks.
    - ``spatial=True`` means lens / dla must pool spatial activations
      before applying the head (vision classifiers).
    """
    # Spatial flag is independent of family: it's true iff this is a
    # vision-style classifier. Computed once and reused below.
    has_attention_blocks = any(b.has_attention for b in blocks)
    has_conv = _has_conv_layer(model)
    spatial = bool(is_classification and (has_attention_blocks or has_conv))

    cls = type(model).__name__

    # Layer 1: type-name overrides take precedence over everything.
    if cls in _TYPE_NAME_FAMILY_OVERRIDES:
        return _TYPE_NAME_FAMILY_OVERRIDES[cls], spatial

    # Encoder-decoder is structural; config.architectures cannot override it.
    if is_encoder_decoder:
        return ArchFamily.SEQ2SEQ_LM, False

    # Layer 2: config.architectures suffix match.
    archs = getattr(getattr(model, "config", None), "architectures", None) or []
    suffixes = [a for a in archs if isinstance(a, str)]
    family_from_config = _family_from_arch_suffixes(suffixes)

    # Layer 3: type-name + structure fallback (also covers vision /
    # CNN / encoder-only branches that Layer 2's LM-suffix scope misses).
    family_from_type = _family_from_type_name(
        model, blocks,
        is_encoder_decoder=is_encoder_decoder,
        has_lm_head=has_lm_head,
        is_classification=is_classification,
    )

    family = family_from_config or family_from_type or ArchFamily.UNKNOWN

    # Audit trail: warn once when sources disagree on the LM family.
    # Skip the warning for spatial classifiers (Layer 2 doesn't carry
    # vision-classifier suffixes; the type-name path is canonical there).
    if (
        family_from_config is not None
        and family_from_type is not None
        and family_from_config != family_from_type
        and not spatial
    ):
        key = (cls, tuple(suffixes))
        if key not in _FAMILY_DISAGREEMENT_WARNED:
            _FAMILY_DISAGREEMENT_WARNED.add(key)
            logger.warning(
                "family classifier picked %s from config.architectures=%s, "
                "but model class is %r (type-name fallback would classify "
                "as %s). Pass arch_override={'family': '<value>'} if this "
                "is wrong.",
                family_from_config.value,
                suffixes,
                cls,
                family_from_type.value,
            )

    return family, spatial


def _is_mlm_model(model: nn.Module) -> bool:
    """True iff the HF model class name ends with ``ForMaskedLM``.

    Pure suffix-match check. Retained for use by ``_find_mlm_head_module``
    and tests; the family classifier itself does not call this directly.
    """
    return type(model).__name__.endswith("ForMaskedLM")


def _resolve_topology(
    config: Any,
) -> Literal["pre_ln", "post_ln", "seq2seq", "parallel"]:
    """Pick the residual-stream topology for this model.

    Config-aware dispatch — never reads ``model_type`` in isolation
    when the topology depends on a per-checkpoint config flag.

    - ``opt``: ``do_layer_norm_before=True`` is pre-LN (opt-125m);
      ``False`` is post-LN (opt-350m). Topology cannot be inferred
      from ``model_type == "opt"`` alone.
    - ``bloom``: residual is added INSIDE the attention/MLP submodules
      (see ``BloomBlock.forward``). The block-level topology is pre-LN
      regardless of ``apply_residual_connection_post_layernorm``; the
      hook-target adjustment (subtract block input from each submodule
      output) is what makes the schema correct on BLOOM.
    - BERT-family ``_POST_LN_MODEL_TYPES``: classic post-LN blocks.
    - Default: pre-LN.

    Encoder-decoder models return ``"seq2seq"`` regardless of
    model_type so the residual schema selector dispatches to the
    seq2seq adapter (which roots itself on ``arch.lm_blocks`` =
    decoder blocks).
    """
    if config is None:
        return "pre_ln"
    if getattr(config, "is_encoder_decoder", False):
        return "seq2seq"
    model_type = getattr(config, "model_type", None)
    if model_type == "opt":
        return "pre_ln" if getattr(config, "do_layer_norm_before", True) else "post_ln"
    if model_type == "bloom":
        # Block-level topology is pre-LN; the hook-target adjustment
        # lives in PreLNResidual's BLOOM branch.
        return "pre_ln"
    if model_type in _POST_LN_MODEL_TYPES_FOR_TOPOLOGY:
        return "post_ln"
    return "pre_ln"


def _is_encoder_only(model: nn.Module) -> bool:
    """Encoder-only model: HF base class (no task head) for an encoder.

    True for ``BertModel``, ``RobertaModel``, ``DistilBertModel``, etc.
    when loaded via ``AutoModel`` (no task suffix).
    """
    class_name = type(model).__name__
    config = getattr(model, "config", None)
    is_decoder = getattr(config, "is_decoder", False) if config is not None else False
    if is_decoder:
        return False
    # Match the "raw" base classes — no task suffix.
    encoder_only_classes = {
        "BertModel", "RobertaModel", "DistilBertModel", "AlbertModel",
        "ElectraModel", "DebertaModel", "DebertaV2Model",
        "XLMRobertaModel", "MobileBertModel", "ConvBertModel",
        "CamembertModel", "XLMRobertaXLModel", "BigBirdModel",
        "ErnieModel", "LukeModel", "RemBertModel",
    }
    return class_name in encoder_only_classes


def _has_conv_layer(model: nn.Module) -> bool:
    """Quick scan: any 2D convolution module under *model*."""
    return any(isinstance(m, nn.Conv2d) for m in model.modules())


def _detect_cls_token(module: nn.Module) -> bool:
    """ViT models have a learned [CLS] token concatenated to the patch sequence.

    Heuristic: look for a parameter named ``cls_token`` on the model.
    """
    return any(
        name.endswith("cls_token") or name == "cls_token"
        for name, _p in module.named_parameters()
    )


def _detect_shared_layers(model: nn.Module) -> bool:
    """Detect ALBERT-style architectures that share a single physical block.

    True when:
      - HF config exposes ``num_hidden_groups`` and it's set to ``1``.
      - ``num_hidden_layers`` (or ``num_layers``) > 1 (otherwise sharing
        is moot).

    ALBERT is the canonical case (``num_hidden_groups=1, inner_group_num=1,
    num_hidden_layers=12``); any future architecture that uses the same
    config convention picks up the flag automatically without code edits.
    """
    config = getattr(model, "config", None)
    if config is None:
        return False
    num_groups = getattr(config, "num_hidden_groups", None)
    if num_groups is None or num_groups != 1:
        return False
    n_layers = (
        getattr(config, "num_hidden_layers", None)
        or getattr(config, "num_layers", None)
    )
    if n_layers is None or n_layers <= 1:
        return False
    return True


def _detect_disentangled_attention(model: nn.Module) -> bool:
    """Detect DeBERTa-v3-style ``DisentangledSelfAttention`` modules.

    True when the model contains any submodule whose class name contains
    "DisentangledSelfAttention". DeBERTa-v3's relative-position-bias
    code path has a known broadcast bug in HF transformers when forward
    hooks fire on the attention module — affects trace / decompose /
    attribute / head_activations / steer / probe / diff / ov_scores /
    qk_scores. Surfaced by ``check_op_supported`` so users get a clean
    ``OperationNotSupportedForArchitecture`` rather than a cryptic
    ``RuntimeError: tensor (512) must match (7)`` deep in the forward.
    """
    return any(
        "DisentangledSelfAttention" in type(mod).__name__
        for mod in model.modules()
    )
