"""Architecture data contract: the single resolved-architecture type.

One canonical :class:`ArchInfo` describes everything ops need to know
about a model: family classification, residual-stream topology, block
list, head/embedding/pre-head/project-out wiring, and the per-layer
structural details (``layer_infos`` / ``modules``) that ops read.

Before the consolidation there were two aggregate types — a discovery
``ModelArchInfo`` and a resolver ``ArchInfo`` subclass — stitched
together. They are now one class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import torch.nn as nn

from interpkit.core.exceptions import ArchitectureSpecMismatch

# ---------------------------------------------------------------------------
# Family classification
# ---------------------------------------------------------------------------


class ArchFamily(str, Enum):
    """Architectural family used by the per-op support matrix.

    The classification is purely about which operations make sense, not
    about HF library taxonomy. ``CAUSAL_LM`` and ``SEQ2SEQ_LM`` are
    decoder transformers; ``MLM`` is encoder-only with a token-prediction
    head (BERT-family); ``ENCODER_ONLY`` is encoder-only without an LM
    head (classification-only); ``VISION_TRANSFORMER`` projects through a
    classifier head; ``CNN_RESIDUAL`` has skip connections;
    ``CNN_PLAIN`` does not.
    """

    CAUSAL_LM = "causal_lm"
    SEQ2SEQ_LM = "seq2seq_lm"
    MLM = "mlm"
    ENCODER_ONLY = "encoder_only"
    VISION_TRANSFORMER = "vision_transformer"
    CNN_RESIDUAL = "cnn_residual"
    CNN_PLAIN = "cnn_plain"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Sub-structures
# ---------------------------------------------------------------------------


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
    layer_type: str = "standard"
    attn_path: str | None = None
    mlp_path: str | None = None
    o_proj_path: str | None = None
    q_proj_path: str | None = None
    k_proj_path: str | None = None
    v_proj_path: str | None = None
    qkv_proj_path: str | None = None
    qkv_style: str = "unknown"
    qkv_layout: str = "concatenated"
    # N-007: deepest submodule of attn_path whose forward output is the
    # pre-residual, pre-LN attention output (i.e. ``W_O @ context + bias``).
    # For BERT-family wrappers this resolves to ``BertSelfOutput`` (the
    # parent of o_proj). For GPT/Llama-style "thin" attention modules
    # this equals ``attn_path`` because the wrapper IS the attention.
    # Guarantees the audit invariant
    # ``Σ_h (per_head[h] @ W_O[h]) + W_O.bias == output(attn_inner_path)``
    # holds across every family.
    attn_inner_path: str | None = None


@dataclass
class BlockSpec:
    """One transformer/CNN block in the model's flat block list.

    Hierarchical CNN architectures (stages-of-blocks) are flattened into
    a single list with per-block ``stage`` metadata so vision-aware ops
    can group / colour by stage when useful.
    """

    path: str
    stage: int | None = None
    has_attention: bool = False
    has_residual: bool = True
    # Computational mechanism of this block, detected structurally (not by
    # model class): "attention" | "recurrent" | "ssm" | "conv" | "mlp" |
    # "unknown". Lets hybrids (e.g. Griffin) and pure-SSM models (Mamba) be
    # described accurately and lets ops report per-mechanism coverage.
    mechanism: str = "unknown"


# ---------------------------------------------------------------------------
# The resolved-architecture contract
# ---------------------------------------------------------------------------


@dataclass
class ArchInfo:
    """Resolved architecture description used by every op.

    Carries both the resolver-driven fields (``family``, ``blocks``,
    ``head_module``, ``pre_head_module``, ...) and the per-layer
    structural fields (``layer_names``, ``layer_infos``, ``modules``)
    that ops read directly.

    Single source of truth for the block stack: for a flat (non-CNN)
    block stack ``layer_names`` is *derived from* ``blocks`` during
    resolution (see ``discover`` / ``_layer_names_from_blocks``), so the
    two views cannot drift. The legacy regex grouping is only used as a
    fallback when ``blocks`` is empty (shared-layer models whose physical
    block is wrapped in length-1 ``ModuleList``s) or hierarchical (CNN
    stages, which keep their per-stage ``layer_names`` view). New code
    should prefer the ``blocks`` / ``lm_blocks`` API; ``layer_infos`` /
    ``layer_names`` / ``modules`` remain for the ops that consume the
    per-layer attn/mlp/QKV detail.
    """

    # ------------------------------------------------------------------
    # HF config-derived metadata
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Resolver-driven fields
    # ------------------------------------------------------------------
    family: ArchFamily = ArchFamily.UNKNOWN
    spatial: bool = False

    embed_module: nn.Module | None = None
    embed_path: str | None = None

    head_module: nn.Module | None = None
    head_path: str | None = None

    blocks: list[BlockSpec] = field(default_factory=list)

    # N-002: encoder-decoder models populate ``decoder_blocks`` separately
    # so lens / dla can hook the decoder stack and ignore the encoder
    # (the head is wired to decoder hidden states for T5/BART/Flan-T5).
    # Empty list for non-seq2seq models.
    decoder_blocks: list[BlockSpec] = field(default_factory=list)

    pre_head_module: nn.Module | None = None
    pre_head_path: str | None = None

    project_out_module: nn.Module | None = None
    project_out_path: str | None = None

    # N-002: BERT-style MLM models route hidden states through a
    # multi-step head pipeline (dense → activation → LayerNorm → decoder)
    # before the token-projection. The single ``head_module`` (the
    # output ``decoder`` Linear) is not enough to reproduce the model's
    # logits — we need the entire wrapper. ``mlm_head_module`` is that
    # wrapper for BERT/RoBERTa/DeBERTa/ALBERT (a single module that
    # applies the full cascade in its forward). For DistilBERT (which
    # has no wrapper module) we record the components separately so
    # ``_apply_mlm_head`` can stitch them together.
    mlm_head_module: nn.Module | None = None
    mlm_head_path: str | None = None
    # DistilBERT-style: components live as siblings on the model root
    # rather than inside a single wrapper.
    distilbert_vocab_transform: nn.Module | None = None
    distilbert_vocab_layer_norm: nn.Module | None = None
    distilbert_vocab_projector: nn.Module | None = None

    # N-005: ALBERT and similar architectures share a single physical
    # transformer block across N "logical" layers (config.num_hidden_groups
    # == 1, num_hidden_layers > 1). When True, ops like decompose / dla
    # must use forward-call indexing in their hooks rather than path-based
    # storage, otherwise every "layer" key gets overwritten with the
    # final invocation's data.
    is_shared_layers: bool = False

    # N-004: DeBERTa-v3-style ``DisentangledSelfAttention`` has a known
    # broadcast bug in HF transformers' relative-position-bias path
    # that fires under forward hooks (used by trace, decompose,
    # attribute, head_activations, steer, probe, diff, ov_scores,
    # qk_scores). Detected at load time and surfaced via
    # ``check_op_supported`` for the affected ops.
    has_disentangled_attention: bool = False

    has_cls_token: bool = False

    # Residual-stream topology, used by ops/circuits.run_decompose,
    # ops/dla.run_dla, and the residual schema selector to decide
    # whether per-layer block deltas should be captured as separate
    # attn/mlp deltas (pre-LN) or as a single block_delta (post-LN).
    # Detected once in resolve_arch via the config-aware _resolve_topology
    # helper. Orthogonal to is_shared_layers; ops dispatch on the
    # cross product.
    residual_topology: Literal[
        "pre_ln", "post_ln", "seq2seq", "parallel"
    ] = "pre_ln"

    overrides_used: dict[str, Any] = field(default_factory=dict)

    # Cache for pipeline-validation outcome, set by support_matrix.
    _lens_validated: bool = False

    # Reference to the root model, set by ``resolve_arch`` so overrides
    # can re-resolve module paths. Not part of the dataclass schema for
    # comparison / repr purposes.
    _root: nn.Module | None = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Properties / helpers
    # ------------------------------------------------------------------

    @property
    def is_language_model(self) -> bool:
        return self.family in (
            ArchFamily.CAUSAL_LM, ArchFamily.SEQ2SEQ_LM, ArchFamily.MLM,
        )

    @property
    def is_vision_model(self) -> bool:
        return self.family in (
            ArchFamily.VISION_TRANSFORMER,
            ArchFamily.CNN_RESIDUAL,
            ArchFamily.CNN_PLAIN,
        )

    @property
    def is_cnn(self) -> bool:
        return self.family in (ArchFamily.CNN_RESIDUAL, ArchFamily.CNN_PLAIN)

    @property
    def attention_layer_indices(self) -> list[int]:
        """Indices of layers that have a resolved attention submodule."""
        return [li.index for li in self.layer_infos if li.attn_path is not None]

    @property
    def attention_layer_infos(self) -> list[LayerInfo]:
        """LayerInfo objects for layers that have a resolved attention submodule."""
        return [li for li in self.layer_infos if li.attn_path is not None]

    @property
    def is_hybrid(self) -> bool:
        """True when the model contains layers of different types."""
        types = {li.layer_type for li in self.layer_infos}
        return len(types) > 1

    # ------------------------------------------------------------------
    # Structural capabilities — what ops gate on (see support_matrix).
    # These are derived from detected structure, not a hard-coded family
    # list, so op support generalises to any HF architecture with the
    # right shape. The family enum is a human-readable descriptor; these
    # predicates are the authority for op gating.
    # ------------------------------------------------------------------

    @property
    def has_unembedding(self) -> bool:
        """A usable output head that projects hidden states to logits.

        True when a head path was resolved AND the family is not
        ``ENCODER_ONLY``. ``ENCODER_ONLY`` is the resolver's verdict that
        the model has no valid token/class unembedding (e.g. an ELECTRA
        discriminator whose only head is a binary classifier), so
        ``lens`` / ``dla`` must refuse even though a stray Linear may have
        been resolved as ``head_path``.
        """
        return bool(self.head_path) and self.family != ArchFamily.ENCODER_ONLY

    @property
    def has_residual_stream(self) -> bool:
        """A residual stream exists to decompose / project per block.

        True when the LM-path block stack is non-empty and at least one
        block writes a residual update. Distinguishes residual CNNs
        (ResNet) from plain CNNs (VGG), which have no residual stream.
        """
        return bool(self.lm_blocks) and any(b.has_residual for b in self.lm_blocks)

    @property
    def has_attention(self) -> bool:
        """At least one layer has a resolved attention submodule.

        Attention-based ops (attention, qk/ov scores, head_activations,
        circuits) require this. Pure-recurrent / SSM / CNN models report
        ``False``; hybrids (e.g. Griffin) report ``True`` and those ops
        cover the attention-bearing layers (see ``attention_layer_indices``).
        """
        return len(self.attention_layer_indices) > 0

    @property
    def block_mechanisms(self) -> list[str]:
        """Per-block computational mechanism for the LM-path block stack.

        Each entry is one of ``attention | recurrent | ssm | conv | mlp |
        unknown`` (see :class:`BlockSpec`). Useful for provenance / coverage
        reporting on hybrids (e.g. Griffin) and pure-SSM models (Mamba).
        """
        return [b.mechanism for b in self.lm_blocks]

    @property
    def is_generative(self) -> bool:
        """Autoregressive or seq2seq generation (used to gate ``chat``).

        Task-level capability: an LM head alone is not enough (MLM and
        vision classifiers have heads but do not generate).
        """
        return self.family in (ArchFamily.CAUSAL_LM, ArchFamily.SEQ2SEQ_LM)

    @property
    def lm_blocks(self) -> list[BlockSpec]:
        """Blocks where the residual stream that flows into the LM head lives.

        Encoder-decoder models route hidden states from the decoder to the
        LM head (encoder output enters via cross-attention; it is not on
        the path to logits). For seq2seq this returns ``decoder_blocks``;
        for everything else it returns ``blocks``.

        Single source of truth for residual schemas and dla — both
        consume ``arch.lm_blocks`` instead of branching on
        ``arch.family == SEQ2SEQ_LM``.
        """
        if self.family == ArchFamily.SEQ2SEQ_LM and self.decoder_blocks:
            return self.decoder_blocks
        return self.blocks

    @property
    def is_post_ln(self) -> bool:
        """Convenience accessor: True iff residual_topology == 'post_ln'."""
        return self.residual_topology == "post_ln"

    @property
    def needs_decoder_input_ids(self) -> bool:
        """True iff a ``decoder_input_ids`` must be injected for a forward pass.

        Encoder-decoder (seq2seq) models require a decoder start token when
        the caller supplies only encoder ``input_ids``. This is the single
        source of truth for that quirk (C2) — previously the same
        ``config.is_encoder_decoder`` check was duplicated in
        ``Model._inject_decoder_ids`` and ``attribute._attribute_from_encoded``.
        """
        return bool(self.is_encoder_decoder)

    def all_paths(self) -> list[str]:
        """Return every known module path for `model.arch_info.all_paths()`.

        Used by :func:`interpkit.core.paths.validate_module_path` to
        suggest close matches when the user passes a typo'd path.
        """
        paths: set[str] = set()
        for mi in self.modules:
            if mi.name:
                paths.add(mi.name)
        if self.embed_path:
            paths.add(self.embed_path)
        if self.head_path:
            paths.add(self.head_path)
        if self.pre_head_path:
            paths.add(self.pre_head_path)
        if self.project_out_path:
            paths.add(self.project_out_path)
        if self.mlm_head_path:
            paths.add(self.mlm_head_path)
        for b in self.blocks:
            paths.add(b.path)
        for b in self.decoder_blocks:
            paths.add(b.path)
        return sorted(paths)

    def layer_of(self, path: str) -> int | None:
        """Return the block-list index of the block that contains *path*.

        This is the canonical way to recover a layer index from a module
        path — ops should never call ``path.split('.')`` to do this.
        Returns ``None`` if *path* lives outside the block stack
        (embedding, head, final norm, etc.).
        """
        if not self.blocks:
            return None
        # Look for the longest matching block path prefix to handle
        # nested module paths like "transformer.h.4.attn.c_attn".
        best_match: tuple[int, int] | None = None
        for i, block in enumerate(self.blocks):
            if (path == block.path or path.startswith(block.path + ".")) and (
                best_match is None or len(block.path) > best_match[1]
            ):
                best_match = (i, len(block.path))
        return best_match[0] if best_match else None

    def with_overrides(self, overrides: dict[str, Any]) -> ArchInfo:
        """Apply user overrides to the resolved info, returning the modified instance.

        Delegates to the free function :func:`interpkit.core.arch.resolve.apply_overrides`
        so the override-application logic does not require an ``_root`` back-
        reference on the dataclass. Kept for backwards compatibility with
        ``resolve_arch``'s internal call site; new code should call
        :func:`apply_overrides` directly.
        """
        if not overrides:
            return self
        if self._root is None:
            raise ArchitectureSpecMismatch(
                "ArchInfo.with_overrides requires _root reference to be set "
                "(by resolve_arch). Call apply_overrides(model, arch, overrides) "
                "directly if you have ArchInfo detached from its container."
            )
        from interpkit.core.arch.resolve import apply_overrides

        return apply_overrides(self._root, self, overrides)

    def discovery_summary(self) -> str:
        """Return a human-readable summary of resolution outcomes."""
        lines = [
            f"Family: {self.family.value}",
            f"Residual topology: {self.residual_topology}",
            f"Spatial: {self.spatial}",
            f"Encoder-decoder: {self.is_encoder_decoder}",
        ]
        if self.is_shared_layers:
            lines.append("Shared-weight layers: True")
        if self.arch_family:
            lines.append(f"Architecture class: {self.arch_family}")
        if self.num_layers is not None:
            lines.append(f"Layers: {self.num_layers}")
        if self.hidden_size is not None:
            lines.append(f"Hidden size: {self.hidden_size}")
        if self.num_attention_heads is not None:
            lines.append(f"Attention heads: {self.num_attention_heads}")
        if self.num_key_value_heads is not None and self.num_key_value_heads != self.num_attention_heads:
            lines.append(f"KV heads: {self.num_key_value_heads}")
        lines.append(
            f"Head: {self.head_path or '(unresolved)'} → "
            f"{type(self.head_module).__name__ if self.head_module else 'None'}"
        )
        lines.append(
            f"Embedding: {self.embed_path or '(unresolved)'} → "
            f"{type(self.embed_module).__name__ if self.embed_module else 'None'}"
        )
        lines.append(
            f"Pre-head: {self.pre_head_path or '(unresolved)'} → "
            f"{type(self.pre_head_module).__name__ if self.pre_head_module else 'None'}"
        )
        if self.project_out_path:
            lines.append(f"project_out: {self.project_out_path}")
        lines.append(f"Blocks: {len(self.blocks)}")
        if self.overrides_used:
            lines.append(f"Overrides applied: {sorted(self.overrides_used)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ArchInfo(family={self.family.value!r}, "
            f"residual_topology={self.residual_topology!r}, "
            f"arch_family={self.arch_family!r}, "
            f"blocks={len(self.blocks)}, "
            f"head={self.head_path!r}, "
            f"is_encoder_decoder={self.is_encoder_decoder})"
        )
