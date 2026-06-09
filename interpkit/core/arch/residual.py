"""Residual-stream decomposition schemas.

Single source of truth for ``run_decompose``. Each schema captures the
per-layer components such that ``Σ components ≈ residual`` to fp32 epsilon
by construction — no audit-driven branches inside the ops.

The four schemas:

- :class:`PreLNResidual` — GPT-2, Llama, Qwen, Pythia, OPT-125m. Per-
  layer components are ``attn`` and ``mlp`` captured as sublayer
  outputs (which ARE the residual deltas in pre-LN). The schema also
  handles BLOOM via a hook-target adjustment: BLOOM's
  ``self_attention`` and ``mlp`` submodules return ``residual + delta``
  (residual is passed positionally and added internally). For BLOOM
  we subtract the captured block input from each submodule output
  before treating it as a delta. Verified against
  ``transformers.models.bloom.modeling_bloom.BloomBlock.forward``
  (audit2 REAUDIT_2026-05-09.md P0c finding).

- :class:`PostLNResidual` — BERT, RoBERTa, DistilBERT, ELECTRA, ALBERT,
  OPT-350m, post-LN OPT variants. Per-layer components are single
  ``block_delta`` entries ``block_output_i − block_output_{i-1}``.
  ``Σ block_deltas + embed = block_output_N = residual`` by
  telescoping. Loses ``attn``/``mlp`` granularity in the components
  list — explicit tradeoff documented in the run_decompose API
  contract.

- :class:`SharedLayerResidual` — composes Pre-LN or Post-LN with
  per-call indexing for shared-weight architectures (ALBERT). The
  physical block module is invoked N times per forward; per-call
  hooks (:func:`interpkit.ops._hooks.register_per_call_hook`) capture
  each invocation as a separate logical layer.

- :class:`Seq2seqResidual` — T5, Flan-T5, BART. Rooted at
  ``arch.lm_blocks`` (which equals ``arch.decoder_blocks`` for seq2seq).
  The pre-Phase-2 ``run_decompose`` iterated both encoder AND decoder
  blocks against the decoder-final residual (audit2 P0a stack
  mismatch); this schema constrains hooks to the decoder stack only.

:func:`residual_schema_for` dispatches on
``(arch.residual_topology, arch.is_shared_layers, arch.family)`` and
returns the appropriate schema. Returns ``None`` for unsupported
combinations so ``run_decompose`` can raise a clean
``OperationNotSupportedForArchitecture``.

Shared-weight contract: shared-layer models (ALBERT) always have a
non-empty ``arch.blocks`` after ``resolve_arch`` synthesises N logical
blocks pointing at the one physical block. This module no longer
re-derives that fallback; if ``arch.lm_blocks`` is empty the schema
selector returns ``None`` (caller raises).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import torch
import torch.nn as nn

from interpkit.ops._hooks import register_capture_hook, register_per_call_hook

if TYPE_CHECKING:
    from interpkit.core.arch.types import ArchInfo, BlockSpec, LayerInfo
    from interpkit.core.model import Model


__all__ = [
    "Component",
    "ResidualSchema",
    "PreLNResidual",
    "PostLNResidual",
    "SharedLayerResidual",
    "Seq2seqResidual",
    "residual_schema_for",
]


@dataclass
class Component:
    """One row in :func:`run_decompose`'s ``components`` list.

    ``type`` is one of ``"embed"``, ``"attn"``, ``"mlp"``, ``"block_delta"``.
    Per-topology API contract:

    - Pre-LN models (and BLOOM): ``c["type"] in {"embed", "attn", "mlp"}``.
    - Post-LN models: ``c["type"] in {"embed", "block_delta"}``.
    - Seq2seq: matches the underlying topology (pre-LN for T5/BART).

    ``layer`` is -1 for ``embed`` and 0..N-1 otherwise. ``vector`` is
    the captured tensor at the analyzed position (sliced from the
    full activation captured during forward).
    """
    name: str
    layer: int
    type: str
    vector: torch.Tensor

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "layer": self.layer,
            "type": self.type,
            "vector": self.vector,
            "norm": self.vector.norm().item(),
        }


class ResidualSchema(Protocol):
    """Protocol for residual-stream decomposition schemas.

    Each schema is responsible for running one forward pass with the
    appropriate hooks, then returning the (embed, components, final)
    triple. Ops that consume this protocol do not branch on topology;
    they call :func:`residual_schema_for` once and dispatch.
    """

    def decompose(
        self,
        model: Model,
        prepared_input: Any,
        *,
        position: int,
    ) -> tuple[torch.Tensor, list[Component], torch.Tensor]:
        """Return ``(embed_vector, components, residual_vector)`` at ``position``.

        All tensors are in fp32 (caller is responsible for casting to
        the model's native dtype if needed; the schema accumulates in
        fp32 for numerical correctness).
        """
        ...


def _slice_position(t: torch.Tensor, position: int) -> torch.Tensor:
    """Slice the position dimension out of a (B?, S, D) or (S, D) or (D,) tensor."""
    if t.dim() == 3:
        return t[0, position, :]
    if t.dim() == 2:
        return t[position, :]
    return t


def _get_module(model: nn.Module, path: str) -> nn.Module:
    """Resolve a dotted path; ``ModuleList`` indices are integer-coerced."""
    obj: Any = model
    for part in path.split("."):
        if part.isdigit() and isinstance(obj, (nn.ModuleList, nn.Sequential)):
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


# ---------------------------------------------------------------------------
# PreLNResidual
# ---------------------------------------------------------------------------


class PreLNResidual:
    """Pre-LN schema with BLOOM hook-target adjustment.

    Pre-LN blocks (GPT-2, Llama, Qwen, Pythia, OPT-125m, BLOOM) compute
    ``block_output = block_input + attn_delta + mlp_delta``. Per-layer
    components are ``attn`` (the attn submodule output) and ``mlp``
    (the mlp submodule output).

    BLOOM is a special case at the hook-target level: its
    ``self_attention`` and ``mlp`` submodules return ``residual + delta``
    (residual is passed positionally inside the block and added
    internally). The schema detects BLOOM via ``model_type == "bloom"``
    on the config and subtracts the captured block input from each
    submodule output before treating it as a delta. Result: same
    user-facing ``attn`` / ``mlp`` components, ``Σ ≈ residual`` to
    fp32 epsilon on both standard pre-LN and BLOOM.
    """

    def __init__(self, blocks: list[BlockSpec], *, is_bloom: bool = False):
        self._blocks = blocks
        self._is_bloom = is_bloom

    def decompose(
        self,
        model: Model,
        prepared_input: Any,
        *,
        position: int,
    ) -> tuple[torch.Tensor, list[Component], torch.Tensor]:
        attn_caps: dict[str, torch.Tensor] = {}
        mlp_caps: dict[str, torch.Tensor] = {}
        block_inputs: dict[str, torch.Tensor] = {}
        block_outputs: dict[str, torch.Tensor] = {}

        handles: list[torch.utils.hooks.RemovableHandle] = []

        # Map block-paths to their layer_infos so we can find attn/mlp anchors.
        path_to_li: dict[str, LayerInfo] = {}
        for li in model.arch_info.layer_infos:
            path_to_li[li.name] = li

        # Hook block input + output (always; BLOOM needs input for the
        # delta-subtraction, telescoping fallback needs both).
        for block in self._blocks:
            mod = _get_module(model._model, block.path)
            handles.append(_register_pre_input_hook(mod, block_inputs, block.path))
            handles.append(register_capture_hook(mod, block_outputs, block.path))

            li = path_to_li.get(block.path)
            if li is None:
                # No discovered layer info → no attn/mlp anchors → skip
                # the sublayer captures for this block. The block-level
                # input/output is still captured.
                continue

            if li.attn_path:
                try:
                    attn_mod = _get_module(model._model, li.attn_path)
                except (AttributeError, IndexError, KeyError, TypeError):
                    attn_mod = None
                if attn_mod is not None:
                    handles.append(
                        register_capture_hook(attn_mod, attn_caps, block.path),
                    )
            if li.mlp_path:
                try:
                    mlp_mod = _get_module(model._model, li.mlp_path)
                except (AttributeError, IndexError, KeyError, TypeError):
                    mlp_mod = None
                if mlp_mod is not None:
                    handles.append(
                        register_capture_hook(mlp_mod, mlp_caps, block.path),
                    )

        try:
            with torch.no_grad():
                model._forward(prepared_input)
        finally:
            for h in handles:
                h.remove()

        # Embed is the input to the first block (post token + position +
        # post-embed-LN; family-agnostic anchor).
        first_block_path = self._blocks[0].path
        if first_block_path not in block_inputs:
            raise RuntimeError(
                f"PreLNResidual: did not capture input to first block "
                f"{first_block_path!r}; the forward did not invoke it."
            )
        embed_full = block_inputs[first_block_path].float()
        embed_vec = _slice_position(embed_full, position)

        # Final residual is the output of the last block.
        last_block_path = self._blocks[-1].path
        if last_block_path not in block_outputs:
            raise RuntimeError(
                f"PreLNResidual: did not capture output of last block "
                f"{last_block_path!r}."
            )
        final_full = block_outputs[last_block_path].float()
        final_vec = _slice_position(final_full, position)

        components: list[Component] = []
        for idx, block in enumerate(self._blocks):
            inp_full = block_inputs.get(block.path)
            if inp_full is None:
                continue
            inp_full = inp_full.float()

            attn_full = attn_caps.get(block.path)
            mlp_full = mlp_caps.get(block.path)

            if attn_full is not None:
                attn_full = attn_full.float()
                if self._is_bloom:
                    # BLOOM: self_attention returns input + attn_delta.
                    delta = attn_full - inp_full
                else:
                    # Standard pre-LN: attn submodule returns just delta.
                    delta = attn_full
                components.append(Component(
                    name=f"L{idx}.attn",
                    layer=idx,
                    type="attn",
                    vector=_slice_position(delta, position),
                ))

            if mlp_full is not None:
                mlp_full = mlp_full.float()
                if self._is_bloom:
                    # BLOOM: mlp returns attn_output + mlp_delta. The
                    # post-attn baseline equals self_attention output.
                    if attn_full is not None:
                        baseline = attn_full
                    else:
                        baseline = inp_full
                    delta = mlp_full - baseline
                else:
                    delta = mlp_full
                components.append(Component(
                    name=f"L{idx}.mlp",
                    layer=idx,
                    type="mlp",
                    vector=_slice_position(delta, position),
                ))

        return embed_vec, components, final_vec


# ---------------------------------------------------------------------------
# PostLNResidual
# ---------------------------------------------------------------------------


class PostLNResidual:
    """Post-LN schema. One ``block_delta`` per layer.

    Post-LN blocks (BERT, RoBERTa, DistilBERT, ELECTRA, OPT-350m,
    post-LN OPT variants) compute
    ``block_output = LN(attn + LN(input) + mlp_input)`` — there is no
    algebraic split of the LN nonlinearity that yields ``attn`` and
    ``mlp`` summands whose pair-sum equals the residual delta.

    Instead we emit a single ``block_delta = block_output_i -
    block_output_{i-1}`` per layer. ``Σ block_deltas + embed =
    block_output_N = residual`` by telescoping. Loses ``attn`` / ``mlp``
    granularity — explicit tradeoff in the user-facing API.

    For ALBERT (post-LN + shared layers), use
    :class:`SharedLayerResidual` wrapping this class.
    """

    def __init__(self, blocks: list[BlockSpec]):
        self._blocks = blocks

    def decompose(
        self,
        model: Model,
        prepared_input: Any,
        *,
        position: int,
    ) -> tuple[torch.Tensor, list[Component], torch.Tensor]:
        block_inputs: dict[str, torch.Tensor] = {}
        block_outputs: dict[str, torch.Tensor] = {}

        handles: list[torch.utils.hooks.RemovableHandle] = []
        for block in self._blocks:
            mod = _get_module(model._model, block.path)
            handles.append(_register_pre_input_hook(mod, block_inputs, block.path))
            handles.append(register_capture_hook(mod, block_outputs, block.path))

        try:
            with torch.no_grad():
                model._forward(prepared_input)
        finally:
            for h in handles:
                h.remove()

        first_path = self._blocks[0].path
        last_path = self._blocks[-1].path

        if first_path not in block_inputs or last_path not in block_outputs:
            raise RuntimeError(
                "PostLNResidual: failed to capture block input/output "
                "for the configured block range."
            )

        embed_full = block_inputs[first_path].float()
        embed_vec = _slice_position(embed_full, position)
        final_full = block_outputs[last_path].float()
        final_vec = _slice_position(final_full, position)

        components: list[Component] = []
        prev_full = embed_full
        for idx, block in enumerate(self._blocks):
            out_full = block_outputs.get(block.path)
            if out_full is None:
                continue
            out_full = out_full.float()
            delta_full = out_full - prev_full
            components.append(Component(
                name=f"L{idx}.block_delta",
                layer=idx,
                type="block_delta",
                vector=_slice_position(delta_full, position),
            ))
            prev_full = out_full

        return embed_vec, components, final_vec


# ---------------------------------------------------------------------------
# SharedLayerResidual
# ---------------------------------------------------------------------------


class SharedLayerResidual:
    """Wrap Pre-LN or Post-LN with per-call indexing for shared-weight models.

    ALBERT and similar architectures (``config.num_hidden_groups=1,
    num_hidden_layers>1``) invoke a single physical block N times per
    forward pass. Path-keyed hooks would overwrite — per-call indexing
    via :func:`interpkit.ops._hooks.register_per_call_hook` writes each
    invocation to a different ``"{path}#{counter}"`` key.

    Composes with :class:`PreLNResidual` or :class:`PostLNResidual` for
    the per-layer-delta math; this class is just bookkeeping for the
    shared-module hook discipline.
    """

    def __init__(
        self,
        blocks: list[BlockSpec],
        *,
        topology: str,
        is_bloom: bool = False,
    ):
        if topology not in ("pre_ln", "post_ln"):
            raise ValueError(
                f"SharedLayerResidual: topology={topology!r} not supported; "
                "expected 'pre_ln' or 'post_ln'."
            )
        self._blocks = blocks
        self._topology = topology
        self._is_bloom = is_bloom

    def decompose(
        self,
        model: Model,
        prepared_input: Any,
        *,
        position: int,
    ) -> tuple[torch.Tensor, list[Component], torch.Tensor]:
        # Deduplicate by id(): all logical blocks share the same physical
        # module under shared-weight architectures.
        physical_blocks: list[tuple[nn.Module, str]] = []
        seen_ids: set[int] = set()
        for block in self._blocks:
            mod = _get_module(model._model, block.path)
            if id(mod) in seen_ids:
                continue
            seen_ids.add(id(mod))
            physical_blocks.append((mod, block.path))

        block_inputs: dict[str, torch.Tensor] = {}
        block_outputs: dict[str, torch.Tensor] = {}

        handles: list[torch.utils.hooks.RemovableHandle] = []
        for mod, path in physical_blocks:
            handles.append(_register_pre_input_per_call_hook(mod, block_inputs, path))
            handles.append(register_per_call_hook(mod, block_outputs, path))

        try:
            with torch.no_grad():
                model._forward(prepared_input)
        finally:
            for h in handles:
                h.remove()

        # Collect captures sorted by call counter — order matters.
        physical_path = physical_blocks[0][1]
        N = len(self._blocks)
        captured_inputs = [block_inputs.get(f"{physical_path}#{i}") for i in range(N)]
        captured_outputs = [block_outputs.get(f"{physical_path}#{i}") for i in range(N)]

        if captured_inputs[0] is None or captured_outputs[-1] is None:
            raise RuntimeError(
                f"SharedLayerResidual: did not capture all {N} invocations "
                f"of {physical_path!r}; only got "
                f"{sum(1 for x in captured_inputs if x is not None)} inputs."
            )

        embed_full = captured_inputs[0].float()
        embed_vec = _slice_position(embed_full, position)
        final_full = captured_outputs[-1].float()
        final_vec = _slice_position(final_full, position)

        components: list[Component] = []

        if self._topology == "post_ln":
            prev_full = embed_full
            for idx in range(N):
                out_full = captured_outputs[idx]
                if out_full is None:
                    continue
                out_full = out_full.float()
                delta = out_full - prev_full
                components.append(Component(
                    name=f"L{idx}.block_delta",
                    layer=idx,
                    type="block_delta",
                    vector=_slice_position(delta, position),
                ))
                prev_full = out_full
        else:
            # pre_ln shared (rare; placeholder; no known model hits this combo)
            for idx in range(N):
                inp = captured_inputs[idx]
                out = captured_outputs[idx]
                if inp is None or out is None:
                    continue
                inp = inp.float()
                out = out.float()
                # Block-delta proxy (we cannot split attn/mlp without
                # per-submodule per-call hooks; defer that wiring).
                delta = out - inp
                components.append(Component(
                    name=f"L{idx}.block_delta",
                    layer=idx,
                    type="block_delta",
                    vector=_slice_position(delta, position),
                ))

        return embed_vec, components, final_vec


# ---------------------------------------------------------------------------
# Seq2seqResidual
# ---------------------------------------------------------------------------


class Seq2seqResidual:
    """Telescoping schema rooted at the decoder block list.

    For seq2seq models (T5, Flan-T5, BART), the residual stream that
    flows into ``lm_head`` lives on the decoder side. Encoder hidden
    states enter via cross-attention; they are not on the path to
    logits. Composes :class:`PostLNResidual` (telescoping block
    deltas) over ``arch.lm_blocks`` (= ``arch.decoder_blocks``).

    P0a stack-mismatch finding: the pre-Phase-2 ``run_decompose``
    iterated ``arch.layer_names`` (encoder + decoder) against the
    decoder-final residual, producing rel=180-778 on T5/Flan-T5/BART.
    This schema constrains hooks to the decoder stack.

    We use telescoping (one ``block_delta`` per decoder layer)
    instead of attn/mlp granularity because the resolver does not
    discover decoder-side ``layer_infos`` (only encoder-side). The
    invariant ``Σ block_deltas + embed = residual`` holds by
    telescoping regardless of pre/post-LN structure; the attn/mlp
    split would require a separate per-decoder-block sublayer
    resolution pass (deferred to a follow-up).

    T5 / Flan-T5 / BART do not require an embed-scale or lm_head
    scaling adjustment at the residual-stream level — verified in
    P0a (scale_embedding=None on T5/Flan-T5; scale_embedding=False on
    bart-base).
    """

    def __init__(self, decoder_blocks: list[BlockSpec]):
        self._impl = PostLNResidual(decoder_blocks)

    def decompose(
        self,
        model: Model,
        prepared_input: Any,
        *,
        position: int,
    ) -> tuple[torch.Tensor, list[Component], torch.Tensor]:
        return self._impl.decompose(model, prepared_input, position=position)


# ---------------------------------------------------------------------------
# Forward-pre hook helpers (internal — capture block inputs)
# ---------------------------------------------------------------------------


def _register_pre_input_hook(
    module: nn.Module,
    store: dict[str, torch.Tensor],
    key: str,
) -> torch.utils.hooks.RemovableHandle:
    """Register a forward-pre hook capturing the first positional argument.

    Used for ``block_input`` capture; mirrors :func:`register_capture_hook`
    but reads from the input side of the forward call.
    """
    def pre_hook(_mod: nn.Module, inp: tuple) -> None:
        if not inp:
            return
        head = inp[0]
        if isinstance(head, torch.Tensor):
            store[key] = head.detach()

    return module.register_forward_pre_hook(pre_hook)


def _register_pre_input_per_call_hook(
    module: nn.Module,
    store: dict[str, torch.Tensor],
    key_prefix: str,
) -> torch.utils.hooks.RemovableHandle:
    """Per-call forward-pre hook (mirror of :func:`register_per_call_hook`)."""
    counter = [0]

    def pre_hook(_mod: nn.Module, inp: tuple) -> None:
        if not inp:
            return
        head = inp[0]
        if isinstance(head, torch.Tensor):
            store[f"{key_prefix}#{counter[0]}"] = head.detach()
            counter[0] += 1

    return module.register_forward_pre_hook(pre_hook)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def residual_schema_for(arch: ArchInfo) -> ResidualSchema | None:
    """Pick a residual schema for *arch* based on (topology, is_shared, family).

    Returns ``None`` when no schema applies (caller should raise
    ``OperationNotSupportedForArchitecture``).

    Shared-weight models (ALBERT) already have ``arch.lm_blocks``
    populated by ``resolve_arch`` (N logical blocks pointing at the one
    physical block), so there is no fallback synthesis here: an empty
    block list always means "no schema".
    """
    topology = arch.residual_topology
    is_shared = arch.is_shared_layers

    if topology == "seq2seq":
        blocks = arch.lm_blocks
        if not blocks:
            return None
        return Seq2seqResidual(blocks)

    blocks = arch.lm_blocks
    if not blocks:
        return None

    # BLOOM detection: ``model_type == "bloom"`` produces pre_ln topology
    # in the resolver but needs the hook-target adjustment inside
    # PreLNResidual to subtract residual from submodule outputs.
    is_bloom = "bloom" in (arch.arch_family or "").lower()

    if topology == "pre_ln":
        if is_shared:
            return SharedLayerResidual(blocks, topology="pre_ln", is_bloom=is_bloom)
        return PreLNResidual(blocks, is_bloom=is_bloom)

    if topology == "post_ln":
        if is_shared:
            return SharedLayerResidual(blocks, topology="post_ln")
        return PostLNResidual(blocks)

    if topology == "parallel":
        # GPT-J / GPT-NeoX placeholder. No current model in our
        # support matrix uses this; return None for now.
        return None

    return None
