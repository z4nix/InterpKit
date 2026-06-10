"""lens — logit lens with family-aware projection (F-003 / F-004 / F-005).

Pre-1.0 interpkit's lens read ``model(x, output_hidden_states=True).hidden_states[-1]``
and projected each layer through ``ln_f`` + ``lm_head``. This had three
correctness bugs:

- F-003: on encoder-decoder models (T5/BART), ``hidden_states`` returns
  the encoder hidden states, but the projection used the decoder's lm_head.
  Result: garbage token rankings.
- F-004: HuggingFace's ``hidden_states[-1]`` semantics differ across
  architectures. For OPT, HF applies ``final_layer_norm`` *in-forward*
  before storing the last hidden state; interpkit then applied it
  *again* (double-LN) when projecting. Top-1 disagreed with model logits.
- F-005: GPT-2 lens disagreed with TransformerLens at the final layer.
  This is a known TL-side reformulation difference (TL folds ``unembed.b``
  into ``ln_final``); not fixable in interpkit. Documented.

The 1.0 fix: hook the **output of the last block directly** (which is
unambiguously pre-final-norm on every family), then apply the
family-appropriate projection pipeline:

- LM (causal / seq2seq): ``pre_head`` (LayerNorm) → ``project_out`` (OPT only)
  → ``head`` (lm_head) → token logits.
- Vision transformer (ViT): pool spatial dims (CLS token or mean) → optional
  ``pre_head`` LN → ``head`` (classifier) → class logits.
- CNN: pool spatial dims (mean) → ``head`` (classifier) → class logits.

Same code structure for every family; only the projection step differs.
The validation contract (Phase 0e) auto-runs on first use to catch any
resolver drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from rich.console import Console

from interpkit.core.exceptions import LensPipelineMismatch
from interpkit.core.inputs import input_seq_len
from interpkit.core.paths import validate_position
from interpkit.core.support_matrix import (
    check_op_supported,
    lens_blocks,
    validate_lens_pipeline,
)

if TYPE_CHECKING:
    from interpkit.core.model import Model

console = Console()


def run_lens(
    model: Model,
    text: Any,
    *,
    save: str | None = None,
    html: str | None = None,
    position: int | None = None,
    kind: str = "logit",
    tuned_lens: Any = None,
) -> list[dict[str, Any]] | None:
    """Project each block's output through the head pipeline.

    The same primary code path works for LMs, ViTs, and CNNs — only the
    projection step differs by family (Phase 3 of the plan).

    For LMs: returns layer-by-layer next-token predictions. For vision
    models: returns layer-by-layer class-probability evolution (the
    "vision lens" technique).

    Parameters
    ----------
    text:
        Input. String for text models; image path for vision models.
    position:
        Token position to report (LMs only; ignored for vision).
        Default ``None`` reports all positions; ``-1`` for last token.
    kind:
        ``"logit"`` (default) — raw logit lens. ``"tuned"`` — apply
        trained per-block translators (Belrose et al. 2023) before the
        head projection; requires *tuned_lens*.
    tuned_lens:
        A :class:`~interpkit.ops.tuned_lens.TunedLens`, or a path to a
        saved one. Required when ``kind="tuned"``; an untrained lens
        reproduces the logit lens exactly (identity translators).

    Returns
    -------
    list[dict] | None
        Per-block dicts with ``layer_name``, ``lens_kind``,
        ``top1_token``, ``top1_prob``, ``top5_tokens``, ``top5_probs``,
        and ``positions`` (LM only). Returns ``None`` if the model has
        no head we can project through (e.g. headless encoder).
    """
    from interpkit.core.enums import VALID_LENS_KINDS, _validate_enum
    from interpkit.core.render import render_lens

    _validate_enum(kind, VALID_LENS_KINDS, "kind")
    arch = model.arch_info
    check_op_supported("lens", arch)

    translator_lens = None
    if kind == "tuned":
        from interpkit.ops.tuned_lens import TunedLens, load_tuned_lens

        if tuned_lens is None:
            raise ValueError(
                "lens(kind='tuned') needs a tuned lens — pass tuned_lens="
                "<TunedLens or path>. Train one with "
                "model.train_tuned_lens(corpus, save=...)."
            )
        if isinstance(tuned_lens, TunedLens):
            translator_lens = tuned_lens
        else:
            translator_lens = load_tuned_lens(tuned_lens, model=model)
    elif tuned_lens is not None:
        raise ValueError(
            "tuned_lens was passed but kind='logit' — did you mean "
            "kind='tuned'?"
        )

    # N-002: pick the family-appropriate block list. Seq2seq lens hooks
    # the decoder stack; MLM/causal/vision use ``arch.blocks`` directly.
    blocks = lens_blocks(arch)

    has_head = arch.head_module is not None or arch.mlm_head_module is not None
    if not has_head or not blocks:
        console.print(
            f"\n  [yellow]lens not available:[/yellow] no head/block detected"
            f" for {arch.arch_family or 'this model'}.\n"
        )
        return None

    if translator_lens is not None:
        translator_lens.validate_against(arch, [b.path for b in blocks])

    # N-009: prepare the input *before* the lens validation contract so that
    # empty / whitespace-only / type-error inputs surface as the same
    # ``ValueError`` users get from every other op, never as the much
    # less actionable ``LensPipelineMismatch``.
    text_input = model._prepare(text)

    if position is not None:
        _seq_len = input_seq_len(text_input)
        if _seq_len is not None:
            position = validate_position(position, _seq_len, op="lens")

    # Validation contract (Phase 0e) — on first use, assert lens-at-last-block
    # matches model output. Catches any resolver drift loudly.
    try:
        validate_lens_pipeline(model)
    except LensPipelineMismatch:
        # Re-raise so the user sees the actionable diagnostic.
        raise

    # Capture each block's output via hooks (only the family-appropriate ones)
    block_outputs = _capture_block_outputs(model, arch, text_input, blocks=blocks)

    if not block_outputs:
        console.print("\n  [yellow]lens:[/yellow] no block outputs captured.\n")
        return None

    # Recover input tokens for labelling (LMs only)
    input_tokens: list[str] | None = None
    if isinstance(text, str) and model._tokenizer is not None and not arch.spatial:
        try:
            encoded = model._tokenizer(text, return_tensors="pt")
            input_tokens = model._tokenizer.convert_ids_to_tokens(
                encoded["input_ids"][0].tolist()
            )
        except Exception:
            input_tokens = None

    predictions: list[dict[str, Any]] = []
    for block_idx, block in enumerate(blocks):
        if block.path not in block_outputs:
            continue
        block_out = block_outputs[block.path].float()
        if translator_lens is not None:
            # Tuned lens: per-block affine translator between the captured
            # hidden state and the (unchanged) head projection pipeline.
            with torch.no_grad():
                block_out = translator_lens.translators[block_idx](
                    block_out.to(next(translator_lens.parameters()).device)
                )
        logits = _project_through_head(arch, block_out)
        if logits is None:
            continue
        entry = _build_prediction_entry(
            block.path, logits, model, position=position,
            input_tokens=input_tokens, spatial=arch.spatial,
        )
        if entry is not None:
            entry["lens_kind"] = kind
            predictions.append(entry)

    if not predictions:
        console.print("\n  [yellow]lens:[/yellow] no projections succeeded.\n")
        return None

    model_name = arch.arch_family or "model"
    if kind == "tuned":
        model_name += " (tuned lens)"
    render_lens(predictions, model_name)

    if save is not None:
        from interpkit.core.plot import plot_lens

        plot_lens(predictions, save_path=save, input_tokens=input_tokens)

    if html is not None:
        from interpkit.core.html import html_lens as gen_html_lens
        from interpkit.core.html import save_html

        flat_preds = []
        for li_idx, pred in enumerate(predictions):
            for pos_data in pred.get("positions", [{"pos": 0, "top1_token": pred.get("top1_token", "?"), "top1_prob": pred.get("top1_prob", 0.0)}]):
                flat_preds.append({
                    "layer": li_idx,
                    "position": pos_data.get("pos", 0),
                    "prediction": pos_data.get("top1_token", "?"),
                    "prob": pos_data.get("top1_prob", 0.0),
                })
        save_html(gen_html_lens(flat_preds, input_tokens), html)

    return predictions


def run_encoder_lens(
    model: Model,
    text: Any,
    *,
    position: int | None = None,
) -> list[dict[str, Any]] | None:
    """Encoder-side lens for seq2seq models (N-002).

    Mirrors :func:`run_lens` but explicitly hooks ``arch.blocks`` (the
    encoder stack on T5/BART, since ``_find_blocks`` picks the encoder
    first when both stacks have equal layer counts) and projects through
    the same head pipeline. The model's lm_head is typically tied to
    both encoder and decoder embeddings on these models, so the same
    projection is meaningful for encoder hidden states.
    """
    from interpkit.core.arch import ArchFamily
    from interpkit.core.exceptions import OperationNotSupportedForArchitecture
    from interpkit.core.render import render_lens

    arch = model.arch_info
    if arch.family != ArchFamily.SEQ2SEQ_LM:
        raise OperationNotSupportedForArchitecture(
            f"`encoder_lens` only applies to seq2seq models; "
            f"this model is family={arch.family.value!r}. "
            f"Use `lens()` for non-encoder-decoder models."
        )
    if arch.head_module is None or not arch.blocks:
        console.print(
            "\n  [yellow]encoder_lens not available:[/yellow] "
            "no head/encoder block detected.\n"
        )
        return None

    text_input = model._prepare(text)
    block_outputs = _capture_block_outputs(model, arch, text_input, blocks=arch.blocks)
    if not block_outputs:
        console.print("\n  [yellow]encoder_lens:[/yellow] no block outputs captured.\n")
        return None

    input_tokens: list[str] | None = None
    if isinstance(text, str) and model._tokenizer is not None:
        try:
            encoded = model._tokenizer(text, return_tensors="pt")
            input_tokens = model._tokenizer.convert_ids_to_tokens(
                encoded["input_ids"][0].tolist(),
            )
        except Exception:
            input_tokens = None

    predictions: list[dict[str, Any]] = []
    for block in arch.blocks:
        if block.path not in block_outputs:
            continue
        block_out = block_outputs[block.path].float()
        logits = _project_through_head(arch, block_out)
        if logits is None:
            continue
        entry = _build_prediction_entry(
            block.path, logits, model, position=position,
            input_tokens=input_tokens, spatial=False,
        )
        if entry is not None:
            predictions.append(entry)

    if not predictions:
        console.print("\n  [yellow]encoder_lens:[/yellow] no projections succeeded.\n")
        return None

    render_lens(predictions, (arch.arch_family or "model") + " (encoder)")
    return predictions


def _capture_block_outputs(
    model: Model,
    arch: Any,
    text_input: Any,
    *,
    blocks: list | None = None,
) -> dict[str, torch.Tensor]:
    """Hook each block in *blocks* and capture its output tensor.

    Defaults to ``arch.blocks`` when *blocks* is None for backwards
    compatibility. ``run_lens`` passes ``lens_blocks(arch)`` explicitly
    so seq2seq models hook only the decoder side (N-002).
    """
    captured: dict[str, torch.Tensor] = {}
    blocks = blocks if blocks is not None else arch.blocks

    def make_hook(name: str):
        def fn(_m: nn.Module, _inp: Any, out: Any) -> None:
            if isinstance(out, torch.Tensor):
                captured[name] = out.detach()
            elif isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
                captured[name] = out[0].detach()

        return fn

    handles = []
    from interpkit.core.arch import module_at_path as _module_at_path

    for block in blocks:
        try:
            mod = _module_at_path(model._model, block.path)
        except (AttributeError, IndexError, KeyError, TypeError):
            continue
        handles.append(mod.register_forward_hook(make_hook(block.path)))

    try:
        with torch.no_grad():
            model._forward(text_input)
    finally:
        for h in handles:
            h.remove()

    return captured


def _project_through_head(arch: Any, block_output: torch.Tensor) -> torch.Tensor | None:
    """Family-aware projection from a block output to logits.

    Delegates to the canonical implementation in
    :mod:`interpkit.core.support_matrix` so lens, dla, trace, and the
    validation contract all use the exact same pipeline (no chance of
    drift between op-time and validation-time projections).
    """
    from interpkit.core.support_matrix import _project_through_head as _proj
    return _proj(arch, block_output)


def _build_prediction_entry(
    layer_name: str,
    logits: torch.Tensor,
    model: Model,
    *,
    position: int | None,
    input_tokens: list[str] | None,
    spatial: bool,
) -> dict[str, Any] | None:
    """Build a per-layer prediction entry with top-k decoding."""
    if spatial:
        # Vision: logits shape (B, num_classes); single prediction per image.
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        probs = torch.softmax(logits, dim=-1)
        topk = min(5, probs.shape[-1])
        top5_probs_t, top5_ids_t = probs[0].topk(topk)
        top5_probs_list = top5_probs_t.tolist()
        top5_tokens = [_decode_class(model, int(c)) for c in top5_ids_t.tolist()]
        return {
            "layer_name": layer_name,
            "top1_token": top5_tokens[0],
            "top1_prob": top5_probs_list[0],
            "top5_tokens": top5_tokens,
            "top5_probs": top5_probs_list,
            "positions": [],
        }

    # Language: logits shape (B, seq, vocab).
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    probs = torch.softmax(logits, dim=-1)
    seq_len = probs.shape[1]

    if position is not None:
        pos_idx = position if position >= 0 else seq_len + position
        pos_indices = [pos_idx]
    else:
        pos_indices = list(range(seq_len))

    per_position: list[dict[str, Any]] = []
    for pos in pos_indices:
        if pos < 0 or pos >= seq_len:
            continue
        top5_probs_t, top5_ids_t = probs[0, pos].topk(min(5, probs.shape[-1]))
        if model._tokenizer is not None:
            top5_tokens = [model._tokenizer.decode([tid]) for tid in top5_ids_t.tolist()]
        else:
            top5_tokens = [str(int(tid)) for tid in top5_ids_t.tolist()]
        top5_probs_list = top5_probs_t.tolist()
        per_position.append({
            "pos": pos,
            "top1_token": top5_tokens[0],
            "top1_prob": top5_probs_list[0],
            "top5_tokens": top5_tokens,
            "top5_probs": top5_probs_list,
        })

    last = per_position[-1] if per_position else {
        "top1_token": "", "top1_prob": 0.0, "top5_tokens": [], "top5_probs": [],
    }
    entry: dict[str, Any] = {
        "layer_name": layer_name,
        "top1_token": last["top1_token"],
        "top1_prob": last["top1_prob"],
        "top5_tokens": last["top5_tokens"],
        "top5_probs": last["top5_probs"],
        "positions": per_position,
    }
    if input_tokens is not None:
        entry["tokens"] = input_tokens
    return entry


def _decode_class(model: Model, class_idx: int) -> str:
    """Decode a class index to a label using the model's id2label if available."""
    config = getattr(model._model, "config", None)
    id2label = getattr(config, "id2label", None) if config is not None else None
    if id2label is not None:
        return str(id2label.get(class_idx, f"class_{class_idx}"))
    return f"class_{class_idx}"
