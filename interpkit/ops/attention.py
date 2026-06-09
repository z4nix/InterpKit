"""attention — eager-attention weights for transformer models (F-001 + F-002 + N-003).

Pre-1.0 interpkit had two correctness bugs in attention:

- F-001: ``output_attentions=True`` was set BEFORE ``_attn_implementation="eager"``,
  but in transformers 5.x the ``output_attentions`` setter inspects the
  current attn implementation and raises if it's still SDPA. The exception
  was swallowed and the code silently fell back to a QK-reconstruction
  path that returned RoPE/ALiBi-less weights — wrong by orders of magnitude.

- F-002: the QK-reconstruction fallback captured Q/K BEFORE the model's
  RoPE/ALiBi/softcap was applied. Reconstructing post-positional Q/K
  correctly requires family-specific code (RoPE via apply_rotary_pos_emb,
  ALiBi via per-head slopes, Gemma softcap, DeBERTa disentangled bias) —
  ~100 lines per family, a permanent source of subtle bugs.

- N-003: encoder-decoder models (T5/Flan-T5/BART) populate
  ``decoder_attentions``, ``cross_attentions``, and ``encoder_attentions``
  on their forward output — never the flat ``attentions`` field. Reading
  only ``attentions`` produced ``AttentionBackendUnavailable`` on every
  seq2seq model. The ``kind=`` parameter routes to the appropriate field
  per family.

The 1.0 fix: write order is correct, every call goes through eager
(via :meth:`Model._ensure_eager_attention` which lazily loads a second
model copy with ``attn_implementation="eager"``), and the QK-reconstruction
fallback is *deleted*. If eager is unavailable (ancient transformers,
custom architecture without eager support), we raise
:class:`AttentionBackendUnavailable` with a clear message rather than
return wrong weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn.functional as F
from rich.console import Console

from interpkit.core.exceptions import AttentionBackendUnavailable
from interpkit.core.support_matrix import check_op_supported

if TYPE_CHECKING:
    from interpkit.core.model import Model

console = Console()


# N-003: valid attention-tensor kinds. ``self`` is decoder-self for
# seq2seq, plain self-attention for causal/MLM/vision. ``cross`` is
# decoder→encoder cross-attention (seq2seq only). ``encoder`` is
# encoder-self attention on a seq2seq model.
_VALID_ATTENTION_KINDS = ("self", "cross", "encoder")


def run_attention(
    model: Model,
    input_data: Any,
    *,
    layer: int | None = None,
    head: int | None = None,
    causal: bool | None = None,
    kind: Literal["self", "cross", "encoder"] = "self",
    save: str | None = None,
    html: str | None = None,
) -> list[dict[str, Any]] | None:
    """Capture eager-attention weights and display a summary.

    Always uses HuggingFace's eager attention implementation for
    correctness — modern transformers default to SDPA / FlashAttention
    which don't return weights without an eager backend.

    Parameters
    ----------
    causal:
        Whether to apply a causal (triangular) attention mask.  Passed
        to the attention layer when relevant; ignored when reading
        eager attentions directly (the model already applies its own mask).
    kind:
        Which attention tensor to return for encoder-decoder models
        (T5/BART). Ignored on causal-LM, MLM, and vision models, which
        only have a single self-attention stack.

        - ``"self"`` (default): decoder self-attention (seq2seq) or the
          model's only self-attention stack (everything else).
        - ``"cross"``: decoder→encoder cross-attention (seq2seq only).
        - ``"encoder"``: encoder self-attention (seq2seq only).

        Each result row carries ``attention_kind`` so callers can verify
        which tensor was returned.
    """
    from interpkit.core.render import render_attention

    if kind not in _VALID_ATTENTION_KINDS:
        raise ValueError(
            f"attention(kind={kind!r}) invalid — must be one of "
            f"{_VALID_ATTENTION_KINDS}."
        )

    arch = model.arch_info
    check_op_supported("attention", arch)

    attn_modules = [m for m in arch.modules if m.role == "attention"]
    if not attn_modules:
        console.print(
            "\n  [yellow]attention not available:[/yellow] no attention modules detected"
            f" for {arch.arch_family or 'this model'}.\n"
        )
        return None

    if causal is None:
        config = getattr(model._model, "config", None)
        if config is not None:
            is_decoder = getattr(config, "is_decoder", None)
            is_enc_dec = getattr(config, "is_encoder_decoder", None)
            causal = not (is_decoder is False and not is_enc_dec)
        else:
            causal = True

    eager_model = model._ensure_eager_attention()
    model_input = model._prepare(input_data)

    tokens: list[str] | None = None
    if model._tokenizer is not None and isinstance(input_data, str):
        encoded = model._tokenizer(input_data, return_tensors="pt")
        token_ids = encoded["input_ids"][0].tolist()
        tokens = model._tokenizer.convert_ids_to_tokens(token_ids)

    config = getattr(eager_model, "config", None)
    if config is None:
        raise AttentionBackendUnavailable(
            f"Cannot enable eager attention on a model without a `.config` "
            f"attribute (type {type(eager_model).__name__}). Pass a HuggingFace "
            f"PreTrainedModel via interpkit.load() to use attention()."
        )

    # Save and restore previous config flags around the eager forward pass.
    old_output_attn = getattr(config, "output_attentions", None)
    old_attn_impl = getattr(config, "_attn_implementation", None)

    try:
        # F-001: order matters — set the implementation first, THEN turn on
        # output_attentions. The setter inspects the attn implementation in
        # transformers 5.x and raises if it's still SDPA.
        config._attn_implementation = "eager"
        config.output_attentions = True
        # N-003: also pass output_attentions / return_dict as forward kwargs.
        # Some HF model classes (notably T5/BART encoder-decoders in modern
        # transformers) ignore the config attribute and only honor the kwarg.
        # ``return_dict=True`` ensures the output object exposes named
        # ``decoder_attentions`` / ``cross_attentions`` fields.
        forward_kwargs: dict[str, Any] = {
            "output_attentions": True,
            "return_dict": True,
        }
        with torch.no_grad():
            if isinstance(model_input, dict):
                out = eager_model(**model_input, **forward_kwargs)
            else:
                out = eager_model(model_input, **forward_kwargs)
        # N-003: encoder-decoder models populate decoder_/cross_/encoder_
        # attentions separately; the flat ``attentions`` field is None.
        # Route per-family + user-requested ``kind`` to the right field.
        attentions, attention_kind_used = _extract_attentions(
            out, arch=arch, kind=kind, model_class=type(eager_model).__name__,
        )
    finally:
        if old_output_attn is None:
            try:
                del config.output_attentions
            except AttributeError:
                pass
        else:
            config.output_attentions = old_output_attn
        if old_attn_impl is None:
            try:
                del config._attn_implementation
            except AttributeError:
                pass
        else:
            config._attn_implementation = old_attn_impl

    results: list[dict[str, Any]] = []
    for li, attn_tensor in enumerate(attentions):
        if layer is not None and li != layer:
            continue
        # attn_tensor shape: (batch, num_heads, seq, seq) — drop batch dim.
        aw = attn_tensor[0].detach()
        for head_idx in range(aw.shape[0]):
            if head is not None and head_idx != head:
                continue
            head_attn = aw[head_idx]
            top_pairs = _get_top_pairs(head_attn, k=5)
            entropy = _attention_entropy(head_attn)
            results.append({
                "layer": li,
                "head": head_idx,
                "top_pairs": top_pairs,
                "entropy": entropy,
                "weights": head_attn,
                # F-001/F-002 metadata — fields retained for API stability
                # even though they're now constants. Documents that the
                # weights are real (not RoPE/ALiBi-less reconstructions).
                "source": "eager",
                "positional_encoding_applied": True,
                # N-003: which attention tensor was returned (always one of
                # "self" / "cross" / "encoder"). For decoder-only / MLM /
                # vision this is always "self" — the field disambiguates
                # only for seq2seq.
                "attention_kind": attention_kind_used,
            })

    if not results:
        console.print(
            f"\n  [yellow]attention:[/yellow] no attention layers matched the "
            f"filter (layer={layer!r}, head={head!r}).\n"
        )
        return None

    model_name = arch.arch_family or "model"
    render_attention(results, tokens, model_name)

    if save is not None:
        from interpkit.core.plot import plot_attention, plot_attention_multi

        if layer is not None and head is not None and len(results) == 1:
            plot_attention(
                results[0]["weights"], tokens, layer=results[0]["layer"],
                head=results[0]["head"], save_path=save,
            )
        else:
            plot_attention_multi(results, tokens, save_path=save)

    if html is not None:
        from interpkit.core.html import html_attention as gen_html_attention
        from interpkit.core.html import save_html

        serializable = []
        for r in results:
            entry = {**r}
            w = r.get("weights")
            if isinstance(w, torch.Tensor):
                entry["weights"] = w.tolist()
            serializable.append(entry)
        save_html(gen_html_attention(serializable, tokens), html)

    return results


def _extract_attentions(
    out: Any,
    *,
    arch: Any,
    kind: str,
    model_class: str,
) -> tuple[tuple[torch.Tensor, ...], str]:
    """Pick the right attention tensor stack from an HF forward output.

    Encoder-decoder models (T5/Flan-T5/BART/Marian/Pegasus/MBart) populate
    ``decoder_attentions``, ``cross_attentions``, and ``encoder_attentions``
    on the forward output. They never set the flat ``attentions`` field.
    Reading that field directly produced ``AttentionBackendUnavailable``
    on every seq2seq audit run (N-003).

    Routing rules:
      - Causal-LM / MLM / vision: only ``out.attentions`` exists. The
        ``kind=`` argument is ignored (informational); we always return
        ``("self", out.attentions)``.
      - Seq2seq + ``kind="self"``: ``out.decoder_attentions``.
      - Seq2seq + ``kind="cross"``: ``out.cross_attentions``.
      - Seq2seq + ``kind="encoder"``: ``out.encoder_attentions``.

    Returns a (tensors, kind_label) pair. Raises
    ``AttentionBackendUnavailable`` only when the family-appropriate
    field is unavailable on the output object.
    """
    is_enc_dec = bool(getattr(arch, "is_encoder_decoder", False))

    if not is_enc_dec:
        attentions = getattr(out, "attentions", None)
        if attentions is None or len(attentions) == 0:
            raise AttentionBackendUnavailable(
                f"Eager attention forward returned no `attentions` field "
                f"(model={model_class}). The model may not "
                f"support output_attentions=True. Try a different attention-"
                f"having model, or file an issue with the model id."
            )
        return tuple(attentions), "self"

    field_for_kind = {
        "self": "decoder_attentions",
        "cross": "cross_attentions",
        "encoder": "encoder_attentions",
    }
    field = field_for_kind[kind]
    attentions = getattr(out, field, None)

    if attentions is None or len(attentions) == 0:
        raise AttentionBackendUnavailable(
            f"Eager attention forward on encoder-decoder model "
            f"{model_class} returned no `{field}` "
            f"field for kind={kind!r}. Available output fields: "
            f"{[k for k in ('attentions', 'decoder_attentions', 'cross_attentions', 'encoder_attentions') if getattr(out, k, None) is not None]}."
        )
    return tuple(attentions), kind


def _get_top_pairs(
    attn: torch.Tensor, k: int = 5,
) -> list[tuple[int, int, float]]:
    """Find top-k (source_pos, target_pos, score) pairs in an attention matrix."""
    flat = attn.view(-1)
    topk_vals, topk_idxs = flat.topk(min(k, flat.numel()))
    n_cols = attn.shape[-1]
    pairs = []
    for val, idx in zip(topk_vals.tolist(), topk_idxs.tolist()):
        src = idx // n_cols
        tgt = idx % n_cols
        pairs.append((src, tgt, val))
    return pairs


def _attention_entropy(attn: torch.Tensor) -> float:
    """Mean entropy of attention distributions across query positions."""
    eps = 1e-10
    log_attn = torch.log(attn + eps)
    entropy_per_query = -(attn * log_attn).sum(dim=-1)
    return entropy_per_query.mean().item()


# Suppress unused-import warning for F (kept for backwards-compat re-exports
# from interpkit.ops.attention.F that older scripts might rely on).
_ = F
