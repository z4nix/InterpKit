"""generate — multi-token generation with interventions and per-step capture.

The op that breaks the single-forward-pass barrier: any
:class:`~interpkit.core.interventions.Intervention` (steer / ablate /
patch / fn) stays active across every decode step of
``model.generate(...)``, and ``capture="lens"`` records a per-token
logit-lens trajectory (each generated token's hidden state at every
block, projected through the validated head pipeline).

Position semantics follow :mod:`interpkit.core.interventions`:
``positions`` are absolute and prompt-indexed (generated token *i* sits
at ``prompt_len + i``); a :class:`GenerationContext` translates them
into each decode step's KV-cache window. Note that an intervened output
at step *t* feeds the KV cache and therefore influences all later steps.

Scope (v1, documented deferrals):

- Greedy / sampling generation only (``num_beams=1`` semantics — beam
  search re-feeds tokens, which breaks position tracking).
- Encoder-decoder models may ``generate`` plainly, but interventions /
  capture raise ``OperationNotSupportedForArchitecture``: encoder hooks
  fire once while decoder hooks fire per-step with different position
  semantics. Same deferral-ledger pattern as ``ops/_hooks.py``.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING, Any

import torch

from interpkit.core.enums import _validate_enum
from interpkit.core.exceptions import OperationNotSupportedForArchitecture
from interpkit.core.interventions import (
    GenerationContext,
    Intervention,
    apply_interventions,
    track_positions,
)
from interpkit.ops._hooks import first_tensor

if TYPE_CHECKING:
    from interpkit.core.model import Model

VALID_GENERATE_CAPTURE = frozenset({"lens", "logits"})

__all__ = ["run_generate", "VALID_GENERATE_CAPTURE"]


def run_generate(
    model: Model,
    input_data: Any,
    *,
    max_new_tokens: int = 64,
    interventions: list[Intervention] | None = None,
    capture: str | None = None,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> dict[str, Any]:
    """Generate from *input_data* with optional interventions / per-step capture.

    Parameters
    ----------
    interventions:
        :class:`Intervention` objects kept active for the whole
        generation (prefill + every decode step).
    capture:
        ``"lens"`` — per generated token, project each block's
        last-position hidden state through the head pipeline (logit
        lens over the generation trajectory).
        ``"logits"`` — record each step's final logits via HF
        ``output_scores``.

    Returns
    -------
    dict with ``prompt``, ``response``, ``input_ids``, ``output_ids``,
    ``interventions`` (serialized specs) and — when *capture* is set —
    ``steps``: one entry per generated token.
    """
    from interpkit.core.render import render_generate
    from interpkit.core.support_matrix import check_op_supported

    check_op_supported("generate", model.arch_info)
    if capture is not None:
        _validate_enum(capture, VALID_GENERATE_CAPTURE, "capture")
    ivs = list(interventions or [])
    if ivs:
        check_op_supported("intervene", model.arch_info)

    arch = model.arch_info
    if arch.is_encoder_decoder and (ivs or capture is not None):
        raise OperationNotSupportedForArchitecture(
            "generate(interventions=..., capture=...) is not yet supported on "
            "encoder-decoder models: encoder hooks fire once per generation "
            "while decoder hooks fire per-step with different position "
            "semantics. Plain generate() works; intervention support for "
            "seq2seq is a documented deferral (see core/interventions.py)."
        )

    if not hasattr(model._model, "generate"):
        raise RuntimeError(
            f"Underlying model {type(model._model).__name__} has no "
            "generate() method — cannot run generate()."
        )

    model_input = model._prepare(input_data)
    if not isinstance(model_input, dict) or "input_ids" not in model_input:
        raise ValueError(
            "generate() requires token inputs — pass a text string (with a "
            "tokenizer loaded) or a dict containing 'input_ids'."
        )
    input_ids = model_input["input_ids"]
    attention_mask = model_input.get("attention_mask")
    prompt_len = int(input_ids.shape[-1])

    pad_id = None
    if model._tokenizer is not None:
        pad_id = (
            getattr(model._tokenizer, "pad_token_id", None)
            or getattr(model._tokenizer, "eos_token_id", None)
        )

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
    }
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask
    if capture == "logits":
        gen_kwargs["output_scores"] = True
        gen_kwargs["return_dict_in_generate"] = True

    # Position tracking is needed whenever hooks must know which absolute
    # positions the current decode window covers.
    ctx: GenerationContext | None = None
    if ivs or capture == "lens":
        ctx = GenerationContext(prompt_len)

    # Per-step lens capture: for every block, store the *last position's*
    # hidden state of each forward call (prefill call i=0 produces
    # generated token 0, decode call i produces token i).
    lens_store: dict[str, list[torch.Tensor]] = {}
    block_paths: list[str] = []
    if capture == "lens":
        from interpkit.core.support_matrix import (
            check_op_supported as _check,
        )
        from interpkit.core.support_matrix import (
            lens_blocks,
            validate_lens_pipeline,
        )

        _check("lens", arch)
        validate_lens_pipeline(model)
        block_paths = [b.path for b in lens_blocks(arch)]

    def _make_lens_hook(path: str):
        def hook(_mod: torch.nn.Module, _inp: Any, output: Any) -> None:
            t = first_tensor(output)
            if t is None or t.dim() < 2:
                return
            # Normalise to (B, H): last position of (B, S, H) or (S, H).
            last = t[:, -1, :] if t.dim() == 3 else t[-1:, :]
            lens_store.setdefault(path, []).append(last.detach().float().clone())
        return hook

    embed_module: torch.nn.Module | None = None
    if ctx is not None and arch.embed_path:
        from interpkit.core.paths import get_module

        try:
            embed_module = get_module(model._model, arch.embed_path)
        except AttributeError:
            embed_module = None

    with ExitStack() as stack:
        if ctx is not None:
            stack.enter_context(
                track_positions(model._model, ctx, embed_module=embed_module)
            )
        if block_paths:
            from interpkit.core.paths import get_module

            handles = [
                get_module(model._model, p).register_forward_hook(_make_lens_hook(p))
                for p in block_paths
            ]

            def _remove_lens_hooks() -> None:
                for h in handles:
                    h.remove()

            stack.callback(_remove_lens_hooks)
        if ivs:
            stack.enter_context(apply_interventions(model, ivs, ctx=ctx))

        with torch.no_grad():
            out = model._model.generate(input_ids=input_ids, **gen_kwargs)

    scores: tuple[torch.Tensor, ...] | None = None
    if capture == "logits":
        output_ids = out.sequences
        scores = tuple(out.scores) if out.scores is not None else None
    else:
        output_ids = out

    if arch.is_encoder_decoder:
        new_tokens = output_ids[0]
    else:
        new_tokens = output_ids[0, prompt_len:]

    if model._tokenizer is not None:
        prompt_text = model._tokenizer.decode(input_ids[0], skip_special_tokens=False)
        response = model._tokenizer.decode(new_tokens, skip_special_tokens=True)
    else:
        prompt_text = ""
        response = ""

    result: dict[str, Any] = {
        "prompt": prompt_text,
        "response": response,
        "input_ids": input_ids,
        "output_ids": output_ids,
        "interventions": [iv.describe() for iv in ivs],
    }

    if capture is not None:
        result["steps"] = _build_steps(
            model, new_tokens, capture, lens_store, block_paths, scores,
        )

    render_generate(result)
    return result


def _build_steps(
    model: Model,
    new_tokens: torch.Tensor,
    capture: str,
    lens_store: dict[str, list[torch.Tensor]],
    block_paths: list[str],
    scores: tuple[torch.Tensor, ...] | None,
) -> list[dict[str, Any]]:
    """One entry per generated token; forward call *i* produced token *i*."""
    from interpkit.core.support_matrix import _project_through_head

    arch = model.arch_info
    tok = model._tokenizer
    steps: list[dict[str, Any]] = []

    for i, token_id in enumerate(new_tokens.tolist()):
        entry: dict[str, Any] = {
            "step": i,
            "token_id": int(token_id),
            "token": tok.decode([token_id]) if tok is not None else str(token_id),
        }
        if capture == "lens":
            lens_entries: list[dict[str, Any]] = []
            for path in block_paths:
                captures = lens_store.get(path, [])
                if i >= len(captures):
                    continue
                logits = _project_through_head(arch, captures[i])
                if logits is None:
                    continue
                probs = torch.softmax(logits[0].float(), dim=-1)
                top_prob, top_id = probs.max(dim=-1)
                lens_entries.append({
                    "block": path,
                    "top1_token": (
                        tok.decode([int(top_id.item())]) if tok is not None
                        else str(int(top_id.item()))
                    ),
                    "top1_id": int(top_id.item()),
                    "top1_prob": float(top_prob.item()),
                })
            entry["lens"] = lens_entries
        elif capture == "logits" and scores is not None and i < len(scores):
            step_logits = scores[i]
            probs = torch.softmax(step_logits[0].float(), dim=-1)
            entry["logits"] = step_logits
            entry["prob"] = float(probs[int(token_id)].item())
        steps.append(entry)

    return steps
