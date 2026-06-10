"""maxact — find the dataset examples that most activate a neuron / SAE feature / head.

The feature-browsing workflow: given a unit (a neuron at a module, an
SAE feature, or an attention head) and a corpus, stream batched forwards
and keep the top-k (example, position) records by activation score —
"what does this unit fire on?". Memory stays O(k) via
:class:`~interpkit.core.topk.TopKTracker`; activations are freed per
batch.

Scoring per (example, position):

- ``neuron=i``  — the raw (signed) activation ``acts[..., i]`` at *at*.
- ``feature=i`` — ``sae.encode(acts)[..., i]`` through the provided SAE
  (the existing config-aware encode path from :mod:`interpkit.ops.sae`).
- ``head=i``    — the L2 norm of head *i*'s slice of the pre-output-
  projection activation inside *at* (the per-head output magnitude).

Pad positions are masked out via the attention mask before scoring.

Datasets are a ``list[str]`` or an ``"hf:name[:split[:column]]"`` spec
(lazy ``datasets`` import; install the ``interpkit[data]`` extra).
``max_examples`` is required for HF datasets so a typo'd spec can't
start an unbounded scan.

Documented deferral (same ledger pattern as ``ops/eap.py``): a
disk-backed activation store for re-scoring different units without
re-running the model. Each scan currently recomputes forwards.
"""

from __future__ import annotations

from itertools import islice
from typing import TYPE_CHECKING, Any

import torch
from rich.console import Console
from rich.progress import Progress

from interpkit.core.paths import validate_module_path
from interpkit.core.topk import TopKTracker
from interpkit.ops._hooks import first_tensor, register_capture_hook
from interpkit.ops.patch import _get_module

if TYPE_CHECKING:
    from interpkit.core.model import Model

console = Console()

__all__ = ["run_max_activating"]


def _resolve_dataset(
    dataset: list[str] | str, max_examples: int | None,
) -> tuple[list[str], str]:
    """Return (texts, description). Handles list[str] and "hf:..." specs."""
    if isinstance(dataset, list):
        if not dataset or not all(isinstance(t, str) for t in dataset):
            raise ValueError("dataset must be a non-empty list of strings.")
        texts = dataset[:max_examples] if max_examples else dataset
        return list(texts), f"list[{len(texts)} texts]"

    if isinstance(dataset, str) and dataset.startswith("hf:"):
        if max_examples is None:
            raise ValueError(
                "max_examples is required with hf: datasets — an unbounded "
                "scan of a hub dataset is rarely what you want. Pass e.g. "
                "max_examples=512."
            )
        try:
            import datasets as hf_datasets
        except ImportError as exc:
            raise ImportError(
                "Loading hf: datasets requires the `datasets` package. "
                "Install the extra: pip install 'interpkit[data]'."
            ) from exc

        parts = dataset[3:].rsplit(":", 2)
        if len(parts) == 3:
            name, split, column = parts
        elif len(parts) == 2:
            name, split, column = parts[0], parts[1], "text"
        else:
            name, split, column = parts[0], "train", "text"

        stream = hf_datasets.load_dataset(name, split=split, streaming=True)
        texts = []
        for example in islice(stream, max_examples):
            value = example.get(column)
            if isinstance(value, str) and value.strip():
                texts.append(value)
        if not texts:
            raise ValueError(
                f"hf:{name}:{split}:{column} yielded no non-empty texts in "
                f"the first {max_examples} examples — check the column name."
            )
        return texts, f"hf:{name}:{split}:{column}[{len(texts)}]"

    raise ValueError(
        "dataset must be a list of strings or an 'hf:name[:split[:column]]' "
        "spec."
    )


def run_max_activating(
    model: Model,
    dataset: list[str] | str,
    *,
    at: str,
    neuron: int | None = None,
    feature: int | None = None,
    head: int | None = None,
    sae: Any = None,
    top_k: int = 20,
    batch_size: int = 8,
    max_examples: int | None = None,
    max_length: int = 128,
    context: int = 12,
) -> dict[str, Any]:
    """Scan *dataset* for the inputs that most activate one unit at *at*.

    Exactly one of *neuron* / *feature* / *head* selects the unit;
    *feature* additionally needs *sae* (an :class:`~interpkit.ops.sae.SAE`
    or a HuggingFace repo / local path accepted by ``load_sae``).

    Returns
    -------
    dict with ``unit``, ``examples`` (rank/score/text/position/token +
    a ±*context*-token window with per-token scores), scan counters, and
    ``meta``.
    """
    from interpkit.core.render import render_max_activating
    from interpkit.core.support_matrix import check_op_supported

    check_op_supported("max_activating", model.arch_info)
    validate_module_path(at, model.arch_info)

    units = {"neuron": neuron, "feature": feature, "head": head}
    chosen = {name: idx for name, idx in units.items() if idx is not None}
    if len(chosen) != 1:
        raise ValueError(
            f"Pass exactly one of neuron= / feature= / head= "
            f"(got {sorted(chosen) or 'none'})."
        )
    unit_kind, unit_idx = next(iter(chosen.items()))
    if unit_idx < 0:
        raise ValueError(f"{unit_kind} index must be >= 0, got {unit_idx}.")

    if unit_kind == "feature":
        if sae is None:
            raise ValueError(
                "feature= requires sae= (an SAE object, HF repo ID, or local "
                "path)."
            )
        from interpkit.ops.sae import SAE, _ensure_sae_on_device, load_sae

        if not isinstance(sae, SAE):
            sae = load_sae(str(sae))
        sae = _ensure_sae_on_device(sae, model._device)
        if unit_idx >= sae.d_sae:
            raise ValueError(
                f"feature {unit_idx} out of range for SAE with d_sae="
                f"{sae.d_sae}."
            )
    elif sae is not None:
        raise ValueError("sae= only applies with feature= (SAE feature mode).")

    num_heads = model.arch_info.num_attention_heads
    proj_mod: torch.nn.Module | None = None
    if unit_kind == "head":
        check_op_supported("head_activations", model.arch_info)
        if num_heads is None:
            raise ValueError(
                "head= requires num_attention_heads in the model config."
            )
        if unit_idx >= num_heads:
            raise ValueError(
                f"head {unit_idx} out of range for {num_heads} heads."
            )
        from interpkit.ops.heads import _find_output_proj

        _, _, proj_mod = _find_output_proj(model._model, at)
        if proj_mod is None:
            raise RuntimeError(
                f"head= requires an identifiable output projection inside "
                f"'{at}' (c_proj / out_proj / o_proj / dense)."
            )

    if model._tokenizer is None:
        raise RuntimeError("max_activating requires a tokenizer.")

    texts, dataset_desc = _resolve_dataset(dataset, max_examples)
    tracker = TopKTracker(top_k)
    target_mod = _get_module(model._model, at)
    tok = model._tokenizer

    n_positions = 0
    with Progress(console=console, transient=True) as progress:
        task = progress.add_task("Scanning for max activations", total=len(texts))
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            batch_input = {k: v.to(model._device) for k, v in encoded.items()}
            batch_input = model._inject_decoder_ids(batch_input)
            input_ids = batch_input["input_ids"]
            attention_mask = batch_input.get("attention_mask")

            captured: dict[str, torch.Tensor] = {}
            if unit_kind == "head":
                assert proj_mod is not None

                def _pre_proj_hook(
                    _m: torch.nn.Module, inp: Any, _out: Any,
                    _store: dict[str, torch.Tensor] = captured,
                ) -> None:
                    t = inp[0] if isinstance(inp, tuple) and inp else inp
                    if isinstance(t, torch.Tensor):
                        _store["acts"] = t.detach()

                handle = proj_mod.register_forward_hook(_pre_proj_hook)
            else:
                handle = register_capture_hook(
                    target_mod, captured, "acts", clone=False,
                )
            try:
                with torch.no_grad():
                    model._forward(batch_input)
            finally:
                handle.remove()

            acts = first_tensor(captured.get("acts"))
            if acts is None or acts.dim() < 2:
                progress.advance(task, advance=len(batch))
                continue
            if acts.dim() == 2:  # (S, H) → (1, S, H)
                acts = acts.unsqueeze(0)
            acts = acts.float()

            if unit_kind == "neuron":
                if unit_idx >= acts.shape[-1]:
                    raise ValueError(
                        f"neuron {unit_idx} out of range for activation width "
                        f"{acts.shape[-1]} at '{at}'."
                    )
                scores = acts[..., unit_idx]
            elif unit_kind == "feature":
                with torch.no_grad():
                    scores = sae.encode(acts)[..., unit_idx]
            else:  # head
                assert num_heads is not None
                head_dim = acts.shape[-1] // num_heads
                sl = acts[..., unit_idx * head_dim : (unit_idx + 1) * head_dim]
                scores = sl.norm(dim=-1)

            # Scores may cover fewer rows than the batch (e.g. seq2seq
            # decoder side); align defensively.
            n_rows = min(scores.shape[0], input_ids.shape[0])
            if attention_mask is not None and attention_mask.shape == scores.shape:
                scores = scores.masked_fill(attention_mask == 0, float("-inf"))
            n_positions += int(
                attention_mask.sum().item()
                if attention_mask is not None and attention_mask.shape == scores.shape
                else scores.numel()
            )

            for b in range(n_rows):
                row = scores[b]
                k_here = min(top_k, row.shape[0])
                vals, idxs = row.topk(k_here)
                ids_row = input_ids[b]
                # Clip context windows to the row's real (un-padded) span so
                # displays don't show pad tokens (works for left or right
                # padding).
                span_lo, span_hi = 0, row.shape[0]
                if (
                    attention_mask is not None
                    and attention_mask.shape[0] > b
                    and attention_mask.shape[-1] == row.shape[0]
                ):
                    nz = attention_mask[b].nonzero()
                    if nz.numel() > 0:
                        span_lo = int(nz.min().item())
                        span_hi = int(nz.max().item()) + 1
                for val, pos in zip(vals.tolist(), idxs.tolist()):
                    if val == float("-inf") or val <= tracker.threshold:
                        continue
                    lo = max(span_lo, pos - context)
                    hi = min(span_hi, pos + context + 1)
                    window_scores = [
                        0.0 if s == float("-inf") else s
                        for s in row[lo:hi].tolist()
                    ]
                    tracker.push(val, {
                        "text_idx": start + b,
                        "position": pos,
                        "token_id": int(ids_row[pos].item()),
                        "context_ids": ids_row[lo:hi].tolist(),
                        "context_offset": pos - lo,
                        "context_scores": window_scores,
                    })
            progress.advance(task, advance=len(batch))

    examples: list[dict[str, Any]] = []
    for rank, (score, payload) in enumerate(tracker.items()):
        examples.append({
            "rank": rank,
            "score": score,
            "text": texts[payload["text_idx"]],
            "text_idx": payload["text_idx"],
            "position": payload["position"],
            "token": tok.decode([payload["token_id"]]),
            "context_tokens": [
                tok.decode([tid]) for tid in payload["context_ids"]
            ],
            "context_offset": payload["context_offset"],
            "context_scores": payload["context_scores"],
        })

    result: dict[str, Any] = {
        "unit": {
            "kind": "sae_feature" if unit_kind == "feature" else unit_kind,
            "at": at,
            "index": unit_idx,
        },
        "examples": examples,
        "n_examples_scanned": len(texts),
        "n_positions_scanned": n_positions,
        "meta": {
            "batch_size": batch_size,
            "max_length": max_length,
            "context": context,
            "dataset": dataset_desc,
            "score_definition": {
                "neuron": "raw (signed) activation at the neuron",
                "sae_feature": "SAE feature activation (post encode)",
                "head": "L2 norm of the head's pre-projection output slice",
            }["sae_feature" if unit_kind == "feature" else unit_kind],
        },
    }

    render_max_activating(result)
    return result
