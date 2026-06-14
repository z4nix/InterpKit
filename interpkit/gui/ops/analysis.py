"""Analysis ops: lens, dla, attribute, attention, activations, trace,
patch, ablate, decompose, diff, probe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from interpkit.gui.ops.base import JobContext, OpSpec, decoded_tokens, lines_list, ui

_METRICS = Literal["logit_diff", "kl_div", "target_prob", "l2_prob"]


def _summarize_logits(model: Any, result: dict[str, Any], k: int = 5) -> dict[str, Any]:
    """Replace full ``*_logits`` tensors with last-position top-k tokens.

    patch/ablate return three (1, seq, vocab) tensors that are megabytes
    of JSON; the GUI shows the prediction shift instead.
    """
    import torch

    tokenizer = getattr(model, "_tokenizer", None)
    out = dict(result)
    for key in list(out.keys()):
        if not key.endswith("_logits"):
            continue
        logits = out.pop(key)
        if tokenizer is None or not isinstance(logits, torch.Tensor):
            continue
        last = logits[0, -1] if logits.dim() == 3 else logits[-1] if logits.dim() == 2 else logits.view(-1)
        probs = torch.softmax(last.float(), dim=-1)
        top_probs, top_ids = probs.topk(k)
        out[key.replace("_logits", "_top")] = [
            [tokenizer.decode([tid]), float(p)] for tid, p in zip(top_ids.tolist(), top_probs.tolist())
        ]
    return out


# ---------------------------------------------------------------------------
# lens
# ---------------------------------------------------------------------------


class LensParams(BaseModel):
    text: str = Field(
        ...,
        description="Input text",
        json_schema_extra=ui(widget="textarea", rows=2, placeholder="The Eiffel Tower is in"),
    )
    position: int | None = Field(
        None,
        description="Single token position to analyse (-1 = last). Leave empty for all positions.",
    )
    tuned_lens: str | None = Field(
        None,
        title="tuned lens path",
        description="Path to a saved tuned lens (switches to tuned-lens mode). Train one under Advanced.",
        json_schema_extra=ui(widget="path", advanced=True),
    )


def _run_lens(model: Any, p: LensParams, ctx: JobContext) -> Any:
    kind = "tuned" if p.tuned_lens else "logit"
    result = model.lens(p.text, position=p.position, kind=kind, tuned_lens=p.tuned_lens)
    if result is None:
        raise ValueError("lens: no projections succeeded for this input.")
    return {"results": result, "tokens": decoded_tokens(model, p.text), "kind": kind}


# ---------------------------------------------------------------------------
# dla
# ---------------------------------------------------------------------------


class DlaParams(BaseModel):
    text: str = Field(
        ...,
        description="Input text",
        json_schema_extra=ui(widget="textarea", rows=2, placeholder="The capital of France is"),
    )
    token: str | None = Field(
        None,
        description="Target token (string or id). Uses the top-1 prediction if omitted.",
    )
    position: int = Field(-1, description="Token position to analyse (-1 = last)")
    top_k: int = Field(10, ge=1, description="Top/bottom contributors to show")
    sae: str | None = Field(
        None,
        title="SAE",
        description="SAE source: HF repo ID, local path, or 'org/repo/subfolder' (decomposes a component into features)",
        json_schema_extra=ui(group="SAE feature breakdown", advanced=True),
    )
    sae_at: str | None = Field(
        None,
        title="SAE at",
        description="Module to decompose through the SAE",
        json_schema_extra=ui(widget="module-picker", group="SAE feature breakdown", advanced=True),
    )
    sae_subfolder: str | None = Field(
        None,
        description="Subfolder inside the SAE repo",
        json_schema_extra=ui(group="SAE feature breakdown", advanced=True),
    )


def _run_dla(model: Any, p: DlaParams, ctx: JobContext) -> Any:
    token: int | str | None = None
    if p.token is not None and p.token != "":
        try:
            token = int(p.token)
        except ValueError:
            token = p.token
    return model.dla(
        p.text,
        token=token,
        position=p.position,
        top_k=p.top_k,
        sae=p.sae,
        sae_at=p.sae_at,
        sae_subfolder=p.sae_subfolder,
    )


# ---------------------------------------------------------------------------
# attribute
# ---------------------------------------------------------------------------


class AttributeParams(BaseModel):
    text: str = Field(
        ...,
        title="input",
        description="Input text (or a server-side image path for vision models)",
        json_schema_extra=ui(widget="textarea", rows=2),
    )
    method: Literal["integrated_gradients", "gradient", "gradient_x_input"] = Field(
        "integrated_gradients", description="Attribution method"
    )
    target: int | None = Field(
        None,
        description="Target class / token id (defaults to the model's prediction)",
        json_schema_extra=ui(advanced=True),
    )


def _run_attribute(model: Any, p: AttributeParams, ctx: JobContext) -> Any:
    return model.attribute(p.text, target=p.target, method=p.method)


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------


class AttentionParams(BaseModel):
    text: str = Field(..., description="Input text", json_schema_extra=ui(widget="textarea", rows=2))
    layer: int | None = Field(
        None,
        description="Layer to show (empty = all layers)",
        json_schema_extra=ui(widget="layer-select"),
    )
    head: int | None = Field(
        None,
        description="Head to show (empty = all heads)",
        json_schema_extra=ui(widget="head-select"),
    )


def _run_attention(model: Any, p: AttentionParams, ctx: JobContext) -> Any:
    result = model.attention(p.text, layer=p.layer, head=p.head)
    if result is None:
        raise ValueError(f"attention: no attention layers matched (layer={p.layer!r}, head={p.head!r}).")
    return {"tokens": decoded_tokens(model, p.text), "heads": result}


# ---------------------------------------------------------------------------
# activations
# ---------------------------------------------------------------------------


class ActivationsParams(BaseModel):
    text: str = Field(
        ...,
        title="input",
        description="Input text (or a server-side image path for vision models)",
        json_schema_extra=ui(widget="textarea", rows=2),
    )
    at: str = Field(
        ...,
        description="Module path(s) to capture, comma-separated",
        json_schema_extra=ui(widget="module-picker"),
    )


def _run_activations(model: Any, p: ActivationsParams, ctx: JobContext) -> Any:
    modules = [s.strip() for s in p.at.split(",") if s.strip()]
    raw = model.activations(p.text, at=modules[0] if len(modules) == 1 else modules)
    tensors = raw if isinstance(raw, dict) else {modules[0]: raw}
    # Stats, not raw tensors: a single activation can be seq x hidden floats,
    # and the GUI's job of "show me what this module does" is statistical.
    stats = []
    for name, t in tensors.items():
        tf = t.detach().float()
        stats.append(
            {
                "module": name,
                "shape": list(t.shape),
                "mean": tf.mean().item(),
                "std": tf.std().item(),
                "min": tf.min().item(),
                "max": tf.max().item(),
                "l2_norm": tf.norm().item(),
            }
        )
    return {"stats": stats}


# ---------------------------------------------------------------------------
# trace
# ---------------------------------------------------------------------------


class TraceParams(BaseModel):
    clean: str = Field(
        ...,
        description="Clean input",
        json_schema_extra=ui(widget="textarea", rows=2, placeholder="The Eiffel Tower is in Paris"),
    )
    corrupted: str = Field(
        ...,
        description="Corrupted input",
        json_schema_extra=ui(widget="textarea", rows=2, placeholder="The Eiffel Tower is in Rome"),
    )
    mode: Literal["module", "position"] = Field(
        "module", description="'module' ranks modules; 'position' is the Meng et al. layers x positions heatmap"
    )
    top_k: int = Field(20, ge=0, description="Scan the top-K modules by proxy score (0 = all)")
    metric: _METRICS = Field("logit_diff", description="Effect metric", json_schema_extra=ui(advanced=True))


def _run_trace(model: Any, p: TraceParams, ctx: JobContext) -> Any:
    result = model.trace(
        p.clean,
        p.corrupted,
        top_k=p.top_k if p.top_k > 0 else None,
        mode=p.mode,
        metric=p.metric,
        progress_callback=ctx.progress_callback,
    )
    return result if isinstance(result, dict) else {"results": result}


# ---------------------------------------------------------------------------
# patch
# ---------------------------------------------------------------------------


class PatchParams(BaseModel):
    clean: str = Field(..., description="Clean input", json_schema_extra=ui(widget="textarea", rows=2))
    corrupted: str = Field(..., description="Corrupted input", json_schema_extra=ui(widget="textarea", rows=2))
    at: str = Field(..., description="Module to patch", json_schema_extra=ui(widget="module-picker"))
    head: int | None = Field(
        None,
        description="Specific attention head to patch",
        json_schema_extra=ui(widget="head-select", advanced=True),
    )
    positions: str | None = Field(
        None,
        description="Comma-separated token positions to patch (e.g. 3,4,5)",
        json_schema_extra=ui(advanced=True),
    )
    metric: _METRICS = Field("logit_diff", description="Effect metric", json_schema_extra=ui(advanced=True))


def _run_patch(model: Any, p: PatchParams, ctx: JobContext) -> Any:
    pos_list = [int(s.strip()) for s in p.positions.split(",")] if p.positions else None
    result = model.patch(p.clean, p.corrupted, at=p.at, head=p.head, positions=pos_list, metric=p.metric)
    return _summarize_logits(model, result)


# ---------------------------------------------------------------------------
# ablate
# ---------------------------------------------------------------------------


class AblateParams(BaseModel):
    text: str = Field(..., title="input", description="Input text", json_schema_extra=ui(widget="textarea", rows=2))
    at: str = Field(..., description="Module to ablate", json_schema_extra=ui(widget="module-picker"))
    method: Literal["zero", "mean", "resample"] = Field("zero", description="Ablation method")
    reference: str | None = Field(
        None,
        description="Reference input (required for resample ablation)",
        json_schema_extra=ui(widget="textarea", rows=2, show_if={"method": "resample"}),
    )


def _run_ablate(model: Any, p: AblateParams, ctx: JobContext) -> Any:
    result = model.ablate(p.text, at=p.at, method=p.method, reference=p.reference)
    return _summarize_logits(model, result)


# ---------------------------------------------------------------------------
# decompose
# ---------------------------------------------------------------------------


class DecomposeParams(BaseModel):
    text: str = Field(..., title="input", description="Input text", json_schema_extra=ui(widget="textarea", rows=2))
    position: int = Field(-1, description="Token position to decompose (-1 = last)")
    exact: bool = Field(False, description="Exact decomposition (slower)", json_schema_extra=ui(advanced=True))


def _run_decompose(model: Any, p: DecomposeParams, ctx: JobContext) -> Any:
    result = model.decompose(p.text, position=p.position, exact=p.exact)
    # The residual vector is hidden_size floats of little display value.
    result.pop("residual", None)
    return result


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


class DiffParams(BaseModel):
    model_b: str = Field(
        ...,
        title="model B",
        description="Second model to compare against (HF model ID — loaded for this run, then released)",
        json_schema_extra=ui(placeholder="distilgpt2"),
    )
    text: str = Field(..., description="Input text to compare on", json_schema_extra=ui(widget="textarea", rows=2))


def _run_diff(model: Any, p: DiffParams, ctx: JobContext) -> Any:
    import gc

    import interpkit
    from interpkit.core.loader import load

    ctx.report(0, 2, f"Loading {p.model_b}...")
    model_b = load(p.model_b, device=model.device)
    try:
        ctx.report(1, 2, "Comparing models...")
        return interpkit.diff(model, model_b, p.text)
    finally:
        del model_b
        gc.collect()


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


class ProbeParams(BaseModel):
    at: str = Field(..., description="Module to probe", json_schema_extra=ui(widget="module-picker"))
    texts: str | None = Field(
        None,
        description="Examples, one per line (pair with labels below)",
        json_schema_extra=ui(widget="textarea", rows=6),
    )
    labels: str | None = Field(
        None,
        description="Integer labels, comma- or space-separated, one per example line",
        json_schema_extra=ui(placeholder="0, 0, 1, 1"),
    )
    data_path: str | None = Field(
        None,
        description='Alternative: server-side JSON file with {"texts": [...], "labels": [...]}',
        json_schema_extra=ui(widget="path", advanced=True),
    )


def _run_probe(model: Any, p: ProbeParams, ctx: JobContext) -> Any:
    if p.data_path:
        data = json.loads(Path(p.data_path).read_text())
        texts, labels = data["texts"], data["labels"]
    else:
        if not p.texts or not p.labels:
            raise ValueError("probe: provide texts + labels, or a data_path JSON file.")
        texts = lines_list(p.texts)
        labels = [int(tok) for tok in p.labels.replace(",", " ").split()]
        if len(texts) != len(labels):
            raise ValueError(f"probe: {len(texts)} texts but {len(labels)} labels — counts must match.")
    return model.probe(texts=texts, labels=labels, at=p.at)


SPECS: list[OpSpec] = [
    OpSpec(
        name="lens",
        category="analysis",
        title="Logit lens",
        description="Project each layer's hidden state into vocabulary space to watch the prediction form.",
        params=LensParams,
        run=_run_lens,
        support_key="lens",
    ),
    OpSpec(
        name="dla",
        category="analysis",
        title="Direct logit attribution",
        description="Decompose the output logit into per-component contributions (embeddings, attention heads, MLPs).",
        params=DlaParams,
        run=_run_dla,
        support_key="dla",
    ),
    OpSpec(
        name="attribute",
        category="analysis",
        title="Input attribution",
        description="Gradient-based saliency over input tokens (or pixels): which inputs drove the prediction?",
        params=AttributeParams,
        run=_run_attribute,
        support_key="attribute",
    ),
    OpSpec(
        name="attention",
        category="analysis",
        title="Attention patterns",
        description="Per-head attention heatmaps with entropy and the strongest source-target token pairs.",
        params=AttentionParams,
        run=_run_attention,
        support_key="attention",
    ),
    OpSpec(
        name="activations",
        category="analysis",
        title="Activations",
        description="Capture activation statistics at any module in the model.",
        params=ActivationsParams,
        run=_run_activations,
        support_key="activations",
    ),
    OpSpec(
        name="trace",
        category="analysis",
        title="Causal trace",
        description="Patch clean activations into a corrupted run, module by module, to rank causal importance.",
        params=TraceParams,
        run=_run_trace,
        long_running=True,
        support_key="trace",
    ),
    OpSpec(
        name="patch",
        category="analysis",
        title="Activation patch",
        description="Swap one module's activations from the clean run into the corrupted run and measure the effect.",
        params=PatchParams,
        run=_run_patch,
        support_key="patch",
    ),
    OpSpec(
        name="ablate",
        category="analysis",
        title="Ablate",
        description="Zero, mean, or resample a module's output and measure how the prediction changes.",
        params=AblateParams,
        run=_run_ablate,
        support_key="ablate",
    ),
    OpSpec(
        name="decompose",
        category="analysis",
        title="Decompose",
        description="Break the residual stream at one position into per-component contributions.",
        params=DecomposeParams,
        run=_run_decompose,
        support_key="decompose",
    ),
    OpSpec(
        name="diff",
        category="analysis",
        title="Model diff",
        description="Compare per-layer activations between this model and another on the same input.",
        params=DiffParams,
        run=_run_diff,
        long_running=True,
        support_key="diff",
    ),
    OpSpec(
        name="probe",
        category="analysis",
        title="Linear probe",
        description="Train a linear probe on activations to test whether a concept is linearly decodable.",
        params=ProbeParams,
        run=_run_probe,
        support_key="probe",
    ),
]
