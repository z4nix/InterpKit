"""Circuit-discovery ops: find-circuit, atp, eap, maxact."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from interpkit.gui.ops.base import JobContext, OpSpec, lines_list, lines_or_text, ui

_METRICS = Literal["logit_diff", "kl_div", "target_prob", "l2_prob"]


# ---------------------------------------------------------------------------
# find-circuit
# ---------------------------------------------------------------------------


class FindCircuitParams(BaseModel):
    clean: str = Field(
        ...,
        description="Clean input(s), one per line",
        json_schema_extra=ui(widget="textarea", rows=2, placeholder="When John and Mary went to the store, John gave a drink to"),
    )
    corrupted: str = Field(
        ...,
        description="Corrupted input(s), one per line (paired with clean lines)",
        json_schema_extra=ui(widget="textarea", rows=2),
    )
    threshold: float = Field(0.01, description="Minimum ablation effect to include in the circuit (0-1)")
    method: Literal["mean", "zero", "resample", "eap", "eap-ig"] = Field(
        "mean", description="Selection method: ablation-based (mean/zero/resample) or gradient-based (eap/eap-ig, much faster)"
    )
    metric: _METRICS = Field("logit_diff", description="Effect metric", json_schema_extra=ui(advanced=True))


def _run_find_circuit(model: Any, p: FindCircuitParams, ctx: JobContext) -> Any:
    return model.find_circuit(
        lines_or_text(p.clean),
        lines_or_text(p.corrupted),
        threshold=p.threshold,
        method=p.method,
        metric=p.metric,
    )


# ---------------------------------------------------------------------------
# atp
# ---------------------------------------------------------------------------


class AtpParams(BaseModel):
    clean: str = Field(..., description="Clean input", json_schema_extra=ui(widget="textarea", rows=2))
    corrupted: str = Field(..., description="Corrupted input", json_schema_extra=ui(widget="textarea", rows=2))
    top_k: int = Field(20, ge=0, description="Top modules to report by absolute score (0 = all)")
    metric: _METRICS = Field("logit_diff", description="Effect metric", json_schema_extra=ui(advanced=True))


def _run_atp(model: Any, p: AtpParams, ctx: JobContext) -> Any:
    return model.atp(p.clean, p.corrupted, top_k=p.top_k if p.top_k > 0 else None, metric=p.metric)


# ---------------------------------------------------------------------------
# eap
# ---------------------------------------------------------------------------


class EapParams(BaseModel):
    clean: str = Field(
        ...,
        description="Clean input (must tokenize to the same length as corrupted)",
        json_schema_extra=ui(widget="textarea", rows=2),
    )
    corrupted: str = Field(..., description="Corrupted input", json_schema_extra=ui(widget="textarea", rows=2))
    ig_steps: int = Field(0, ge=0, description="EAP-IG interpolation steps (0 = plain EAP; try 5)")
    top_k_edges: int = Field(30, ge=0, description="Top edges to report by absolute score (0 = all)")
    metric: _METRICS = Field("logit_diff", description="Effect metric", json_schema_extra=ui(advanced=True))


def _run_eap(model: Any, p: EapParams, ctx: JobContext) -> Any:
    return model.eap(
        p.clean,
        p.corrupted,
        ig_steps=p.ig_steps,
        top_k_edges=p.top_k_edges if p.top_k_edges > 0 else None,
        metric=p.metric,
    )


# ---------------------------------------------------------------------------
# maxact
# ---------------------------------------------------------------------------


class MaxactParams(BaseModel):
    at: str = Field(..., description="Module whose activations to scan", json_schema_extra=ui(widget="module-picker"))
    texts: str | None = Field(
        None,
        description="Corpus examples, one per line",
        json_schema_extra=ui(widget="textarea", rows=6),
    )
    texts_path: str | None = Field(
        None,
        description="Alternative: server-side text file with one example per line",
        json_schema_extra=ui(widget="path"),
    )
    dataset: str | None = Field(
        None,
        description="Alternative: HF dataset spec hf:name[:split[:column]] (needs interpkit[data] + max examples)",
        json_schema_extra=ui(advanced=True),
    )
    neuron: int | None = Field(None, description="Neuron index at the module")
    feature: int | None = Field(None, description="SAE feature index (requires SAE)")
    head: int | None = Field(None, description="Attention head index", json_schema_extra=ui(widget="head-select"))
    sae: str | None = Field(None, title="SAE", description="SAE repo ID or local path (with feature)")
    top_k: int = Field(20, ge=1, description="Top examples to keep")
    batch_size: int = Field(8, ge=1, description="Forward batch size", json_schema_extra=ui(advanced=True))
    max_examples: int | None = Field(None, description="Cap on examples scanned", json_schema_extra=ui(advanced=True))
    max_length: int = Field(128, ge=1, description="Token truncation length", json_schema_extra=ui(advanced=True))


def _run_maxact(model: Any, p: MaxactParams, ctx: JobContext) -> Any:
    sources = [s for s in (p.texts, p.texts_path, p.dataset) if s]
    if len(sources) != 1:
        raise ValueError("maxact: provide exactly one of texts, texts_path, or dataset.")

    data: list[str] | str
    if p.texts:
        data = lines_list(p.texts)
    elif p.texts_path:
        from interpkit.core.inputs import read_examples_file

        data = read_examples_file(p.texts_path)
    else:
        data = p.dataset  # hf:... spec, streamed by the op

    return model.max_activating(
        data,
        at=p.at,
        neuron=p.neuron,
        feature=p.feature,
        head=p.head,
        sae=p.sae,
        top_k=p.top_k,
        batch_size=p.batch_size,
        max_examples=p.max_examples,
        max_length=p.max_length,
        progress_callback=ctx.progress_callback,
    )


SPECS: list[OpSpec] = [
    OpSpec(
        name="find-circuit",
        category="circuits",
        title="Find circuit",
        description="Automated circuit discovery: the minimal set of components that explains a behaviour, causally verified.",
        params=FindCircuitParams,
        run=_run_find_circuit,
        long_running=True,
        support_key="find_circuit",
    ),
    OpSpec(
        name="atp",
        category="circuits",
        title="Attribution patching",
        description="First-order patch-effect scores for every module from just three passes — the fast first look.",
        params=AtpParams,
        run=_run_atp,
        support_key="atp",
    ),
    OpSpec(
        name="eap",
        category="circuits",
        title="Edge attribution patching",
        description="Gradient-based scores for every component-to-residual edge — circuit discovery at edge granularity.",
        params=EapParams,
        run=_run_eap,
        support_key="eap",
    ),
    OpSpec(
        name="maxact",
        category="circuits",
        title="Max-activating examples",
        description="Scan a corpus for the examples that most activate a neuron, SAE feature, or attention head.",
        params=MaxactParams,
        run=_run_maxact,
        long_running=True,
        support_key="max_activating",
    ),
]
