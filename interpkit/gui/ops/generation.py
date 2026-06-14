"""Steering & generation ops: steer, generate, chat, features."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from interpkit.gui.ops.base import JobContext, OpSpec, lines_list, lines_or_text, ui

_STEER_GROUP = "Contrastive steering"
_SAE_GROUP = "SAE feature steering"


def _build_interventions(
    model: Any,
    *,
    positive: str | None,
    negative: str | None,
    at: str | None,
    scale: float,
    sae: str | None,
    sae_subfolder: str | None,
    feature: int | None,
    feature_mode: str,
    strength: float,
    ablate_at: str | None = None,
    ablate_method: str = "zero",
) -> list[Any]:
    """Shared CLI-parity validation + intervention construction for generate."""
    from interpkit.core.interventions import (
        AblateIntervention,
        SAEFeatureIntervention,
        SteerIntervention,
    )

    wants_steering = bool(positive or negative)
    wants_feature = feature is not None or sae is not None
    if wants_steering and wants_feature:
        raise ValueError(
            "Contrastive steering (positive/negative) and SAE feature steering "
            "(sae/feature) are mutually exclusive."
        )
    if wants_feature and (feature is None or sae is None):
        raise ValueError("SAE feature steering requires both an SAE and a feature index.")
    if (wants_steering or wants_feature) and not at:
        raise ValueError("Steering requires a module to apply it at ('at').")

    interventions: list[Any] = []
    if wants_feature:
        from interpkit.ops.sae import _ensure_sae_on_device, load_sae

        loaded_sae = _ensure_sae_on_device(
            load_sae(sae, device=model._device, subfolder=sae_subfolder), model._device
        )
        interventions.append(
            SAEFeatureIntervention(at, sae=loaded_sae, feature=feature, strength=strength, mode=feature_mode)
        )
    if wants_steering:
        if not positive or not negative:
            raise ValueError("Contrastive steering requires both positive and negative examples.")
        vector = model.steer_vector(lines_or_text(positive), lines_or_text(negative), at=at)
        interventions.append(SteerIntervention(at, vector=vector, scale=scale))
    if ablate_at:
        interventions.append(AblateIntervention(ablate_at, method=ablate_method))
    return interventions


# ---------------------------------------------------------------------------
# steer
# ---------------------------------------------------------------------------


class SteerParams(BaseModel):
    text: str = Field(
        ...,
        title="input",
        description="Input text to steer",
        json_schema_extra=ui(widget="textarea", rows=2, placeholder="My favorite place in the world is"),
    )
    at: str = Field(..., description="Module to apply steering at", json_schema_extra=ui(widget="module-picker"))
    positive: str | None = Field(
        None,
        description="Positive direction example(s), one per line",
        json_schema_extra=ui(widget="textarea", rows=2, group=_STEER_GROUP, placeholder="I love this!"),
    )
    negative: str | None = Field(
        None,
        description="Negative direction example(s), one per line",
        json_schema_extra=ui(widget="textarea", rows=2, group=_STEER_GROUP, placeholder="I hate this!"),
    )
    scale: float = Field(2.0, description="Steering vector scale", json_schema_extra=ui(group=_STEER_GROUP))
    sae: str | None = Field(
        None,
        title="SAE",
        description="SAE source: HF repo ID, local path, or 'org/repo/subfolder'",
        json_schema_extra=ui(group=_SAE_GROUP),
    )
    feature: int | None = Field(None, description="SAE feature index", json_schema_extra=ui(group=_SAE_GROUP))
    feature_mode: Literal["clamp", "add"] = Field(
        "clamp",
        description="'clamp' pins the feature's activation (Golden Gate style); 'add' injects the decoder direction",
        json_schema_extra=ui(group=_SAE_GROUP),
    )
    strength: float = Field(
        10.0,
        description="Feature target activation (clamp) or added activation (add)",
        json_schema_extra=ui(group=_SAE_GROUP),
    )
    sae_subfolder: str | None = Field(
        None, description="Subfolder inside the SAE repo", json_schema_extra=ui(group=_SAE_GROUP, advanced=True)
    )


def _run_steer(model: Any, p: SteerParams, ctx: JobContext) -> Any:
    wants_contrastive = bool(p.positive or p.negative)
    wants_feature = p.feature is not None or p.sae is not None
    if wants_contrastive and wants_feature:
        raise ValueError(
            "Contrastive steering (positive/negative) and SAE feature steering "
            "(sae/feature) are mutually exclusive."
        )

    if wants_feature:
        if p.feature is None or p.sae is None:
            raise ValueError("SAE feature steering requires both an SAE and a feature index.")
        result = model.steer(
            p.text,
            at=p.at,
            sae=p.sae,
            feature=p.feature,
            mode=p.feature_mode,
            strength=p.strength,
            sae_subfolder=p.sae_subfolder,
        )
    else:
        if not p.positive or not p.negative:
            raise ValueError("Provide positive and negative examples, or an SAE + feature.")
        vector = model.steer_vector(lines_or_text(p.positive), lines_or_text(p.negative), at=p.at)
        result = model.steer(p.text, vector=vector, at=p.at, scale=p.scale)

    # Full (1, seq, vocab) logit tensors ride along in the Python API but
    # are megabytes of noise in a JSON job result.
    return {k: v for k, v in result.items() if k not in ("original_logits", "steered_logits")}


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


class GenerateParams(BaseModel):
    prompt: str = Field(..., description="Prompt to generate from", json_schema_extra=ui(widget="textarea", rows=2))
    max_new_tokens: int = Field(64, ge=1, description="Max generation length")
    sample: bool = Field(False, description="Sample instead of greedy decoding")
    temperature: float = Field(
        1.0, description="Sampling temperature", json_schema_extra=ui(show_if={"sample": True})
    )
    top_p: float = Field(
        1.0, description="Nucleus sampling cutoff", json_schema_extra=ui(show_if={"sample": True})
    )
    capture: Literal["lens", "logits"] | None = Field(
        None,
        description="Record per-token trajectories: 'lens' (logit-lens through every block) or 'logits'",
        json_schema_extra=ui(advanced=True),
    )
    at: str | None = Field(
        None,
        description="Module to apply steering at (required for steering)",
        json_schema_extra=ui(widget="module-picker", group="Steering (optional)"),
    )
    positive: str | None = Field(
        None,
        description="Positive steering example(s), one per line",
        json_schema_extra=ui(widget="textarea", rows=2, group="Steering (optional)"),
    )
    negative: str | None = Field(
        None,
        description="Negative steering example(s), one per line",
        json_schema_extra=ui(widget="textarea", rows=2, group="Steering (optional)"),
    )
    scale: float = Field(2.0, description="Steering vector scale", json_schema_extra=ui(group="Steering (optional)"))
    sae: str | None = Field(
        None, title="SAE", description="SAE source (with feature)", json_schema_extra=ui(group="Steering (optional)")
    )
    feature: int | None = Field(
        None, description="SAE feature index", json_schema_extra=ui(group="Steering (optional)")
    )
    feature_mode: Literal["clamp", "add"] = Field(
        "clamp", description="SAE feature steering mode", json_schema_extra=ui(group="Steering (optional)", advanced=True)
    )
    strength: float = Field(
        10.0, description="SAE feature strength", json_schema_extra=ui(group="Steering (optional)")
    )
    sae_subfolder: str | None = Field(
        None, description="Subfolder inside the SAE repo", json_schema_extra=ui(group="Steering (optional)", advanced=True)
    )
    ablate_at: str | None = Field(
        None,
        description="Module to ablate during generation",
        json_schema_extra=ui(widget="module-picker", group="Ablation (optional)"),
    )
    ablate_method: Literal["zero", "mean"] = Field(
        "zero", description="Ablation method", json_schema_extra=ui(group="Ablation (optional)")
    )


def _run_generate(model: Any, p: GenerateParams, ctx: JobContext) -> Any:
    interventions = _build_interventions(
        model,
        positive=p.positive,
        negative=p.negative,
        at=p.at,
        scale=p.scale,
        sae=p.sae,
        sae_subfolder=p.sae_subfolder,
        feature=p.feature,
        feature_mode=p.feature_mode,
        strength=p.strength,
        ablate_at=p.ablate_at,
        ablate_method=p.ablate_method,
    )
    result = model.generate(
        p.prompt,
        max_new_tokens=p.max_new_tokens,
        interventions=interventions or None,
        capture=p.capture,
        do_sample=p.sample,
        temperature=p.temperature,
        top_p=p.top_p,
    )
    # Same JSON trimming as the CLI: drop token-id tensors and per-step
    # (1, vocab) logits.
    out = {k: v for k, v in result.items() if k not in ("input_ids", "output_ids")}
    if "steps" in out:
        out["steps"] = [{k: v for k, v in step.items() if k != "logits"} for step in out["steps"]]
    return out


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


class ChatTurn(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatParams(BaseModel):
    message: str = Field(..., description="User message", json_schema_extra=ui(widget="textarea", rows=2))
    system: str | None = Field(None, description="System prompt", json_schema_extra=ui(advanced=True))
    history: list[ChatTurn] = Field(
        default_factory=list,
        description="Prior conversation turns (managed by the chat view)",
        json_schema_extra=ui(widget="hidden"),
    )
    max_new_tokens: int = Field(128, ge=1, description="Max generation length")
    sample: bool = Field(False, description="Sample instead of greedy decoding")
    temperature: float = Field(1.0, description="Sampling temperature", json_schema_extra=ui(show_if={"sample": True}))
    top_p: float = Field(1.0, description="Nucleus sampling cutoff", json_schema_extra=ui(show_if={"sample": True}))


def _run_chat(model: Any, p: ChatParams, ctx: JobContext) -> Any:
    if p.history:
        messages = [t.model_dump() for t in p.history] + [{"role": "user", "content": p.message}]
        result = model.chat(
            messages,
            max_new_tokens=p.max_new_tokens,
            do_sample=p.sample,
            temperature=p.temperature,
            top_p=p.top_p,
        )
    else:
        result = model.chat(
            p.message,
            system=p.system,
            max_new_tokens=p.max_new_tokens,
            do_sample=p.sample,
            temperature=p.temperature,
            top_p=p.top_p,
        )
    return {k: v for k, v in result.items() if k not in ("input_ids", "output_ids")}


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------


class FeaturesParams(BaseModel):
    sae: str = Field(
        ...,
        title="SAE",
        description="SAE source: HF repo ID, local path, or 'org/repo/subfolder'",
        json_schema_extra=ui(placeholder="jbloom/GPT2-Small-SAEs-Reformatted/blocks.8.hook_resid_pre"),
    )
    at: str = Field(..., description="Module to decompose", json_schema_extra=ui(widget="module-picker"))
    text: str | None = Field(
        None,
        title="input",
        description="Input text (leave empty when using contrastive mode below)",
        json_schema_extra=ui(widget="textarea", rows=2),
    )
    top_k: int = Field(20, ge=1, description="Top features to show")
    positive: str | None = Field(
        None,
        description="Positive examples for contrastive analysis, one per line",
        json_schema_extra=ui(widget="textarea", rows=3, group="Contrastive mode", advanced=True),
    )
    negative: str | None = Field(
        None,
        description="Negative examples for contrastive analysis, one per line",
        json_schema_extra=ui(widget="textarea", rows=3, group="Contrastive mode", advanced=True),
    )
    sae_subfolder: str | None = Field(
        None, description="Subfolder inside the SAE repo", json_schema_extra=ui(advanced=True)
    )


def _run_features(model: Any, p: FeaturesParams, ctx: JobContext) -> Any:
    contrastive = bool(p.positive or p.negative)
    if contrastive:
        if not p.positive or not p.negative:
            raise ValueError("Contrastive mode needs both positive and negative examples.")
        return model.contrastive_features(
            lines_list(p.positive),
            lines_list(p.negative),
            at=p.at,
            sae=p.sae,
            top_k=p.top_k,
            sae_subfolder=p.sae_subfolder,
        )
    if not p.text:
        raise ValueError("Provide an input text, or positive + negative examples for contrastive mode.")
    return model.features(p.text, at=p.at, sae=p.sae, top_k=p.top_k, sae_subfolder=p.sae_subfolder)


SPECS: list[OpSpec] = [
    OpSpec(
        name="steer",
        category="generation",
        title="Steer",
        description="Push the model along a direction — a contrastive vector or an SAE feature — and compare predictions.",
        params=SteerParams,
        run=_run_steer,
        support_key="steer",
    ),
    OpSpec(
        name="generate",
        category="generation",
        title="Generate",
        description="Generate text with steering or ablation active at every decode step.",
        params=GenerateParams,
        run=_run_generate,
        long_running=True,
        support_key="generate",
    ),
    OpSpec(
        name="chat",
        category="generation",
        title="Chat",
        description="Converse with an instruct model through its chat template — interventions and all ops apply.",
        params=ChatParams,
        run=_run_chat,
        long_running=True,
        support_key="chat",
    ),
    OpSpec(
        name="features",
        category="generation",
        title="SAE features",
        description="Decompose activations through a Sparse Autoencoder into interpretable features.",
        params=FeaturesParams,
        run=_run_features,
        support_key="features",
    ),
]
