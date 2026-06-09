"""attribute — gradient saliency over input tokens or pixels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
from rich.console import Console
from rich.progress import Progress

from interpkit.core.enums import VALID_ATTR_METHODS, _validate_enum

if TYPE_CHECKING:
    from interpkit.core.model import Model

console = Console()

# Integrated-gradients interpolation steps are batched this many at a time
# through a single forward+backward (see ``_ig_one_pass``). Trades memory
# (~_IG_BATCH× one forward's activations) for ≈ that many fewer passes. 32
# keeps the common n_steps range (16–64) to one or two batches while staying
# well within memory for interpretability-scale models.
_IG_BATCH = 32

# N-008: integration schemes for IG.
VALID_QUADRATURES = frozenset({"riemann_midpoint", "trapezoidal", "gauss_legendre"})


def _quadrature_nodes_and_weights(
    quadrature: str, n_steps: int,
) -> tuple[list[float], list[float]]:
    """Return ``(alphas, weights)`` for the requested quadrature scheme.

    Each alpha lies in ``[0, 1]``; weights sum to 1 so the integral
    estimator is ``Σ_i w_i * grad(baseline + alpha_i * delta)`` and the
    final IG attribution is ``delta * estimator``.

    - ``riemann_midpoint``: alpha_i = (i + 0.5) / n, w_i = 1/n.
    - ``trapezoidal``: alpha_i = i / (n-1), w_i = 1/(n-1) with
      endpoint weights of 0.5/(n-1). Strictly more accurate than
      midpoint on monotone-ish gradients (N-008).
    - ``gauss_legendre``: Gauss–Legendre nodes mapped from [-1, 1] to
      [0, 1]. Optimal accuracy for smooth integrands; uses
      ``numpy.polynomial.legendre.leggauss`` when numpy is available
      and falls back to trapezoidal otherwise.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if quadrature == "riemann_midpoint":
        alphas = [(i + 0.5) / n_steps for i in range(n_steps)]
        weights = [1.0 / n_steps] * n_steps
        return alphas, weights
    if quadrature == "trapezoidal":
        if n_steps == 1:
            return [0.5], [1.0]
        alphas = [i / (n_steps - 1) for i in range(n_steps)]
        w = 1.0 / (n_steps - 1)
        weights = [w] * n_steps
        weights[0] *= 0.5
        weights[-1] *= 0.5
        return alphas, weights
    if quadrature == "gauss_legendre":
        try:
            import numpy as np

            nodes, w = np.polynomial.legendre.leggauss(n_steps)
            # Map [-1, 1] → [0, 1]
            alphas = np.asarray((nodes + 1.0) / 2.0, dtype=float).tolist()
            weights = np.asarray(w / 2.0, dtype=float).tolist()
            return alphas, weights
        except ImportError:
            return _quadrature_nodes_and_weights("trapezoidal", n_steps)
    raise ValueError(
        f"Unknown quadrature {quadrature!r}; valid: {sorted(VALID_QUADRATURES)}"
    )


def run_attribute(
    model: Model,
    input_data: Any,
    *,
    target: int | None = None,
    method: str = "integrated_gradients",
    n_steps: int = 128,
    baseline: str | torch.Tensor = "pad",
    quadrature: Literal["riemann_midpoint", "trapezoidal", "gauss_legendre"] = "trapezoidal",
    auto_bump: bool = True,
    max_n_steps: int = 512,
    save: str | None = None,
    html: str | None = None,
) -> dict[str, Any]:
    """Compute gradient-based attribution and render results.

    Parameters
    ----------
    method:
        ``"integrated_gradients"`` (default) — Sundararajan et al. 2017.
        ``"gradient"`` — vanilla gradient saliency.
        ``"gradient_x_input"`` — gradient times input embedding.
    n_steps:
        Interpolation steps for integrated gradients (default 128 in 1.0;
        was 50 pre-1.0). The audit found ~17 nat completeness error at
        50 steps on modern decoders (F-011) — 128 brings this under 1
        nat for most models.
    baseline:
        IG baseline embedding (F-011). One of:

        - ``"pad"`` (default): the PAD-token embedding repeated. Stays
          in-distribution and dramatically improves completeness.
        - ``"zero"``: the all-zero embedding (legacy, out-of-distribution).
        - ``"mean"``: the mean of the embedding matrix.
        - a ``torch.Tensor``: shape must match the input embedding.
    quadrature:
        N-008 — integration scheme for IG. Default ``"trapezoidal"`` is
        strictly more accurate than midpoint on monotone gradients at
        the same ``n_steps``; ``"gauss_legendre"`` converges faster on
        smooth integrands at the cost of a numpy dependency. Only
        affects ``method="integrated_gradients"``.
    auto_bump:
        N-008 — when True (default), if the completeness axiom fails
        on the initial run, automatically re-run with double the
        ``n_steps`` once (capped at ``max_n_steps``). The
        ``ig_diagnostics`` block reports whether the bump was attempted
        and the final n_steps used.
    max_n_steps:
        Cap on ``auto_bump``. Default 512.

    Methods can disagree by significant amounts on the same input
    (F-012); ``integrated_gradients`` satisfies the completeness axiom
    in the limit but ``gradient_x_input`` does not. For faithfulness
    analyses prefer IG; for local-behaviour studies use gradient_x_input.

    For text inputs: returns ``{"tokens", "scores", "target", "method",
    "interpretation", "ig_diagnostics"}`` with per-token importance and a
    diagnostics block reporting baseline, n_steps, completeness error, and
    pass/fail status (F-011).

    ``result["interpretation"]`` is the programmatic ranking-vs-magnitude
    contract (A3) ∈ ``{"quantitative", "ranking_only"}``:

    - ``"quantitative"`` — integrated_gradients whose completeness error is
      within tolerance; the per-token scores are additive contribution
      magnitudes.
    - ``"ranking_only"`` — either a non-IG method (``gradient`` /
      ``gradient_x_input`` are saliency, never additive), or IG whose
      completeness error exceeds 50% of the output gap (e.g. Qwen2/2.5/3 and
      SmolLM-family models, which do not converge even at large ``n_steps``).
      The scores are a valid token-importance *ranking* but must not be read
      as contribution magnitudes. Branch on this field instead of parsing the
      warning text.

    For image inputs: returns ``{"grad", "target", "interpretation"}`` with the
    pixel-gradient tensor (``interpretation`` is always ``"ranking_only"`` —
    saliency).
    For tensor inputs: returns ``{"labels", "scores", "target", "interpretation"}``.
    """
    # F-018: validate method at the entry. Pre-1.0 silently fell back to
    # the default (gradient) on typo'd method strings, producing wrong
    # attributions under benign-looking inputs.
    _validate_enum(method, VALID_ATTR_METHODS, "method")
    if quadrature not in VALID_QUADRATURES:
        raise ValueError(
            f"quadrature={quadrature!r} invalid; valid: {sorted(VALID_QUADRATURES)}"
        )

    # N-004: gate DeBERTa-v3 (DisentangledSelfAttention) — its
    # relative-position-bias path crashes under the gradient-tracking
    # forward used by IG. Surface a clean
    # ``OperationNotSupportedForArchitecture`` instead.
    from interpkit.core.support_matrix import check_op_supported
    check_op_supported("attribute", model.arch_info)

    from interpkit.core.inputs import _is_message_list, _looks_like_image_path

    is_text = isinstance(input_data, str) and not _looks_like_image_path(input_data)
    is_image = isinstance(input_data, str) and _looks_like_image_path(input_data)
    is_messages = _is_message_list(input_data)

    if is_text:
        return _attribute_text(
            model, input_data, target=target, method=method,
            n_steps=n_steps, baseline=baseline, quadrature=quadrature,
            auto_bump=auto_bump, max_n_steps=max_n_steps,
            save=save, html=html,
        )
    elif is_messages:
        return _attribute_messages(
            model, input_data, target=target, method=method,
            n_steps=n_steps, baseline=baseline, quadrature=quadrature,
            auto_bump=auto_bump, max_n_steps=max_n_steps,
            save=save, html=html,
        )
    elif is_image:
        return _attribute_image(model, input_data, target=target, save=save)
    else:
        return _attribute_tensor(model, input_data, target=target)


def _attribute_text(
    model: Model,
    text: str,
    *,
    target: int | None,
    method: str = "integrated_gradients",
    n_steps: int = 128,
    baseline: str | torch.Tensor = "pad",
    quadrature: str = "trapezoidal",
    auto_bump: bool = True,
    max_n_steps: int = 512,
    save: str | None = None,
    html: str | None = None,
) -> dict[str, Any]:
    if model._tokenizer is None:
        raise ValueError("No tokenizer available for text attribution.")

    # N-009: ``run_attribute`` bypasses ``prepare_input`` for text and goes
    # straight to the tokenizer to avoid double-tokenization. Mirror the
    # same pre-tokenization empty-string guard so ``attribute('')`` raises
    # the same friendly ``ValueError`` users get from every other op.
    if not text.strip():
        raise ValueError(
            "Input is empty or whitespace-only. "
            "Pass at least one non-whitespace character."
        )

    encoded = model._tokenizer(text, return_tensors="pt")
    return _attribute_from_encoded(
        model, encoded,
        target=target, method=method, n_steps=n_steps, baseline=baseline,
        quadrature=quadrature, auto_bump=auto_bump, max_n_steps=max_n_steps,
        save=save, html=html,
    )


def _attribute_messages(
    model: Model,
    messages: list[dict[str, Any]],
    *,
    target: int | None,
    method: str = "integrated_gradients",
    n_steps: int = 128,
    baseline: str | torch.Tensor = "pad",
    quadrature: str = "trapezoidal",
    auto_bump: bool = True,
    max_n_steps: int = 512,
    save: str | None = None,
    html: str | None = None,
) -> dict[str, Any]:
    """Attribute over chat-template-formatted messages.

    Routes through the tokenizer's chat template so models like
    Llama-3-Instruct, SmolLM2-Instruct, or Qwen-Chat receive the
    expected role/turn markers before gradient attribution runs.
    """
    from interpkit.core.inputs import _apply_chat_template

    if model._tokenizer is None:
        raise ValueError("No tokenizer available for chat-message attribution.")

    encoded = _apply_chat_template(
        messages,
        tokenizer=model._tokenizer,
        device="cpu",
    )
    return _attribute_from_encoded(
        model, encoded,
        target=target, method=method, n_steps=n_steps, baseline=baseline,
        quadrature=quadrature, auto_bump=auto_bump, max_n_steps=max_n_steps,
        save=save, html=html,
    )


def _attribute_from_encoded(
    model: Model,
    encoded: dict[str, torch.Tensor],
    *,
    target: int | None,
    method: str = "integrated_gradients",
    n_steps: int = 128,
    baseline: str | torch.Tensor = "pad",
    quadrature: str = "trapezoidal",
    auto_bump: bool = True,
    max_n_steps: int = 512,
    save: str | None = None,
    html: str | None = None,
) -> dict[str, Any]:
    """Run gradient attribution over an already-tokenized input dict.

    Shared backend for :func:`_attribute_text` and
    :func:`_attribute_messages`.  Expects ``encoded`` to contain at least
    ``input_ids``; ``attention_mask`` and other keys are forwarded to the
    underlying model.
    """
    from interpkit.core.render import render_attribution_tokens

    assert model._tokenizer is not None
    encoded = dict(encoded)
    input_ids = encoded["input_ids"].to(model._device)
    encoded["input_ids"] = input_ids

    # C2: route the seq2seq decoder-id quirk through the single accessor
    # (was a duplicate of Model._inject_decoder_ids).
    if model.arch_info.needs_decoder_input_ids and "decoder_input_ids" not in encoded:
        config = getattr(model._model, "config", None)
        decoder_start = getattr(config, "decoder_start_token_id", 0) or 0
        encoded["decoder_input_ids"] = torch.tensor(
            [[decoder_start]], dtype=torch.long,
        )

    embed_layer = _find_embedding(model._model)
    if embed_layer is None:
        raise RuntimeError("Could not find embedding layer for gradient attribution.")

    base_embeddings = embed_layer(input_ids).detach()
    original_forward = embed_layer.forward

    # Determine target class on a clean forward pass
    if target is None:
        with torch.no_grad():
            model_input_clean = {k: v.to(model._device) for k, v in encoded.items()}
            logits_clean = model._forward(model_input_clean)
            if logits_clean.dim() == 3:
                target = int(logits_clean[0, -1, :].argmax().item())
            else:
                target = int(logits_clean[0].argmax().item())

    model_input = {k: v.to(model._device) for k, v in encoded.items()}

    # Initialise IG diagnostics; populated by the IG path below.
    ig_diagnostics: dict[str, Any] | None = None

    # A3: programmatic ranking-vs-magnitude contract. Default "ranking_only":
    # gradient / gradient_x_input are saliency rankings, never additive
    # contribution magnitudes. The IG path upgrades this to "quantitative"
    # only when completeness holds (see below).
    interpretation = "ranking_only"

    if method == "integrated_gradients":
        # F-011: choose baseline embedding rather than the legacy zero baseline.
        # Zero embeddings are wildly out-of-distribution for HF transformers and
        # cause IG completeness error of 17+ nats on common LMs at 50 steps.
        baseline_embeddings, baseline_label, baseline_token_id = _resolve_baseline(
            baseline, embed_layer, model._tokenizer, base_embeddings,
        )

        # Compute f(x) and f(baseline) so we can check the completeness axiom.
        # Completeness: sum(IG attributions) ≈ f(x) - f(baseline). Required to
        # diagnose whether n_steps is sufficient for the chosen baseline.
        with torch.no_grad():
            f_x = _forward_score(model, model_input, embed_layer, original_forward, base_embeddings, target)
            f_baseline = _forward_score(model, model_input, embed_layer, original_forward, baseline_embeddings, target)
        output_gap = f_x - f_baseline

        delta = base_embeddings - baseline_embeddings

        # N-008: integrate using the user-selected quadrature scheme.
        # On completeness failure, retry once with double n_steps (capped
        # at max_n_steps) when ``auto_bump`` is enabled.
        ig, token_scores, token_scores_signed, completeness_error, used_n_steps, auto_bump_attempted = _run_ig_with_optional_bump(
            model=model,
            model_input=model_input,
            embed_layer=embed_layer,
            original_forward=original_forward,
            baseline_embeddings=baseline_embeddings,
            delta=delta,
            target=target,
            n_steps=n_steps,
            quadrature=quadrature,
            output_gap=output_gap,
            auto_bump=auto_bump,
            max_n_steps=max_n_steps,
        )

        completeness_tolerance = 0.1 * max(abs(output_gap), 1e-6)
        completeness_passed = completeness_error <= completeness_tolerance

        # P3b: surface meaningful completeness failure at two thresholds.
        # The 0.1 tier fires the existing "try a different baseline"
        # advice. The 0.5 tier — explicitly informed by the P0b sweep
        # finding that several Qwen/SmolLM models cannot converge with
        # any tractable n_steps — issues a stronger warning that the
        # IG result is qualitatively meaningful (ranking) but not
        # quantitatively faithful (magnitudes).
        rel_error = (
            abs(completeness_error) / max(abs(output_gap), 1e-6)
        )
        # A3: the strong-warning threshold is also the quantitative-vs-ranking
        # cutline. Above it the completeness axiom is so far from holding that
        # the per-token magnitudes are not additive contributions, only a
        # ranking; at or below it the IG scores are quantitatively meaningful.
        interpretation = "ranking_only" if rel_error > 0.5 else "quantitative"
        if rel_error > 0.5:
            import warnings as _warnings
            _warnings.warn(
                f"IG completeness error {completeness_error:.3f} is "
                f"{rel_error:.2f}× the output gap ({output_gap:.3f}) "
                f"after {used_n_steps} steps ({quadrature}). The per-token "
                f"attribution scores are meaningful as RANKINGS but should "
                f"not be interpreted as quantitative contribution magnitudes. "
                f"Consider method='gradient_x_input' for a different "
                f"trade-off (no completeness guarantee but cheaper).",
                UserWarning,
                stacklevel=3,
            )
        elif not completeness_passed:
            import warnings as _warnings
            extra = ""
            if quadrature != "gauss_legendre":
                extra = (
                    " Try quadrature='gauss_legendre' for faster convergence "
                    "on smooth integrands."
                )
            _warnings.warn(
                f"IG completeness error {completeness_error:.3f} exceeds "
                f"{100 * 0.1:.0f}% of output gap ({output_gap:.3f}) "
                f"after {used_n_steps} steps ({quadrature}). "
                f"Try a different baseline or further increase n_steps.{extra}",
                UserWarning,
                stacklevel=3,
            )

        ig_diagnostics = {
            "method": quadrature,
            "n_steps": used_n_steps,
            "n_steps_initial": n_steps,
            "auto_bump_attempted": auto_bump_attempted,
            "baseline": baseline_label,
            "baseline_token_id": baseline_token_id,
            "output_gap": float(output_gap),
            "completeness_error": float(completeness_error),
            "completeness_tolerance": float(completeness_tolerance),
            "completeness_passed": completeness_passed,
        }

    elif method == "gradient_x_input":
        embeddings = base_embeddings.requires_grad_(True)

        def _patched_forward_gxi(*args: Any, _emb: torch.Tensor = embeddings, **kwargs: Any) -> torch.Tensor:
            return _emb

        embed_layer.forward = _patched_forward_gxi  # type: ignore[assignment]
        try:
            logits = model._forward_with_grad(model_input).float()
            if logits.dim() == 3:
                score = logits[0, -1, target]
            else:
                score = logits[0, target]

            (grad,) = torch.autograd.grad(score, embeddings)
        finally:
            embed_layer.forward = original_forward  # type: ignore[assignment]

        gxi = grad[0] * base_embeddings[0]
        token_scores = gxi.norm(dim=-1).tolist()

    else:  # "gradient" — vanilla saliency
        embeddings = base_embeddings.requires_grad_(True)

        def _patched_forward_grad(*args: Any, _emb: torch.Tensor = embeddings, **kwargs: Any) -> torch.Tensor:
            return _emb

        embed_layer.forward = _patched_forward_grad  # type: ignore[assignment]
        try:
            logits = model._forward_with_grad(model_input).float()
            if logits.dim() == 3:
                score = logits[0, -1, target]
            else:
                score = logits[0, target]

            (grad,) = torch.autograd.grad(score, embeddings)
        finally:
            embed_layer.forward = original_forward  # type: ignore[assignment]

        token_scores = grad[0].norm(dim=-1).tolist()

    tokens = model._tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    render_attribution_tokens(tokens, token_scores)

    if save is not None:
        from interpkit.core.plot import plot_attribution

        plot_attribution(tokens, token_scores, save_path=save)

    if html is not None:
        from interpkit.core.html import html_attribution as gen_html_attribution
        from interpkit.core.html import save_html

        save_html(gen_html_attribution(tokens, token_scores), html)

    result = {
        "tokens": tokens,
        "scores": token_scores,
        "target": target,
        "method": method,
        "interpretation": interpretation,
    }
    if ig_diagnostics is not None:
        result["ig_diagnostics"] = ig_diagnostics
    return result


def _ig_one_pass(
    *,
    model: Model,
    model_input: dict[str, torch.Tensor],
    embed_layer: torch.nn.Module,
    original_forward: Any,
    baseline_embeddings: torch.Tensor,
    delta: torch.Tensor,
    target: int,
    n_steps: int,
    quadrature: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """One IG integration pass at the given (n_steps, quadrature) settings.

    Returns ``(ig, token_scores_norm, token_scores_signed, completeness_error)``.
    """
    alphas, weights = _quadrature_nodes_and_weights(quadrature, n_steps)
    accumulated_grads = torch.zeros_like(baseline_embeddings)

    device = baseline_embeddings.device
    dtype = baseline_embeddings.dtype
    alphas_t = torch.tensor(alphas, device=device, dtype=dtype)
    weights_t = torch.tensor(weights, device=device, dtype=dtype)

    # Batch the interpolation steps: run ONE forward+backward per chunk of
    # ``_IG_BATCH`` steps instead of one per step (≈ n_steps× fewer passes —
    # the dominant cost of IG). Chunking keeps peak memory at ~_IG_BATCH× a
    # single forward regardless of n_steps. This is numerically identical to
    # the sequential estimator Σ_i w_i · grad_i: each batch row's score
    # depends only on its own interpolated input, so the gradient of
    # ``Σ_i w_i · score_i`` w.r.t. the batch is ``[w_i · grad_i]`` and summing
    # over the batch dim reproduces the accumulation exactly.
    with Progress(console=console, transient=True) as progress:
        task = progress.add_task(
            f"Integrated gradients ({quadrature}, n={n_steps})",
            total=n_steps,
        )
        for start in range(0, n_steps, _IG_BATCH):
            a_chunk = alphas_t[start : start + _IG_BATCH]
            w_chunk = weights_t[start : start + _IG_BATCH]
            nb = int(a_chunk.shape[0])

            # (nb, seq, d): baseline_embeddings / delta are (1, seq, d) and broadcast.
            interpolated = (
                baseline_embeddings + a_chunk.view(nb, 1, 1) * delta
            ).requires_grad_(True)

            def _patched_forward_ig(*args: Any, _interp: torch.Tensor = interpolated, **kwargs: Any) -> torch.Tensor:
                return _interp

            # Expand the batch-1 model inputs to nb so attention masks /
            # position ids line up with the nb interpolated rows.
            batched_input = {
                k: (v.expand(nb, *v.shape[1:])
                    if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == 1
                    else v)
                for k, v in model_input.items()
            }

            embed_layer.forward = _patched_forward_ig  # type: ignore[assignment]
            try:
                logits = model._forward_with_grad(batched_input).float()
                if logits.dim() == 3:
                    scores = logits[:, -1, target]
                else:
                    scores = logits[:, target]

                total = (w_chunk * scores).sum()
                (grad,) = torch.autograd.grad(total, interpolated)
                accumulated_grads += grad.detach().sum(dim=0, keepdim=True)
            finally:
                embed_layer.forward = original_forward  # type: ignore[assignment]

            progress.advance(task, nb)

    # IG = delta * Σ w_i * grad_i (the weights already encode the 1/n
    # factor for midpoint and the proper trapezoidal/GL weights).
    ig = delta * accumulated_grads
    token_scores_signed = ig[0].sum(dim=-1)
    token_scores_norm = ig[0].norm(dim=-1)
    sum_attributions = float(token_scores_signed.sum().item())
    return ig, token_scores_norm, token_scores_signed, sum_attributions


def _run_ig_with_optional_bump(
    *,
    model: Model,
    model_input: dict[str, torch.Tensor],
    embed_layer: torch.nn.Module,
    original_forward: Any,
    baseline_embeddings: torch.Tensor,
    delta: torch.Tensor,
    target: int,
    n_steps: int,
    quadrature: str,
    output_gap: float,
    auto_bump: bool,
    max_n_steps: int,
) -> tuple[torch.Tensor, list[float], torch.Tensor, float, int, bool]:
    """Run IG once; if completeness fails and ``auto_bump`` is on, retry
    with double n_steps (capped). Returns the final pass's data plus
    bookkeeping fields the diagnostics block reports.
    """
    ig, token_scores_norm, token_scores_signed, sum_attr = _ig_one_pass(
        model=model, model_input=model_input,
        embed_layer=embed_layer, original_forward=original_forward,
        baseline_embeddings=baseline_embeddings, delta=delta,
        target=target, n_steps=n_steps, quadrature=quadrature,
    )
    completeness_error = abs(sum_attr - output_gap)
    completeness_tolerance = 0.1 * max(abs(output_gap), 1e-6)

    auto_bump_attempted = False
    used_n_steps = n_steps
    if (
        auto_bump
        and completeness_error > completeness_tolerance
        and n_steps < max_n_steps
    ):
        bumped = min(n_steps * 2, max_n_steps)
        auto_bump_attempted = True
        used_n_steps = bumped
        ig, token_scores_norm, token_scores_signed, sum_attr = _ig_one_pass(
            model=model, model_input=model_input,
            embed_layer=embed_layer, original_forward=original_forward,
            baseline_embeddings=baseline_embeddings, delta=delta,
            target=target, n_steps=bumped, quadrature=quadrature,
        )
        completeness_error = abs(sum_attr - output_gap)

    return (
        ig,
        token_scores_norm.tolist(),
        token_scores_signed,
        completeness_error,
        used_n_steps,
        auto_bump_attempted,
    )


def _resolve_baseline(
    baseline: str | torch.Tensor,
    embed_layer: torch.nn.Module,
    tokenizer: Any | None,
    base_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, str, int | None]:
    """Resolve the IG baseline embedding (F-011).

    Returns ``(baseline_embeddings, label, token_id_if_applicable)``.
    """
    if isinstance(baseline, torch.Tensor):
        if baseline.shape != base_embeddings.shape:
            raise ValueError(
                f"baseline tensor shape {tuple(baseline.shape)} does not match "
                f"base embeddings shape {tuple(base_embeddings.shape)}."
            )
        return baseline.to(base_embeddings.device, dtype=base_embeddings.dtype), "tensor", None

    if baseline == "zero":
        return torch.zeros_like(base_embeddings), "zero", None

    if baseline == "mean":
        # Mean of the embedding matrix — in-distribution but content-free.
        if hasattr(embed_layer, "weight"):
            mean_emb = embed_layer.weight.detach().mean(dim=0, keepdim=True)
            tile = mean_emb.expand_as(base_embeddings[0]).unsqueeze(0)
            return tile.to(base_embeddings.device, dtype=base_embeddings.dtype), "mean", None
        return torch.zeros_like(base_embeddings), "zero", None  # fallback

    if baseline == "pad":
        # PAD-token embedding: in-distribution and well-defined for HF tokenisers.
        pad_id: int | None = None
        if tokenizer is not None:
            pad_id = getattr(tokenizer, "pad_token_id", None)
            if pad_id is None:
                pad_id = getattr(tokenizer, "eos_token_id", None)
        if pad_id is None or not hasattr(embed_layer, "weight"):
            # Fall back to mean if no PAD token is defined.
            return _resolve_baseline("mean", embed_layer, tokenizer, base_embeddings)
        pad_emb = embed_layer.weight.detach()[pad_id]
        tile = pad_emb.unsqueeze(0).expand_as(base_embeddings[0]).unsqueeze(0)
        return tile.to(base_embeddings.device, dtype=base_embeddings.dtype), "pad_token", int(pad_id)

    raise ValueError(
        f"Unknown baseline {baseline!r}. Use 'pad', 'zero', 'mean', or pass a torch.Tensor."
    )


def _forward_score(
    model: Model,
    model_input: dict[str, torch.Tensor],
    embed_layer: torch.nn.Module,
    original_forward: Any,
    embeddings: torch.Tensor,
    target: int,
) -> float:
    """Run the model with *embeddings* swapped in for the embedding layer.

    Used by IG completeness check to evaluate ``f(x)`` and ``f(baseline)``
    along the same path the IG integration uses.
    """

    def _patched(*args: Any, _e: torch.Tensor = embeddings, **kwargs: Any) -> torch.Tensor:
        return _e

    embed_layer.forward = _patched  # type: ignore[assignment]
    try:
        with torch.no_grad():
            logits = model._forward(model_input).float()
        if logits.dim() == 3:
            return float(logits[0, -1, target].item())
        return float(logits[0, target].item())
    finally:
        embed_layer.forward = original_forward  # type: ignore[assignment]


def _attribute_image(model: Model, image_path: str, *, target: int | None, save: str | None = None) -> dict[str, Any]:
    from interpkit.core.inputs import _load_image
    from interpkit.core.render import render_attribution_heatmap

    processed = _load_image(
        image_path,
        image_processor=model._image_processor,
        device=model._device,
    )

    if isinstance(processed, dict):
        pixel_key = "pixel_values" if "pixel_values" in processed else list(processed.keys())[0]
        pixel_values = processed[pixel_key].requires_grad_(True)
        model_input: dict[str, torch.Tensor] | torch.Tensor = {**processed, pixel_key: pixel_values}
    else:
        pixel_values = processed.requires_grad_(True)
        model_input = pixel_values

    logits = model._forward_with_grad(model_input)

    if logits.dim() > 1:
        logits_flat = logits[0]
    else:
        logits_flat = logits

    if target is None:
        target = int(logits_flat.argmax().item())

    score = logits_flat[target]
    score.backward()

    if pixel_values.grad is None:
        raise RuntimeError("Gradient computation failed — no gradients on pixel values.")

    grad = pixel_values.grad[0].detach()

    if save is not None:
        render_attribution_heatmap(grad, output_path=save)

    # A3: image attribution is gradient saliency — a ranking, never additive
    # contribution magnitudes. Carry the same interpretation contract as text.
    return {"grad": grad, "target": target, "interpretation": "ranking_only"}


def _attribute_tensor(model: Model, tensor_input: Any, *, target: int | None) -> dict[str, Any]:
    from interpkit.core.render import render_attribution_tokens

    inp = model._prepare(tensor_input)

    if isinstance(inp, dict):
        for k, v in inp.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                inp[k] = v.requires_grad_(True)
                grad_tensor = v
                break
        else:
            raise ValueError("No floating-point tensor found in input dict.")
    else:
        inp = inp.requires_grad_(True)
        grad_tensor = inp

    logits = model._forward_with_grad(inp)

    if logits.dim() == 3:
        logits_last = logits[0, -1, :]
    elif logits.dim() == 2:
        logits_last = logits[0]
    else:
        logits_last = logits.view(-1)

    if target is None:
        target = int(logits_last.argmax().item())

    score = logits_last[target]
    score.backward()

    if grad_tensor.grad is None:
        raise RuntimeError("Gradient computation failed.")

    grad = grad_tensor.grad.detach().float()
    if grad.dim() > 1:
        feature_scores = grad.view(grad.shape[0], -1).norm(dim=0).tolist()
    else:
        feature_scores = grad.abs().tolist()

    labels = [f"feat_{i}" for i in range(len(feature_scores))]
    render_attribution_tokens(labels, feature_scores)

    # A3: tensor/feature attribution is gradient saliency — a ranking.
    return {
        "labels": labels, "scores": feature_scores, "target": target,
        "interpretation": "ranking_only",
    }


def _find_embedding(model: torch.nn.Module) -> torch.nn.Module | None:
    """Find the token embedding layer.

    Prefers an embedding whose name contains "token" or "wte".  Falls back
    to the largest embedding (by num_embeddings), which is almost always the
    token embedding rather than a position embedding.
    """
    # Prefer explicitly named token/word embeddings, excluding token_type
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Embedding):
            lower = name.lower()
            if "token_type" in lower or "segment" in lower:
                continue
            if "word" in lower or "token" in lower or "wte" in lower:
                return mod

    # Fall back to the largest embedding (token > position in practice)
    best: torch.nn.Module | None = None
    best_size = 0
    for _name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Embedding) and mod.num_embeddings > best_size:
            best = mod
            best_size = mod.num_embeddings

    return best


def _is_image_path(s: str) -> bool:
    """Deprecated — use ``interpkit.core.inputs._looks_like_image_path``."""
    from interpkit.core.inputs import _looks_like_image_path

    return _looks_like_image_path(s)
