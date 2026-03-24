"""attribute — gradient saliency over input tokens or pixels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from rich.console import Console
from rich.progress import Progress

if TYPE_CHECKING:
    from interpkit.core.model import Model

console = Console()


def run_attribute(
    model: "Model",
    input_data: Any,
    *,
    target: int | None = None,
    method: str = "integrated_gradients",
    n_steps: int = 50,
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
        Interpolation steps for integrated gradients (default 50).

    For text inputs: returns ``{"tokens", "scores", "target"}`` with per-token importance.
    For image inputs: returns ``{"grad", "target"}`` with the pixel-gradient tensor.
    For tensor inputs: returns ``{"labels", "scores", "target"}``.
    """
    from interpkit.core.inputs import _looks_like_image_path

    is_text = isinstance(input_data, str) and not _looks_like_image_path(input_data)
    is_image = isinstance(input_data, str) and _looks_like_image_path(input_data)

    if is_text:
        return _attribute_text(model, input_data, target=target, method=method, n_steps=n_steps, save=save, html=html)
    elif is_image:
        return _attribute_image(model, input_data, target=target, save=save)
    else:
        return _attribute_tensor(model, input_data, target=target)


def _attribute_text(
    model: "Model",
    text: str,
    *,
    target: int | None,
    method: str = "integrated_gradients",
    n_steps: int = 50,
    save: str | None = None,
    html: str | None = None,
) -> dict[str, Any]:
    from interpkit.core.render import render_attribution_tokens

    if model._tokenizer is None:
        raise ValueError("No tokenizer available for text attribution.")

    encoded = model._tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model._device)

    config = getattr(model._model, "config", None)
    if getattr(config, "is_encoder_decoder", False) and "decoder_input_ids" not in encoded:
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
                target = logits_clean[0, -1, :].argmax().item()
            else:
                target = logits_clean[0].argmax().item()

    model_input = {k: v.to(model._device) for k, v in encoded.items()}

    if method == "integrated_gradients":
        # Sundararajan et al. 2017: integrate gradients along path from
        # zero baseline to actual embeddings
        accumulated_grads = torch.zeros_like(base_embeddings)
        with Progress(console=console, transient=True) as progress:
            task = progress.add_task("Integrated gradients", total=n_steps)
            for step in range(n_steps):
                alpha = (step + 0.5) / n_steps
                interpolated = (alpha * base_embeddings).requires_grad_(True)

                def _patched_forward_ig(*args: Any, **kwargs: Any) -> torch.Tensor:
                    return interpolated

                embed_layer.forward = _patched_forward_ig  # type: ignore[assignment]
                try:
                    logits = model._forward_with_grad(model_input).float()
                    if logits.dim() == 3:
                        score = logits[0, -1, target]
                    else:
                        score = logits[0, target]

                    (grad,) = torch.autograd.grad(score, interpolated)
                    accumulated_grads += grad.detach()
                finally:
                    embed_layer.forward = original_forward  # type: ignore[assignment]

                progress.advance(task)

        ig = (base_embeddings / n_steps) * accumulated_grads
        token_scores = ig[0].norm(dim=-1).tolist()

    elif method == "gradient_x_input":
        embeddings = base_embeddings.requires_grad_(True)

        def _patched_forward_gxi(*args: Any, **kwargs: Any) -> torch.Tensor:
            return embeddings

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

        def _patched_forward_grad(*args: Any, **kwargs: Any) -> torch.Tensor:
            return embeddings

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

    return {"tokens": tokens, "scores": token_scores, "target": target, "method": method}


def _attribute_image(model: "Model", image_path: str, *, target: int | None, save: str | None = None) -> dict[str, Any]:
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
        model_input = {**processed, pixel_key: pixel_values}
    else:
        pixel_values = processed.requires_grad_(True)
        model_input = pixel_values

    logits = model._forward_with_grad(model_input)

    if logits.dim() > 1:
        logits_flat = logits[0]
    else:
        logits_flat = logits

    if target is None:
        target = logits_flat.argmax().item()

    score = logits_flat[target]
    score.backward()

    if pixel_values.grad is None:
        raise RuntimeError("Gradient computation failed — no gradients on pixel values.")

    grad = pixel_values.grad[0].detach()

    if save is not None:
        render_attribution_heatmap(grad, output_path=save)

    return {"grad": grad, "target": target}


def _attribute_tensor(model: "Model", tensor_input: Any, *, target: int | None) -> dict[str, Any]:
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
        target = logits_last.argmax().item()

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

    return {"labels": labels, "scores": feature_scores, "target": target}


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
