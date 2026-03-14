"""attribute — gradient saliency over input tokens or pixels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from interpkit.core.model import Model


def run_attribute(
    model: "Model",
    input_data: Any,
    *,
    target: int | None = None,
    save: str | None = None,
    html: str | None = None,
) -> None:
    """Compute gradient-based saliency and render results.

    For text inputs: shows coloured tokens by importance.
    For image/tensor inputs: saves a heatmap to disk.
    """
    is_text = isinstance(input_data, str) and not _is_image_path(input_data)
    is_image = isinstance(input_data, str) and _is_image_path(input_data)

    if is_text:
        _attribute_text(model, input_data, target=target, save=save, html=html)
    elif is_image:
        _attribute_image(model, input_data, target=target, save=save)
    else:
        _attribute_tensor(model, input_data, target=target)


def _attribute_text(model: "Model", text: str, *, target: int | None, save: str | None = None, html: str | None = None) -> None:
    from interpkit.core.render import render_attribution_tokens

    if model._tokenizer is None:
        raise ValueError("No tokenizer available for text attribution.")

    encoded = model._tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model._device)

    # Get embedding layer
    embed_layer = _find_embedding(model._model)
    if embed_layer is None:
        raise RuntimeError("Could not find embedding layer for gradient attribution.")

    embeddings = embed_layer(input_ids)
    embeddings = embeddings.detach().requires_grad_(True)

    # Replace embedding output with our gradient-tracked version
    original_forward = embed_layer.forward

    def _patched_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
        return embeddings

    embed_layer.forward = _patched_forward  # type: ignore[assignment]

    try:
        model_kwargs = {k: v.to(model._device) for k, v in encoded.items() if k != "input_ids"}
        out = model._model(input_ids, **model_kwargs)

        logits = out.logits if hasattr(out, "logits") else (out[0] if isinstance(out, (tuple, list)) else out)

        # Pick target: last-position argmax if not specified
        if logits.dim() == 3:
            logits_last = logits[0, -1, :]
        else:
            logits_last = logits[0]

        if target is None:
            target = logits_last.argmax().item()

        score = logits_last[target]
        score.backward()
    finally:
        embed_layer.forward = original_forward  # type: ignore[assignment]

    if embeddings.grad is None:
        raise RuntimeError("Gradient computation failed — no gradients on embeddings.")

    # Per-token importance: L2 norm of gradient over the embedding dimension
    token_grads = embeddings.grad[0]  # (seq_len, hidden)
    token_scores = token_grads.norm(dim=-1).tolist()  # (seq_len,)

    tokens = model._tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    render_attribution_tokens(tokens, token_scores)

    if save is not None:
        from interpkit.core.plot import plot_attribution

        plot_attribution(tokens, token_scores, save_path=save)

    if html is not None:
        from interpkit.core.html import html_attribution as gen_html_attribution
        from interpkit.core.html import save_html

        save_html(gen_html_attribution(tokens, token_scores), html)


def _attribute_image(model: "Model", image_path: str, *, target: int | None, save: str | None = None) -> None:
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
        out = model._model(**model_input)
    else:
        pixel_values = processed.requires_grad_(True)
        out = model._model(pixel_values)

    logits = out.logits if hasattr(out, "logits") else (out[0] if isinstance(out, (tuple, list)) else out)

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

    out_path = save or "attribution_heatmap.png"
    render_attribution_heatmap(pixel_values.grad[0], output_path=out_path)


def _attribute_tensor(model: "Model", tensor_input: Any, *, target: int | None) -> None:
    from interpkit.core.render import render_attribution_tokens

    inp = model._prepare(tensor_input)

    if isinstance(inp, dict):
        # Pick the first tensor-valued entry
        for k, v in inp.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                inp[k] = v.requires_grad_(True)
                grad_tensor = v
                break
        else:
            raise ValueError("No floating-point tensor found in input dict.")
        out = model._model(**inp)
    else:
        inp = inp.requires_grad_(True)
        grad_tensor = inp
        out = model._model(inp)

    logits = out.logits if hasattr(out, "logits") else (out[0] if isinstance(out, (tuple, list)) else out)

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

    # Flatten to feature importance
    grad = grad_tensor.grad.detach().float()
    if grad.dim() > 1:
        feature_scores = grad.view(grad.shape[0], -1).norm(dim=0).tolist()
    else:
        feature_scores = grad.abs().tolist()

    labels = [f"feat_{i}" for i in range(len(feature_scores))]
    render_attribution_tokens(labels, feature_scores)


def _find_embedding(model: torch.nn.Module) -> torch.nn.Module | None:
    """Find the token embedding layer.

    Prefers an embedding whose name contains "token" or "wte".  Falls back
    to the largest embedding (by num_embeddings), which is almost always the
    token embedding rather than a position embedding.
    """
    # Prefer explicitly named token embeddings
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Embedding):
            if "token" in name.lower() or "wte" in name.lower():
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
    import os
    return os.path.splitext(s)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
