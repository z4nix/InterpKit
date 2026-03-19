"""trace — causal tracing across modules, ranked by causal effect."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from rich.console import Console
from rich.progress import Progress

from interpkit.ops.patch import _compute_effect, _get_module

if TYPE_CHECKING:
    from interpkit.core.model import Model

console = Console()


def run_trace(
    model: "Model",
    clean: Any,
    corrupted: Any,
    *,
    top_k: int | None = 20,
    mode: str = "module",
    metric: str = "logit_diff",
    save: str | None = None,
    html: str | None = None,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Causal tracing.

    Parameters
    ----------
    mode:
        ``"module"`` (default) — two-phase module-level tracing (fast proxy
        then full patch).  Returns a sorted list of dicts.

        ``"position"`` — Meng et al. style position-aware tracing.  For
        each (layer, token-position) pair, restores that single position's
        clean activation during the corrupted run and measures recovery.
        Returns a dict with ``"effects"`` (tensor of shape
        ``num_layers x seq_len``), ``"layer_names"``, ``"tokens"``.
    metric:
        Effect metric passed to ``_compute_effect``.  One of
        ``"logit_diff"`` (default), ``"kl_div"``, ``"target_prob"``,
        ``"l2_prob"``.
    """
    if mode == "position":
        return _run_position_trace(
            model, clean, corrupted, metric=metric, save=save, html=html,
        )
    return _run_module_trace(
        model, clean, corrupted, top_k=top_k, metric=metric, save=save, html=html,
    )


# ------------------------------------------------------------------
# Module-level tracing (original implementation)
# ------------------------------------------------------------------

def _run_module_trace(
    model: "Model",
    clean: Any,
    corrupted: Any,
    *,
    top_k: int | None = 20,
    metric: str = "logit_diff",
    save: str | None = None,
    html: str | None = None,
) -> list[dict[str, Any]]:
    """Two-phase causal tracing at the module level."""
    from interpkit.core.render import render_trace

    clean_input, corrupted_input = model._prepare_pair(clean, corrupted)

    candidates = [
        m for m in model.arch_info.modules
        if m.param_count > 0
    ]

    total_modules = len(candidates)

    if top_k == 0:
        top_k = None

    # Phase 1: fast proxy — activation norm delta
    clean_norms: dict[str, float] = {}
    corrupted_norms: dict[str, float] = {}

    def _make_norm_hook(store: dict[str, float], name: str):
        def hook_fn(_mod: torch.nn.Module, _inp: Any, output: Any) -> None:
            t = output if isinstance(output, torch.Tensor) else (
                output[0] if isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor) else None
            )
            if t is not None:
                store[name] = t.detach().float().norm().item()
        return hook_fn

    hooks = []
    for m in candidates:
        mod = _get_module(model._model, m.name)
        hooks.append(mod.register_forward_hook(_make_norm_hook(clean_norms, m.name)))
    clean_logits = model._forward(clean_input)
    for h in hooks:
        h.remove()

    hooks = []
    for m in candidates:
        mod = _get_module(model._model, m.name)
        hooks.append(mod.register_forward_hook(_make_norm_hook(corrupted_norms, m.name)))
    corrupted_logits = model._forward(corrupted_input)
    for h in hooks:
        h.remove()

    proxy_scores: list[tuple[str, float]] = []
    for m in candidates:
        cn = clean_norms.get(m.name, 0.0)
        crn = corrupted_norms.get(m.name, 0.0)
        proxy_scores.append((m.name, abs(cn - crn)))

    proxy_scores.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None and top_k < total_modules:
        selected_names = {name for name, _ in proxy_scores[:top_k]}
        console.print(
            f"\n  Scanning top {top_k} of {total_modules} modules by proxy score."
        )
    else:
        selected_names = {name for name, _ in proxy_scores}

    # Phase 2: full causal patching on selected modules
    clean_cache: dict[str, torch.Tensor] = {}

    def _make_cache_hook(name: str):
        def hook_fn(_mod: torch.nn.Module, _inp: Any, output: Any) -> None:
            t = output if isinstance(output, torch.Tensor) else (
                output[0] if isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor) else None
            )
            if t is not None:
                clean_cache[name] = t.detach().clone()
        return hook_fn

    hooks = []
    for m in candidates:
        if m.name in selected_names:
            mod = _get_module(model._model, m.name)
            hooks.append(mod.register_forward_hook(_make_cache_hook(m.name)))
    clean_logits = model._forward(clean_input)
    for h in hooks:
        h.remove()

    results: list[dict[str, Any]] = []
    module_role_map = {m.name: m.role for m in candidates}

    with Progress(console=console, transient=True) as progress:
        task = progress.add_task("Causal tracing", total=len(selected_names))
        for name in selected_names:
            if name not in clean_cache:
                progress.advance(task)
                continue

            target_mod = _get_module(model._model, name)

            def _make_patch_hook(cached: torch.Tensor):
                def hook_fn(_mod: torch.nn.Module, _inp: Any, output: Any) -> Any:
                    if isinstance(output, torch.Tensor):
                        return cached
                    elif isinstance(output, (tuple, list)):
                        return (cached,) + tuple(output[1:])
                    return output
                return hook_fn

            handle = target_mod.register_forward_hook(_make_patch_hook(clean_cache[name]))
            patched_logits = model._forward(corrupted_input)
            handle.remove()

            effect = _compute_effect(clean_logits, corrupted_logits, patched_logits, metric=metric)
            results.append({
                "module": name,
                "role": module_role_map.get(name),
                "effect": effect,
            })
            progress.advance(task)

    results.sort(key=lambda x: x["effect"], reverse=True)

    model_name = model.arch_info.arch_family or "model"
    render_trace(results, model_name, total_modules, top_k)

    if save is not None:
        from interpkit.core.plot import plot_trace

        plot_trace(results, model_name=model_name, save_path=save)

    if html is not None:
        from interpkit.core.html import html_trace as gen_html_trace
        from interpkit.core.html import save_html

        save_html(gen_html_trace(results), html)

    return results


# ------------------------------------------------------------------
# Position-aware tracing (Meng et al.)
# ------------------------------------------------------------------

def _run_position_trace(
    model: "Model",
    clean: Any,
    corrupted: Any,
    *,
    metric: str = "logit_diff",
    save: str | None = None,
    html: str | None = None,
) -> dict[str, Any]:
    """Position-aware causal tracing (Meng et al. 2022).

    For each (layer, position) pair, runs the corrupted input but restores
    the clean hidden state at that specific position in that specific layer.
    Measures how much the output probability of the correct token recovers.

    Returns a dict with:
        ``effects``: tensor (num_layers, seq_len)
        ``layer_names``: list[str]
        ``tokens``: list[str] | None
    """
    from interpkit.core.render import render_position_trace

    arch = model.arch_info
    if not arch.layer_names:
        raise ValueError("Position-aware tracing requires detected layer structure.")

    clean_input, corrupted_input = model._prepare_pair(clean, corrupted)

    layer_names = arch.layer_names

    # Recover input tokens for labels
    tokens: list[str] | None = None
    if isinstance(clean, str) and model._tokenizer is not None:
        enc = model._tokenizer(clean, return_tensors="pt")
        tokens = model._tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())

    # Cache clean activations at every layer
    clean_cache: dict[str, torch.Tensor] = {}

    def _make_cache_hook(name: str):
        def hook_fn(_mod: torch.nn.Module, _inp: Any, output: Any) -> None:
            t = output if isinstance(output, torch.Tensor) else (
                output[0] if isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor) else None
            )
            if t is not None:
                clean_cache[name] = t.detach().clone()
        return hook_fn

    hooks = []
    for ln in layer_names:
        try:
            mod = _get_module(model._model, ln)
            hooks.append(mod.register_forward_hook(_make_cache_hook(ln)))
        except AttributeError:
            continue
    clean_logits = model._forward(clean_input)
    for h in hooks:
        h.remove()

    corrupted_logits = model._forward(corrupted_input)

    # Determine seq_len from cached activations
    sample_cached = next(iter(clean_cache.values()))
    if sample_cached.dim() == 3:
        seq_len = sample_cached.shape[1]
    elif sample_cached.dim() == 2:
        seq_len = sample_cached.shape[0]
    else:
        seq_len = 1

    num_layers = len(layer_names)
    effects = torch.zeros(num_layers, seq_len)

    total_iters = num_layers * seq_len
    with Progress(console=console, transient=True) as progress:
        task = progress.add_task("Position tracing", total=total_iters)
        for li, ln in enumerate(layer_names):
            if ln not in clean_cache:
                progress.advance(task, advance=seq_len)
                continue

            clean_act = clean_cache[ln]
            target_mod = _get_module(model._model, ln)

            for pos in range(seq_len):
                def _make_pos_patch_hook(cached: torch.Tensor, p: int):
                    def hook_fn(_mod: torch.nn.Module, _inp: Any, output: Any) -> Any:
                        t = output if isinstance(output, torch.Tensor) else (
                            output[0] if isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor) else None
                        )
                        if t is None:
                            return output

                        patched = t.clone()
                        if patched.dim() == 3:
                            patched[:, p, :] = cached[:, p, :]
                        elif patched.dim() == 2:
                            patched[p, :] = cached[p, :]
                        else:
                            return output

                        if isinstance(output, torch.Tensor):
                            return patched
                        return (patched,) + tuple(output[1:])
                    return hook_fn

                handle = target_mod.register_forward_hook(
                    _make_pos_patch_hook(clean_act, pos)
                )
                patched_logits = model._forward(corrupted_input)
                handle.remove()

                effects[li, pos] = _compute_effect(
                    clean_logits, corrupted_logits, patched_logits, metric=metric
                )
                progress.advance(task)

    result = {
        "effects": effects,
        "layer_names": layer_names,
        "tokens": tokens,
    }

    render_position_trace(result)

    if save is not None:
        from interpkit.core.plot import plot_position_trace

        plot_position_trace(result, save_path=save)

    if html is not None:
        from interpkit.core.html import html_position_trace as gen_html_pt, save_html

        save_html(gen_html_pt(result), html)

    return result
