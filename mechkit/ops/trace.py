"""trace — two-phase causal tracing across all modules, ranked by causal effect."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import torch
from rich.console import Console
from rich.progress import Progress

from mechkit.ops.patch import _compute_effect, _get_module

if TYPE_CHECKING:
    from mechkit.core.model import Model

console = Console()


def run_trace(
    model: "Model",
    clean: Any,
    corrupted: Any,
    *,
    top_k: int | None = 20,
    save: str | None = None,
    html: str | None = None,
) -> list[dict[str, Any]]:
    """Two-phase causal tracing.

    Phase 1 (fast proxy): run clean and corrupted forward passes, capture activation
    norms at every module, rank by norm delta.

    Phase 2 (expensive): for the top-K modules by proxy score, run full
    patch-and-measure to get true causal effect.
    """
    from mechkit.core.render import render_trace

    clean_input, corrupted_input = model._prepare_pair(clean, corrupted)

    # Filter to leaf modules with parameters (skip containers)
    candidates = [
        m for m in model.arch_info.modules
        if m.param_count > 0
    ]

    total_modules = len(candidates)

    if top_k == 0:
        top_k = None

    # ----------------------------------------------------------------
    # Phase 1: fast proxy — activation norm delta
    # ----------------------------------------------------------------
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

    # Clean pass
    hooks = []
    for m in candidates:
        mod = _get_module(model._model, m.name)
        hooks.append(mod.register_forward_hook(_make_norm_hook(clean_norms, m.name)))
    with torch.no_grad():
        clean_logits = model._forward(clean_input)
    for h in hooks:
        h.remove()

    # Corrupted pass
    hooks = []
    for m in candidates:
        mod = _get_module(model._model, m.name)
        hooks.append(mod.register_forward_hook(_make_norm_hook(corrupted_norms, m.name)))
    with torch.no_grad():
        corrupted_logits = model._forward(corrupted_input)
    for h in hooks:
        h.remove()

    # Rank by proxy: absolute norm difference
    proxy_scores: list[tuple[str, float]] = []
    for m in candidates:
        cn = clean_norms.get(m.name, 0.0)
        crn = corrupted_norms.get(m.name, 0.0)
        proxy_scores.append((m.name, abs(cn - crn)))

    proxy_scores.sort(key=lambda x: x[1], reverse=True)

    # Select top-K for expensive phase
    if top_k is not None and top_k < total_modules:
        selected_names = {name for name, _ in proxy_scores[:top_k]}
        console.print(
            f"\n  Scanning top {top_k} of {total_modules} modules by proxy score."
        )
    else:
        selected_names = {name for name, _ in proxy_scores}

    # ----------------------------------------------------------------
    # Phase 2: full causal patching on selected modules
    # ----------------------------------------------------------------

    # Cache all clean activations in one pass
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
    with torch.no_grad():
        clean_logits = model._forward(clean_input)
    for h in hooks:
        h.remove()

    # Patch each selected module one at a time
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
            with torch.no_grad():
                patched_logits = model._forward(corrupted_input)
            handle.remove()

            effect = _compute_effect(clean_logits, corrupted_logits, patched_logits)
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
        from mechkit.core.plot import plot_trace

        plot_trace(results, model_name=model_name, save_path=save)

    if html is not None:
        from mechkit.core.html import html_trace as gen_html_trace
        from mechkit.core.html import save_html

        save_html(gen_html_trace(results), html)

    return results
