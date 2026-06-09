"""trace — causal tracing across modules, ranked by causal effect (F-015).

Pre-1.0 ``trace(top_k=K)`` returned the top-K modules by an
activation-norm proxy, then ran full patch only on those K. The audit
showed this routinely missed important modules: e.g. ``transformer.wte``
ranked low by activation-norm delta but tied for top-1 by true causal
effect on a France→Italy patch (F-015). The "top-K" was silently wrong.

The 1.0 fix is a **three-tier dispatcher**:

- **Tier A (exhaustive)** — full patch on every candidate, return true
  top-K. Used by default for small models (≤ ``exhaustive_threshold``).
- **Tier B (approximate)** — Attribution Patching (see :mod:`_atp`)
  shortlists candidates, then full patch on the union of (top
  ``top_k_search`` AtP-ranked) and (always-pinned modules: embed /
  unembed / final_norm / pos_embed). Re-rank by true effect.
- **Tier C (auto)** — pick A or B based on candidate count vs threshold.

Per-candidate provenance is reported (effect, measurement_method,
atp_score, atp_rank). A meta block at the top level reports the algorithm
chosen, total candidates measured, etc.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import torch
from rich.console import Console
from rich.progress import Progress

from interpkit.core.enums import (
    VALID_METRICS,
    VALID_TRACE_METHODS,
    VALID_TRACE_MODES,
    _validate_enum,
)
from interpkit.core.paths import validate_module_path
from interpkit.ops.patch import _compute_effect_value as _compute_effect
from interpkit.ops.patch import _get_module

if TYPE_CHECKING:
    from interpkit.core.model import Model

console = Console()


def run_trace(
    model: Model,
    clean: Any,
    corrupted: Any,
    *,
    top_k: int | None = 10,
    mode: str = "module",
    method: str = "auto",
    metric: str = "logit_diff",
    exhaustive_threshold: int = 500,
    top_k_search: int | None = None,
    pin_modules: list[str] | None = None,
    save: str | None = None,
    html: str | None = None,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Causal tracing with three-tier dispatcher (F-015).

    Parameters
    ----------
    top_k:
        Number of top modules to return after measurement. ``None`` or 0
        returns all candidates measured.
    mode:
        ``"module"`` (default) — module-level tracing.
        ``"position"`` — Meng et al. style position-aware tracing
        (currently always exhaustive; AtP for position mode is a future
        iteration).
    method:
        Tier dispatch:

        - ``"auto"`` (default): exhaustive when ``total_candidates ≤
          exhaustive_threshold``, else approximate.
        - ``"exhaustive"`` / ``"exhaustive_forced"``: full patch on every
          candidate. Guaranteed-correct ranking.
        - ``"approximate"``: Attribution Patching shortlist + full-patch
          confirmation on the shortlist. Faster on large models; ranking
          quality depends on how well AtP approximates the true effect.
    exhaustive_threshold:
        Threshold for the ``"auto"`` dispatch (default 500). Models with
        fewer modules than this run exhaustive; larger models go
        approximate.
    top_k_search:
        Number of AtP-shortlisted candidates to confirm with full patching
        (approximate mode only). Defaults to ``4 * top_k``.
    pin_modules:
        Modules to always include in the full-patch confirmation regardless
        of their AtP score (approximate mode only). Defaults to embed /
        unembed / final_norm / pos_embed when present.
    metric:
        Effect metric. One of ``"logit_diff"`` (default), ``"kl_div"``,
        ``"target_prob"``, ``"target_prob_effect"``, ``"l2_prob"``.

    Returns
    -------
    list[dict] | dict
        For module mode: ``{"results": list[dict], "meta": dict}`` with
        per-module provenance (``effect``, ``measurement_method``,
        ``atp_score``, ``atp_rank``) and a meta block describing the
        algorithm. The dict-with-meta shape replaces the pre-1.0 raw
        list. (Position mode still returns its existing dict shape.)

    Raises
    ------
    ValueError
        If *mode*, *method*, or *metric* is unknown (F-018 / F-019).
    """
    _validate_enum(mode, VALID_TRACE_MODES, "mode")
    _validate_enum(method, VALID_TRACE_METHODS, "method")
    _validate_enum(metric, VALID_METRICS, "metric")

    # N-004: gate DeBERTa-v3 — trace runs forward hooks on every module
    # which fires the broken relative-position-bias broadcast path.
    from interpkit.core.support_matrix import check_op_supported
    check_op_supported("trace", model.arch_info)

    # F-022: validate user-supplied pin_modules up-front. Auto-discovered
    # candidates are intentionally tolerated by the catch-all in
    # `_run_module_trace` (some named_modules entries are uninstantiable
    # leaf nodes), but anything the user passes in must be a real path.
    if pin_modules is not None:
        for path in pin_modules:
            validate_module_path(path, model.arch_info)

    if mode == "position":
        return _run_position_trace(
            model, clean, corrupted, metric=metric, save=save, html=html,
        )
    return _run_module_trace(
        model, clean, corrupted,
        top_k=top_k, method=method, metric=metric,
        exhaustive_threshold=exhaustive_threshold,
        top_k_search=top_k_search, pin_modules=pin_modules,
        save=save, html=html,
    )


# ------------------------------------------------------------------
# Module-level tracing (original implementation)
# ------------------------------------------------------------------

def _run_module_trace(
    model: Model,
    clean: Any,
    corrupted: Any,
    *,
    top_k: int | None = 10,
    method: str = "auto",
    metric: str = "logit_diff",
    exhaustive_threshold: int = 500,
    top_k_search: int | None = None,
    pin_modules: list[str] | None = None,
    save: str | None = None,
    html: str | None = None,
) -> dict[str, Any]:
    """Module-level causal tracing with three-tier dispatcher (F-015).

    See :func:`run_trace` for parameter documentation.
    """
    from interpkit.core.render import render_trace

    arch = model.arch_info
    clean_input, corrupted_input = model._prepare_pair(clean, corrupted)

    candidates = [m for m in arch.modules if m.param_count > 0]
    total_modules = len(candidates)
    candidate_names = [m.name for m in candidates]
    module_role_map = {m.name: m.role for m in candidates}

    if top_k == 0 or top_k is None:
        top_k = total_modules

    # Resolve dispatch tier
    if method == "auto":
        chosen = "exhaustive" if total_modules <= exhaustive_threshold else "approximate"
    else:
        chosen = method  # exhaustive / exhaustive_forced / approximate
    if chosen == "exhaustive_forced":
        chosen = "exhaustive"

    # Default pin set — modules that historically dominate the top of the
    # ranking (F-015 noted ``transformer.wte`` was silently excluded). We
    # always include them in the full-patch confirmation regardless of AtP.
    if pin_modules is None:
        pin_modules = _default_pin_modules(arch, candidate_names)

    start_time = time.time()
    atp_scores: dict[str, float] = {}
    atp_rank_map: dict[str, int] = {}

    if chosen == "approximate":
        # Phase 1: AtP shortlist
        from interpkit.ops._atp import compute_atp_scores

        atp_scores = compute_atp_scores(
            model, clean_input, corrupted_input, candidate_names, metric=metric,
        )
        # Rank by absolute AtP score (sign-agnostic)
        ranked = sorted(
            atp_scores.items(),
            key=lambda kv: abs(kv[1]) if kv[1] == kv[1] else -1.0,  # NaN sorts last
            reverse=True,
        )
        atp_rank_map = {name: i for i, (name, _) in enumerate(ranked)}

        if top_k_search is None:
            top_k_search = 4 * top_k
        shortlist = {name for name, _ in ranked[:top_k_search]}
        # Always include pinned modules
        shortlist |= set(pin_modules)
        selected_names = shortlist & set(candidate_names)
    else:
        selected_names = set(candidate_names)
        if pin_modules:
            selected_names |= set(pin_modules)

    # Phase 2: full patching on selected modules with cached clean activations
    clean_cache: dict[str, torch.Tensor] = {}

    def _make_cache_hook(name: str):
        def hook_fn(_mod: torch.nn.Module, _inp: Any, output: Any) -> None:
            t = output if isinstance(output, torch.Tensor) else (
                output[0] if isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor) else None
            )
            if t is not None:
                clean_cache[name] = t.detach().clone()

        return hook_fn

    cache_handles = []
    for name in selected_names:
        try:
            mod = _get_module(model._model, name)
        except (AttributeError, IndexError, KeyError, TypeError):
            continue
        cache_handles.append(mod.register_forward_hook(_make_cache_hook(name)))
    clean_logits = model._forward(clean_input)
    for h in cache_handles:
        h.remove()

    corrupted_logits = model._forward(corrupted_input)

    results: list[dict[str, Any]] = []
    with Progress(console=console, transient=True) as progress:
        task = progress.add_task(
            f"Causal tracing ({chosen})", total=len(selected_names),
        )
        for name in selected_names:
            if name not in clean_cache:
                progress.advance(task)
                continue
            try:
                target_mod = _get_module(model._model, name)
            except (AttributeError, IndexError, KeyError, TypeError):
                progress.advance(task)
                continue

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
            entry = {
                "module": name,
                "role": module_role_map.get(name),
                "effect": effect,
                "measurement_method": "full_patch",
            }
            if chosen == "approximate":
                entry["atp_score"] = atp_scores.get(name, float("nan"))
                entry["atp_rank"] = atp_rank_map.get(name)
                entry["pinned"] = name in pin_modules
            results.append(entry)
            progress.advance(task)

    results.sort(
        key=lambda x: abs(x["effect"]) if x["effect"] == x["effect"] else -1.0,
        reverse=True,
    )

    if top_k is not None and len(results) > top_k:
        results = results[:top_k]

    elapsed = time.time() - start_time
    meta = {
        "mode": "module",
        "algorithm": chosen,
        "total_candidates": total_modules,
        "candidates_full_patched": len(selected_names),
        "candidates_atp_only": (
            total_modules - len(selected_names) if chosen == "approximate" else 0
        ),
        "pinned_modules": list(pin_modules),
        "exhaustive_threshold": exhaustive_threshold,
        "runtime_seconds": elapsed,
        "memory_fallback_triggered": False,  # placeholder for future grad-checkpointing
    }

    output = {"results": results, "meta": meta}

    model_name = arch.arch_family or "model"
    render_trace(results, model_name, total_modules, top_k)

    if save is not None:
        from interpkit.core.plot import plot_trace

        plot_trace(results, model_name=model_name, save_path=save)

    if html is not None:
        from interpkit.core.html import html_trace as gen_html_trace
        from interpkit.core.html import save_html

        save_html(gen_html_trace(results), html)

    return output


def _default_pin_modules(arch: Any, candidate_names: list[str]) -> list[str]:
    """Default 'always-include' modules for the approximate-mode shortlist.

    F-015's specific failure was ``transformer.wte`` being excluded by the
    activation-norm proxy despite tying for top-1 by true effect. The
    pinned set covers the structurally-important modules the resolver
    knows about: embedding, head, pre-head norm, project_out, position
    embed (when present).
    """
    pinned: list[str] = []
    candidates = set(candidate_names)
    for path in (
        getattr(arch, "embed_path", None),
        getattr(arch, "head_path", None),
        getattr(arch, "pre_head_path", None),
        getattr(arch, "project_out_path", None),
    ):
        if path and path in candidates:
            pinned.append(path)
    return pinned


# ------------------------------------------------------------------
# Position-aware tracing (Meng et al.)
# ------------------------------------------------------------------

def _run_position_trace(
    model: Model,
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
    if not clean_cache:
        raise ValueError(
            "No layer activations were captured during the clean forward pass. "
            "Check that layer_names match actual modules in the model."
        )
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
        from interpkit.core.html import html_position_trace as gen_html_pt
        from interpkit.core.html import save_html

        save_html(gen_html_pt(result), html)

    return result
