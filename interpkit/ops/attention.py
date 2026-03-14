"""attention — capture and display attention patterns for transformer models."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from rich.console import Console

from interpkit.ops.patch import _get_module

if TYPE_CHECKING:
    from interpkit.core.model import Model

console = Console()


def run_attention(
    model: "Model",
    input_data: Any,
    *,
    layer: int | None = None,
    head: int | None = None,
    save: str | None = None,
    html: str | None = None,
) -> list[dict[str, Any]] | None:
    """Capture attention weights and display a summary.

    Computes attention weights manually from Q/K projections via hooks,
    since modern transformer implementations (SDPA, FlashAttention) don't
    return attention weights.
    """
    from interpkit.core.render import render_attention

    arch = model.arch_info
    attn_modules = [m for m in arch.modules if m.role == "attention"]

    if not attn_modules:
        console.print(
            "\n  [yellow]attention not available:[/yellow] no attention modules detected"
            f" for {arch.arch_family or 'this model'}.\n"
        )
        return None

    model_input = model._prepare(input_data)

    tokens = None
    if model._tokenizer is not None and isinstance(input_data, str):
        encoded = model._tokenizer(input_data, return_tensors="pt")
        token_ids = encoded["input_ids"][0].tolist()
        tokens = model._tokenizer.convert_ids_to_tokens(token_ids)

    # Capture Q and K projections to compute attention weights manually
    qk_cache: dict[str, dict[str, torch.Tensor]] = {}
    hooks = []

    for mod_info in attn_modules:
        if layer is not None:
            layer_match = re.search(r"\.(\d+)\.", mod_info.name)
            if layer_match and int(layer_match.group(1)) != layer:
                continue

        attn_mod = _get_module(model._model, mod_info.name)

        # Find Q/K projection submodules
        for child_name, child_mod in attn_mod.named_modules():
            full_name = f"{mod_info.name}.{child_name}" if child_name else mod_info.name
            is_qkv = any(p in child_name.lower() for p in ("c_attn", "qkv", "q_proj", "query"))
            is_k = any(p in child_name.lower() for p in ("k_proj", "key"))

            if is_qkv or is_k:
                def _make_qk_hook(name: str, attn_name: str):
                    def hook_fn(_mod: torch.nn.Module, _inp: Any, output: Any) -> None:
                        t = output if isinstance(output, torch.Tensor) else (
                            output[0] if isinstance(output, (tuple, list)) else None
                        )
                        if t is not None:
                            qk_cache.setdefault(attn_name, {})[name] = t.detach()
                    return hook_fn

                hooks.append(child_mod.register_forward_hook(_make_qk_hook(child_name, mod_info.name)))

    with torch.no_grad():
        model._forward(model_input)

    for h in hooks:
        h.remove()

    # Compute attention weights from cached Q/K
    results: list[dict[str, Any]] = []

    for attn_name, projections in qk_cache.items():
        layer_match = re.search(r"\.(\d+)\.", attn_name)
        layer_idx = int(layer_match.group(1)) if layer_match else 0

        attn_weights = _compute_attention_from_projections(
            projections, arch.num_attention_heads or 12
        )

        if attn_weights is None:
            continue

        num_heads = attn_weights.shape[0]
        for head_idx in range(num_heads):
            if head is not None and head_idx != head:
                continue

            head_attn = attn_weights[head_idx]
            top_pairs = _get_top_pairs(head_attn, k=5)
            entropy = _attention_entropy(head_attn)

            results.append({
                "layer": layer_idx,
                "head": head_idx,
                "top_pairs": top_pairs,
                "entropy": entropy,
                "weights": head_attn,
            })

    if not results:
        console.print(
            "\n  [yellow]attention:[/yellow] could not compute attention weights.\n"
        )
        return None

    model_name = arch.arch_family or "model"
    render_attention(results, tokens, model_name)

    if save is not None:
        from interpkit.core.plot import plot_attention, plot_attention_multi

        if layer is not None and head is not None and len(results) == 1:
            plot_attention(results[0]["weights"], tokens, layer=results[0]["layer"],
                           head=results[0]["head"], save_path=save)
        else:
            plot_attention_multi(results, tokens, save_path=save)

    if html is not None:
        from interpkit.core.html import html_attention as gen_html_attention
        from interpkit.core.html import save_html

        serializable = []
        for r in results:
            entry = {**r}
            w = r.get("weights")
            if isinstance(w, torch.Tensor):
                entry["weights"] = w.tolist()
            serializable.append(entry)
        save_html(gen_html_attention(serializable, tokens), html)

    return results


def _compute_attention_from_projections(
    projections: dict[str, torch.Tensor],
    num_heads: int,
) -> torch.Tensor | None:
    """Compute attention weights from captured QKV or Q/K projections."""
    # GPT-2 style: c_attn produces [Q, K, V] concatenated
    for key, tensor in projections.items():
        if "c_attn" in key or "qkv" in key:
            # tensor shape: (batch, seq, 3 * hidden) or (seq, 3 * hidden)
            if tensor.dim() == 3:
                tensor = tensor[0]  # drop batch
            hidden = tensor.shape[-1] // 3
            q, k, _v = tensor.split(hidden, dim=-1)
            return _qk_to_attention(q, k, num_heads)

    # Separate Q and K projections
    q_tensor = None
    k_tensor = None
    for key, tensor in projections.items():
        if "q_proj" in key or "query" in key:
            q_tensor = tensor
        elif "k_proj" in key or "key" in key:
            k_tensor = tensor

    if q_tensor is not None and k_tensor is not None:
        if q_tensor.dim() == 3:
            q_tensor = q_tensor[0]
        if k_tensor.dim() == 3:
            k_tensor = k_tensor[0]
        return _qk_to_attention(q_tensor, k_tensor, num_heads)

    return None


def _qk_to_attention(
    q: torch.Tensor, k: torch.Tensor, num_heads: int,
) -> torch.Tensor:
    """Compute attention weights from Q and K tensors.

    q, k: (seq_len, hidden_size)
    Returns: (num_heads, seq_len, seq_len)
    """
    seq_len, hidden = q.shape
    head_dim = hidden // num_heads

    q = q.view(seq_len, num_heads, head_dim).transpose(0, 1)  # (heads, seq, head_dim)
    k = k.view(seq_len, num_heads, head_dim).transpose(0, 1)

    scale = head_dim ** 0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (heads, seq, seq)

    # Apply causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device), diagonal=1)
    scores.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))

    return F.softmax(scores, dim=-1)


def _get_top_pairs(
    attn: torch.Tensor, k: int = 5,
) -> list[tuple[int, int, float]]:
    """Find top-k (source_pos, target_pos, score) pairs in an attention matrix."""
    flat = attn.view(-1)
    topk_vals, topk_idxs = flat.topk(min(k, flat.numel()))
    seq_len = attn.shape[-1]
    pairs = []
    for val, idx in zip(topk_vals.tolist(), topk_idxs.tolist()):
        src = idx // seq_len
        tgt = idx % seq_len
        pairs.append((src, tgt, val))
    return pairs


def _attention_entropy(attn: torch.Tensor) -> float:
    """Mean entropy of attention distributions across query positions."""
    eps = 1e-10
    log_attn = torch.log(attn + eps)
    entropy_per_query = -(attn * log_attn).sum(dim=-1)
    return entropy_per_query.mean().item()
