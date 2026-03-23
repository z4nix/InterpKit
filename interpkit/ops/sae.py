"""sae — load pre-trained Sparse Autoencoders and decompose activations into features."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from rich.console import Console
from rich.progress import Progress

from interpkit.ops.patch import _get_module

if TYPE_CHECKING:
    from interpkit.core.model import Model

console = Console()


@dataclass
class SAE:
    """A loaded Sparse Autoencoder with weights ready for inference."""

    W_enc: torch.Tensor
    W_dec: torch.Tensor
    b_enc: torch.Tensor
    b_dec: torch.Tensor
    d_in: int = 0
    d_sae: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (features, reconstruction)."""
        features = self.encode(x)
        x_hat = self.decode(features)
        return features, x_hat


def load_sae(hf_id: str, *, device: str | torch.device = "cpu") -> SAE:
    """Download and load a Sparse Autoencoder from HuggingFace.

    Expects the repo to contain ``sae_weights.safetensors`` (or ``.pt``)
    and optionally ``cfg.json`` with metadata.
    """
    from huggingface_hub import hf_hub_download

    device = torch.device(device)

    # Try safetensors first, fall back to .pt
    weights = _download_weights(hf_id)

    required_keys = {"W_enc", "W_dec", "b_enc", "b_dec"}
    missing = required_keys - set(weights.keys())
    if missing:
        raise KeyError(
            f"SAE weights from {hf_id!r} are missing keys: {missing}. "
            f"Found keys: {list(weights.keys())}. "
            f"interpkit expects the SAELens format (W_enc, W_dec, b_enc, b_dec)."
        )

    W_enc = weights["W_enc"].to(device).float()
    W_dec = weights["W_dec"].to(device).float()
    b_enc = weights["b_enc"].to(device).float()
    b_dec = weights["b_dec"].to(device).float()

    metadata = _download_config(hf_id)

    d_in = W_enc.shape[0]
    d_sae = W_enc.shape[1]

    return SAE(
        W_enc=W_enc,
        W_dec=W_dec,
        b_enc=b_enc,
        b_dec=b_dec,
        d_in=d_in,
        d_sae=d_sae,
        metadata=metadata,
    )


def load_sae_from_tensors(
    W_enc: torch.Tensor,
    W_dec: torch.Tensor,
    b_enc: torch.Tensor,
    b_dec: torch.Tensor,
    *,
    metadata: dict[str, Any] | None = None,
) -> SAE:
    """Create an SAE from raw weight tensors (useful for testing)."""
    return SAE(
        W_enc=W_enc,
        W_dec=W_dec,
        b_enc=b_enc,
        b_dec=b_dec,
        d_in=W_enc.shape[0],
        d_sae=W_enc.shape[1],
        metadata=metadata or {},
    )


def run_features(
    model: "Model",
    input_data: Any,
    *,
    at: str,
    sae: SAE,
    top_k: int = 20,
    attribute: bool = False,
    print_results: bool = True,
) -> dict[str, Any]:
    """Decompose activations at *at* through the SAE and return top features.

    Parameters
    ----------
    attribute:
        When ``True``, compute each feature's logit contribution by
        projecting ``feature_activation * W_dec_row`` through the model's
        unembedding matrix toward the predicted token.  Adds
        ``feature_attributions`` to the result dict.

    Returns a dict with ``top_features``, ``reconstruction_error``, ``sparsity``.
    """
    from interpkit.ops.activations import run_activations

    act = run_activations(model, input_data, at=at, print_stats=False)
    if not isinstance(act, torch.Tensor):
        raise TypeError(f"Expected tensor from activations, got {type(act).__name__}")

    if act.dim() == 1:
        flat = act.unsqueeze(0).float()
    else:
        flat = act.view(-1, act.shape[-1]).float()

    if flat.shape[-1] != sae.d_in:
        raise ValueError(
            f"Activation dim ({flat.shape[-1]}) does not match SAE input dim ({sae.d_in}). "
            f"Make sure the SAE was trained on the same layer."
        )

    features, x_hat = sae.forward(flat)

    recon_error = (flat - x_hat).norm(dim=-1).mean().item()
    sparsity = (features == 0).float().mean().item()

    mean_activations = features.mean(dim=0)
    topk_vals, topk_idxs = mean_activations.topk(min(top_k, sae.d_sae))

    top_features = [
        (idx.item(), val.item())
        for idx, val in zip(topk_idxs, topk_vals)
    ]

    result: dict[str, Any] = {
        "module": at,
        "top_features": top_features,
        "reconstruction_error": recon_error,
        "sparsity": sparsity,
        "num_active_features": int((mean_activations > 0).sum().item()),
        "total_features": sae.d_sae,
        "feature_activations": features.detach(),
    }

    if attribute:
        result["feature_attributions"] = _compute_feature_attribution(
            model, sae, features, topk_idxs,
        )

    if print_results:
        from interpkit.core.render import render_features

        render_features(result)

    return result


def run_contrastive_features(
    model: "Model",
    positive_inputs: list[Any],
    negative_inputs: list[Any],
    *,
    at: str,
    sae: SAE,
    top_k: int = 20,
    print_results: bool = True,
) -> dict[str, Any]:
    """Compare SAE feature activations between two groups of inputs.

    Returns features ranked by absolute differential activation
    (positive mean - negative mean), surfacing features that distinguish
    the two concepts.
    """
    from interpkit.ops.activations import run_activations

    def _group_features(inputs: list[Any], progress: Progress, task) -> torch.Tensor:
        feat_sum: torch.Tensor | None = None
        for inp in inputs:
            act = run_activations(model, inp, at=at, print_stats=False)
            if not isinstance(act, torch.Tensor):
                raise TypeError(f"Expected tensor, got {type(act).__name__}")
            flat = act.view(-1, act.shape[-1]).float() if act.dim() > 1 else act.unsqueeze(0).float()
            if flat.shape[-1] != sae.d_in:
                raise ValueError(
                    f"Activation dim ({flat.shape[-1]}) != SAE input dim ({sae.d_in}). "
                    f"Make sure the SAE was trained on the same layer."
                )
            feats, _ = sae.forward(flat)
            mean_feats = feats.mean(dim=0)
            feat_sum = mean_feats if feat_sum is None else feat_sum + mean_feats
            progress.advance(task)
        return feat_sum / len(inputs)  # type: ignore[operator]

    total = len(positive_inputs) + len(negative_inputs)
    with Progress(console=console, transient=True) as progress:
        task = progress.add_task("Processing examples", total=total)
        pos_mean = _group_features(positive_inputs, progress, task)
        neg_mean = _group_features(negative_inputs, progress, task)
    diff = pos_mean - neg_mean

    abs_diff = diff.abs()
    k = min(top_k, sae.d_sae)
    topk_vals, topk_idxs = abs_diff.topk(k)

    top_features = []
    for idx, _abs_val in zip(topk_idxs.tolist(), topk_vals.tolist()):
        top_features.append({
            "feature_idx": idx,
            "positive_mean": pos_mean[idx].item(),
            "negative_mean": neg_mean[idx].item(),
            "diff": diff[idx].item(),
        })

    result: dict[str, Any] = {
        "module": at,
        "top_differential_features": top_features,
        "num_positive": len(positive_inputs),
        "num_negative": len(negative_inputs),
        "total_features": sae.d_sae,
    }

    if print_results:
        from interpkit.core.render import render_contrastive_features

        render_contrastive_features(result)

    return result


def _compute_feature_attribution(
    model: "Model",
    sae: SAE,
    features: torch.Tensor,
    topk_idxs: torch.Tensor,
) -> list[dict[str, Any]]:
    """Project each top feature through the decoder → unembedding to get logit contributions."""
    arch = model.arch_info
    if not arch.unembedding_name:
        return []

    from interpkit.ops.patch import _get_module

    unembed_mod = _get_module(model._model, arch.unembedding_name)
    w_unembed = None
    for name, param in unembed_mod.named_parameters():
        if "weight" in name:
            w_unembed = param.data.float()
            break

    if w_unembed is None:
        return []

    # w_unembed: (vocab_size, d_model) for nn.Linear, transposed for Conv1D
    if type(unembed_mod).__name__ == "Conv1D":
        w_unembed = w_unembed.T  # -> (vocab_size, d_model)

    last_pos_feats = features[-1]  # (d_sae,)

    attributions = []
    tokenizer = model._tokenizer

    for feat_idx in topk_idxs.tolist():
        feat_act = last_pos_feats[feat_idx].item()
        if feat_act == 0:
            attributions.append({
                "feature_idx": feat_idx,
                "activation": 0.0,
                "top_logit_contributions": [],
            })
            continue

        # feature contribution to residual stream: feat_act * W_dec[feat_idx]
        dec_row = sae.W_dec[feat_idx].float()  # (d_model,)
        logit_contribution = feat_act * (w_unembed @ dec_row)  # (vocab_size,)

        top_vals, top_ids = logit_contribution.topk(5)
        bot_vals, bot_ids = logit_contribution.topk(5, largest=False)

        top_contribs = []
        for tok_id, val in zip(top_ids.tolist(), top_vals.tolist()):
            tok_str = tokenizer.decode([tok_id]) if tokenizer else str(tok_id)
            top_contribs.append({"token": tok_str, "token_id": tok_id, "logit": val})

        bot_contribs = []
        for tok_id, val in zip(bot_ids.tolist(), bot_vals.tolist()):
            tok_str = tokenizer.decode([tok_id]) if tokenizer else str(tok_id)
            bot_contribs.append({"token": tok_str, "token_id": tok_id, "logit": val})

        attributions.append({
            "feature_idx": feat_idx,
            "activation": feat_act,
            "top_logit_contributions": top_contribs,
            "bottom_logit_contributions": bot_contribs,
        })

    return attributions


def _download_weights(hf_id: str) -> dict[str, torch.Tensor]:
    """Download SAE weights from HuggingFace."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    # Try safetensors first
    try:
        path = hf_hub_download(hf_id, filename="sae_weights.safetensors")
        from safetensors.torch import load_file

        return load_file(path)
    except (EntryNotFoundError, FileNotFoundError):
        pass
    except RepositoryNotFoundError:
        raise FileNotFoundError(
            f"HuggingFace repository {hf_id!r} not found. "
            f"Check the repo ID and your network/auth settings."
        )

    # Fall back to .pt
    try:
        path = hf_hub_download(hf_id, filename="sae_weights.pt")
        return torch.load(path, map_location="cpu", weights_only=True)
    except (EntryNotFoundError, FileNotFoundError):
        pass

    raise FileNotFoundError(
        f"Could not find sae_weights.safetensors or sae_weights.pt in {hf_id!r}. "
        f"The HF repo should contain one of these files."
    )


def _download_config(hf_id: str) -> dict[str, Any]:
    """Download SAE config from HuggingFace (optional)."""
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(hf_id, filename="cfg.json")
        return json.loads(Path(path).read_text())
    except Exception:
        return {}
