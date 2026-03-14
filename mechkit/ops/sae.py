"""sae — load pre-trained Sparse Autoencoders and decompose activations into features."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from mechkit.ops.patch import _get_module

if TYPE_CHECKING:
    from mechkit.core.model import Model


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
            f"mechkit expects the SAELens format (W_enc, W_dec, b_enc, b_dec)."
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
    print_results: bool = True,
) -> dict[str, Any]:
    """Decompose activations at *at* through the SAE and return top features.

    Returns a dict with ``top_features``, ``reconstruction_error``, ``sparsity``.
    """
    from mechkit.ops.activations import run_activations

    act = run_activations(model, input_data, at=at, print_stats=False)
    if not isinstance(act, torch.Tensor):
        raise TypeError(f"Expected tensor from activations, got {type(act).__name__}")

    # Flatten to 2D: (batch * seq, d_model)
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

    # Reconstruction error (mean L2 across positions)
    recon_error = (flat - x_hat).norm(dim=-1).mean().item()

    # Sparsity: fraction of features that are zero
    sparsity = (features == 0).float().mean().item()

    # Top-K features by mean activation (across all positions)
    mean_activations = features.mean(dim=0)
    topk_vals, topk_idxs = mean_activations.topk(min(top_k, sae.d_sae))

    top_features = [
        (idx.item(), val.item())
        for idx, val in zip(topk_idxs, topk_vals)
    ]

    result = {
        "module": at,
        "top_features": top_features,
        "reconstruction_error": recon_error,
        "sparsity": sparsity,
        "num_active_features": int((mean_activations > 0).sum().item()),
        "total_features": sae.d_sae,
        "feature_activations": features.detach(),
    }

    if print_results:
        from mechkit.core.render import render_features

        render_features(result)

    return result


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
