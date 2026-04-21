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


def _ensure_sae_on_device(sae: SAE, target_device: torch.device) -> SAE:
    """Return *sae* with its tensors on *target_device*.

    If the SAE's weights already live on the target device the original
    object is returned unchanged.  Otherwise a new :class:`SAE` instance
    is constructed with all four weight tensors moved to the device, so
    that callers can safely run ``sae.encode`` / ``sae.forward`` against
    activations produced on a different device (e.g. SAE loaded on CPU
    but model running on MPS).
    """
    if sae.W_enc.device == target_device:
        return sae
    return SAE(
        W_enc=sae.W_enc.to(target_device),
        W_dec=sae.W_dec.to(target_device),
        b_enc=sae.b_enc.to(target_device),
        b_dec=sae.b_dec.to(target_device),
        d_in=sae.d_in,
        d_sae=sae.d_sae,
        metadata=sae.metadata,
    )


def load_sae(
    source: str,
    *,
    device: str | torch.device = "cpu",
    subfolder: str | None = None,
) -> SAE:
    """Load a Sparse Autoencoder from a HuggingFace repo ID or a local file path.

    *source* is interpreted as a **local path** when it points to an existing
    file or ends with ``.safetensors`` / ``.pt``.  Otherwise it is treated as
    a HuggingFace repo ID and weights are downloaded.

    For HuggingFace repos the repo should contain
    ``sae_weights.safetensors`` (or ``.pt``) and optionally ``cfg.json``.

    Parameters
    ----------
    source:
        Either a local path or a HuggingFace repo ID.  A shorthand is
        supported: ``"<org>/<repo>/<subfolder>"`` is parsed as repo
        ``"<org>/<repo>"`` with subfolder ``"<subfolder>"`` (used by
        repos like ``jbloom/GPT2-Small-SAEs-Reformatted`` that store
        per-layer weights in subdirectories).
    device:
        Device the resulting SAE tensors are placed on.
    subfolder:
        Explicit subfolder within a HuggingFace repo.  Mutually
        exclusive with the shorthand syntax above; passing both raises
        :class:`ValueError`.

    Raises
    ------
    ValueError
        If both ``subfolder=`` and the shorthand syntax are used.
    FileNotFoundError
        If the weights cannot be found in the repo / subfolder.
    """
    p = Path(source)
    if p.is_file() or p.suffix in (".safetensors", ".pt"):
        return load_sae_from_path(p, device=device)

    repo_id, parsed_subfolder = _split_repo_and_subfolder(source)
    if parsed_subfolder is not None:
        if subfolder is not None and subfolder != parsed_subfolder:
            raise ValueError(
                f"Conflicting subfolder: source shorthand specifies "
                f"{parsed_subfolder!r} but explicit subfolder= is "
                f"{subfolder!r}. Pass only one."
            )
        subfolder = parsed_subfolder

    return _load_sae_from_hf(repo_id, device=device, subfolder=subfolder)


def _split_repo_and_subfolder(source: str) -> tuple[str, str | None]:
    """Parse ``"<org>/<repo>/<subfolder>"`` shorthand into ``(repo_id, subfolder)``.

    The shorthand is accepted only when *source*:

    - Has 3+ slash-separated segments;
    - Is not an existing local path (file or directory);
    - Does not start with ``"./"``, ``"../"``, ``"~"``, or ``"/"``;
    - The first two segments look like a HuggingFace ``org/name`` (no
      whitespace, no path separators within either segment).

    Otherwise the original *source* is returned with ``subfolder=None``.
    """
    if not source or "/" not in source:
        return source, None

    if source.startswith(("/", "./", "../", "~")):
        return source, None

    if Path(source).exists():
        return source, None

    parts = source.split("/")
    if len(parts) < 3:
        return source, None

    org, repo, *rest = parts
    if not org or not repo:
        return source, None
    if any(ch.isspace() for ch in org) or any(ch.isspace() for ch in repo):
        return source, None

    subfolder = "/".join(rest)
    if not subfolder:
        return source, None

    return f"{org}/{repo}", subfolder


def load_sae_from_path(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> SAE:
    """Load a Sparse Autoencoder from a local ``.safetensors`` or ``.pt`` file.

    Optionally loads ``cfg.json`` from the same directory for metadata.
    """
    path = Path(path)
    device = torch.device(device)

    if not path.is_file():
        raise FileNotFoundError(f"SAE weights file not found: {path}")

    weights = _load_local_weights(path)

    required_keys = {"W_enc", "W_dec", "b_enc", "b_dec"}
    missing = required_keys - set(weights.keys())
    if missing:
        raise KeyError(
            f"SAE weights from {path} are missing keys: {missing}. "
            f"Found keys: {list(weights.keys())}. "
            f"interpkit expects the SAELens format (W_enc, W_dec, b_enc, b_dec)."
        )

    W_enc = weights["W_enc"].to(device).float()
    W_dec = weights["W_dec"].to(device).float()
    b_enc = weights["b_enc"].to(device).float()
    b_dec = weights["b_dec"].to(device).float()

    cfg_path = path.parent / "cfg.json"
    metadata: dict[str, Any] = {}
    if cfg_path.is_file():
        metadata = json.loads(cfg_path.read_text())

    return SAE(
        W_enc=W_enc, W_dec=W_dec, b_enc=b_enc, b_dec=b_dec,
        d_in=W_enc.shape[0], d_sae=W_enc.shape[1],
        metadata=metadata,
    )


def _load_local_weights(path: Path) -> dict[str, torch.Tensor]:
    """Load weight tensors from a local .safetensors or .pt file."""
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path))
    elif path.suffix == ".pt":
        result: dict[str, torch.Tensor] = torch.load(str(path), map_location="cpu", weights_only=True)
        return result
    else:
        raise ValueError(
            f"Unsupported SAE weight file format: {path.suffix!r}. "
            f"Expected .safetensors or .pt"
        )


def _load_sae_from_hf(
    hf_id: str,
    *,
    device: str | torch.device = "cpu",
    subfolder: str | None = None,
) -> SAE:
    """Download and load a Sparse Autoencoder from HuggingFace."""
    device = torch.device(device)

    weights = _download_weights(hf_id, subfolder=subfolder)

    required_keys = {"W_enc", "W_dec", "b_enc", "b_dec"}
    missing = required_keys - set(weights.keys())
    if missing:
        location = f"{hf_id!r}" + (f" (subfolder={subfolder!r})" if subfolder else "")
        raise KeyError(
            f"SAE weights from {location} are missing keys: {missing}. "
            f"Found keys: {list(weights.keys())}. "
            f"interpkit expects the SAELens format (W_enc, W_dec, b_enc, b_dec)."
        )

    W_enc = weights["W_enc"].to(device).float()
    W_dec = weights["W_dec"].to(device).float()
    b_enc = weights["b_enc"].to(device).float()
    b_dec = weights["b_dec"].to(device).float()

    metadata = _download_config(hf_id, subfolder=subfolder)

    return SAE(
        W_enc=W_enc, W_dec=W_dec, b_enc=b_enc, b_dec=b_dec,
        d_in=W_enc.shape[0], d_sae=W_enc.shape[1],
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
    model: Model,
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

    sae = _ensure_sae_on_device(sae, flat.device)

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
    model: Model,
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

    if not positive_inputs:
        raise ValueError("At least one positive input is required for contrastive features.")
    if not negative_inputs:
        raise ValueError("At least one negative input is required for contrastive features.")

    from interpkit.core.inputs import (
        normalize_input_group,
        warn_if_leading_space_better,
    )

    # Treat a bare chat-message list as a single example rather than as a
    # batch of message dicts (which would crash in run_activations with
    # ``ValueError: You must specify exactly one of input_ids or inputs_embeds``).
    positive_inputs = normalize_input_group(positive_inputs)
    negative_inputs = normalize_input_group(negative_inputs)

    warned: list[int] = [0]
    for p in positive_inputs:
        warn_if_leading_space_better(
            getattr(model, "_tokenizer", None),
            p,
            op_label="features",
            role="positive",
            warned_count=warned,
            console=console,
        )
    for n in negative_inputs:
        warn_if_leading_space_better(
            getattr(model, "_tokenizer", None),
            n,
            op_label="features",
            role="negative",
            warned_count=warned,
            console=console,
        )

    sae_local = sae

    def _group_features(inputs: list[Any], progress: Progress, task) -> torch.Tensor:
        nonlocal sae_local
        feat_sum: torch.Tensor | None = None
        for inp in inputs:
            act = run_activations(model, inp, at=at, print_stats=False)
            if not isinstance(act, torch.Tensor):
                raise TypeError(f"Expected tensor, got {type(act).__name__}")
            flat = act.view(-1, act.shape[-1]).float() if act.dim() > 1 else act.unsqueeze(0).float()
            if flat.shape[-1] != sae_local.d_in:
                raise ValueError(
                    f"Activation dim ({flat.shape[-1]}) != SAE input dim ({sae_local.d_in}). "
                    f"Make sure the SAE was trained on the same layer."
                )
            sae_local = _ensure_sae_on_device(sae_local, flat.device)
            feats, _ = sae_local.forward(flat)
            mean_feats = feats.mean(dim=0)
            feat_sum = mean_feats if feat_sum is None else feat_sum + mean_feats
            progress.advance(task)
        assert isinstance(feat_sum, torch.Tensor)
        return feat_sum / len(inputs)

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
    model: Model,
    sae: SAE,
    features: torch.Tensor,
    topk_idxs: torch.Tensor,
) -> list[dict[str, Any]]:
    """Project each top feature through the decoder → unembedding to get logit contributions."""
    arch = model.arch_info
    if not arch.unembedding_name:
        return []


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


def _download_weights(
    hf_id: str,
    *,
    subfolder: str | None = None,
) -> dict[str, torch.Tensor]:
    """Download SAE weights from HuggingFace, optionally from a subfolder."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    extra: dict[str, Any] = {}
    if subfolder is not None:
        extra["subfolder"] = subfolder

    # Try safetensors first
    try:
        path = hf_hub_download(hf_id, filename="sae_weights.safetensors", **extra)
        from safetensors.torch import load_file

        return load_file(path)
    except (EntryNotFoundError, FileNotFoundError):
        pass
    except RepositoryNotFoundError:
        raise FileNotFoundError(
            f"HuggingFace repository {hf_id!r} not found. "
            f"Check the repo ID and your network/auth settings."
        ) from None

    # Fall back to .pt
    try:
        path = hf_hub_download(hf_id, filename="sae_weights.pt", **extra)
        weights: dict[str, torch.Tensor] = torch.load(path, map_location="cpu", weights_only=True)
        return weights
    except (EntryNotFoundError, FileNotFoundError):
        pass
    except RepositoryNotFoundError:
        raise FileNotFoundError(
            f"HuggingFace repository {hf_id!r} not found. "
            f"Check the repo ID and your network/auth settings."
        ) from None

    location = f"{hf_id!r}" + (f" (subfolder={subfolder!r})" if subfolder else "")
    hint = (
        " If the repo organises weights into per-layer subdirectories "
        "(e.g. 'blocks.8.hook_resid_pre/sae_weights.safetensors'), pass "
        "the subfolder either via subfolder=... or as part of the source "
        "string: 'org/repo/subfolder'."
    )
    raise FileNotFoundError(
        f"Could not find sae_weights.safetensors or sae_weights.pt in {location}. "
        f"The HF repo should contain one of these files." + (hint if subfolder is None else "")
    )


def _download_config(
    hf_id: str,
    *,
    subfolder: str | None = None,
) -> dict[str, Any]:
    """Download SAE config from HuggingFace (optional).

    Missing ``cfg.json`` files are common for community SAE uploads, so a
    not-found result is silently swallowed.  Other failures (auth /
    network / corrupt JSON) print a one-line warning and still return
    an empty dict so weight loading can proceed.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    extra: dict[str, Any] = {}
    if subfolder is not None:
        extra["subfolder"] = subfolder

    try:
        path = hf_hub_download(hf_id, filename="cfg.json", **extra)
        cfg: dict[str, Any] = json.loads(Path(path).read_text())
        return cfg
    except (EntryNotFoundError, FileNotFoundError, RepositoryNotFoundError):
        return {}
    except (json.JSONDecodeError, OSError) as exc:
        console.print(
            f"  [yellow]sae:[/yellow] could not parse cfg.json for "
            f"{hf_id!r} ({type(exc).__name__}: {exc}). "
            "Continuing with empty config — d_in / d_sae will be inferred "
            "from weight shapes."
        )
        return {}
