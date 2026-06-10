"""tuned_lens — trained per-block affine translators for unbiased lens readout.

The raw logit lens projects intermediate hidden states through the
*final* norm + unembedding, which is biased for layers far from the
output: early-layer readouts are dominated by basis misalignment rather
than the layer's actual prediction. The tuned lens (Belrose et al.
2023, "Eliciting Latent Predictions from Transformers with the Tuned
Lens") fixes this by learning one affine translator per block, trained
so that ``head(translator_l(h_l))`` matches the model's *own* final
distribution under KL — the model is frozen; only the translators
(``n_blocks × (d² + d)`` parameters) train.

Identity contract: a freshly initialised :class:`TunedLens` has
identity-weight translators, so ``lens(kind="tuned")`` with an untrained
lens reproduces ``lens(kind="logit")`` exactly. Training only ever
moves the readout *toward* the model's true distribution (the final
block's translator stays ≈ identity because its KL is already ≈ 0).

Artifacts are saved as safetensors + a JSON sidecar (hidden size, block
paths, source model) under ``~/.cache/interpkit/tuned_lens/<model>/`` by
default; loading validates the metadata against the live ``arch_info``.

Scope (documented deferrals, same ledger pattern as ``ops/eap.py``):
text models with a resolvable head pipeline. Vision models (spatial)
and shared-weight models (ALBERT-style logical blocks aliasing one
physical module) raise ``OperationNotSupportedForArchitecture``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.progress import Progress

from interpkit.core.exceptions import InterpkitError, OperationNotSupportedForArchitecture

if TYPE_CHECKING:
    from interpkit.core.model import Model

console = Console()

__all__ = [
    "TunedLens",
    "train_tuned_lens",
    "save_tuned_lens",
    "load_tuned_lens",
    "tuned_lens_loss",
    "default_tuned_lens_dir",
]


class TunedLens(nn.Module):
    """Per-block affine translators: ``translator_l(h_l)`` → head pipeline.

    ``meta`` carries everything needed to validate compatibility at
    load/use time: hidden size, the exact block paths the translators
    were trained against, and the source model id.
    """

    def __init__(
        self,
        hidden_size: int,
        block_paths: list[str],
        *,
        model_id: str | None = None,
    ) -> None:
        super().__init__()
        self.translators = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size, bias=True) for _ in block_paths
        )
        with torch.no_grad():
            for t in self.translators:
                t.weight.copy_(torch.eye(hidden_size))
                t.bias.zero_()
        self.meta: dict[str, Any] = {
            "hidden_size": hidden_size,
            "block_paths": list(block_paths),
            "model_id": model_id,
            "trained": False,
            "train_config": None,
        }

    def __len__(self) -> int:
        return len(self.translators)

    def forward(self, layer_idx: int, hidden: torch.Tensor) -> torch.Tensor:
        return self.translators[layer_idx](hidden)

    def validate_against(self, arch: Any, block_paths: list[str]) -> None:
        """Raise :class:`InterpkitError` if this lens doesn't fit *arch*."""
        hidden = getattr(arch, "hidden_size", None)
        if hidden is not None and hidden != self.meta["hidden_size"]:
            raise InterpkitError(
                f"TunedLens was trained for hidden_size="
                f"{self.meta['hidden_size']} but this model has "
                f"hidden_size={hidden}. Train a lens for this model with "
                f"model.train_tuned_lens(corpus)."
            )
        if len(block_paths) != len(self.translators):
            raise InterpkitError(
                f"TunedLens has {len(self.translators)} translators but this "
                f"model lenses {len(block_paths)} blocks. Train a lens for "
                f"this model with model.train_tuned_lens(corpus)."
            )
        trained_paths = self.meta.get("block_paths") or []
        if trained_paths and trained_paths != list(block_paths):
            raise InterpkitError(
                "TunedLens block paths do not match this model's lens blocks "
                f"(trained: {trained_paths[:2]}…, model: {list(block_paths)[:2]}…). "
                "It was likely trained on a different model or interpkit "
                "resolved the architecture differently. Retrain with "
                "model.train_tuned_lens(corpus)."
            )


def _lens_setup(model: Model) -> tuple[Any, list, list[str]]:
    """Shared gating: resolve lens blocks and reject unsupported families."""
    from interpkit.core.support_matrix import lens_blocks

    arch = model.arch_info
    if getattr(arch, "spatial", False):
        raise OperationNotSupportedForArchitecture(
            "tuned lens is not yet supported on vision models. "
            "Documented deferral (see ops/tuned_lens.py)."
        )
    blocks = lens_blocks(arch)
    block_paths = [b.path for b in blocks]
    if len(set(block_paths)) < len(block_paths):
        raise OperationNotSupportedForArchitecture(
            "tuned lens is not yet supported on shared-weight models "
            "(ALBERT-style): logical blocks alias one physical module, so "
            "per-block captures collide. Documented deferral (see "
            "ops/tuned_lens.py)."
        )
    if not block_paths:
        raise OperationNotSupportedForArchitecture(
            "tuned lens requires resolvable blocks; this model resolved none."
        )
    return arch, blocks, block_paths


def _capture_for_training(
    model: Model, block_paths: list[str], batch_input: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """One frozen forward: detached block outputs + final logits."""
    from interpkit.ops._hooks import register_capture_hook
    from interpkit.ops.patch import _get_module

    captured: dict[str, torch.Tensor] = {}
    handles = [
        register_capture_hook(_get_module(model._model, p), captured, p)
        for p in block_paths
    ]
    try:
        with torch.no_grad():
            logits = model._forward(batch_input)
    finally:
        for h in handles:
            h.remove()
    return captured, logits.float().detach()


def _batch_kl(
    model: Model,
    lens: TunedLens,
    block_paths: list[str],
    captured: dict[str, torch.Tensor],
    final_logits: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Mean-over-blocks masked KL(final ‖ tuned readout), fp32."""
    from interpkit.core.support_matrix import _project_through_head

    arch = model.arch_info
    target = F.softmax(final_logits, dim=-1)

    mask: torch.Tensor | None = None
    if (
        attention_mask is not None
        and final_logits.dim() == 3
        and attention_mask.shape[-1] == final_logits.shape[1]
    ):
        mask = attention_mask.float()

    losses: list[torch.Tensor] = []
    for i, path in enumerate(block_paths):
        h = captured.get(path)
        if h is None:
            continue
        z = lens.translators[i](h.float())
        logits_i = _project_through_head(arch, z)
        if logits_i is None or logits_i.shape != final_logits.shape:
            continue
        kl = F.kl_div(
            F.log_softmax(logits_i, dim=-1), target, reduction="none",
        ).sum(dim=-1)
        if mask is not None:
            kl = (kl * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            kl = kl.mean()
        losses.append(kl)
    if not losses:
        raise InterpkitError(
            "tuned-lens training captured no usable block outputs — the "
            "head pipeline rejected every block projection."
        )
    return torch.stack(losses).mean()


def train_tuned_lens(
    model: Model,
    corpus: list[str],
    *,
    steps: int = 200,
    batch_size: int = 4,
    lr: float = 1e-3,
    max_length: int = 64,
    seed: int = 0,
    save: str | None = None,
    progress: bool = True,
) -> TunedLens:
    """Train per-block affine translators against the model's own logits.

    The model is frozen; only the translators train (Adam, fp32 KL).
    Compute expectations: gpt2 on CPU at the defaults (200 steps × batch
    4 × 64 tokens) takes a few minutes; seconds on GPU. Translators are
    ``n_blocks × (hidden² + hidden)`` parameters.

    Parameters
    ----------
    corpus:
        Texts to train on — a few hundred diverse sentences is plenty
        for small models. Batches cycle (reshuffling each epoch with
        *seed*), so a small corpus simply repeats.
    save:
        Optional path (directory or ``.safetensors`` file). Defaults to
        not saving; use :func:`save_tuned_lens` or the ``save=`` of this
        function, and ``lens(kind="tuned", tuned_lens=<path>)`` to load.

    Returns the trained :class:`TunedLens` (also usable directly via
    ``model.lens(kind="tuned", tuned_lens=lens)``).
    """
    from interpkit.core.support_matrix import check_op_supported, validate_lens_pipeline

    check_op_supported("train_tuned_lens", model.arch_info)
    arch, _blocks, block_paths = _lens_setup(model)
    if model._tokenizer is None:
        raise RuntimeError(
            "tuned-lens training requires a tokenizer — load the model with "
            "one (interpkit.load does this automatically for HF text models)."
        )
    if not corpus or not isinstance(corpus, list):
        raise ValueError("corpus must be a non-empty list of strings.")
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}.")
    validate_lens_pipeline(model)

    hidden = arch.hidden_size
    if hidden is None:
        raise InterpkitError("tuned lens requires arch_info.hidden_size.")

    config = getattr(model._model, "config", None)
    model_id = getattr(config, "_name_or_path", None) if config is not None else None

    torch.manual_seed(seed)
    lens = TunedLens(hidden, block_paths, model_id=model_id).to(model._device)
    optimizer = torch.optim.Adam(lens.parameters(), lr=lr)
    rng = random.Random(seed)

    order: list[str] = []
    losses: list[float] = []

    def _next_batch() -> list[str]:
        nonlocal order
        batch: list[str] = []
        while len(batch) < batch_size:
            if not order:
                order = list(corpus)
                rng.shuffle(order)
            batch.append(order.pop())
        return batch

    progress_ctx = Progress(console=console, transient=True) if progress else None
    task = None
    if progress_ctx is not None:
        progress_ctx.start()
        task = progress_ctx.add_task("Training tuned lens", total=steps)
    try:
        for _step in range(steps):
            texts = _next_batch()
            encoded = model._tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            batch_input = {k: v.to(model._device) for k, v in encoded.items()}
            batch_input = model._inject_decoder_ids(batch_input)

            captured, final_logits = _capture_for_training(
                model, block_paths, batch_input,
            )
            loss = _batch_kl(
                model, lens, block_paths, captured, final_logits,
                batch_input.get("attention_mask"),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            if progress_ctx is not None and task is not None:
                progress_ctx.advance(task)
    finally:
        if progress_ctx is not None:
            progress_ctx.stop()
        # Projection runs through frozen model modules; drop any grad
        # buffers backward left on them.
        model._model.zero_grad(set_to_none=True)

    lens.meta["trained"] = True
    lens.meta["train_config"] = {
        "steps": steps, "batch_size": batch_size, "lr": lr,
        "max_length": max_length, "seed": seed, "corpus_size": len(corpus),
        "first_loss": losses[0], "final_loss": losses[-1],
    }
    console.print(
        f"  [bold green]Tuned lens trained[/bold green] — KL "
        f"{losses[0]:.4f} → {losses[-1]:.4f} over {steps} steps."
    )

    if save is not None:
        path = save_tuned_lens(lens, save)
        console.print(f"  Saved to [bold]{path}[/bold]")
    return lens


def tuned_lens_loss(
    model: Model,
    lens: TunedLens,
    texts: list[str],
    *,
    max_length: int = 64,
) -> float:
    """Mean KL(final ‖ tuned readout) over *texts* — evaluation helper."""
    _arch, _blocks, block_paths = _lens_setup(model)
    lens.validate_against(model.arch_info, block_paths)
    assert model._tokenizer is not None
    encoded = model._tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_length,
    )
    batch_input = {k: v.to(model._device) for k, v in encoded.items()}
    batch_input = model._inject_decoder_ids(batch_input)
    captured, final_logits = _capture_for_training(model, block_paths, batch_input)
    with torch.no_grad():
        loss = _batch_kl(
            model, lens, block_paths, captured, final_logits,
            batch_input.get("attention_mask"),
        )
    return float(loss.item())


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def default_tuned_lens_dir(model_id: str) -> Path:
    """Cache location for a model's tuned lens artifacts."""
    sanitized = model_id.replace("/", "--")
    return Path.home() / ".cache" / "interpkit" / "tuned_lens" / sanitized


def _resolve_paths(path: str | Path) -> tuple[Path, Path]:
    """Map a directory or .safetensors path to (weights, sidecar) paths."""
    p = Path(path).expanduser()
    if p.suffix == ".safetensors":
        return p, p.with_suffix(".json")
    return p / "tuned_lens.safetensors", p / "tuned_lens.json"


def save_tuned_lens(lens: TunedLens, path: str | Path) -> Path:
    """Save translator weights (safetensors) + metadata sidecar (JSON)."""
    from safetensors.torch import save_file

    weights_path, meta_path = _resolve_paths(path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.contiguous() for k, v in lens.state_dict().items()},
        str(weights_path),
    )
    meta_path.write_text(json.dumps(lens.meta, indent=2))
    return weights_path


def load_tuned_lens(
    path: str | Path,
    *,
    model: Model | None = None,
) -> TunedLens:
    """Load a tuned lens; validate against *model*'s architecture if given."""
    from safetensors.torch import load_file

    weights_path, meta_path = _resolve_paths(path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"No tuned lens at {weights_path}. Train one with "
            f"model.train_tuned_lens(corpus, save=...)."
        )
    if not meta_path.exists():
        raise InterpkitError(
            f"Tuned lens weights at {weights_path} have no metadata sidecar "
            f"({meta_path.name}) — cannot validate compatibility."
        )
    meta = json.loads(meta_path.read_text())
    lens = TunedLens(
        int(meta["hidden_size"]),
        list(meta["block_paths"]),
        model_id=meta.get("model_id"),
    )
    lens.load_state_dict(load_file(str(weights_path)))
    lens.meta.update(meta)
    if model is not None:
        from interpkit.core.support_matrix import lens_blocks

        block_paths = [b.path for b in lens_blocks(model.arch_info)]
        lens.validate_against(model.arch_info, block_paths)
        lens.to(model._device)
    return lens
