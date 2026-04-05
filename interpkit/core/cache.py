"""Activation cache helpers — hashing and device memory management."""

from __future__ import annotations

import hashlib

import torch


def hash_input(model_input: dict[str, torch.Tensor] | torch.Tensor) -> int:
    """Compute a hash of a model input for cache key comparison.

    Uses the raw byte content of tensors to avoid collisions from inputs
    that happen to share the same sum/shape.
    """
    h = hashlib.sha256()
    if isinstance(model_input, dict):
        for k in sorted(model_input.keys()):
            v = model_input[k]
            h.update(k.encode())
            if isinstance(v, torch.Tensor):
                h.update(v.detach().cpu().contiguous().numpy().tobytes())
            else:
                h.update(repr(v).encode())
    else:
        h.update(model_input.detach().cpu().contiguous().numpy().tobytes())
    return int.from_bytes(h.digest()[:8], "little")


def empty_device_cache(device: torch.device | str) -> None:
    """Release cached memory for the given device backend."""
    dev = str(device)
    if dev.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif dev == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
