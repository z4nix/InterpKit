"""Universal input loader — text, images, raw tensors."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def prepare_input(
    raw: str | torch.Tensor | Any,
    *,
    tokenizer: Any | None = None,
    image_processor: Any | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor] | torch.Tensor:
    """Normalise a user-provided input into model-ready tensors.

    Dispatch order:
    1. ``torch.Tensor`` → return as-is (moved to *device*).
    2. ``dict`` of tensors → return as-is (moved to *device*).
    3. ``str`` that looks like an image path → load image and preprocess.
    4. ``str`` → tokenize with *tokenizer*.
    """
    if isinstance(raw, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in raw.items()}

    if isinstance(raw, torch.Tensor):
        return raw.to(device)

    if isinstance(raw, str):
        # Image file?
        if _looks_like_image_path(raw):
            return _load_image(raw, image_processor=image_processor, device=device)

        # .pt tensor file?
        if raw.endswith(".pt"):
            try:
                return torch.load(raw, map_location=device, weights_only=True)
            except TypeError:
                return torch.load(raw, map_location=device)

        # Text
        if tokenizer is None:
            raise ValueError(
                f"Cannot tokenize string input — no tokenizer available. "
                f"Pass a tokenizer when loading the model or provide a torch.Tensor directly."
            )
        encoded = tokenizer(raw, return_tensors="pt")
        return {k: v.to(device) for k, v in encoded.items()}

    raise TypeError(f"Unsupported input type: {type(raw).__name__}")


def prepare_pair(
    raw_a: str | torch.Tensor | Any,
    raw_b: str | torch.Tensor | Any,
    *,
    tokenizer: Any | None = None,
    image_processor: Any | None = None,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, torch.Tensor] | torch.Tensor, dict[str, torch.Tensor] | torch.Tensor]:
    """Prepare two inputs for paired operations (patching, tracing).

    For text inputs, both are tokenized together with padding so they
    have the same sequence length — required for activation patching.
    """
    both_text = isinstance(raw_a, str) and isinstance(raw_b, str)
    both_text = both_text and not _looks_like_image_path(raw_a) and not _looks_like_image_path(raw_b)
    both_text = both_text and not raw_a.endswith(".pt") and not raw_b.endswith(".pt")

    if both_text and tokenizer is not None:
        encoded = tokenizer(
            [raw_a, raw_b],
            return_tensors="pt",
            padding=True,
        )
        input_a = {k: v[0:1].to(device) for k, v in encoded.items()}
        input_b = {k: v[1:2].to(device) for k, v in encoded.items()}
        return input_a, input_b

    a = prepare_input(raw_a, tokenizer=tokenizer, image_processor=image_processor, device=device)
    b = prepare_input(raw_b, tokenizer=tokenizer, image_processor=image_processor, device=device)
    return a, b


def read_examples_file(path: str) -> list[str]:
    """Read a text file with one example per line, skipping blank lines."""
    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    return [line.strip() for line in lines if line.strip()]


def _looks_like_image_path(s: str) -> bool:
    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    ext = os.path.splitext(s)[1].lower()
    return ext in _IMAGE_EXTS and Path(s).exists()


def _load_image(
    path: str,
    *,
    image_processor: Any | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor] | torch.Tensor:
    from PIL import Image

    img = Image.open(path).convert("RGB")

    if image_processor is not None:
        processed = image_processor(images=img, return_tensors="pt")
        return {k: v.to(device) for k, v in processed.items()}

    from torchvision import transforms  # type: ignore[import-untyped]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0).to(device)
