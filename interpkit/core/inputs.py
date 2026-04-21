"""Universal input loader — text, images, raw tensors, chat messages."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import torch

NO_CHAT_TEMPLATE_MSG = (
    "Tokenizer for this model has no chat template — pass a plain string instead, "
    "or load a chat/instruct variant of the model."
)


MAX_LEADING_SPACE_WARNINGS = 4


def warn_if_leading_space_better(
    tokenizer: Any | None,
    text: Any,
    *,
    op_label: str,
    role: str,
    warned_count: list[int],
    max_warnings: int = MAX_LEADING_SPACE_WARNINGS,
    console: Any | None = None,
) -> None:
    """Emit a one-line warning when *text* tokenises differently with a leading space.

    Many LM tokenisers (BPE / GPT-2 / Llama) treat ``"love"`` and
    ``" love"`` as different vocabulary entries.  Single-word inputs
    without a leading space often map to a token the model rarely sees
    in normal text, which weakens steering / contrastive directions.
    This helper detects single-token leading-space variants and surfaces
    a yellow tip; it is a no-op for tensor / list inputs, missing
    tokenizers, empty strings, or strings that already begin with
    whitespace.

    Parameters
    ----------
    tokenizer:
        The tokenizer to consult.  No-op when ``None``.
    text:
        Candidate input.  Only ``str`` inputs are checked.
    op_label:
        Short prefix such as ``"steer"`` or ``"features"`` shown in the
        warning so users can identify which operation triggered it.
    role:
        Free-form role descriptor (e.g. ``"positive"``, ``"negative"``)
        appended to the warning so users can identify which contrast
        term is suspect.
    warned_count:
        Single-element mutable counter shared across one operation
        invocation.  Caps the total number of warnings emitted at
        *max_warnings* to avoid spamming the console for very long
        input lists.
    max_warnings:
        Upper bound on warnings per call.  Defaults to
        :data:`MAX_LEADING_SPACE_WARNINGS`.
    console:
        Optional :class:`rich.console.Console` instance to print
        through.  Defaults to a fresh ``Console()``.  Pass an op's
        module-level ``console`` so test fixtures that monkeypatch
        ``steer.console.print`` (or similar) still observe the warning.
    """
    if warned_count[0] >= max_warnings:
        return
    if not isinstance(text, str):
        return
    if tokenizer is None:
        return
    if not text or text.startswith((" ", "\t", "\n")):
        return

    try:
        ids_plain = tokenizer.encode(text, add_special_tokens=False)
        ids_spaced = tokenizer.encode(" " + text, add_special_tokens=False)
    except (TypeError, ValueError, RuntimeError):
        return

    if not ids_plain or not ids_spaced:
        return
    if ids_plain == ids_spaced:
        return
    if len(ids_spaced) != 1:
        return

    if console is None:
        from rich.console import Console as _Console

        console = _Console()

    console.print(
        f"  [yellow]{op_label}:[/yellow] {role} input {text!r} tokenizes to "
        f"{len(ids_plain)} token(s) {ids_plain}, but "
        f"{(' ' + text)!r} is a single token {ids_spaced}. "
        f"Consider using {(' ' + text)!r} for a stronger contrast "
        f"(BPE leading-space convention)."
    )
    warned_count[0] += 1


def _is_message_list(raw: Any) -> bool:
    """Return True iff *raw* is a non-empty list of chat-message dicts.

    A chat-message dict has at minimum string ``role`` and string ``content``
    keys.  Extra keys are allowed for forward-compatibility with newer
    chat-template formats (tool calls, attachments, ...).
    """
    if not isinstance(raw, list) or not raw:
        return False
    for entry in raw:
        if not isinstance(entry, dict):
            return False
        role = entry.get("role")
        content = entry.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            return False
    return True


def normalize_input_group(raw: Any) -> list[Any]:
    """Normalise a steer/contrast/circuit input into a list of model inputs.

    Operations like :func:`interpkit.ops.steer.run_steer_vector` accept either
    a single example or a list of examples for averaging.  Naive
    ``isinstance(raw, list)`` disambiguation is wrong because a chat-message
    list (``[{"role": "user", "content": "..."}]``) is *also* a Python
    ``list`` — treating it as "one chat per dict" silently breaks chat-aware
    steering / circuits with the cryptic
    ``ValueError: You must specify exactly one of input_ids or inputs_embeds``.

    Rules:

    * ``None``                          → ``[None]``  (defer rejection to caller)
    * ``str`` / ``torch.Tensor`` / dict → ``[raw]``
    * Chat-message list                 → ``[raw]``  (one chat = one example)
    * Any other ``list``                → ``raw``    (already a batch)
    * Anything else                     → ``[raw]``  (let downstream raise)

    Examples
    --------
    >>> normalize_input_group("hi")
    ['hi']
    >>> normalize_input_group([{"role": "user", "content": "hi"}])
    [[{'role': 'user', 'content': 'hi'}]]
    >>> normalize_input_group(["one", "two"])
    ['one', 'two']
    """
    if isinstance(raw, list):
        if _is_message_list(raw):
            return [raw]
        return raw
    return [raw]


def _apply_chat_template(
    messages: list[dict[str, Any]],
    *,
    tokenizer: Any,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Apply *tokenizer*'s chat template and return ``{input_ids, attention_mask}``.

    Raises :class:`ValueError` when the tokenizer has no chat template
    configured.
    """
    template = getattr(tokenizer, "chat_template", None)
    if template is None and not getattr(tokenizer, "default_chat_template", None):
        raise ValueError(NO_CHAT_TEMPLATE_MSG)

    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    except TypeError:
        ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        encoded = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    if isinstance(encoded, torch.Tensor):
        encoded = {"input_ids": encoded, "attention_mask": torch.ones_like(encoded)}

    if "attention_mask" not in encoded:
        encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])

    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in encoded.items()
    }


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
    3. ``list[dict]`` shaped like chat messages → apply the tokenizer's
       chat template (with ``add_generation_prompt=True``).
    4. ``str`` that looks like an image path → load image and preprocess.
    5. ``str`` → tokenize with *tokenizer*.
    """
    if isinstance(raw, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in raw.items()}

    if isinstance(raw, torch.Tensor):
        return raw.to(device)

    if _is_message_list(raw):
        if tokenizer is None:
            raise ValueError(
                "Cannot apply chat template — no tokenizer available. "
                "Pass a tokenizer when loading the model."
            )
        return _apply_chat_template(
            cast("list[dict[str, Any]]", raw),
            tokenizer=tokenizer,
            device=device,
        )

    if isinstance(raw, str):
        # Image file?
        if _looks_like_image_path(raw):
            return _load_image(raw, image_processor=image_processor, device=device)

        # .pt tensor file?
        if raw.endswith(".pt"):
            try:
                loaded: dict[str, torch.Tensor] | torch.Tensor = torch.load(raw, map_location=device, weights_only=True)
                return loaded
            except TypeError:
                loaded = torch.load(raw, map_location=device)
                return loaded

        # Text
        if tokenizer is None:
            raise ValueError(
                "Cannot tokenize string input — no tokenizer available. "
                "Pass a tokenizer when loading the model or provide a torch.Tensor directly."
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
    Chat-message lists are templated independently and then padded to
    a common length so the same paired-op math keeps working.
    """
    if (
        isinstance(raw_a, str)
        and isinstance(raw_b, str)
        and not _looks_like_image_path(raw_a)
        and not _looks_like_image_path(raw_b)
        and not raw_a.endswith(".pt")
        and not raw_b.endswith(".pt")
        and tokenizer is not None
    ):
        encoded = tokenizer(
            [raw_a, raw_b],
            return_tensors="pt",
            padding=True,
        )
        input_a = {k: v[0:1].to(device) for k, v in encoded.items()}
        input_b = {k: v[1:2].to(device) for k, v in encoded.items()}
        return input_a, input_b

    if _is_message_list(raw_a) and _is_message_list(raw_b) and tokenizer is not None:
        templated_a = _apply_chat_template(
            cast("list[dict[str, Any]]", raw_a), tokenizer=tokenizer, device="cpu",
        )
        templated_b = _apply_chat_template(
            cast("list[dict[str, Any]]", raw_b), tokenizer=tokenizer, device="cpu",
        )
        padded = _pad_to_match(
            templated_a, templated_b, tokenizer=tokenizer, device=device,
        )
        return padded

    a = prepare_input(raw_a, tokenizer=tokenizer, image_processor=image_processor, device=device)
    b = prepare_input(raw_b, tokenizer=tokenizer, image_processor=image_processor, device=device)
    return a, b


def _pad_to_match(
    a: dict[str, torch.Tensor],
    b: dict[str, torch.Tensor],
    *,
    tokenizer: Any,
    device: torch.device | str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Right-pad two ``{input_ids, attention_mask}`` dicts to the same length.

    Uses ``tokenizer.pad_token_id`` (falling back to ``eos_token_id`` then
    ``0``).  Both returned tensors are moved to *device*.
    """
    pad_id = (
        getattr(tokenizer, "pad_token_id", None)
        or getattr(tokenizer, "eos_token_id", None)
        or 0
    )

    ids_a = a["input_ids"]
    ids_b = b["input_ids"]
    mask_a = a.get("attention_mask", torch.ones_like(ids_a))
    mask_b = b.get("attention_mask", torch.ones_like(ids_b))

    target = max(ids_a.shape[-1], ids_b.shape[-1])

    def _pad(ids: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gap = target - ids.shape[-1]
        if gap == 0:
            return ids, mask
        pad_ids = torch.full(
            (*ids.shape[:-1], gap), pad_id, dtype=ids.dtype, device=ids.device,
        )
        pad_mask = torch.zeros(
            (*mask.shape[:-1], gap), dtype=mask.dtype, device=mask.device,
        )
        return torch.cat([ids, pad_ids], dim=-1), torch.cat([mask, pad_mask], dim=-1)

    ids_a, mask_a = _pad(ids_a, mask_a)
    ids_b, mask_b = _pad(ids_b, mask_b)

    out_a = {"input_ids": ids_a.to(device), "attention_mask": mask_a.to(device)}
    out_b = {"input_ids": ids_b.to(device), "attention_mask": mask_b.to(device)}
    return out_a, out_b


def read_examples_file(path: str) -> list[str]:
    """Read a text file with one example per line, skipping blank lines."""
    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    return [line.strip() for line in lines if line.strip()]


def _looks_like_image_path(s: str) -> bool:
    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    ext = os.path.splitext(s)[1].lower()
    return ext in _IMAGE_EXTS and Path(s).exists()


TORCHVISION_REQUIRED_MSG = (
    "Loading raw image files without an HF image_processor requires torchvision. "
    "Install it with 'pip install interpkit[vision]', or pass an "
    "image_processor=... when calling interpkit.load()."
)


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

    try:
        from torchvision import transforms  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(TORCHVISION_REQUIRED_MSG) from exc

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    result: torch.Tensor = transform(img).unsqueeze(0).to(device)
    return result
