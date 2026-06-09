"""Module-path validation with friendly typo suggestions.

Replaces the opaque HuggingFace ``AttributeError: 'GPT2LMHeadModel' object
has no attribute 'completely'`` that pre-1.0 interpkit propagated when
users typo'd a module path. ``validate_module_path`` raises a clear
``KeyError`` listing the closest matches found via difflib.

Used at the entry of every op that accepts an ``at=`` path:
``activations``, ``patch``, ``ablate``, ``steer``, ``trace``, etc.
"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from interpkit.core.arch import ArchInfo


def validate_module_path(path: str, arch_info: ArchInfo) -> None:
    """Raise :class:`KeyError` if *path* is not a known module of the model.

    The error message includes the closest matches via :func:`difflib.get_close_matches`
    so users can quickly fix typos without grepping the module tree.

    Examples
    --------
    >>> validate_module_path("transformer.h.4.attn", arch)
    # passes silently if the path exists
    >>> validate_module_path("transformr.h.4.attn", arch)
    KeyError: Module path 'transformr.h.4.attn' not found on this model.
        Did you mean: ['transformer.h.4.attn', 'transformer.h.4', 'transformer']?
        See model.arch_info.all_paths() for the full list.
    """
    if not isinstance(path, str):
        raise TypeError(
            f"Module path must be a string, got {type(path).__name__}. "
            f"Pass a dotted module path like 'transformer.h.4.attn' "
            f"(see model.arch_info.all_paths())."
        )
    known = arch_info.all_paths()
    if path in known:
        return
    suggestions = difflib.get_close_matches(path, known, n=3, cutoff=0.5)
    hint = f" Did you mean: {suggestions}?" if suggestions else ""
    raise KeyError(
        f"Module path {path!r} not found on this model.{hint} "
        f"See model.arch_info.all_paths() for the full list."
    )


def validate_position(position: int, seq_len: int, *, op: str | None = None) -> int:
    """Validate a token *position* against *seq_len*; return it unchanged.

    Accepts Python-style negative indices (``-1`` = last token). Raises a
    clear :class:`ValueError` for out-of-range positions instead of letting a
    raw ``IndexError`` ("index 999 is out of bounds for dimension 1 with size
    6") surface from deep inside a tensor index, which is opaque to users.
    """
    if seq_len > 0 and -seq_len <= position < seq_len:
        return position
    where = f"`{op}`: " if op else ""
    raise ValueError(
        f"{where}position {position} is out of range for a {seq_len}-token "
        f"sequence; valid positions are {-seq_len}..{seq_len - 1} "
        f"(negative indexes count from the end, e.g. -1 = last token)."
    )


__all__ = ["validate_module_path", "validate_position"]
