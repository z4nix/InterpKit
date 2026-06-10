"""Composable, declarative interventions — the single home for hook *write* plumbing.

:mod:`interpkit.ops._hooks` owns the read-side capture patterns
(:func:`~interpkit.ops._hooks.first_tensor`, ``register_capture_hook``).
This module owns the write side: replacing or modifying a module's
forward output. Before 1.1 every intervening op (``steer``, ``ablate``,
``patch``, ``trace``, ``find_circuit``) carried its own inline hook
closure with subtly-divergent copies of the tensor-vs-tuple writeback
and the F-008 dtype/device cast. Those closures now compile from the
declarative :class:`Intervention` objects defined here, and the same
objects can be applied during multi-token generation via
:meth:`interpkit.Model.generate` / :meth:`interpkit.Model.intervene`.

Position semantics
------------------
``positions`` are **absolute, prompt-indexed** token positions:
generated token *i* sits at position ``prompt_len + i``. During a
single forward pass the absolute index equals the sequence index. During
incremental decoding with a KV cache each decode step presents a
length-1 window; :class:`GenerationContext` maps absolute positions into
the current window (positions outside the window are skipped). With
``positions=None`` an intervention applies to every position of every
forward — matching single-forward op semantics.

Note that an intervened block output at decode step *t* feeds the KV
cache and therefore influences all later steps; positional steering has
downstream effect by design.

Deliberate deferrals (same ledger pattern as ``ops/_hooks.py``):

- ``ops/patch.py``'s head-level path registers *pre*-hooks on the
  attention output projection and performs input surgery (slicing the
  pre-projection activation per head). That is a different contract
  from the output-replacement hooks expressed here; it stays inline
  until per-head nodes land with EAP (roadmap phase 2).
- ``ops/dla.py``'s capture factories remain inline (see
  ``ops/_hooks.py`` docstring).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

# Read-side canonical helper. `ops._hooks` is dependency-free (torch only),
# so this import does not invert the core→ops layering in practice.
from interpkit.ops._hooks import first_tensor

if TYPE_CHECKING:
    from interpkit.core.model import Model

__all__ = [
    "replace_in_output",
    "cast_like",
    "Intervention",
    "SteerIntervention",
    "AblateIntervention",
    "PatchIntervention",
    "FnIntervention",
    "CaptureProbe",
    "GenerationContext",
    "apply_interventions",
    "track_positions",
]


# ---------------------------------------------------------------------------
# Canonical write-side helpers
# ---------------------------------------------------------------------------


def replace_in_output(output: Any, new: torch.Tensor) -> Any:
    """Write *new* back into a forward-hook ``output``, preserving its shape.

    The single canonical writeback (mirror of :func:`first_tensor`):

    - ``Tensor`` → ``new``.
    - ``tuple`` / ``list`` with a leading ``Tensor`` → ``(new, *rest)``
      (tail elements — e.g. ``present_key_value`` caches — preserved).
    - Anything else → *output* unchanged.
    """
    if isinstance(output, torch.Tensor):
        return new
    if isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(
        output[0], torch.Tensor,
    ):
        return (new,) + tuple(output[1:])
    return output


def cast_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """F-008 canonical cast: move *src* to *ref*'s device **and** dtype.

    Surgery frequently happens in fp32 while the model runs fp16/bf16;
    casting only the device (the pre-1.0 pattern) produced
    Float-vs-Half mismatches inside the next module.
    """
    return src.to(device=ref.device, dtype=ref.dtype)


def _seq_len_of(t: torch.Tensor) -> int:
    """Sequence length of a (B, S, H) or (S, H) activation."""
    if t.dim() >= 2:
        return int(t.shape[-2])
    return 1


def _local_positions(
    positions: Sequence[int],
    t: torch.Tensor,
    ctx: GenerationContext | None,
) -> list[int]:
    """Translate absolute *positions* into indices of the current window.

    Without a :class:`GenerationContext` the window starts at 0 (single
    forward). Out-of-window positions are skipped — the same silent
    bounds behaviour ``ops/patch.py`` always had for ``p >= seq_len``.
    """
    offset = ctx.offset if ctx is not None else 0
    seq_len = _seq_len_of(t)
    out: list[int] = []
    for p in positions:
        local = p - offset
        if 0 <= local < seq_len:
            out.append(local)
    return out


def _assign_at_positions(
    target: torch.Tensor, source: torch.Tensor, local_positions: list[int],
) -> torch.Tensor:
    """Copy *source* rows into *target* at *local_positions* (in place).

    Handles the (B, S, H) and (S, H) layouts exactly as ``ops/patch.py``'s
    position hook did.
    """
    for p in local_positions:
        if target.dim() == 3:
            target[:, p, :] = source[:, p, :] if source.dim() == 3 else source
        elif target.dim() == 2:
            target[p, :] = source[p, :] if source.dim() == 2 else source
    return target


# ---------------------------------------------------------------------------
# Generation position tracking
# ---------------------------------------------------------------------------


class GenerationContext:
    """Tracks absolute token positions across incremental decoding.

    ``offset`` is the absolute position of the first token in the
    current forward window; ``step`` is ``-1`` during prefill and
    ``0, 1, …`` for subsequent decode steps. Advanced once per forward
    by the pre-hook installed via :func:`track_positions`.
    """

    def __init__(self, prompt_len: int = 0) -> None:
        self.prompt_len = prompt_len
        self.offset = 0
        self.step = -1
        self._total = 0
        self._calls = 0

    def advance(self, seq_len: int) -> None:
        new_offset = self._total
        if new_offset < self.offset:
            raise RuntimeError(
                "GenerationContext offset went backwards — the model re-fed "
                "earlier tokens (beam search or assistant prefill?). "
                "Intervention position tracking supports greedy/sampling "
                "generation with num_beams=1 only."
            )
        self.offset = new_offset
        self.step = self._calls - 1  # first forward (prefill) → -1
        self._total += seq_len
        self._calls += 1


@contextmanager
def track_positions(
    model_module: nn.Module,
    ctx: GenerationContext,
    *,
    embed_module: nn.Module | None = None,
) -> Iterator[None]:
    """Advance *ctx* once per forward pass of *model_module*.

    Prefers a pre-hook on the embedding module (fires exactly once per
    forward, sees the raw ``input_ids``); falls back to a kwargs-aware
    pre-hook on the top-level module.
    """

    def _seq_len_from_ids(ids: Any) -> int | None:
        if isinstance(ids, torch.Tensor) and ids.dim() >= 1:
            if ids.dtype in (torch.long, torch.int, torch.int32, torch.int64):
                return int(ids.shape[-1])
            if ids.dim() >= 2:
                return int(ids.shape[-2])
        return None

    if embed_module is not None:
        def _embed_pre_hook(_mod: nn.Module, inputs: tuple[Any, ...]) -> None:
            ids = inputs[0] if inputs else None
            seq_len = _seq_len_from_ids(ids)
            if seq_len is not None:
                ctx.advance(seq_len)

        handle = embed_module.register_forward_pre_hook(_embed_pre_hook)
    else:
        def _top_pre_hook(
            _mod: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any],
        ) -> None:
            ids = kwargs.get("input_ids")
            if ids is None and args:
                ids = args[0]
            seq_len = _seq_len_from_ids(ids)
            if seq_len is not None:
                ctx.advance(seq_len)

        handle = model_module.register_forward_pre_hook(
            _top_pre_hook, with_kwargs=True,
        )
    try:
        yield
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# Declarative interventions
# ---------------------------------------------------------------------------


# ``eq=False``: tensor-valued fields make generated ``__eq__`` ambiguous.
@dataclass(frozen=True, eq=False)
class Intervention:
    """A declarative modification of one module's forward output.

    Subclasses implement :meth:`build_hook`, which compiles the
    intervention into a standard forward-hook callable. Every hook
    follows the same pipeline: ``first_tensor(output)`` → compute
    replacement → :func:`cast_like` → :func:`replace_in_output`.
    """

    at: str
    positions: tuple[int, ...] | None = field(default=None, kw_only=True)

    def build_hook(
        self, ctx: GenerationContext | None = None,
    ) -> Callable[[nn.Module, Any, Any], Any]:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        """JSON-safe summary for result dicts (tensors elided)."""
        out: dict[str, Any] = {"type": _SPEC_NAMES.get(type(self).__name__, type(self).__name__)}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, torch.Tensor):
                out[f.name] = f"<tensor shape={tuple(value.shape)}>"
            elif isinstance(value, dict) or callable(value):
                continue
            else:
                out[f.name] = value
        return out


_SPEC_NAMES = {
    "SteerIntervention": "steer",
    "AblateIntervention": "ablate",
    "PatchIntervention": "patch",
    "FnIntervention": "fn",
    "CaptureProbe": "capture",
}


@dataclass(frozen=True, eq=False)
class SteerIntervention(Intervention):
    """Add ``scale * vector`` to the module output (steering)."""

    vector: torch.Tensor = field(default=None)  # type: ignore[assignment]
    scale: float = 2.0

    def __post_init__(self) -> None:
        if self.vector is None:
            raise ValueError("SteerIntervention requires a `vector` tensor.")

    def build_hook(
        self, ctx: GenerationContext | None = None,
    ) -> Callable[[nn.Module, Any, Any], Any]:
        def hook(_mod: nn.Module, _inp: Any, output: Any) -> Any:
            t = first_tensor(output)
            if t is None:
                return output
            if t.shape[-1] != self.vector.shape[-1]:
                raise ValueError(
                    f"Steering vector dimension ({self.vector.shape[-1]}) does not match "
                    f"module output dimension ({t.shape[-1]}) at '{self.at}'."
                )
            vec = cast_like(self.vector, t) * self.scale
            if self.positions is None:
                new = t + vec
            else:
                local = _local_positions(self.positions, t, ctx)
                if not local:
                    return output
                new = t.clone()
                for p in local:
                    if new.dim() == 3:
                        new[:, p, :] = new[:, p, :] + vec
                    elif new.dim() == 2:
                        new[p, :] = new[p, :] + vec
            return replace_in_output(output, new)

        return hook


@dataclass(frozen=True, eq=False)
class AblateIntervention(Intervention):
    """Replace the module output with zeros / its sequence mean / a reference."""

    method: str = "zero"
    replacement: torch.Tensor | None = None

    def __post_init__(self) -> None:
        from interpkit.core.enums import VALID_ABLATE_METHODS, _validate_enum

        _validate_enum(self.method, VALID_ABLATE_METHODS, "method")

    def build_hook(
        self, ctx: GenerationContext | None = None,
    ) -> Callable[[nn.Module, Any, Any], Any]:
        def hook(_mod: nn.Module, _inp: Any, output: Any) -> Any:
            t = first_tensor(output)
            if t is None:
                return output
            if self.method == "zero":
                repl = torch.zeros_like(t)
            elif self.method == "mean":
                if t.dim() >= 3:
                    repl = t.mean(dim=-2, keepdim=True).expand_as(t)
                else:
                    repl = t.mean(dim=0, keepdim=True).expand_as(t)
            else:  # "resample"
                repl = (
                    cast_like(self.replacement, t)
                    if self.replacement is not None
                    else torch.zeros_like(t)
                )
            if self.positions is not None:
                local = _local_positions(self.positions, t, ctx)
                if not local:
                    return output
                new = _assign_at_positions(t.clone(), repl, local)
            else:
                new = repl
            return replace_in_output(output, new)

        return hook


@dataclass(frozen=True, eq=False)
class PatchIntervention(Intervention):
    """Replace the module output with *source* (a cached clean activation)."""

    source: torch.Tensor = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.source is None:
            raise ValueError("PatchIntervention requires a `source` tensor.")

    def build_hook(
        self, ctx: GenerationContext | None = None,
    ) -> Callable[[nn.Module, Any, Any], Any]:
        def hook(_mod: nn.Module, _inp: Any, output: Any) -> Any:
            t = first_tensor(output)
            if t is None:
                return output
            src = cast_like(self.source, t)
            if self.positions is None:
                new = src
            else:
                local = _local_positions(self.positions, t, ctx)
                if not local:
                    return output
                new = _assign_at_positions(t.clone(), src, local)
            return replace_in_output(output, new)

        return hook


@dataclass(frozen=True, eq=False)
class FnIntervention(Intervention):
    """Escape hatch: arbitrary ``fn(tensor, ctx) -> tensor`` on the output."""

    fn: Callable[[torch.Tensor, GenerationContext | None], torch.Tensor] = field(
        default=None,  # type: ignore[assignment]
    )

    def __post_init__(self) -> None:
        if self.fn is None:
            raise ValueError("FnIntervention requires an `fn` callable.")
        if self.positions is not None:
            raise ValueError(
                "FnIntervention does not take `positions` — slice inside `fn` "
                "(it receives the GenerationContext for offset translation)."
            )

    def build_hook(
        self, ctx: GenerationContext | None = None,
    ) -> Callable[[nn.Module, Any, Any], Any]:
        def hook(_mod: nn.Module, _inp: Any, output: Any) -> Any:
            t = first_tensor(output)
            if t is None:
                return output
            new = self.fn(t, ctx)
            if not isinstance(new, torch.Tensor):
                return output
            return replace_in_output(output, cast_like(new, t))

        return hook


@dataclass(frozen=True, eq=False)
class CaptureProbe(Intervention):
    """Read-only capture of the module output into ``store[key]``."""

    store: dict[str, torch.Tensor] = field(default=None)  # type: ignore[assignment]
    key: str = ""
    clone: bool = True
    detach: bool = True

    def __post_init__(self) -> None:
        if self.store is None:
            raise ValueError("CaptureProbe requires a `store` dict.")
        if self.positions is not None:
            raise ValueError("CaptureProbe captures the full output; `positions` is unsupported.")

    def build_hook(
        self, ctx: GenerationContext | None = None,
    ) -> Callable[[nn.Module, Any, Any], Any]:
        def hook(_mod: nn.Module, _inp: Any, output: Any) -> None:
            t = first_tensor(output)
            if t is None:
                return
            if self.detach:
                t = t.detach()
            if self.clone:
                t = t.clone()
            self.store[self.key or self.at] = t

        return hook


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


@contextmanager
def apply_interventions(
    model: Model,
    interventions: Sequence[Intervention],
    *,
    ctx: GenerationContext | None = None,
) -> Iterator[None]:
    """Register hooks for *interventions* on *model*; remove all on exit.

    Module paths are validated up-front (F-022) so a typo'd ``at`` fails
    with a friendly ``KeyError`` before any hook is registered.
    """
    from interpkit.core.paths import get_module, validate_module_path

    arch = getattr(model, "arch_info", None)
    handles: list[torch.utils.hooks.RemovableHandle] = []
    try:
        for iv in interventions:
            if arch is not None:
                validate_module_path(iv.at, arch)
            module = get_module(model._model, iv.at)
            handles.append(module.register_forward_hook(iv.build_hook(ctx)))
        yield
    finally:
        for h in handles:
            h.remove()
