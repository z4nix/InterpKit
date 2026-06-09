"""Hook helpers shared by ops + the residual schema that capture forward outputs.

The single source of truth for the forward-hook capture patterns that were
previously copied inline (with subtle variance: some checked
``len(output) > 0``, some asserted ``isinstance(output[0], Tensor)`` without
it, some routed shared-layer per-call indexing inline).

Adopted by :mod:`interpkit.core.arch.residual` and
:mod:`interpkit.ops.find_circuit`. Some op call sites intentionally remain
inline where these helpers do not (yet) express their exact contract — most
notably ``dla.py``'s capture factories, which cast to ``float()`` (a dtype
contract these helpers deliberately leave to the caller, see NR-001) and
embed the per-call counter mid-key (``"{path}#{i}::attn"`` rather than the
trailing ``"{prefix}#{i}"`` this module produces). Extending the helpers to
cover those is a follow-up; until then those sites stay as-is rather than be
force-fit.

Four helpers:

- :func:`first_tensor` extracts the leading ``Tensor`` from a forward
  hook's ``output`` argument across the supported shapes (``Tensor``,
  ``tuple[Tensor, ...]``, ``list[Tensor, ...]``, ``None``).
- :func:`register_capture_hook` wires up the standard one-shot
  capture pattern (``store[key] = first_tensor(output).detach()``).
- :func:`register_per_call_hook` wires up per-call indexing for
  shared-weight modules (ALBERT and similar): each invocation of the
  same physical block writes to a different ``f"{key_prefix}#{counter}"``
  key.
- :func:`eager_attention_forward` is a context manager wrapping the
  ``output_attentions=True`` + ``_attn_implementation="eager"``
  save/restore boilerplate previously inlined at
  ``attention.py:138-182``.

All helpers handle dtype-agnostic capture (caller decides whether to
``.float()`` the captured tensor) and explicit detach (no caller leaks
the autograd graph through a hook closure).
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn

__all__ = [
    "first_tensor",
    "register_capture_hook",
    "register_per_call_hook",
    "eager_attention_forward",
]


def first_tensor(output: Any) -> torch.Tensor | None:
    """Extract the leading :class:`torch.Tensor` from a forward-hook output.

    Forward hooks receive `output` in one of four shapes:

    - ``Tensor`` — return as-is.
    - ``tuple[Tensor, ...]`` or ``list[Tensor, ...]`` — return the first
      element if it is a ``Tensor``.
    - ``tuple[None, ...]`` / empty container / anything else — return
      ``None``.

    Replaces the ~18 inline copies of the ``output if isinstance(output,
    Tensor) else output[0] if ...`` pattern across the ops. The inline
    copies had subtle variance: ``circuits.py`` checked
    ``isinstance(output[0], Tensor)`` without a length guard,
    ``find_circuit.py`` added ``len(output) > 0`` first, ``trace.py``
    used a different control-flow shape entirely.

    This helper is the single canonical implementation. All op files
    migrate to call it directly.
    """
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and len(output) > 0:
        head = output[0]
        if isinstance(head, torch.Tensor):
            return head
    return None


def register_capture_hook(
    module: nn.Module,
    store: dict[str, torch.Tensor],
    key: str,
    *,
    clone: bool = True,
    detach: bool = True,
) -> torch.utils.hooks.RemovableHandle:
    """Register a forward hook that captures *module*'s output into ``store[key]``.

    Standard one-shot capture pattern used by every op that needs to
    grab a layer / submodule output during a forward pass. Replaces
    inline ``def hook(_m, _i, output): store[key] = output.detach()``
    factories at every call site.

    Parameters
    ----------
    module:
        The submodule to hook.
    store:
        Destination dict; ``first_tensor(output)`` is written under
        *key* on every forward call. Non-tensor outputs are dropped.
    key:
        Storage key in *store*.
    clone:
        If True (default), the captured tensor is ``.clone()`` ed so
        downstream code can safely mutate it without affecting the
        original activation. Set to False to skip the clone when the
        caller knows it will not mutate.
    detach:
        If True (default), the captured tensor is ``.detach()`` ed,
        breaking the autograd graph. Set to False to retain the graph
        (needed by IG-style gradient-flow ops).

    Returns
    -------
    The :class:`torch.utils.hooks.RemovableHandle` returned by
    ``module.register_forward_hook``. Caller is responsible for
    ``handle.remove()`` in a ``finally`` block.
    """
    def hook(_mod: nn.Module, _inp: Any, output: Any) -> None:
        t = first_tensor(output)
        if t is None:
            return
        if detach:
            t = t.detach()
        if clone:
            t = t.clone()
        store[key] = t

    return module.register_forward_hook(hook)


def register_per_call_hook(
    module: nn.Module,
    store: dict[str, torch.Tensor],
    key_prefix: str,
    *,
    clone: bool = True,
    detach: bool = True,
) -> torch.utils.hooks.RemovableHandle:
    """Register a forward hook that indexes each invocation of *module* separately.

    For shared-weight architectures (ALBERT and similar with
    ``config.num_hidden_groups=1, num_hidden_layers>1``), the same
    physical block module is invoked N times per forward pass. A
    regular :func:`register_capture_hook` would overwrite the previous
    invocation's capture; this helper instead writes to
    ``f"{key_prefix}#{i}"`` where ``i`` is the per-instance call
    counter, preserving every invocation's output.

    For non-shared models the counter still advances but each call
    fires at most once per forward, so the dict ends up with a single
    ``f"{key_prefix}#0"`` entry — callers reading the dict back can
    handle both cases uniformly by globbing on the prefix.

    Note: ``dla.py``'s shared-weight factories are NOT yet migrated to this
    helper — they cast captures to ``float()`` and place the counter mid-key
    (``"{path}#{i}::attn"``), neither of which this helper expresses. See the
    module docstring.

    Returns the removable handle for ``module.register_forward_hook``.
    """
    counter = [0]  # mutable closure cell

    def hook(_mod: nn.Module, _inp: Any, output: Any) -> None:
        t = first_tensor(output)
        if t is None:
            return
        if detach:
            t = t.detach()
        if clone:
            t = t.clone()
        store[f"{key_prefix}#{counter[0]}"] = t
        counter[0] += 1

    return module.register_forward_hook(hook)


@contextmanager
def eager_attention_forward(model: nn.Module) -> Iterator[None]:
    """Context manager that flips ``model.config`` to eager attention.

    Wraps the ``output_attentions=True`` + ``_attn_implementation="eager"``
    save/restore boilerplate inlined at ``attention.py:138-182``. The
    contract:

    - On enter: save current ``output_attentions`` and
      ``_attn_implementation`` config values, then set both to
      ``True`` / ``"eager"`` respectively.
    - On exit: restore (or delete, if the attribute was unset before).

    Caller is still responsible for passing ``output_attentions=True``
    and ``return_dict=True`` as forward kwargs — modern transformers
    (5.x) ignore the config attribute for T5/BART encoder-decoders.
    """
    config = getattr(model, "config", None)
    if config is None:
        # No config to flip — yield unconditionally so callers can
        # still try the forward (they will get a friendly error from
        # the downstream attention op if eager is unsupported).
        yield
        return

    old_output_attn = getattr(config, "output_attentions", None)
    old_attn_impl = getattr(config, "_attn_implementation", None)

    try:
        # Order matters in transformers 5.x: set the implementation
        # first, then turn on output_attentions. The setter inspects
        # the attn implementation and raises if it's still SDPA.
        config._attn_implementation = "eager"
        config.output_attentions = True
        yield
    finally:
        if old_output_attn is None:
            try:
                del config.output_attentions
            except AttributeError:
                pass
        else:
            config.output_attentions = old_output_attn
        if old_attn_impl is None:
            try:
                del config._attn_implementation
            except AttributeError:
                pass
        else:
            config._attn_implementation = old_attn_impl
