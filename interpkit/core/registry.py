"""Manual annotation registry for custom nn.Module models.

Usage::

    import interpkit

    interpkit.register(
        my_model,
        layers=["blocks.0", "blocks.1", "blocks.2"],
        output_head="head",
    )

    model = interpkit.load(my_model, tokenizer=my_tokenizer)
    model.trace(tensor_a, tensor_b)
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn

_REGISTRY: dict[int, "Registration"] = {}


@dataclass
class Registration:
    layers: list[str] = field(default_factory=list)
    output_head: str | None = None
    attention_modules: list[str] = field(default_factory=list)
    mlp_modules: list[str] = field(default_factory=list)


def register(
    model: nn.Module,
    *,
    layers: list[str] | None = None,
    output_head: str | None = None,
    attention_modules: list[str] | None = None,
    mlp_modules: list[str] | None = None,
) -> None:
    """Annotate a custom ``nn.Module`` so interpkit knows its structure.

    Parameters
    ----------
    model:
        The model instance to annotate.
    layers:
        Ordered list of module names that constitute the repeated layer blocks
        (e.g. ``["blocks.0", "blocks.1", ...]``).
    output_head:
        Module name of the output / LM head.
    attention_modules:
        Module names that should be treated as attention.
    mlp_modules:
        Module names that should be treated as MLPs.
    """
    reg = Registration(
        layers=layers or [],
        output_head=output_head,
        attention_modules=attention_modules or [],
        mlp_modules=mlp_modules or [],
    )
    model_id = id(model)
    _REGISTRY[model_id] = reg

    # Clean up when the model is garbage-collected
    def _cleanup(ref: weakref.ref) -> None:  # noqa: ARG001
        _REGISTRY.pop(model_id, None)

    try:
        weakref.finalize(model, _cleanup, weakref.ref(model))
    except TypeError:
        pass


def get_registration(model: nn.Module) -> Registration | None:
    """Return the registration for *model*, if any."""
    return _REGISTRY.get(id(model))
