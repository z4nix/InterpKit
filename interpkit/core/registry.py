"""Manual annotation registry for custom nn.Module models.

When interpkit's auto-discovery cannot detect your model's structure
(you'll see a warning like "Could not auto-detect layer structure"),
use ``register()`` to tell interpkit where the layers, attention, MLP,
and output head live.

Example — custom GPT-style model::

    import interpkit

    interpkit.register(
        my_model,
        layers=["blocks.0", "blocks.1", "blocks.2"],
        attention_modules=["blocks.0.attn", "blocks.1.attn", "blocks.2.attn"],
        mlp_modules=["blocks.0.ffn", "blocks.1.ffn", "blocks.2.ffn"],
        output_head="head",
    )

    model = interpkit.load(my_model, tokenizer=my_tokenizer)
    model.trace(tensor_a, tensor_b)

To find the right module names, call ``model.inspect()`` after loading
or use ``dict(model.named_modules())`` on your raw ``nn.Module``.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn

_REGISTRY: dict[int, Registration] = {}


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

    Call this **before** ``interpkit.load()`` when auto-discovery fails or
    misidentifies your model's components.

    Parameters
    ----------
    model:
        The model instance to annotate.
    layers:
        Ordered list of module names that constitute the repeated layer blocks
        (e.g. ``["blocks.0", "blocks.1", ...]``).
    output_head:
        Module name of the output / LM head (enables DLA, logit lens, etc.).
    attention_modules:
        Module names that should be treated as attention (enables patching,
        tracing, head-level analysis).
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
