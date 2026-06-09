"""interpkit — mech interp for any HuggingFace model."""

from interpkit.core.arch import (
    ArchFamily,
    ArchInfo,
    BlockSpec,
    LayerInfo,
    ModuleInfo,
    resolve_arch,
)
from interpkit.core.exceptions import (
    ArchitectureNotSupported,
    AttentionBackendUnavailable,
    InterpkitError,
    LensPipelineMismatch,
    OperationNotSupportedForArchitecture,
    WrongInputType,
)
from interpkit.core.loader import load, load_module
from interpkit.core.model import Model
from interpkit.core.registry import register
from interpkit.core.tl_compat import (
    list_roundtrippable_hooks,
    list_tl_hooks,
    to_native_name,
    to_tl_name,
)


def diff(model_a, model_b, input_data, *, save=None):
    """Compare activations between two models on the same input."""
    from interpkit.ops.diff import run_diff

    return run_diff(model_a, model_b, input_data, save=save)


__all__ = [
    # Loaders
    "load",
    "load_module",
    "Model",
    # Architecture types
    "ArchInfo",
    "ArchFamily",
    "BlockSpec",
    "resolve_arch",
    # Per-layer structural types
    "LayerInfo",
    "ModuleInfo",
    # Exception types
    "InterpkitError",
    "ArchitectureNotSupported",
    "AttentionBackendUnavailable",
    "LensPipelineMismatch",
    "OperationNotSupportedForArchitecture",
    "WrongInputType",
    # Operations
    "register",
    "diff",
    # TL compat
    "to_tl_name",
    "to_native_name",
    "list_tl_hooks",
    "list_roundtrippable_hooks",
]
