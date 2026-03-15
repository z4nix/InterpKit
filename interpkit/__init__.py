"""interpkit — mech interp for any HuggingFace model."""

from interpkit.core.discovery import ModelArchInfo, ModuleInfo
from interpkit.core.model import Model, load
from interpkit.core.registry import register
from interpkit.core.tl_compat import list_tl_hooks, to_native_name, to_tl_name


def diff(model_a, model_b, input_data, *, save=None):
    """Compare activations between two models on the same input."""
    from interpkit.ops.diff import run_diff

    return run_diff(model_a, model_b, input_data, save=save)


__all__ = [
    "load",
    "Model",
    "ModelArchInfo",
    "ModuleInfo",
    "register",
    "diff",
    "to_tl_name",
    "to_native_name",
    "list_tl_hooks",
]
