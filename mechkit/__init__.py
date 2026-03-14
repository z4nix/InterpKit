"""mechkit — mech interp for any HuggingFace model."""

from mechkit.core.model import load
from mechkit.core.registry import register
from mechkit.core.tl_compat import list_tl_hooks, to_native_name, to_tl_name


def diff(model_a, model_b, input_data, *, save=None):
    """Compare activations between two models on the same input."""
    from mechkit.ops.diff import run_diff

    return run_diff(model_a, model_b, input_data, save=save)


__all__ = ["load", "register", "diff", "to_tl_name", "to_native_name", "list_tl_hooks"]
