"""Enum-validation helpers for op-method/metric/mode strings.

Pre-1.0 interpkit silently fell back to the default when a user passed
an unknown method/metric/mode (F-018, F-019). With this module, every op
calls :func:`_validate_enum` at the top of its dispatch and raises a
clear ``ValueError`` listing the supported values.

Frozensets of canonical values are defined here so they can be imported
both for validation and for documentation generation.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Canonical enum sets — single source of truth across the codebase
# ---------------------------------------------------------------------------

VALID_METRICS = frozenset({
    "logit_diff",
    "kl_div",
    "target_prob",
    "target_prob_effect",  # added in 1.0 (F-009)
    "l2_prob",
})

VALID_ATTR_METHODS = frozenset({
    "gradient",
    "gradient_x_input",
    "integrated_gradients",
})

VALID_TRACE_MODES = frozenset({
    "module",
    "position",
})

VALID_TRACE_METHODS = frozenset({
    "auto",
    "exhaustive",
    "exhaustive_forced",
    "approximate",
})

VALID_ABLATE_METHODS = frozenset({
    "zero",
    "mean",
    "resample",
})

VALID_FIND_CIRCUIT_METHODS = frozenset({
    "zero",
    "mean",
    "resample",
})

VALID_IG_METHODS = frozenset({
    "riemann_midpoint",
    # Future-proof for adaptive / gauss-legendre additions.
})

VALID_IG_BASELINES = frozenset({
    "pad",
    "zero",
    "mean",
})


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def _validate_enum(value: str, valid: frozenset[str], param_name: str) -> None:
    """Raise :class:`ValueError` with a clear message if *value* is not in *valid*.

    Used at the top of every op that accepts a method / metric / mode
    string parameter. Replaces the pre-1.0 silent-fallback behaviour
    that produced wrong results for typo'd flags.

    Examples
    --------
    >>> _validate_enum("logit_diff", VALID_METRICS, "metric")
    # passes silently
    >>> _validate_enum("logit_dif", VALID_METRICS, "metric")
    ValueError: Unknown metric 'logit_dif'. Must be one of [
        'kl_div', 'l2_prob', 'logit_diff', 'target_prob', 'target_prob_effect'
    ].
    """
    if value not in valid:
        raise ValueError(
            f"Unknown {param_name} {value!r}. Must be one of {sorted(valid)}."
        )


__all__ = [
    "VALID_METRICS",
    "VALID_ATTR_METHODS",
    "VALID_TRACE_MODES",
    "VALID_TRACE_METHODS",
    "VALID_ABLATE_METHODS",
    "VALID_FIND_CIRCUIT_METHODS",
    "VALID_IG_METHODS",
    "VALID_IG_BASELINES",
    "_validate_enum",
]
