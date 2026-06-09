"""Custom exception types for interpkit's fail-loud philosophy.

Every silent-fallback or silent-redirect code path in pre-1.0 interpkit
has been replaced with one of these exceptions, raised at the earliest
point where the wrong-result risk surfaces. The error messages always
include either an actionable workaround (override hint) or a list of
supported alternatives.
"""

from __future__ import annotations


class InterpkitError(Exception):
    """Base class for all interpkit-raised exceptions."""


class ArchitectureNotSupported(InterpkitError):
    """Raised when the resolver cannot identify a model's architecture.

    Always carries an actionable ``arch_override`` hint so the user can
    bypass detection.
    """


class ArchitectureSpecMismatch(InterpkitError):
    """Raised when the architecture override / detected paths do not match
    the actual model module tree (path missing, wrong type, etc.)."""


class OperationNotSupportedForArchitecture(InterpkitError):
    """Raised when an op (lens, attention, circuits, ...) is invoked on a
    model family it does not apply to (e.g. attention on a CNN).

    The message lists the supported families and suggests an alternative op.
    """


class LensPipelineMismatch(InterpkitError):
    """Raised when the lens-at-last-block validation contract fails.

    This is the universal correctness safety net for the lens / DLA
    pipeline: even if the resolver picks wrong paths, this assertion
    fires loudly at first use rather than producing silent wrong results.
    """


class AttentionBackendUnavailable(InterpkitError):
    """Raised when ``model.attention()`` cannot obtain real (eager) attention
    weights and refuses to silently return RoPE/ALiBi-less reconstructions.
    """


class WrongInputType(InterpkitError):
    """Raised when an op-supported family received the wrong input type
    (e.g. a text string passed to a vision model).

    This is a category distinct from
    :class:`OperationNotSupportedForArchitecture`: the operation *is*
    supported for the family, but the caller passed an input the model
    cannot consume. The message names the family and the accepted inputs.
    """


class DegenerateMetricGap(InterpkitError):
    """Raised when an effect-ratio metric (logit_diff, target_prob_effect)
    is undefined because the clean/corrupted gap is below the numeric guard.

    Note: most call sites instead return ``float("nan")`` and add a
    ``warnings`` field to the result dict; this exception is reserved for
    cases where NaN propagation is unsafe.
    """


__all__ = [
    "InterpkitError",
    "ArchitectureNotSupported",
    "ArchitectureSpecMismatch",
    "OperationNotSupportedForArchitecture",
    "LensPipelineMismatch",
    "AttentionBackendUnavailable",
    "WrongInputType",
    "DegenerateMetricGap",
]
