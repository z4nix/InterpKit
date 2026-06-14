"""The GUI op contract: one ``OpSpec`` per CLI command.

An op declares a pydantic params model (which both validates requests and
generates the frontend form) and a runner. The server dispatches every op
through one generic endpoint, so adding op N+1 to the GUI is exactly: one
``OpSpec`` here + one result renderer in ``static/js/renderers/``.

Form generation deliberately does *not* ship raw JSON Schema to the
browser — pydantic's ``anyOf``-heavy output would push schema-walking
complexity into vanilla JS. Instead :func:`fields_from_model` flattens the
model into a simple field list the form-builder consumes directly.
"""

from __future__ import annotations

import types
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

# Sidebar order. Keys are OpSpec.category values.
CATEGORIES: list[tuple[str, str]] = [
    ("overview", "Overview"),
    ("analysis", "Analysis"),
    ("generation", "Steering & Generation"),
    ("circuits", "Circuits"),
    ("advanced", "Advanced"),
]


class JobCancelled(Exception):
    """Raised inside a runner when the user cancelled the job."""


class JobContext:
    """Progress/cancel handle passed to every runner.

    Runners for long ops call :meth:`report` at checkpoint boundaries;
    that both surfaces progress to the polling frontend and gives the
    cooperative-cancel flag a place to fire.
    """

    def __init__(self, job: Any, *, model_id: str = "") -> None:
        self._job = job
        self.model_id = model_id

    @property
    def cancelled(self) -> bool:
        return self._job.cancel_event.is_set()

    def report(self, current: int, total: int, message: str = "") -> None:
        if self.cancelled:
            raise JobCancelled()
        self._job.progress = {
            "current": current,
            "total": total,
            "fraction": (current / total) if total else None,
            "message": message,
        }

    def progress_callback(self, current: int, total: int, message: str = "") -> None:
        """Adapter matching the library's ``progress_callback`` kwarg shape."""
        self.report(current, total, message)


@dataclass(frozen=True)
class OpSpec:
    """One GUI operation: params model + runner + presentation metadata."""

    name: str  # matches the CLI command name, e.g. "find-circuit"
    category: str  # one of CATEGORIES keys
    title: str
    description: str  # one friendly sentence shown atop the op panel
    params: type[BaseModel]
    run: Callable[[Any, BaseModel, JobContext], Any]
    long_running: bool = False
    # Key into interpkit.core.support_matrix.SUPPORT_MATRIX ("all"-style ops
    # may still be probed — DeBERTa-v3 gating applies per op key).
    support_key: str | None = None

    def catalog_entry(self) -> dict[str, Any]:
        """JSON description consumed by the frontend (GET /api/ops)."""
        return {
            "name": self.name,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "long_running": self.long_running,
            "fields": fields_from_model(self.params),
        }


def ui(
    *,
    widget: str | None = None,
    placeholder: str | None = None,
    rows: int | None = None,
    show_if: dict[str, Any] | None = None,
    group: str | None = None,
    advanced: bool = False,
) -> dict[str, Any]:
    """Build the ``json_schema_extra`` UI-hint dict for a pydantic Field.

    Widgets the form-builder understands: ``textarea``, ``text``,
    ``number``, ``checkbox``, ``select``, ``path``, ``module-picker``,
    ``layer-select``, ``head-select``. ``group`` clusters fields under a
    sub-heading; ``advanced`` collapses them behind "Advanced options";
    ``show_if`` hides a field until another field has a given value.
    """
    hints: dict[str, Any] = {}
    if widget is not None:
        hints["widget"] = widget
    if placeholder is not None:
        hints["placeholder"] = placeholder
    if rows is not None:
        hints["rows"] = rows
    if show_if is not None:
        hints["show_if"] = show_if
    if group is not None:
        hints["group"] = group
    if advanced:
        hints["advanced"] = True
    return {"x-ui": hints}


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Return ``(inner_type, optional)`` for ``X | None`` annotations."""
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return annotation, False


def _field_type(annotation: Any) -> tuple[str, list[Any] | None]:
    """Map a python annotation to a form field type + optional choices."""
    if typing.get_origin(annotation) is Literal:
        return "enum", list(typing.get_args(annotation))
    if annotation is bool:
        return "boolean", None
    if annotation is int:
        return "integer", None
    if annotation is float:
        return "number", None
    return "string", None


def fields_from_model(model_cls: type[BaseModel]) -> list[dict[str, Any]]:
    """Flatten a pydantic model into the form-builder's field list."""
    fields: list[dict[str, Any]] = []
    for name, info in model_cls.model_fields.items():
        inner, optional = _unwrap_optional(info.annotation)
        ftype, choices = _field_type(inner)
        extra = info.json_schema_extra or {}
        ui_hints = dict(extra.get("x-ui", {})) if isinstance(extra, dict) else {}
        default = info.get_default(call_default_factory=True)
        if default is PydanticUndefined:
            default = None
        fields.append(
            {
                "name": name,
                "label": info.title or name.replace("_", " "),
                "type": ftype,
                "choices": choices,
                "required": info.is_required() and not optional,
                "optional": optional or not info.is_required(),
                "default": default,
                "help": info.description,
                "ui": ui_hints,
            }
        )
    return fields


def lines_or_text(value: str) -> str | list[str]:
    """Interpret a textarea value: one line -> str, many lines -> list[str].

    Mirrors the CLI's single-example flags vs ``--*-file`` line lists with
    a single input surface.
    """
    lines = [ln.strip() for ln in value.splitlines() if ln.strip()]
    if len(lines) <= 1:
        return lines[0] if lines else ""
    return lines


def lines_list(value: str) -> list[str]:
    """Textarea value -> non-empty line list."""
    return [ln.strip() for ln in value.splitlines() if ln.strip()]


def decoded_tokens(model: Any, text: Any) -> list[str] | None:
    """Per-position decoded token strings for axis labels, if tokenizable."""
    tokenizer = getattr(model, "_tokenizer", None)
    if tokenizer is None or not isinstance(text, str):
        return None
    try:
        ids = tokenizer(text)["input_ids"]
        return [tokenizer.decode([i]) for i in ids]
    except Exception:
        return None
