"""Local web GUI for interpkit — FastAPI backend + no-build frontend.

The GUI is an optional surface on top of the library: a local server
(``interpkit gui``) that loads models into named sessions and runs every
CLI operation through a job queue, rendering results natively in the
browser. Web dependencies are isolated behind the ``[gui]`` extra so the
core library stays lean::

    pip install "interpkit[gui]"
    interpkit gui

``create_app`` / ``serve`` are imported lazily so that importing
``interpkit.gui`` (e.g. for the CLI's friendly missing-extra message)
never requires fastapi/uvicorn at module-import time.
"""

from __future__ import annotations

from typing import Any

__all__ = ["create_app", "serve", "require_gui_deps"]


def require_gui_deps() -> None:
    """Raise a clear, actionable ImportError when the [gui] extra is missing."""
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The interpkit GUI requires the optional [gui] extra. "
            'Install it with: pip install "interpkit[gui]"'
        ) from exc


def __getattr__(name: str) -> Any:
    if name in ("create_app", "serve"):
        require_gui_deps()
        from interpkit.gui import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
