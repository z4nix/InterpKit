"""GUI op registry: every CLI operation as an OpSpec.

The registry is the single source of truth the server dispatches from and
the frontend builds its sidebar + forms from. A parity test asserts that
every CLI command (except ``gui`` itself) has an entry here.
"""

from __future__ import annotations

from typing import Any

from interpkit.gui.ops import advanced, analysis, circuits, generation, overview
from interpkit.gui.ops.base import CATEGORIES, JobCancelled, JobContext, OpSpec

OP_REGISTRY: dict[str, OpSpec] = {
    spec.name: spec
    for module in (overview, analysis, generation, circuits, advanced)
    for spec in module.SPECS
}


def catalog() -> dict[str, Any]:
    """Payload of ``GET /api/ops``."""
    return {
        "categories": [{"id": cid, "label": label} for cid, label in CATEGORIES],
        "ops": [spec.catalog_entry() for spec in OP_REGISTRY.values()],
    }


__all__ = ["OP_REGISTRY", "catalog", "OpSpec", "JobContext", "JobCancelled", "CATEGORIES"]
