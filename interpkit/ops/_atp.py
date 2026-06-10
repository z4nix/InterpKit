"""Backwards-compat shim — the AtP implementation moved to :mod:`interpkit.ops.atp`.

AtP shipped in 1.0 as trace's private shortlister at this path; it was
promoted to a public op in phase 2 (same re-export pattern as
``ops/patch._get_module``). Importers of ``compute_atp_scores`` from
here keep working.
"""

from __future__ import annotations

from interpkit.ops.atp import compute_atp_scores

__all__ = ["compute_atp_scores"]
