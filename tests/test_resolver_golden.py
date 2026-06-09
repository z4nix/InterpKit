"""Workstream D1 — frozen ArchInfo golden snapshots.

Loads every in-scope audit2 model, runs the resolver, and diffs the resolved
ArchInfo against a committed JSON snapshot. This is the safety net that makes
the Workstream D resolver split behavior-preserving: any unintended change to
a resolved path / family / flag fails here.

The snapshot records resolution *outcomes* (paths + family + flags + block
lists). The ``pre_head_path`` / ``project_out_path`` fields are the observable
result of the pre-head fallback, so a Workstream E change that picks a
different module shows up as a diff (the "provenance" the plan asks for: a
silent resolution change becomes a loud snapshot diff).

Regenerate after an intended change with::

    INTERPKIT_REGEN_GOLDEN=1 pytest tests/test_resolver_golden.py

Models that cannot be loaded offline are skipped (so a no-cache CI degrades
gracefully rather than failing).
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

import interpkit
from audit2 import _models

GOLDEN_DIR = pathlib.Path(__file__).parent / "golden" / "archinfo"
REGEN = os.environ.get("INTERPKIT_REGEN_GOLDEN") == "1"

ALL_MODEL_IDS = [m.model_id for m in _models.ALL_MODELS]


def _safe_id(model_id: str) -> str:
    return model_id.replace("/", "__")


def _snapshot(arch) -> dict:
    """Stable, JSON-serialisable view of the resolved architecture."""
    def _paths(blocks):
        return [b.path for b in (blocks or [])]

    family = arch.family.value if hasattr(arch.family, "value") else str(arch.family)
    return {
        "family": family,
        "spatial": bool(arch.spatial),
        "is_shared_layers": bool(getattr(arch, "is_shared_layers", False)),
        "is_encoder_decoder": bool(getattr(arch, "is_encoder_decoder", False)),
        "has_disentangled_attention": bool(getattr(arch, "has_disentangled_attention", False)),
        "has_cls_token": bool(getattr(arch, "has_cls_token", False)),
        "residual_topology": arch.residual_topology,
        "embed_path": arch.embed_path,
        "head_path": arch.head_path,
        "pre_head_path": arch.pre_head_path,
        "project_out_path": arch.project_out_path,
        "mlm_head_path": getattr(arch, "mlm_head_path", None),
        "blocks": _paths(arch.blocks),
        "decoder_blocks": _paths(getattr(arch, "decoder_blocks", None)),
        # Capability-based op gating + structural mechanism taxonomy: pin the
        # detected capabilities and per-block mechanism so a detection change
        # (e.g. a model that stops resolving an attention layer, or whose
        # blocks get re-labelled) shows up as a loud golden diff.
        "capabilities": {
            "has_unembedding": bool(arch.has_unembedding),
            "has_residual_stream": bool(arch.has_residual_stream),
            "has_attention": bool(arch.has_attention),
            "is_generative": bool(arch.is_generative),
        },
        "block_mechanisms": list(arch.block_mechanisms),
    }


def _load_arch(model_id: str):
    m = interpkit.load(model_id, device="cpu")
    return m.arch_info


@pytest.mark.parametrize("model_id", ALL_MODEL_IDS)
def test_archinfo_matches_golden(model_id):
    path = GOLDEN_DIR / f"{_safe_id(model_id)}.json"

    # Load first — this doubles as the offline-availability gate.
    try:
        arch = _load_arch(model_id)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{model_id} unavailable: {type(exc).__name__}")

    # DeBERTa-v3 (DisentangledSelfAttention) resolves its family / head
    # pipeline via hook-based probe forwards, which hit the known N-004
    # transformers broadcast bug. That bug behaves differently across torch
    # versions (e.g. 2.10 vs 2.11), so the resolved ArchInfo — family,
    # mlm_head/pre_head/project_out paths, has_attention — is not stable enough
    # to pin. The model is fully gated (every meaningful op raises
    # OperationNotSupportedForArchitecture), so these details are functionally
    # irrelevant. We assert only the gating-relevant flag and don't pin the
    # rest (no golden file is written for it).
    if getattr(arch, "has_disentangled_attention", False):
        assert arch.has_disentangled_attention is True
        pytest.skip(
            f"{model_id}: DisentangledSelfAttention resolution is hook-based and "
            f"torch-version-unstable (N-004); model is gated, ArchInfo not pinned."
        )

    if REGEN:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_snapshot(arch), indent=2, sort_keys=True) + "\n")
        return

    if not path.exists():
        pytest.skip(f"no golden snapshot for {model_id} (run with INTERPKIT_REGEN_GOLDEN=1)")

    expected = json.loads(path.read_text())
    actual = _snapshot(arch)
    assert actual == expected, (
        f"ArchInfo for {model_id} diverged from golden snapshot.\n"
        f"Run INTERPKIT_REGEN_GOLDEN=1 to re-baseline if the change is intended."
    )
