"""Architecture resolution: one cohesive package.

Consolidates what used to be three entangled modules — ``discovery``,
the ``resolve`` package, and ``residual`` — into a single
``interpkit.core.arch`` package with one :class:`ArchInfo` contract.

Submodule layout:

- ``names``    — module-name vocabulary + regexes.
- ``types``    — ``ArchInfo``, ``ArchFamily``, ``BlockSpec``, ``LayerInfo``, ``ModuleInfo``.
- ``tree``     — static module-tree primitives + weight extraction.
- ``probe``    — runtime forward-hook probes.
- ``family``   — family classification, topology, config parsing.
- ``blocks``   — block / decoder-block discovery.
- ``layers``   — per-layer attn/mlp/qkv resolution + role assignment.
- ``heads``    — head / unembedding / project-out / MLM / pre-head discovery.
- ``resolve``  — ``resolve_arch`` orchestrator + ``discover`` + overrides.
- ``residual`` — residual-stream decomposition schemas.
"""

from __future__ import annotations

from interpkit.core.arch.names import (
    ALL_QKV_NAMES,
    ATTN_NAMES,
    ATTN_RE,
    FUSED_QKV_NAMES,
    K_PROJ_NAMES,
    MLP_NAMES,
    MLP_RE,
    O_PROJ_NAMES,
    Q_PROJ_NAMES,
    V_PROJ_NAMES,
    names_to_regex,
)
from interpkit.core.arch.residual import (
    Component,
    PostLNResidual,
    PreLNResidual,
    ResidualSchema,
    Seq2seqResidual,
    SharedLayerResidual,
    residual_schema_for,
)
from interpkit.core.arch.resolve import (
    ARCH_OVERRIDES,
    apply_overrides,
    discover,
    resolve_arch,
)
from interpkit.core.arch.tree import (
    canonical_linear_weight,
    extract_proj_weight,
    get_weight,
    module_at_path,
)
from interpkit.core.arch.types import (
    ArchFamily,
    ArchInfo,
    BlockSpec,
    LayerInfo,
    ModuleInfo,
)

__all__ = [
    # Types
    "ArchInfo",
    "ArchFamily",
    "BlockSpec",
    "LayerInfo",
    "ModuleInfo",
    # Resolution
    "resolve_arch",
    "discover",
    "apply_overrides",
    "ARCH_OVERRIDES",
    # Tree / weight helpers
    "module_at_path",
    "get_weight",
    "extract_proj_weight",
    "canonical_linear_weight",
    # Module-name vocabulary
    "ATTN_NAMES",
    "MLP_NAMES",
    "FUSED_QKV_NAMES",
    "Q_PROJ_NAMES",
    "K_PROJ_NAMES",
    "V_PROJ_NAMES",
    "ALL_QKV_NAMES",
    "O_PROJ_NAMES",
    "ATTN_RE",
    "MLP_RE",
    "names_to_regex",
    # Residual schemas
    "Component",
    "ResidualSchema",
    "PreLNResidual",
    "PostLNResidual",
    "SharedLayerResidual",
    "Seq2seqResidual",
    "residual_schema_for",
]
