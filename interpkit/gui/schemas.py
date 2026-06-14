"""Request models and JSON serializers for the GUI API.

``ArchInfo`` mixes JSON-safe metadata with live ``nn.Module`` references,
so the GUI serializes an explicit whitelist (never the dataclass wholesale).
The serialized arch powers the frontend's module pickers and layer/head
selectors — GUI users never type module paths from memory.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    """Body of ``POST /api/sessions``."""

    model_id: str = Field(..., min_length=1, description="HuggingFace model ID or local path")
    device: str | None = None
    dtype: str | None = None
    device_map: str | None = None


def serialize_arch(model: Any) -> dict[str, Any]:
    """Whitelisted JSON view of ``model.arch_info`` for the frontend."""
    arch = model.arch_info
    layer_infos = [
        {
            "index": li.index,
            "name": li.name,
            "layer_type": li.layer_type,
            "attn_path": li.attn_path,
            "mlp_path": li.mlp_path,
        }
        for li in arch.layer_infos
    ]
    return {
        "family": arch.family.value if hasattr(arch.family, "value") else str(arch.family),
        "arch_family": arch.arch_family,
        "num_layers": arch.num_layers,
        "hidden_size": arch.hidden_size,
        "num_attention_heads": arch.num_attention_heads,
        "num_key_value_heads": arch.num_key_value_heads,
        "vocab_size": arch.vocab_size,
        "is_encoder_decoder": arch.is_encoder_decoder,
        "is_language_model": arch.is_language_model,
        "is_vision_model": arch.is_vision_model,
        "is_generative": arch.is_generative,
        "spatial": arch.spatial,
        "residual_topology": arch.residual_topology,
        "head_path": arch.head_path,
        "embed_path": arch.embed_path,
        "pre_head_path": arch.pre_head_path,
        "blocks": [
            {
                "path": b.path,
                "stage": b.stage,
                "has_attention": b.has_attention,
                "has_residual": b.has_residual,
                "mechanism": b.mechanism,
            }
            for b in arch.blocks
        ],
        "layer_infos": layer_infos,
        "layer_names": list(arch.layer_names),
        "attention_layer_indices": list(arch.attention_layer_indices),
        # Grouped path lists for the module-picker widget.
        "paths": {
            "blocks": [b.path for b in arch.blocks],
            "attention": [li.attn_path for li in arch.layer_infos if li.attn_path],
            "mlp": [li.mlp_path for li in arch.layer_infos if li.mlp_path],
            "all": arch.all_paths(),
        },
    }


def serialize_support(model: Any, specs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-GUI-op support map: ``{op_name: {supported, reason}}``.

    Probes :func:`interpkit.core.support_matrix.check_op_supported` for
    each op's support key so the sidebar can grey out unsupported ops with
    the library's own explanation, instead of letting the user discover an
    ``OperationNotSupportedForArchitecture`` after filling in a form.
    """
    from interpkit.core.exceptions import OperationNotSupportedForArchitecture
    from interpkit.core.support_matrix import check_op_supported

    arch = model.arch_info
    support: dict[str, dict[str, Any]] = {}
    for name, spec in specs.items():
        key = spec.support_key
        if key is None:
            support[name] = {"supported": True, "reason": None}
            continue
        try:
            check_op_supported(key, arch)
            support[name] = {"supported": True, "reason": None}
        except OperationNotSupportedForArchitecture as exc:
            support[name] = {"supported": False, "reason": str(exc)}
    return support


def serialize_inspect(model: Any) -> dict[str, Any]:
    """Structured architecture description for the inspect op (mirrors the
    CLI's ``inspect --format json`` payload, plus the discovery summary)."""
    arch = model.arch_info
    total_params = sum(p.numel() for p in model._model.parameters())
    return {
        "summary": arch.discovery_summary(),
        "total_params": total_params,
        "device": model.device,
        "dtype": str(model.dtype),
        **serialize_arch(model),
        "modules": [
            {"name": m.name, "type": m.type_name, "param_count": m.param_count, "role": m.role}
            for m in arch.modules
        ],
    }
