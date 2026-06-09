"""Head, unembedding, project-out, MLM-head, and pre-head discovery.

Model-level (not per-layer) resolution: the classifier / LM head, the
embedding, the pre-head module that feeds the head, the dimension-bridge
``project_out`` Linear, and the multi-step MLM head cascade.
"""

from __future__ import annotations

import re
from typing import Any, cast

import torch
import torch.nn as nn

from interpkit.core.arch.blocks import _likely_block_prefixes
from interpkit.core.arch.tree import _safe_forward, module_at_path, path_of

# Generic head suffix names used for role assignment and the relaxed
# unembedding search.
_HEAD_BASE_NAMES = frozenset({
    "lm_head", "head", "classifier", "output_projection",
    "qa_outputs", "embed_out",
})

_HEAD_ATTRS = ("lm_head", "fc", "head", "classifier", "score", "output_projection", "embed_out")
_EMBED_ATTRS = ("patch_embed", "stem", "embeddings", "embed", "wte")

_LM_HEAD_PATTERNS = re.compile(
    r"(^|\.)(lm_head|output_projection|embed_out)(\.|\b)", re.IGNORECASE
)

_MLM_WRAPPER_ATTRS = (
    # The wrapper module is callable as wrapper(hidden_states) → vocab logits
    # and applies the full dense → activation → LayerNorm → decoder cascade.
    # We try these attribute paths in order.
    ("cls", "predictions"),     # BertOnlyMLMHead.predictions
    ("lm_head",),               # RobertaLMHead, BartForCausalLM-style
    ("predictions",),           # AlbertMLMHead
    ("generator_predictions",), # ElectraForMaskedLM
)


# ---------------------------------------------------------------------------
# Convention-based head / embedding lookup
# ---------------------------------------------------------------------------


def _convention_find_head(module: nn.Module, *, num_classes: int | None = None) -> nn.Module | None:
    """Try common head attribute names in priority order.

    Covers both HF (``lm_head`` / ``classifier`` / ``score``) and timm
    (``fc`` / ``head``) conventions in a single walk. When *num_classes*
    is provided, only returns a head whose ``out_features`` matches
    (avoids false positives on models with multiple Linear heads).
    """
    for attr in _HEAD_ATTRS:
        head = getattr(module, attr, None)
        if head is None:
            continue
        if isinstance(head, (nn.Linear, nn.Conv2d)):
            if num_classes is None or getattr(head, "out_features", None) == num_classes:
                return head
        elif isinstance(head, nn.Sequential):
            # Some heads wrap (norm, dropout, linear) — find the trailing Linear.
            for child in reversed(list(head.children())):
                if isinstance(child, (nn.Linear, nn.Conv2d)):
                    if num_classes is None or getattr(child, "out_features", None) == num_classes:
                        return head  # return the wrapping module so the projection includes the norm
                    break
    return None


def _convention_find_num_classes(module: nn.Module) -> int | None:
    """Read num_classes / num_labels / vocab_size from the conventional places."""
    for attr in ("num_classes", "num_labels"):
        val = getattr(module, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    config = getattr(module, "config", None)
    if config is not None:
        for attr in ("num_labels", "vocab_size"):
            val = getattr(config, attr, None)
            if isinstance(val, int) and val > 0:
                return val
    return None


def _convention_find_embedding(module: nn.Module) -> nn.Module | None:
    """Find the embedding module via common attribute names.

    Used as a fast path before falling back to the runtime walker.
    """
    for attr in _EMBED_ATTRS:
        embed = getattr(module, attr, None)
        if isinstance(embed, nn.Module):
            return embed
    return None


def _hf_find_classifier_head(module: nn.Module) -> nn.Module | None:
    """Find a classifier head on a HF model when get_output_embeddings returns None.

    HF vision models (ViTForImageClassification, ResNetForImageClassification,
    etc.) implement ``get_output_embeddings()`` returning ``None`` because
    they don't have an LM head; they expose ``model.classifier`` instead.
    """
    config = getattr(module, "config", None)
    num_labels = getattr(config, "num_labels", None) if config is not None else None
    return _convention_find_head(module, num_classes=num_labels)


# ---------------------------------------------------------------------------
# Unembedding / project-out (legacy regex-driven discovery)
# ---------------------------------------------------------------------------


def _find_unembedding(model: nn.Module) -> str | None:
    """Try to find the unembedding / LM head weight matrix.

    Only matches names that are unambiguously language-model heads
    (``lm_head``, ``output_projection``). Generic names like ``head``,
    ``classifier``, and ``qa_outputs`` are excluded to avoid false
    positives on vision and QA models.  If the model has a
    ``config.vocab_size``, a broader search is attempted with a
    shape check as a safety net.
    """
    # Strict pass: unambiguous LM head names
    for name, module in model.named_modules():
        if _LM_HEAD_PATTERNS.search(name) and hasattr(module, "weight"):
            return cast(str, name)

    # Relaxed pass: allow generic head names only when the output
    # dimension matches vocab_size from the config.
    vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    if vocab_size is not None:
        for name, module in model.named_modules():
            base = name.rsplit(".", 1)[-1].lower()
            if base in _HEAD_BASE_NAMES and hasattr(module, "weight"):
                out_features = getattr(module, "out_features", None)
                if out_features == vocab_size:
                    return cast(str, name)

    return None


def _detect_project_out(model: nn.Module) -> str | None:
    """Find a ``project_out`` layer (OPT-style embed_dim != hidden_size)."""
    for name, mod in model.named_modules():
        if "project_out" in name and hasattr(mod, "weight"):
            return cast(str, name)
    return None


# ---------------------------------------------------------------------------
# Pre-head + intermediate (project_out) detection
# ---------------------------------------------------------------------------


def _find_pre_head_module(
    model: nn.Module, head: nn.Module, sample_input: Any,
    *,
    prefer_prefix: str | None = None,
) -> tuple[nn.Module | None, str | None]:
    """Find the module whose output feeds *head*.

    Strategy: in one forward pass, hook *every* module's output AND the
    head's input. Compare by Python identity to find which module's
    output is the same tensor as the head's input.

    For LMs/ViTs this is typically the final ``LayerNorm``; for CNNs it's
    typically an ``AdaptiveAvgPool2d``. Same code path for every family.

    Parameters
    ----------
    prefer_prefix:
        When the structural fallback runs, restrict candidates to module
        paths starting with this prefix. Used for encoder-decoder models
        to bias toward the decoder side (lens projects decoder hidden
        states for seq2seq).
    """
    activations: dict[str, torch.Tensor] = {}
    head_input: dict[str, torch.Tensor] = {}

    def head_pre_hook(_m: nn.Module, inp: tuple) -> None:
        if inp and isinstance(inp[0], torch.Tensor):
            head_input["x"] = inp[0]

    def out_hook(name: str):
        def fn(_m: nn.Module, _inp: Any, out: Any) -> None:
            if isinstance(out, torch.Tensor):
                activations[name] = out
            elif isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
                activations[name] = out[0]

        return fn

    handles = []
    for name, mod in model.named_modules():
        if name and mod is not head:
            handles.append(mod.register_forward_hook(out_hook(name)))
    handles.append(head.register_forward_pre_hook(head_pre_hook))

    try:
        _safe_forward(model, sample_input)
    except Exception:
        for h in handles:
            h.remove()
        return None, None
    finally:
        for h in handles:
            h.remove()

    if "x" not in head_input:
        return None, None

    target = head_input["x"]
    target_id = id(target)

    found_name: str | None = None
    # Layer 1: identity match.
    for name, act in activations.items():
        if (act is target or id(act) == target_id) and (
            found_name is None or len(name) > len(found_name)
        ):
            found_name = name

    # Layer 2: value equality on same-shape outputs (handles
    # contiguous() / view() / to() that breaks identity but preserves value).
    if found_name is None:
        for name, act in activations.items():
            if act.shape != target.shape or act.dtype != target.dtype:
                continue
            try:
                if torch.equal(act, target) and (
                    found_name is None or len(name) > len(found_name)
                ):
                    found_name = name
            except RuntimeError:
                continue

    # Layer 3: shape-reducing match (handles ViT's CLS-token slice,
    # mean-pool, max-pool that produce a rank-2 head input from rank-3
    # module output). Prefer the last such match in module-tree order.
    if found_name is None and target.dim() <= 2:
        candidates = list(activations.items())
        for name, act in candidates:
            if act.dtype != target.dtype:
                continue
            for reduced in _candidate_reductions(act):
                if reduced.shape != target.shape:
                    continue
                try:
                    if torch.equal(reduced, target):
                        if found_name is None or len(name) > len(found_name):
                            found_name = name
                        break
                except RuntimeError:
                    continue

    # Layer 4: structural fallback — pick the trailing norm / pool module
    # that lives outside the block stack and is the closest predecessor
    # to the head in the named_modules() order. Handles T5 (decoder.final_layer_norm
    # → scale → lm_head) and other models with non-trivial wiring between
    # the final norm and head.
    if found_name is None:
        norm_classes = (
            nn.LayerNorm, nn.GroupNorm, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d,
        )
        head_path = path_of(model, head)
        block_prefixes = _likely_block_prefixes(model)
        norm_candidates: list[str] = []
        for name, mod in model.named_modules():
            if not name or mod is head:
                continue
            if prefer_prefix and not name.startswith(prefer_prefix + "."):
                continue
            type_name = type(mod).__name__.lower()
            looks_norm = (
                isinstance(mod, norm_classes)
                or "layernorm" in type_name
                or "rmsnorm" in type_name
                or "groupnorm" in type_name
                or "rms_norm" in type_name
            )
            if not looks_norm:
                continue
            if any(name == prefix or name.startswith(prefix + ".") for prefix in block_prefixes):
                continue
            norm_candidates.append(name)
        # Prefer the candidate that shares the longest path-prefix with
        # the head (i.e. lives in the same parent scope).
        if norm_candidates and head_path:
            head_parts = head_path.split(".")
            def shared_prefix_len(c: str) -> int:
                cparts = c.split(".")
                n = 0
                for a, b in zip(head_parts, cparts):
                    if a != b:
                        break
                    n += 1
                return n

            norm_candidates.sort(key=lambda c: (-shared_prefix_len(c), -len(c)))
            found_name = norm_candidates[0]
        elif norm_candidates:
            found_name = norm_candidates[-1]

    if found_name is None:
        return None, None
    try:
        found_mod = module_at_path(model, found_name)
    except (AttributeError, IndexError, KeyError, TypeError):
        return None, None
    return found_mod, found_name


def _candidate_reductions(act: torch.Tensor) -> list[torch.Tensor]:
    """Generate plausible head-input reductions of *act*.

    Used when the head's input has fewer dims than the captured module
    outputs (e.g. ViT slices the CLS token before the classifier).
    """
    out: list[torch.Tensor] = []
    if act.dim() == 3:
        out.append(act[:, 0, :])  # CLS token
        out.append(act[:, -1, :])  # last token (LM-style)
        out.append(act.mean(dim=1))  # mean pool
    elif act.dim() == 4:
        out.append(act.mean(dim=(-1, -2)))  # spatial pool
        out.append(act.flatten(1))  # flatten
    return out


def _pick_directional_project_out(
    candidates: list[tuple[str, nn.Linear]],
    head: nn.Module,
    hidden_size: int | None = None,
) -> tuple[str, nn.Linear] | None:
    """Disambiguate multiple project_out candidates by direction (NR-005).

    OPT-style architectures expose two sibling Linears around the LM head:
    ``project_in: Linear(embed → hidden)`` and ``project_out: Linear(hidden →
    embed)``. Both qualify as "project*-named Linear next to pre_head", but
    only ``project_out`` produces vectors compatible with ``lm_head``'s
    input. Pre-NR-005 the resolver returned the first/wrong one and DLA
    crashed with ``size mismatch (embed), (embed × hidden)``.

    Selection rules (in order):
      1. Prefer the candidate whose ``out_features == head.in_features``
         (i.e. its output dim matches the LM head's input dim).
      2. NR-008: when ``hidden_size`` is known, a genuine project_out is a
         *dimension bridge* from the residual stream to the head input, so
         its ``in_features`` must equal ``hidden_size``. This rejects a
         same-output Linear that is not such a bridge (e.g. BART's decoder
         ``fc2``, ``Linear(4*d_model → d_model)``: out matches head_in but
         in is the FFN intermediate width, not the residual width).
      3. Among those, prefer name containing ``"out"`` over ``"in"``.
      4. Otherwise: ambiguous — caller should treat as no match.
    """
    if not candidates:
        return None
    head_in = getattr(head, "in_features", None)
    if head_in is None:
        return candidates[0] if len(candidates) == 1 else None

    directional = [c for c in candidates if c[1].out_features == head_in]
    if not directional:
        return None  # No candidate has the right output dim — bridge missing.
    if hidden_size is not None:
        # NR-008: keep only true residual->head bridges.
        bridged = [c for c in directional if c[1].in_features == hidden_size]
        if not bridged:
            return None
        directional = bridged
    if len(directional) == 1:
        return directional[0]
    # Multiple direction-compatible candidates: prefer name "out" over "in".
    out_named = [c for c in directional if "out" in c[0].lower()]
    if len(out_named) == 1:
        return out_named[0]
    in_named = [c for c in directional if "in" in c[0].lower()]
    if out_named:
        return out_named[0]
    if in_named:
        # All candidates are project_in-style — none should be picked.
        return None
    return directional[0]


def _find_intermediate_linear(
    model: nn.Module,
    pre_head: nn.Module | None,
    head: nn.Module,
    hidden_size: int | None = None,
) -> tuple[nn.Module | None, str | None]:
    """Detect an intermediate ``project_out``-style Linear between pre_head and head.

    Used for OPT-style models where ``hidden_size != embed_dim`` and a
    Linear between the final norm and the LM head adapts the dimensions.

    Heuristic: walk modules between *pre_head* and *head* in the same
    parent scope; if exactly one Linear sits between them, that's
    project_out. N-006 extension: also check the pre_head's *own*
    parent for sibling Linears (covers ELECTRA-style heads where the
    cascade is dense → activation → LayerNorm and the bridge Linear
    lives inside the same wrapper as pre_head). NR-005 extension:
    when multiple candidates exist, disambiguate by the direction of
    the projection (out_features must match head.in_features), so
    OPT's ``project_in`` is never mistaken for ``project_out``.

    NR-008 root-cause guard: a ``project_out`` exists only when the head
    consumes vectors of a *different* width than the residual stream. When
    ``head.in_features == hidden_size`` there is no dimension bridge, so any
    same-width Linear nearby (e.g. a seq2seq decoder's FFN out-projection)
    is NOT project_out — return ``None`` up front. Without this guard the
    resolver picked BART's ``decoder.layers.N.fc2`` as project_out, which
    inflated the DLA unembedding direction to the FFN width and crashed
    with ``tensor [768] vs src [3072]``.
    """
    if pre_head is None:
        return None, None
    head_in = getattr(head, "in_features", None)
    if head_in is not None and hidden_size is not None and head_in == hidden_size:
        return None, None
    pre_path = path_of(model, pre_head)
    head_path = path_of(model, head)
    if pre_path is None or head_path is None:
        return None, None
    pre_parts = pre_path.split(".")
    head_parts = head_path.split(".")
    common_prefix_len = 0
    for a, b in zip(pre_parts, head_parts):
        if a != b:
            break
        common_prefix_len += 1
    common = ".".join(pre_parts[:common_prefix_len])
    try:
        parent = module_at_path(model, common) if common else model
    except (AttributeError, IndexError, KeyError, TypeError):
        parent = None

    candidates: list[tuple[str, nn.Linear]] = []
    if parent is not None:
        for name, child in parent.named_children():
            full_name = f"{common}.{name}" if common else name
            if (isinstance(child, nn.Linear) and child is not head and child is not pre_head
                    and ("project" in name.lower() or full_name not in (pre_path, head_path))):
                candidates.append((full_name, child))

    # N-006: ELECTRA-style — pre_head and head don't share a common ancestor
    # other than root, so the loop above misses ``generator_predictions.dense``.
    # Also look at pre_head's immediate-parent siblings.
    if not candidates and "." in pre_path:
        pre_parent_path, _, pre_child_name = pre_path.rpartition(".")
        try:
            pre_parent = module_at_path(model, pre_parent_path)
        except (AttributeError, IndexError, KeyError, TypeError):
            pre_parent = None
        if pre_parent is not None:
            for name, child in pre_parent.named_children():
                if name == pre_child_name:
                    continue
                if isinstance(child, nn.Linear) and child is not head:
                    candidates.append((f"{pre_parent_path}.{name}", child))

    # NR-005: when multiple candidates exist, disambiguate by direction
    # (out_features must match head.in_features). Only fall back to
    # "exactly one" when the head doesn't expose ``in_features`` (i.e.
    # we have no direction information at all). With a single
    # direction-incompatible candidate (OPT's ``project_in``) we'd
    # rather return None and let ``legacy.project_out_path`` (regex on
    # name) resolve the correct module.
    picked = _pick_directional_project_out(candidates, head, hidden_size)
    if picked is None and len(candidates) == 1 and not hasattr(head, "in_features"):
        picked = candidates[0]
    if picked is None:
        return None, None
    name, lin = picked

    # Heuristic shape sanity: when pre_head exposes ``normalized_shape``,
    # accept the candidate if it matches EITHER the candidate's input
    # (OPT-style: hidden→embed flowing pre_head→project_out→head) OR
    # output (ELECTRA-style: project_out→activation→LayerNorm→head)
    # dimension.
    if hasattr(pre_head, "normalized_shape"):
        norm_dim = pre_head.normalized_shape[-1]
        if lin.in_features != norm_dim and lin.out_features != norm_dim:
            return None, None
    return lin, name


# ---------------------------------------------------------------------------
# MLM head cascade resolution (N-002 / N-006)
# ---------------------------------------------------------------------------


def _find_mlm_head_module(model: nn.Module) -> tuple[nn.Module | None, str | None]:
    """Resolve the MLM head wrapper module that maps hidden→vocab logits.

    The HF ``get_output_embeddings()`` returns only the *final* decoder
    Linear layer. Reproducing the model's actual logits requires the
    full cascade (dense → activation → LayerNorm → decoder). For
    BERT/RoBERTa/ALBERT/DeBERTa this whole cascade is encapsulated in a
    single wrapper module on the model (cls.predictions / lm_head / etc.)
    that takes hidden_states and returns vocab logits in its forward.

    Returns ``(module, dotted_path)`` for the first match; ``(None, None)``
    when no wrapper is found (DistilBERT — handled separately).
    """
    for attr_chain in _MLM_WRAPPER_ATTRS:
        try:
            obj: Any = model
            path_parts: list[str] = []
            for attr in attr_chain:
                obj = getattr(obj, attr)
                path_parts.append(attr)
            if isinstance(obj, nn.Module):
                return obj, ".".join(path_parts)
        except AttributeError:
            continue
    return None, None


def _find_mlm_project_out(
    model: nn.Module,
    mlm_head_module: nn.Module | None,
    distilbert_components: tuple[nn.Module | None, nn.Module | None, nn.Module | None],
) -> tuple[nn.Module | None, str | None]:
    """Locate the ``hidden → embedding`` Linear in an MLM head (N-006).

    For BERT/RoBERTa/ALBERT/DeBERTa the canonical name is
    ``mlm_head.dense`` (or ``mlm_head.transform.dense`` for the BERT
    variant where ``cls.predictions`` wraps a ``BertPredictionHeadTransform``).
    For DistilBERT the corresponding Linear is the standalone
    ``vocab_transform`` already captured in *distilbert_components*.

    Returns ``(module, dotted_path)`` or ``(None, None)``.
    """
    if distilbert_components[0] is not None:
        # vocab_transform already lives at model.vocab_transform; surface
        # its dotted path on the model root.
        path = path_of(model, distilbert_components[0])
        if path is not None:
            return distilbert_components[0], path

    if mlm_head_module is None:
        return None, None

    # Try direct attribute first (RoBERTa: lm_head.dense; ALBERT: predictions.dense)
    direct_dense = getattr(mlm_head_module, "dense", None)
    if isinstance(direct_dense, nn.Linear):
        path = path_of(model, direct_dense)
        if path is not None:
            return direct_dense, path

    # BERT-style: cls.predictions.transform.dense
    transform = getattr(mlm_head_module, "transform", None)
    if transform is not None:
        dense = getattr(transform, "dense", None)
        if isinstance(dense, nn.Linear):
            path = path_of(model, dense)
            if path is not None:
                return dense, path

    return None, None


def _find_distilbert_mlm_components(
    model: nn.Module,
) -> tuple[nn.Module | None, nn.Module | None, nn.Module | None]:
    """Resolve DistilBERT's MLM head pieces.

    DistilBERT exposes the head as three separate sibling modules on
    the model root rather than a single wrapper:
      vocab_transform → activation (gelu) → vocab_layer_norm → vocab_projector
    """
    transform = getattr(model, "vocab_transform", None)
    layer_norm = getattr(model, "vocab_layer_norm", None)
    projector = getattr(model, "vocab_projector", None)
    if not all(isinstance(m, nn.Module) for m in (transform, layer_norm, projector)):
        return None, None, None
    return transform, layer_norm, projector
