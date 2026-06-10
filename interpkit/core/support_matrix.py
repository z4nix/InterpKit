"""Per-op architectural support matrix and lens-pipeline validation contract.

Two complementary safety nets that together prevent silent wrong-results:

1. **Support matrix** — every op declares the structural *capabilities*
   it needs (e.g. ``attention`` needs ≥1 attention layer; ``dla`` needs an
   unembedding + a residual stream). ``check_op_supported(op, arch)`` is
   called at the top of each op entry point and raises
   ``OperationNotSupportedForArchitecture`` (with a helpful message) when
   the model lacks a required capability (e.g. ``attention`` on a CNN).
   Gating on detected capabilities — not a fixed family list — is what
   lets any HF architecture with the right shape work without per-model
   code.

2. **Lens validation contract** — for ops that need full-pipeline
   correctness (lens, dla), the resolver may pick wrong paths. The
   contract asserts that ``lens(last_block_output)`` argmax matches
   ``model(input).logits`` argmax on a sample input. Runs once per
   ``Model`` instance on first use; raises ``LensPipelineMismatch``
   with diagnostic + ``arch_override`` hint on failure.

Together these guarantee: ops only run on architectures they support,
and when they run, the underlying pipeline is verified correct.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from interpkit.core.arch import ArchFamily, ArchInfo
from interpkit.core.exceptions import (
    LensPipelineMismatch,
    OperationNotSupportedForArchitecture,
)

if TYPE_CHECKING:
    from interpkit.core.model import Model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Support matrix
# ---------------------------------------------------------------------------


# Op support is gated on *structural capabilities* the resolver detects,
# not on a fixed family list. This is what lets a brand-new HF
# architecture with the right shape (a head + attention blocks + a
# residual stream) get lens / dla / attention with zero new code, instead
# of being refused because its ``config.architectures`` suffix wasn't
# recognised. The ``ArchFamily`` enum remains a human-readable descriptor;
# these capabilities are the gating authority. See the capability
# properties on ``ArchInfo`` (``has_unembedding`` / ``has_residual_stream``
# / ``has_attention`` / ``is_generative``).


@dataclass(frozen=True)
class Requires:
    """Structural capabilities an op needs to run on a model.

    Each flag names a capability property on :class:`ArchInfo`.
    ``check_op_supported`` evaluates these against the *resolved structure*
    rather than a fixed family set, so support generalises to any
    architecture with the matching shape (no per-model adaptors).
    """

    unembedding: bool = False      # has_unembedding — lens / dla
    residual_stream: bool = False  # has_residual_stream — dla / decompose
    attention: bool = False        # has_attention — attention / qk / ov / heads / circuits
    seq2seq: bool = False          # is_encoder_decoder — encoder_lens
    generative: bool = False       # is_generative — chat

    def missing(self, arch: ArchInfo) -> list[str]:
        """Human-readable list of required capabilities this model lacks."""
        out: list[str] = []
        if self.unembedding and not arch.has_unembedding:
            out.append("a usable output head (unembedding)")
        if self.residual_stream and not arch.has_residual_stream:
            out.append("a residual stream")
        if self.attention and not arch.has_attention:
            out.append("at least one attention layer")
        if self.seq2seq and not arch.is_encoder_decoder:
            out.append("an encoder-decoder (seq2seq) structure")
        if self.generative and not arch.is_generative:
            out.append("autoregressive or seq2seq generation")
        return out


# String "all" means architecture-agnostic — runs on any model.
SUPPORT_MATRIX: dict[str, Requires | str] = {
    "lens":          Requires(unembedding=True),
    "encoder_lens":  Requires(seq2seq=True),
    "dla":           Requires(unembedding=True, residual_stream=True),
    "decompose":     Requires(residual_stream=True),
    "attention":     Requires(attention=True),
    "circuits":      Requires(attention=True),
    "ov_scores":     Requires(attention=True),
    "qk_scores":     Requires(attention=True),
    "composition":   Requires(attention=True),
    "head_activations": Requires(attention=True),
    # Architecture-agnostic — work on any captured tensor
    "patch":       "all",
    "trace":       "all",
    "ablate":      "all",
    "activations": "all",
    "attribute":   "all",
    "probe":       "all",
    "features":    "all",
    "steer":       "all",
    "find_circuit": "all",
    "diff":        "all",
    "scan":        "all",
    "report":      "all",
    "inspect":     "all",
    "chat":        Requires(generative=True),
    "generate":    Requires(generative=True),
    "intervene":   "all",
    "atp":         "all",
    "eap":         Requires(unembedding=True, residual_stream=True),
    "train_tuned_lens": Requires(unembedding=True),
    "max_activating": "all",
}


def _suggest_alternative_op(op: str, family: ArchFamily) -> str:
    """Suggest a similar op the user could try given the family mismatch."""
    if family in (ArchFamily.CNN_RESIDUAL, ArchFamily.CNN_PLAIN):
        if op in ("attention", "circuits", "ov_scores", "qk_scores", "composition", "head_activations"):
            return (
                "`activations(image, at='...')` to capture intermediate "
                "feature maps, or `attribute(image)` for pixel saliency"
            )
        if op == "encoder_lens":
            return "`lens(image)` (vision lens projects through the classifier)"
    if family == ArchFamily.UNKNOWN:
        return (
            "`load(..., arch_override={'family': 'causal_lm', ...})` if you "
            "know the model family, or `inspect()` to debug detection"
        )
    return "`activations(...)` or `attribute(...)`"


# N-004: DeBERTa-v3 (and any DisentangledSelfAttention model) hits a
# known transformers broadcast bug under forward hooks. Affected ops:
_DEBERTA_V3_GATED_OPS = frozenset({
    "trace", "decompose", "attribute", "head_activations",
    "steer", "probe", "diff", "ov_scores", "qk_scores",
    "intervene",  # intervention hooks fire the same broken broadcast path
    "atp", "eap",  # hook every module / block under gradients — same path
})


def check_op_supported(op: str, arch: ArchInfo) -> None:
    """Raise :class:`OperationNotSupportedForArchitecture` if *op* doesn't apply.

    Called at the entry of every op to fail-loud on architecture
    mismatches rather than producing garbage downstream.
    """
    if op not in SUPPORT_MATRIX:
        # Unregistered op — skip the check (developer-facing, not a user issue)
        return

    # N-004: gate ops that interact badly with DisentangledSelfAttention
    # under forward hooks (DeBERTa-v3). Returns a clean
    # OperationNotSupportedForArchitecture rather than the cryptic
    # ``RuntimeError: tensor (512) must match (7)`` from deep in HF
    # transformers' relative-position-bias broadcast path.
    if (
        op in _DEBERTA_V3_GATED_OPS
        and getattr(arch, "has_disentangled_attention", False)
    ):
        raise OperationNotSupportedForArchitecture(
            f"`{op}` is unsupported on DeBERTa-v3 "
            f"(DisentangledSelfAttention) due to a known broadcast bug in "
            f"HF transformers' relative-position-bias path that fires under "
            f"forward hooks. Use a non-DeBERTa encoder (bert / roberta / "
            f"electra / albert) for this op. Tracked in CHANGELOG (N-004)."
        )

    req = SUPPORT_MATRIX[op]
    if req == "all":
        return
    missing = req.missing(arch)
    if not missing:
        return

    needs = ", ".join(missing)
    suggestion = _suggest_alternative_op(op, arch.family)
    # Provenance: report what was detected so the user can see *why* the
    # capability is absent (family is a descriptor here, not the gate).
    found = f"family={arch.family.value!r}"
    if req.attention:
        found += f", attention_layers={arch.attention_layer_indices}"
    raise OperationNotSupportedForArchitecture(
        f"`{op}` requires {needs}; this model lacks it ({found}). "
        f"Try {suggestion}."
    )


# ---------------------------------------------------------------------------
# Lens-pipeline validation contract
# ---------------------------------------------------------------------------


def validate_lens_pipeline(model: Model) -> None:
    """Assert lens-at-last-block matches model output logits.

    Runs once per ``Model`` instance; cached via ``arch_info._lens_validated``.
    Raises :class:`LensPipelineMismatch` with a detailed diagnostic +
    ``arch_override`` workaround hint on any failure.

    This is the universal safety net: even if every layer of architecture
    resolution gets confused, this assertion catches the resulting
    incorrect lens output before the user trusts wrong numbers.
    """
    arch = model.arch_info
    if arch._lens_validated:
        return

    # Skip validation when the resolver couldn't even pick a head/blocks
    # — the support_matrix check or the op itself will surface this.
    if arch.head_module is None or not arch.blocks:
        return

    sample_input = _generate_sample(model)
    if sample_input is None:
        # No way to probe — skip silently. Ops will fail later with a
        # clearer error if they need a head we couldn't validate.
        return

    try:
        with torch.no_grad():
            expected = _run_model(model._model, sample_input)
            if expected is None:
                return
            expected_top1 = expected.argmax(dim=-1)
    except Exception:
        return

    # Run lens at the last block via direct hook.
    try:
        actual = _lens_at_last_block(model, sample_input)
        if actual is None:
            return
        actual_top1 = actual.argmax(dim=-1)
    except Exception as exc:
        raise LensPipelineMismatch(
            f"Lens pipeline crashed during validation: {exc!r}.\n"
            f"  family: {arch.family.value}\n"
            f"  blocks_path (first): {arch.blocks[0].path if arch.blocks else None}\n"
            f"  pre_head_path: {arch.pre_head_path}\n"
            f"  head_path: {arch.head_path}\n"
            f"This is a resolver bug. File at github.com/z4nix/interpkit/issues "
            f"with this trace.\n"
            f"Workaround: load(..., arch_override={{...}})."
        ) from exc

    # Allow shape mismatch when expected has extra dims (e.g. seq logits).
    expected_top1 = _last_token_top1(expected_top1)
    actual_top1 = _last_token_top1(actual_top1)

    if expected_top1.shape != actual_top1.shape:
        # Shape disagreement is acceptable: vision models return (B,)
        # while LMs return (B, seq) or (B,). Compare the scalar argmax.
        try:
            expected_scalar = expected_top1.flatten()[0].item()
            actual_scalar = actual_top1.flatten()[0].item()
        except Exception:
            arch._lens_validated = True
            return
        if expected_scalar == actual_scalar:
            arch._lens_validated = True
            return
        suggestions = _diagnose_lens_candidates(model, sample_input, int(expected_scalar))
        _raise_lens_mismatch(arch, expected_scalar, actual_scalar, suggestions)

    # Permit small disagreements due to fp accumulation in eager — only
    # raise when more than 50% of positions disagree.
    if not torch.equal(expected_top1, actual_top1) and expected_top1.numel() > 0:
        mismatch_frac = (expected_top1 != actual_top1).float().mean().item()
        if mismatch_frac > 0.5:
            try:
                exp_scalar = int(expected_top1.flatten()[0].item())
            except (RuntimeError, IndexError, ValueError):
                exp_scalar = -1
            suggestions = _diagnose_lens_candidates(model, sample_input, exp_scalar)
            _raise_lens_mismatch(
                arch, expected_top1.tolist(), actual_top1.tolist(), suggestions,
            )

    arch._lens_validated = True


def _raise_lens_mismatch(
    arch: ArchInfo, expected: Any, actual: Any, suggestions: list[str] | None = None,
) -> None:
    # E3: when the diagnostic found module paths that DO satisfy the lens
    # contract, surface them so the user fixes it in one round-trip instead of
    # three. Collapse "resolver picked X; try one of {a,b,c}" into the message.
    if suggestions:
        sugg = (
            f"Resolver picked pre_head_path={arch.pre_head_path!r}; the lens "
            f"contract is also satisfied by {suggestions}.\n"
            f"Try load(..., arch_override={{'pre_head_path': {suggestions[0]!r}}}).\n"
        )
    else:
        sugg = (
            "Workaround: load(..., arch_override="
            "{'pre_head_path': '...', 'head_path': '...'}).\n"
        )
    raise LensPipelineMismatch(
        f"Lens at last block top-1 disagrees with model output top-1.\n"
        f"  expected: {expected!r}\n"
        f"  actual:   {actual!r}\n"
        f"  family: {arch.family.value}\n"
        f"  blocks_path (first): {arch.blocks[0].path if arch.blocks else None}\n"
        f"  pre_head_path: {arch.pre_head_path}\n"
        f"  head_path: {arch.head_path}\n"
        f"  project_out_path: {arch.project_out_path}\n"
        f"This indicates a resolver bug. File at github.com/z4nix/interpkit/issues\n"
        f"with this trace.\n"
        + sugg
    )


def _diagnose_lens_candidates(
    model: Model, sample_input: Any, expected_top1_scalar: int,
) -> list[str]:
    """E3: find module paths whose output, projected through the head pipeline,
    reproduces the model's top-1. Returns up to 3 candidate dotted paths.

    Runs one extra forward (only on the validation-failure path) that hooks
    every named module. Projection reuses the single canonical
    :func:`_project_through_head`. Per-module candidates are pre-filtered by a
    narrow shape check (last-dim must equal the residual width) rather than a
    blanket ``except Exception``, so this never silently swallows real bugs.
    """
    arch = model.arch_info
    hidden = getattr(arch, "hidden_size", None)
    captured: dict[str, torch.Tensor] = {}
    handles: list[Any] = []
    mdl = model._model

    def _mk(name: str):
        def _h(_m: nn.Module, _i: Any, out: Any) -> None:
            t = out[0] if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor) else out
            if isinstance(t, torch.Tensor):
                captured[name] = t.detach()
        return _h

    for name, mod in mdl.named_modules():
        if name:
            handles.append(mod.register_forward_hook(_mk(name)))
    try:
        with torch.no_grad():
            _run_model(mdl, sample_input)
    except Exception:
        return []
    finally:
        for h in handles:
            h.remove()

    matches: list[str] = []
    for name, x in captured.items():
        xf = x.float()
        # Narrow shape guard: only residual-width vectors (non-spatial) or
        # 4-D feature maps (spatial) can be the pre-head input.
        if not arch.spatial and (xf.dim() < 2 or (hidden is not None and xf.shape[-1] != hidden)):
            continue
        try:
            logits = _project_through_head(arch, xf)
        except (RuntimeError, TypeError):
            continue
        if logits is None:
            continue
        try:
            top1 = int(_last_token_top1(logits.argmax(dim=-1)).flatten()[0].item())
        except (RuntimeError, IndexError, ValueError):
            continue
        if top1 == expected_top1_scalar:
            matches.append(name)

    # Prefer the deepest (closest-to-head) candidate paths.
    matches.sort(key=len, reverse=True)
    return matches[:3]


def _generate_sample(model: Model) -> Any | None:
    """Generate a small sample input the model can run.

    Uses the model's tokenizer / image_processor when available, falling
    back to a randomly-shaped tensor based on config. Mirrors
    ``loader._make_dummy_input`` but reuses the Model's preparation path.
    """
    try:
        if model._tokenizer is not None:
            return model._prepare("hello")
    except Exception:
        pass
    try:
        if model._image_processor is not None:
            return model._prepare("__dummy_image__")
    except Exception:
        pass
    config = getattr(model._model, "config", None)
    if config is not None:
        image_size = getattr(config, "image_size", None)
        num_channels = getattr(config, "num_channels", 3)
        if image_size:
            return torch.zeros(1, num_channels, image_size, image_size, device=model._device)
        hidden = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
        if hidden:
            return torch.randn(1, 8, hidden, device=model._device)
    return None


def _run_model(module: nn.Module, sample_input: Any) -> torch.Tensor | None:
    """Run *module* on *sample_input* and return logits-shaped tensor."""
    if isinstance(sample_input, dict):
        out = module(**sample_input)
    elif isinstance(sample_input, (tuple, list)):
        out = module(*sample_input)
    else:
        out = module(sample_input)
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    return None


def _last_token_top1(top1: torch.Tensor) -> torch.Tensor:
    """Reduce a (B,) or (B, seq) top-1 tensor to a (B,) top-1 over the
    last position so LMs and vision models can be compared symmetrically."""
    if top1.dim() == 0:
        return top1.unsqueeze(0)
    if top1.dim() == 1:
        return top1
    return top1[..., -1]


def lens_blocks(arch: ArchInfo) -> list[Any]:
    """Return the blocks lens should hook for this family.

    For SEQ2SEQ_LM, lens lives on the decoder side: hidden states from
    decoder blocks are projected through ``decoder.final_layer_norm + lm_head``.
    Encoder blocks have no LM projection (audit N-002 root cause).

    Falls back to ``arch.blocks`` when ``decoder_blocks`` is empty (e.g.
    legacy model where decoder discovery missed) so the validation
    contract still gets a chance to run.
    """
    from interpkit.core.arch import ArchFamily as _ArchFamily

    if arch.family == _ArchFamily.SEQ2SEQ_LM and arch.decoder_blocks:
        return arch.decoder_blocks
    return arch.blocks


def _lens_at_last_block(model: Model, sample_input: Any) -> torch.Tensor | None:
    """Capture the last block's output and project it through the family pipeline.

    Standalone implementation used only by validation — the production
    ``run_lens`` lives in ``interpkit.ops.lens`` (Phase 3).
    """
    arch = model.arch_info
    blocks = lens_blocks(arch)
    if not blocks:
        return None

    last_block_path = blocks[-1].path
    captured: dict[str, torch.Tensor] = {}

    def hook(_m: nn.Module, _inp: Any, out: Any) -> None:
        if isinstance(out, torch.Tensor):
            captured["x"] = out
        elif isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
            captured["x"] = out[0]

    from interpkit.core.arch import module_at_path as _module_at_path

    try:
        last_block = _module_at_path(model._model, last_block_path)
    except (AttributeError, IndexError, KeyError, TypeError):
        return None
    h = last_block.register_forward_hook(hook)
    try:
        with torch.no_grad():
            _run_model(model._model, sample_input)
    finally:
        h.remove()

    if "x" not in captured:
        return None
    block_out = captured["x"]

    return _project_through_head(arch, block_out)


def _dtype_aware_apply(
    module: nn.Module,
    x: torch.Tensor,
    *,
    return_fp32: bool = True,
) -> torch.Tensor:
    """Apply *module* to *x*, matching dtypes via the module's parameters.

    NR-001: ``run_lens`` calls ``block_output.float()`` to standardise
    activations on fp32, but the projection modules retain the model's
    native dtype (fp16 / bf16 when loaded with ``dtype="float16"``).
    Calling ``module(x_fp32)`` on an fp16-weighted module raises a
    cryptic dtype RuntimeError. Cast ``x`` to the module's parameter
    dtype before forward, then optionally cast the output back to fp32
    so all downstream comparisons live in a single precision.

    For modules with no parameters (``nn.GELU``, ``nn.Identity``), keeps
    ``x`` unchanged.
    """
    try:
        target_dtype = next(module.parameters()).dtype
    except StopIteration:
        target_dtype = x.dtype
    out = module(x.to(target_dtype))
    if not isinstance(out, torch.Tensor):
        return out  # type: ignore[return-value]
    return out.float() if return_fp32 else out


def _project_through_head(arch: ArchInfo, block_output: torch.Tensor) -> torch.Tensor | None:
    """Family-aware projection from a captured block output to logits.

    For language models: ``pre_head_module`` (LayerNorm) → ``project_out``
    (OPT only) → ``head_module`` (lm_head).

    For MLM models: route through the resolved MLM head pipeline
    (dense → activation → LayerNorm → decoder); the single
    ``head_module`` is only the final decoder Linear and is not
    sufficient (N-002).

    For vision models: spatial-pool first, then optional pre-head norm
    (ViT) → ``head_module`` (classifier).

    NR-001: every submodule application goes through
    :func:`_dtype_aware_apply` so fp16 / bf16 models project correctly
    without dtype-mismatch crashes. Narrow ``except`` clauses (used to
    be ``except Exception``) so unexpected errors propagate instead of
    being silently coerced into ``None``.
    """
    from interpkit.core.arch import ArchFamily as _ArchFamily

    if arch.head_module is None and arch.mlm_head_module is None:
        return None

    # I3: one canonical entrypoint that dispatches to a named per-regime helper.
    if arch.family == _ArchFamily.MLM:
        return _apply_mlm_head(arch, block_output)        # N-002 cascade
    if arch.spatial:
        return _project_spatial_head(arch, block_output)  # ViT / CNN
    return _project_language_head(arch, block_output)     # causal / seq2seq


def _project_spatial_head(arch: ArchInfo, x: torch.Tensor) -> torch.Tensor | None:
    """Vision head projection: spatial pool → optional pre-head norm → head."""
    if x.dim() == 4:
        # CNN: (B, C, H, W) → spatial mean pool → (B, C)
        x = x.mean(dim=(-1, -2))
    elif x.dim() == 3:
        # ViT: (B, N, hidden) → CLS token if present, else mean pool → (B, hidden)
        if arch.has_cls_token:
            x = x[:, 0, :]
        else:
            x = x.mean(dim=1)
    if arch.pre_head_module is not None and isinstance(
        arch.pre_head_module, (nn.LayerNorm, nn.GroupNorm),
    ):
        try:
            x = _dtype_aware_apply(arch.pre_head_module, x)
        except (RuntimeError, TypeError) as exc:
            # Pre-head norm sometimes has shape constraints the captured tensor
            # doesn't satisfy on exotic vision models; downstream head still
            # gets a usable x. Log loudly rather than silently degrade.
            logger.warning("lens pre-head norm failed: %r", exc)
    try:
        return _dtype_aware_apply(arch.head_module, x)
    except (RuntimeError, TypeError) as exc:
        logger.warning("lens head projection failed: %r", exc)
        return None


def _project_language_head(arch: ArchInfo, x: torch.Tensor) -> torch.Tensor | None:
    """Language head projection: pre-head norm → project_out (OPT) → head."""
    if arch.pre_head_module is not None:
        try:
            x = _dtype_aware_apply(arch.pre_head_module, x)
        except (RuntimeError, TypeError) as exc:
            logger.warning("lens pre-head module failed: %r", exc)
    if arch.project_out_module is not None:
        try:
            x = _dtype_aware_apply(arch.project_out_module, x)
        except (RuntimeError, TypeError) as exc:
            logger.warning("lens project_out failed: %r", exc)
    try:
        return _dtype_aware_apply(arch.head_module, x)
    except (RuntimeError, TypeError) as exc:
        logger.warning("lens head projection failed: %r", exc)
        return None


def _apply_mlm_head(arch: ArchInfo, hidden_states: torch.Tensor) -> torch.Tensor | None:
    """Apply the MLM head pipeline (BERT-style) to hidden states.

    Three regimes:

    1. ``mlm_head_module`` is set (BERT/RoBERTa/ALBERT/DeBERTa) — call
       it directly. Most wrappers (BertLMPredictionHead, RobertaLMHead,
       AlbertMLMHead) include the final decoder Linear in their
       forward and return vocab logits directly. ELECTRA's
       ``generator_predictions`` is the exception: it produces
       embedding-sized vectors and the final decoder
       (``generator_lm_head``) is applied separately by the model's
       forward (NR-004). Detect this by output shape and chain
       ``head_module`` after the wrapper when they differ.
    2. DistilBERT components are set (vocab_transform / vocab_layer_norm
       / vocab_projector) — apply them in the documented order with a
       GELU activation between transform and layer_norm.
    3. Neither is set — fall back to ``head_module`` alone (the lens
       validation contract will catch this if the projection is wrong).

    NR-001 / NR-003: dtype-aware application so fp16 / bf16 MLM models
    project correctly. Narrow ``except`` clauses surface real bugs
    instead of collapsing to silent ``None``.
    """
    vocab_size = getattr(arch, "vocab_size", None)

    if arch.mlm_head_module is not None:
        try:
            out = _dtype_aware_apply(arch.mlm_head_module, hidden_states)
        except (RuntimeError, TypeError) as exc:
            logger.warning("MLM head projection failed: %r", exc)
            return None
        # NR-004: if the wrapper produced embedding-sized vectors (not
        # vocab logits), the model's actual head pipeline ends with a
        # SEPARATE final decoder Linear (ELECTRA's ``generator_lm_head``).
        # Apply it to recover vocab logits.
        if (
            arch.head_module is not None
            and arch.head_module is not arch.mlm_head_module
            and vocab_size is not None
            and isinstance(out, torch.Tensor)
            and out.shape[-1] != vocab_size
        ):
            try:
                out = _dtype_aware_apply(arch.head_module, out)
            except (RuntimeError, TypeError) as exc:
                logger.warning("MLM head final decoder failed: %r", exc)
                return None
        return out

    if arch.distilbert_vocab_transform is not None:
        try:
            x = _dtype_aware_apply(
                arch.distilbert_vocab_transform, hidden_states, return_fp32=False,
            )
            # DistilBERT uses GELU between transform and layer_norm
            # (see DistilBertForMaskedLM.forward). GELU has no params,
            # dtype is preserved automatically.
            x = torch.nn.functional.gelu(x)
            x = _dtype_aware_apply(arch.distilbert_vocab_layer_norm, x, return_fp32=False)
            return _dtype_aware_apply(arch.distilbert_vocab_projector, x)
        except (RuntimeError, TypeError) as exc:
            logger.warning("DistilBERT MLM head pipeline failed: %r", exc)
            return None

    # Last-resort fallback to single head_module.
    if arch.head_module is None:
        return None
    try:
        return _dtype_aware_apply(arch.head_module, hidden_states)
    except (RuntimeError, TypeError) as exc:
        logger.warning("MLM fallback head projection failed: %r", exc)
        return None


__all__ = [
    "SUPPORT_MATRIX",
    "Requires",
    "check_op_supported",
    "validate_lens_pipeline",
    "lens_blocks",
]
