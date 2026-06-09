"""Model loading — HuggingFace, TransformerLens, timm, and raw nn.Module.

Two entry points share the same underlying construction:

- :func:`load` takes a HuggingFace model id (or a HF ``PreTrainedModel``
  instance) and uses ``AutoModel.from_pretrained`` for retrieval.
- :func:`load_module` takes any ``nn.Module`` plus a sample input —
  used for timm models, custom research code, or pre-instantiated
  HuggingFace models.

Both call :func:`interpkit.core.arch.resolve_arch` for architecture
discovery and produce a :class:`~interpkit.core.model.Model` with an
``ArchInfo`` populated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from interpkit.core.arch import resolve_arch
from interpkit.core.registry import get_registration

if TYPE_CHECKING:
    from interpkit.core.model import Model


# N-004: track which (model_id) we've already warned about to avoid
# spamming the console when the same DeBERTa-v3 model is loaded multiple
# times in one process.
_DISENTANGLED_WARNED: set[str] = set()

# N-010: track which (model_id) we've warned about an uninitialized head
# for, to avoid spamming on repeated loads in one process.
_UNINIT_HEAD_WARNED: set[str] = set()


def _warn_disentangled_attention_once(name: str) -> None:
    """Emit a single load-time UserWarning for DeBERTa-v3-style models."""
    import warnings as _warnings

    if name in _DISENTANGLED_WARNED:
        return
    _DISENTANGLED_WARNED.add(name)
    _warnings.warn(
        f"Loaded {name!r}: this model uses DisentangledSelfAttention "
        f"(DeBERTa-v3 family). The following ops are unsupported under "
        f"forward hooks due to a known HF transformers broadcast bug: "
        f"trace, decompose, attribute, head_activations, steer, probe, "
        f"diff, ov_scores, qk_scores. Calling any of these will raise "
        f"OperationNotSupportedForArchitecture (N-004).",
        UserWarning,
        stacklevel=3,
    )


def _warn_uninitialized_head_once(
    name: str, arch_info: Any, missing_keys: set[str],
) -> None:
    """N-010: warn when the resolved head pipeline was randomly initialized.

    When a checkpoint's parameter names don't match the chosen HF model
    class (e.g. ``microsoft/deberta-v3-small`` stores its MLM head under
    ``lm_predictions.lm_head.*`` but is loaded as ``DebertaV2ForMaskedLM``,
    whose head lives at ``cls.predictions.*``), transformers silently
    leaves the unmatched head weights randomly initialized. interpkit
    then faithfully projects through those random weights, so ``lens`` /
    ``dla`` produce meaningless token rankings with no error.

    This detects the condition generically (any resolved head / pre-head /
    project-out / unembedding module whose parameters appear in
    ``missing_keys``) and warns once per model, rather than special-casing
    one architecture.
    """
    import warnings as _warnings

    if not missing_keys or name in _UNINIT_HEAD_WARNED:
        return

    # The resolved projection pipeline that lens / dla rely on. A randomly
    # initialized module here means projected logits are meaningless.
    candidate_paths = {
        "head": getattr(arch_info, "head_path", None),
        "unembedding": getattr(arch_info, "unembedding_name", None),
        "pre_head": getattr(arch_info, "pre_head_path", None),
        "project_out": getattr(arch_info, "project_out_path", None),
    }

    affected: dict[str, str] = {}
    for role, path in candidate_paths.items():
        if not path:
            continue
        prefix = path + "."
        if any(k == path or k.startswith(prefix) for k in missing_keys):
            affected[role] = path

    if not affected:
        return

    _UNINIT_HEAD_WARNED.add(name)
    detail = ", ".join(f"{role} ({path})" for role, path in sorted(affected.items()))
    _warnings.warn(
        f"Loaded {name!r}: the resolved output-head pipeline was NOT fully "
        f"initialized from the checkpoint — these modules carry random "
        f"weights: {detail}. This usually means the checkpoint stores its "
        f"head under different parameter names than the HF model class "
        f"interpkit loaded. Operations that project through the head "
        f"(lens, dla) will reflect these random weights and are NOT "
        f"meaningful for this checkpoint (N-010). Activation-level ops "
        f"(activations, attention, ablate, trace) are unaffected.",
        UserWarning,
        stacklevel=3,
    )


_DTYPE_MAP: dict[str, torch.dtype | str] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "auto": "auto",
}


def load(
    model_or_name: str | nn.Module,
    *,
    tokenizer: Any | None = None,
    image_processor: Any | None = None,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype = "float32",
    device_map: str | dict | None = None,
    arch_override: dict[str, Any] | None = None,
    attention_impl: str = "auto",
    validate_pipeline: bool = True,
) -> Model:
    """Load a HuggingFace model for mechanistic interpretability.

    Parameters
    ----------
    model_or_name:
        A HuggingFace model ID (``"gpt2"``, ``"microsoft/resnet-50"``)
        or an existing ``nn.Module`` instance.  For arbitrary
        non-PreTrainedModel modules (e.g. timm models), prefer
        :func:`load_module`.
    tokenizer:
        An explicit tokenizer. Auto-loaded for HF models if not provided.
    image_processor:
        An explicit image processor. Auto-loaded for HF vision models if not provided.
    device:
        Device to run on. Auto-detected (cuda > mps > cpu) if omitted.
        Ignored when *device_map* is set (HF handles placement).
    dtype:
        Model dtype.  Defaults to ``"float32"`` — interpkit defaults to
        full precision so numerical noise doesn't masquerade as
        interpretability findings.  Pass ``"auto"`` explicitly to use
        the dtype stored in HF's checkpoint config (typically fp16/bf16
        for modern models — fast but lossy).  Other supported values:
        ``"float16"``, ``"bfloat16"``, ``"fp16"``, ``"bf16"``, ``"fp32"``,
        or a ``torch.dtype``.  Passing ``None`` raises ``TypeError``.
    device_map:
        HuggingFace ``device_map`` for multi-GPU / offload placement
        (e.g. ``"auto"``).  Requires the ``accelerate`` package.
    arch_override:
        Optional dict of explicit hints applied to the architecture
        resolver.  Use when interpkit's auto-detection is wrong for an
        exotic model.  See :func:`interpkit.core.arch.resolve_arch`
        for the supported keys.
    attention_impl:
        Attention backend to load with. Defaults to ``"auto"`` (let HF
        decide — typically SDPA on modern GPUs). Pass ``"eager"`` to
        load with eager attention up-front; this makes
        :meth:`Model.attention` and :meth:`Model.head_activations` skip
        the lazy second-copy reload (saves ~2× VRAM during attention
        ops on large models). Other accepted values
        (``"sdpa"`` / ``"flash"``) are passed through to HF; their
        availability depends on the installed transformers + the GPU.

        Note: when this kwarg is ``"auto"`` (default), the first call
        to ``model.attention()`` triggers a second model copy with
        eager attention. The copy is cached on the ``Model`` instance.
        Call :meth:`Model.unload_eager_attention` to free it
        explicitly when done with attention ops.
    validate_pipeline:
        E4 — when *True* (default) and the model has a detectable LM/classifier
        head, the lens-at-last-block validation contract runs at load time and
        raises :class:`~interpkit.core.exceptions.LensPipelineMismatch` (with
        top-3 ``arch_override`` suggestions) immediately if the resolver picked
        wrong paths — rather than surfacing mid-analysis on the first ``lens`` /
        ``dla`` call. Pass *False* to skip the probe forward for
        attention-only / inspect-only workflows. Headless and ``UNKNOWN``-family
        models are skipped automatically.
    """
    if dtype is None:
        raise TypeError(
            "load(dtype=None) is not supported. Pass dtype='float32' for full "
            "precision (default), dtype='auto' to use the HF checkpoint's "
            "stored dtype, or an explicit dtype string / torch.dtype."
        )

    valid_attn_impls = {"auto", "eager", "sdpa", "flash"}
    if attention_impl not in valid_attn_impls:
        raise ValueError(
            f"attention_impl={attention_impl!r} invalid; "
            f"valid: {sorted(valid_attn_impls)}."
        )

    torch_dtype = _resolve_dtype(dtype)

    if device is None and device_map is None:
        device = _resolve_device()

    if str(device) == "mps" and torch_dtype is torch.bfloat16:
        import warnings

        warnings.warn(
            "bfloat16 is not fully supported on MPS; falling back to float16.",
            stacklevel=2,
        )
        torch_dtype = torch.float16

    is_tl = False

    if isinstance(model_or_name, str):
        _hf_kwargs: dict[str, Any] = {
            "tokenizer": tokenizer,
            "image_processor": image_processor,
            "device": device,
            "torch_dtype": torch_dtype,
            "device_map": device_map,
        }
        # Only forward the new attention_impl kwarg when the user
        # asked for something non-default so existing mocks of
        # _load_from_hf don't have to update their signature.
        if attention_impl != "auto":
            _hf_kwargs["attention_impl"] = attention_impl
        model, tokenizer, image_processor = _load_from_hf(
            model_or_name, **_hf_kwargs,
        )
    else:
        model = model_or_name
        # User-supplied module — if they asked for eager, flip the
        # config flag in place so Model.attention() skips the reload.
        if attention_impl == "eager":
            config = getattr(model, "config", None)
            if config is not None:
                config._attn_implementation = "eager"

        if _is_hooked_transformer(model):
            is_tl = True
            if tokenizer is None:
                tl_tok = getattr(model, "tokenizer", None)
                if tl_tok is not None:
                    tokenizer = tl_tok
                    if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token

        if device_map is None and device is not None:
            model.to(device)

    model.eval()

    if device is None and device_map is not None:
        device = next(model.parameters()).device

    built = _build_model(
        model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        device=device or "cpu",
        is_tl=is_tl,
        arch_override=arch_override,
    )
    if validate_pipeline:
        _maybe_validate_pipeline_at_load(built)
    return built


def _maybe_validate_pipeline_at_load(model: Model) -> None:
    """E4: run the lens-pipeline validation contract at load time, gated.

    Only when a head exists and the family is known — headless / inspect-only /
    UNKNOWN models pay no probe-forward cost. Raises ``LensPipelineMismatch``
    (with E3 top-3 suggestions) on failure. Sets ``arch._lens_validated`` so the
    lazy first-op-call check becomes a no-op.
    """
    from interpkit.core.arch import ArchFamily
    from interpkit.core.support_matrix import validate_lens_pipeline

    arch = model.arch_info
    if arch.head_module is None or arch.family == ArchFamily.UNKNOWN:
        return
    validate_lens_pipeline(model)


def load_module(
    module: nn.Module,
    sample_input: Any,
    *,
    tokenizer: Any | None = None,
    image_processor: Any | None = None,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype = "float32",
    arch_override: dict[str, Any] | None = None,
) -> Model:
    """Wrap an arbitrary ``nn.Module`` as an interpkit Model.

    Use this for timm models, custom research architectures, or any
    PyTorch module that isn't a HF ``PreTrainedModel``.  HF models can
    also be loaded this way if you've already instantiated them.

    Parameters
    ----------
    module:
        Any PyTorch ``nn.Module`` whose forward can run on
        *sample_input*.
    sample_input:
        A small input the module can run.  Required so the resolver's
        runtime hooks can detect blocks, residuals, and the pre-head
        module.  For vision models, pass ``torch.zeros(1, 3, H, W)``.
        For text models without a wrapped tokenizer, pass a tokenized
        tensor (or a dict of HF inputs).
    tokenizer:
        Optional tokenizer for text-input convenience.  Required for
        ``model.lens("hello")``-style calls; not needed for tensor-input ops.
    image_processor:
        Optional image processor for image-input convenience.
    device:
        Device to move the model to.  Auto-detected if omitted.
    dtype:
        Same semantics as :func:`load`.  Defaults to ``"float32"``.
    arch_override:
        Optional dict of explicit hints applied to the architecture
        resolver.  See :func:`interpkit.core.arch.resolve_arch` for
        supported keys (``head_path``, ``embed_path``, ``pre_head_path``,
        ``project_out_path``, ``blocks_path``, ``family``).

    Examples
    --------
    Loading a timm vision model::

        import timm, torch, interpkit
        m = timm.create_model("resnet50.a1_in1k", pretrained=True)
        model = interpkit.load_module(
            m, sample_input=torch.zeros(1, 3, 224, 224),
        )
        model.activations(image_path, at="layer3.0")

    Loading a custom architecture with override::

        my_model = MyArchitecture()
        model = interpkit.load_module(
            my_model,
            sample_input=torch.zeros(1, 3, 224, 224),
            arch_override={"head_path": "out.linear", "blocks_path": "encoder.stages"},
        )
    """
    if dtype is None:
        raise TypeError(
            "load_module(dtype=None) is not supported. Pass dtype='float32' "
            "(default) or an explicit dtype string / torch.dtype."
        )

    torch_dtype = _resolve_dtype(dtype)

    if device is None:
        device = _resolve_device()

    module = module.to(device)
    if torch_dtype not in (None, "auto") and isinstance(torch_dtype, torch.dtype):
        # Best-effort dtype cast; not all params can be cast (e.g. int buffers).
        try:
            module = module.to(dtype=torch_dtype)
        except (TypeError, RuntimeError):
            pass
    module.eval()

    return _build_model(
        module,
        tokenizer=tokenizer,
        image_processor=image_processor,
        device=device,
        sample_input=sample_input,
        is_tl=_is_hooked_transformer(module),
        arch_override=arch_override,
    )


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype | str:
    if isinstance(dtype, str):
        if dtype not in _DTYPE_MAP:
            raise ValueError(
                f"Unknown dtype {dtype!r}. Allowed values: "
                f"{', '.join(sorted(_DTYPE_MAP.keys()))}"
            )
        return _DTYPE_MAP[dtype]
    return dtype


def _build_model(
    module: nn.Module,
    *,
    tokenizer: Any | None,
    image_processor: Any | None,
    device: torch.device | str,
    sample_input: Any | None = None,
    is_tl: bool = False,
    arch_override: dict[str, Any] | None = None,
) -> Model:
    """Construct a :class:`Model` from an instantiated ``nn.Module``.

    Shared backend for :func:`load` and :func:`load_module`.
    """
    from interpkit.core.model import Model

    registration = get_registration(module)

    if is_tl:
        sample = torch.tensor([[0]], device=device)
    elif sample_input is not None:
        sample = sample_input
    else:
        sample = _make_dummy_input(
            module, tokenizer=tokenizer, image_processor=image_processor,
            device=device,
        )

    arch_info = resolve_arch(module, sample, arch_override=arch_override)
    arch_info.is_tl_model = is_tl

    # N-004: warn at load time when DeBERTa-v3 (DisentangledSelfAttention)
    # is detected so users know up-front which ops are gated. Only warns
    # once per process via a module-level set.
    # NR-002: ``name`` is not a parameter of ``_build_model`` (only
    # ``load()`` had access to the original HF id). Derive a stable
    # label from the loaded module — ``module.name_or_path`` is set by
    # HF's ``from_pretrained``; ``type(module).__name__`` is the
    # universal fallback for ``load_module()`` callers.
    if getattr(arch_info, "has_disentangled_attention", False):
        label = getattr(module, "name_or_path", None) or type(module).__name__
        _warn_disentangled_attention_once(label)

    # N-010: warn if the resolved head pipeline was randomly initialized
    # (checkpoint/head-class name mismatch). Only HF loads populate
    # ``_interpkit_missing_keys``; load_module() callers skip this.
    missing_keys = getattr(module, "_interpkit_missing_keys", None)
    if missing_keys:
        label = getattr(module, "name_or_path", None) or type(module).__name__
        _warn_uninitialized_head_once(label, arch_info, missing_keys)

    if registration is not None:
        if registration.layers:
            arch_info.layer_names = registration.layers
        if registration.output_head:
            arch_info.output_head_name = registration.output_head
            arch_info.unembedding_name = registration.output_head
            arch_info.has_lm_head = True
        for mod_info in arch_info.modules:
            if mod_info.name in registration.attention_modules:
                mod_info.role = "attention"
            elif mod_info.name in registration.mlp_modules:
                mod_info.role = "mlp"

    return Model(
        module,
        tokenizer=tokenizer,
        image_processor=image_processor,
        arch_info=arch_info,
        registration=registration,
        device=device,
    )


def _resolve_device() -> str:
    """Auto-detect the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_from_hf(
    name: str,
    *,
    tokenizer: Any | None,
    image_processor: Any | None,
    device: str | torch.device | None,
    torch_dtype: torch.dtype | str | None = None,
    device_map: str | dict | None = None,
    attention_impl: str = "auto",
) -> tuple[nn.Module, Any | None, Any | None]:
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    extra_kwargs: dict[str, Any] = {}
    if torch_dtype is not None:
        extra_kwargs["torch_dtype"] = torch_dtype
    if device_map is not None:
        extra_kwargs["device_map"] = device_map
    if attention_impl != "auto":
        # P3g: HF accepts attn_implementation as a kwarg to
        # from_pretrained for any PreTrainedModel that supports it.
        # Modern transformers (5.x) propagate this through to the
        # config and pick the matching backend.
        extra_kwargs["attn_implementation"] = attention_impl

    config = AutoConfig.from_pretrained(name)

    _TASK_HINTS: list[tuple[str, str]] = [
        ("questionanswering", "AutoModelForQuestionAnswering"),
        ("tokenclassification", "AutoModelForTokenClassification"),
        ("sequenceclassification", "AutoModelForSequenceClassification"),
        ("maskgeneration", "AutoModelForMaskGeneration"),
        ("objectdetection", "AutoModelForObjectDetection"),
        ("semanticsegmentation", "AutoModelForSemanticSegmentation"),
    ]
    architectures = getattr(config, "architectures", None) or []
    arch_str = " ".join(architectures).lower()

    auto_order: list[str] = []
    for keyword, cls_name in _TASK_HINTS:
        if keyword in arch_str:
            auto_order.append(cls_name)
    is_enc_dec = getattr(config, "is_encoder_decoder", False)
    if is_enc_dec:
        auto_order.extend([
            "AutoModelForSeq2SeqLM",
            "AutoModelForCausalLM",
        ])
    else:
        auto_order.extend([
            "AutoModelForCausalLM",
            "AutoModelForSeq2SeqLM",
        ])
    auto_order.extend([
        "AutoModelForMaskedLM",
        "AutoModelForImageClassification",
        "AutoModel",
    ])

    import transformers

    model = None
    missing_keys: set[str] = set()
    for auto_cls_name in auto_order:
        auto_cls = getattr(transformers, auto_cls_name, None)
        if auto_cls is None:
            continue
        try:
            model, missing_keys = _from_pretrained_with_loading_info(
                auto_cls, name, config=config, **extra_kwargs,
            )
            break
        except (ValueError, OSError, KeyError):
            continue

    if model is None:
        model, missing_keys = _from_pretrained_with_loading_info(
            AutoModel, name, **extra_kwargs,
        )

    # N-010: stash the set of randomly-initialized parameter names so
    # _build_model can warn if the resolved head pipeline is among them.
    try:
        model._interpkit_missing_keys = missing_keys
    except (AttributeError, TypeError):
        pass

    if device_map is None and device is not None:
        model = model.to(device)

    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
        except (OSError, KeyError, ValueError):
            tokenizer = None
        except ImportError as exc:
            from rich.console import Console

            Console().print(
                "  [yellow]load:[/yellow] tokenizer for "
                f"[bold]{name}[/bold] requires an optional dependency "
                f"that is not installed ({exc}). Some text-input ops will "
                "be unavailable."
            )
            tokenizer = None
        except Exception as exc:
            from rich.console import Console

            Console().print(
                "  [yellow]load:[/yellow] AutoTokenizer raised an unexpected "
                f"error for [bold]{name}[/bold] "
                f"({type(exc).__name__}: {exc}). Continuing without a "
                "tokenizer; pass one explicitly via tokenizer=... if needed."
            )
            tokenizer = None

    if tokenizer is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if image_processor is None:
        try:
            from transformers import AutoImageProcessor

            image_processor = AutoImageProcessor.from_pretrained(name)
        except (OSError, KeyError):
            pass
        except ImportError as exc:
            if "torchvision" in str(exc):
                from rich.console import Console

                Console().print(
                    "  [yellow]load:[/yellow] HF image processor for "
                    f"[bold]{name}[/bold] requires torchvision but it is not "
                    "installed. Install with 'pip install interpkit[vision]' "
                    "if you plan to feed raw images to this model."
                )

    return model, tokenizer, image_processor


def _from_pretrained_with_loading_info(
    auto_cls: Any, name: str, **kwargs: Any,
) -> tuple[nn.Module, set[str]]:
    """Call ``from_pretrained`` and capture randomly-initialized keys.

    Requests ``output_loading_info=True`` so we can detect checkpoints
    whose head weights didn't map (N-010). Degrades gracefully: if a
    model class doesn't support the flag, or the return shape is
    unexpected, the model still loads and the missing-key set is empty.
    """
    try:
        result = auto_cls.from_pretrained(name, output_loading_info=True, **kwargs)
    except TypeError:
        # Custom model classes may not accept output_loading_info.
        return auto_cls.from_pretrained(name, **kwargs), set()

    if (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[1], dict)
    ):
        model, info = result
        # transformers returns missing_keys as a list (4.x) or set (5.x).
        return model, set(info.get("missing_keys", []) or [])

    # Unexpected shape (e.g. a bare model) — treat as no info.
    return (result[0] if isinstance(result, tuple) else result), set()


def _make_dummy_input(
    model: nn.Module,
    *,
    tokenizer: Any | None,
    image_processor: Any | None,
    device: str | torch.device,
) -> dict[str, torch.Tensor] | torch.Tensor | None:
    """Create a small dummy input for forward-pass shape enumeration."""
    if tokenizer is not None:
        try:
            encoded = tokenizer("hello", return_tensors="pt")
            result = {k: v.to(device) for k, v in encoded.items()}
            config = getattr(model, "config", None)
            is_enc_dec = getattr(config, "is_encoder_decoder", False)
            if is_enc_dec and "decoder_input_ids" not in result:
                decoder_start = getattr(config, "decoder_start_token_id", 0) or 0
                result["decoder_input_ids"] = torch.tensor(
                    [[decoder_start]], dtype=torch.long, device=device,
                )
            return result
        except (TypeError, ValueError, RuntimeError):
            pass

    if image_processor is not None:
        try:
            from PIL import Image

            dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            processed = image_processor(images=dummy_img, return_tensors="pt")
            return {k: v.to(device) for k, v in processed.items()}
        except (ImportError, TypeError, ValueError, RuntimeError):
            pass

    config = getattr(model, "config", None)
    if config is not None:
        hidden = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
        if hidden:
            return torch.randn(1, 8, hidden, device=device)

    return None


def _is_hooked_transformer(model: nn.Module) -> bool:
    """Detect a TransformerLens HookedTransformer without importing the library."""
    cls_name = type(model).__name__
    if cls_name in ("HookedTransformer", "HookedEncoder", "HookedEncoderDecoder"):
        return True
    if hasattr(model, "hook_dict") and hasattr(model, "cfg"):
        return True
    return False
