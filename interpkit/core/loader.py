"""Model loading — HuggingFace, TransformerLens, and raw nn.Module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from interpkit.core.discovery import discover
from interpkit.core.registry import get_registration

if TYPE_CHECKING:
    from interpkit.core.model import Model


def load(
    model_or_name: str | nn.Module,
    *,
    tokenizer: Any | None = None,
    image_processor: Any | None = None,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    device_map: str | dict | None = None,
) -> Model:
    """Load a model for mechanistic interpretability.

    Parameters
    ----------
    model_or_name:
        A HuggingFace model ID (``"gpt2"``, ``"microsoft/resnet-50"``)
        or an existing ``nn.Module`` instance.
    tokenizer:
        An explicit tokenizer. Auto-loaded for HF models if not provided.
    image_processor:
        An explicit image processor. Auto-loaded for HF vision models if not provided.
    device:
        Device to run on. Auto-detected (cuda > mps > cpu) if omitted.
        Ignored when *device_map* is set (HF handles placement).
    dtype:
        Model dtype: ``"float16"``, ``"bfloat16"``, ``"float32"``,
        ``"auto"``, or a ``torch.dtype``.  Maps to ``torch_dtype`` in
        HuggingFace ``from_pretrained``.
    device_map:
        HuggingFace ``device_map`` for multi-GPU / offload placement
        (e.g. ``"auto"``).  Requires the ``accelerate`` package.
    """
    from interpkit.core.model import Model

    _dtype_map: dict[str, torch.dtype | str] = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "auto": "auto",
    }
    torch_dtype: torch.dtype | str | None = None
    if dtype is not None:
        if isinstance(dtype, str):
            if dtype not in _dtype_map:
                raise ValueError(
                    f"Unknown dtype {dtype!r}. Allowed values: {', '.join(sorted(_dtype_map.keys()))}"
                )
            torch_dtype = _dtype_map[dtype]
        else:
            torch_dtype = dtype

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
        model, tokenizer, image_processor = _load_from_hf(
            model_or_name,
            tokenizer=tokenizer,
            image_processor=image_processor,
            device=device,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
    else:
        model = model_or_name

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
    registration = get_registration(model)

    if is_tl:
        dummy: dict[str, torch.Tensor] | torch.Tensor | None = torch.tensor([[0]], device=device)
    else:
        dummy = _make_dummy_input(model, tokenizer=tokenizer, image_processor=image_processor, device=device or "cpu")
    arch_info = discover(model, dummy_input=dummy)
    arch_info.is_tl_model = is_tl

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
        model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        arch_info=arch_info,
        registration=registration,
        device=device or "cpu",
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
) -> tuple[nn.Module, Any | None, Any | None]:
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    extra_kwargs: dict[str, Any] = {}
    if torch_dtype is not None:
        extra_kwargs["torch_dtype"] = torch_dtype
    if device_map is not None:
        extra_kwargs["device_map"] = device_map

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
    for auto_cls_name in auto_order:
        auto_cls = getattr(transformers, auto_cls_name, None)
        if auto_cls is None:
            continue
        try:
            model = auto_cls.from_pretrained(name, config=config, **extra_kwargs)
            break
        except (ValueError, OSError, KeyError):
            continue

    if model is None:
        model = AutoModel.from_pretrained(name, **extra_kwargs)

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
