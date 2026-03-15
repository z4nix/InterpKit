"""Universal model wrapper — load any HF model or nn.Module and run mech interp ops."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from interpkit.core.discovery import ModelArchInfo, discover
from interpkit.core.inputs import prepare_input, prepare_pair
from interpkit.core.registry import Registration, get_registration


class Model:
    """Wraps a PyTorch model for mechanistic interpretability operations.

    Created via :func:`interpkit.load` — not instantiated directly.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        tokenizer: Any | None = None,
        image_processor: Any | None = None,
        arch_info: ModelArchInfo,
        registration: Registration | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._image_processor = image_processor
        self.arch_info = arch_info
        self._registration = registration
        self._device = torch.device(device)
        self._cache: dict[str, torch.Tensor] = {}
        self._cache_input_hash: int | None = None

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def _prepare(self, raw: str | torch.Tensor | Any) -> dict[str, torch.Tensor] | torch.Tensor:
        return prepare_input(
            raw,
            tokenizer=self._tokenizer,
            image_processor=self._image_processor,
            device=self._device,
        )

    def _prepare_pair(
        self, raw_a: str | torch.Tensor | Any, raw_b: str | torch.Tensor | Any,
    ) -> tuple[dict[str, torch.Tensor] | torch.Tensor, dict[str, torch.Tensor] | torch.Tensor]:
        return prepare_pair(
            raw_a, raw_b,
            tokenizer=self._tokenizer,
            image_processor=self._image_processor,
            device=self._device,
        )

    def _forward(self, model_input: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return the output logits / final tensor."""
        with torch.no_grad():
            return self._forward_with_grad(model_input)

    def _forward_with_grad(self, model_input: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Like ``_forward`` but without ``torch.no_grad()`` — use for gradient-based ops."""
        if isinstance(model_input, dict):
            out = self._model(**model_input)
        else:
            out = self._model(model_input)

        if hasattr(out, "logits"):
            return out.logits
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, (tuple, list)):
            return out[0]
        raise TypeError(f"Unexpected model output type: {type(out).__name__}")

    # ------------------------------------------------------------------
    # Activation cache
    # ------------------------------------------------------------------

    @property
    def cached(self) -> bool:
        """True if the activation cache is populated."""
        return len(self._cache) > 0

    def cache(
        self,
        input_data: str | torch.Tensor | Any,
        *,
        at: list[str] | None = None,
    ) -> "Model":
        """Run a forward pass and cache activations for reuse by other operations.

        Parameters
        ----------
        input_data:
            The input to cache activations for.
        at:
            Module names to cache. If None, caches all modules with parameters.

        Returns ``self`` for chaining.
        """
        from interpkit.ops.activations import run_activations

        model_input = self._prepare(input_data)
        input_hash = _hash_input(model_input)

        if at is None:
            at = [m.name for m in self.arch_info.modules if m.param_count > 0]

        result = run_activations(self, input_data, at=at, print_stats=False)
        self._cache = result if isinstance(result, dict) else {at[0]: result}
        self._cache_input_hash = input_hash
        return self

    def clear_cache(self) -> None:
        """Free cached activation tensors."""
        self._cache.clear()
        self._cache_input_hash = None

    def _get_cached(
        self,
        input_data: str | torch.Tensor | Any,
        module_names: list[str],
        *,
        _prepared_input: dict[str, torch.Tensor] | torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Return cached activations if available for this input, else None.

        Pass *_prepared_input* to avoid re-tokenizing when the caller
        already has the prepared input.
        """
        if not self._cache:
            return None

        model_input = _prepared_input if _prepared_input is not None else self._prepare(input_data)
        input_hash = _hash_input(model_input)

        if input_hash != self._cache_input_hash:
            return None

        if all(name in self._cache for name in module_names):
            return {name: self._cache[name] for name in module_names}

        return None

    # ------------------------------------------------------------------
    # Public operations — delegate to ops/
    # ------------------------------------------------------------------

    def inspect(self) -> None:
        """Print the model's module tree with types, param counts, and detected roles."""
        from interpkit.ops.inspect import run_inspect

        run_inspect(self)

    def activations(
        self,
        input_data: str | torch.Tensor | Any,
        *,
        at: str | list[str],
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """Extract raw activation tensors at one or more named modules.

        Returns a single tensor if *at* is a string, or a dict if *at* is a list.
        """
        from interpkit.ops.activations import run_activations

        return run_activations(self, input_data, at=at)

    def head_activations(
        self,
        input_data: str | torch.Tensor | Any,
        *,
        at: str,
        output_proj: bool = True,
    ) -> dict[str, Any]:
        """Decompose an attention module's output into per-head contributions.

        Returns a dict with ``head_acts`` (tensor of shape
        ``(num_heads, batch, seq, dim)``), ``num_heads``, ``head_dim``,
        and ``module``.

        When *output_proj* is True (default), each head's output is projected
        through its slice of W_o so the result lives in residual-stream space.
        """
        from interpkit.ops.heads import run_head_activations

        return run_head_activations(self, input_data, at=at, output_proj=output_proj)

    def steer_vector(
        self,
        positive: str | torch.Tensor | Any,
        negative: str | torch.Tensor | Any,
        *,
        at: str,
    ) -> torch.Tensor:
        """Extract a steering vector: activation(positive) - activation(negative)."""
        from interpkit.ops.steer import run_steer_vector

        return run_steer_vector(self, positive, negative, at=at)

    def steer(
        self,
        input_data: str | torch.Tensor | Any,
        *,
        vector: torch.Tensor,
        at: str,
        scale: float = 2.0,
        save: str | None = None,
    ) -> dict[str, Any]:
        """Run inference with a steering vector added at module *at*.

        Shows side-by-side comparison of original vs steered top predictions.
        Pass ``save="path.png"`` to export a matplotlib figure.
        """
        from interpkit.ops.steer import run_steer

        return run_steer(self, input_data, vector=vector, at=at, scale=scale, save=save)

    def attention(
        self,
        input_data: str | torch.Tensor | Any,
        *,
        layer: int | None = None,
        head: int | None = None,
        save: str | None = None,
        html: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """Show attention patterns. Returns None for non-transformer models.

        Pass ``save="path.png"`` to export a matplotlib heatmap.
        Pass ``html="path.html"`` to export an interactive HTML page.
        """
        from interpkit.ops.attention import run_attention

        return run_attention(self, input_data, layer=layer, head=head, save=save, html=html)

    def ablate(
        self,
        input_data: str | torch.Tensor | Any,
        *,
        at: str,
        method: str = "zero",
    ) -> dict[str, Any]:
        """Ablate a module (zero or mean) and measure effect on output.

        Returns a dict with ``effect`` (0 = no change, 1 = max change).
        """
        from interpkit.ops.ablate import run_ablate

        return run_ablate(self, input_data, at=at, method=method)

    def patch(
        self,
        clean: str | torch.Tensor | Any,
        corrupted: str | torch.Tensor | Any,
        *,
        at: str,
        head: int | None = None,
        positions: list[int] | None = None,
    ) -> dict[str, Any]:
        """Activation patching: swap a module's output from clean into corrupted.

        Parameters
        ----------
        head:
            Patch only this attention head (requires an attention module with
            a detectable output projection).
        positions:
            Patch only these token positions.

        Returns a dict with ``clean_logits``, ``corrupted_logits``, ``patched_logits``,
        and ``effect`` (normalised scalar measuring how much the patch restored clean behaviour).
        """
        from interpkit.ops.patch import run_patch

        return run_patch(self, clean, corrupted, at=at, head=head, positions=positions)

    def trace(
        self,
        clean: str | torch.Tensor | Any,
        corrupted: str | torch.Tensor | Any,
        *,
        top_k: int | None = 20,
        mode: str = "module",
        save: str | None = None,
        html: str | None = None,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Causal tracing: rank modules by how much patching them restores clean output.

        Parameters
        ----------
        mode:
            ``"module"`` (default) — two-phase module-level tracing.
            ``"position"`` — Meng et al. style (layer x position) heatmap.

        Pass ``save="path.png"`` to export a matplotlib figure.
        Pass ``html="path.html"`` to export an interactive HTML page.
        """
        from interpkit.ops.trace import run_trace

        return run_trace(self, clean, corrupted, top_k=top_k, mode=mode, save=save, html=html)

    def lens(
        self,
        text: str | torch.Tensor | Any,
        *,
        save: str | None = None,
        position: int | None = None,
    ) -> list[dict[str, Any]] | None:
        """Logit lens: project each layer's output to vocabulary space.

        Analyses all token positions by default, producing the classic
        (layers x positions) heatmap.  Pass ``position=-1`` to analyse only
        the last token (original behaviour) or ``position=N`` for any single
        position.

        Pass ``save="path.png"`` to export a matplotlib heatmap.
        """
        from interpkit.ops.lens import run_lens

        return run_lens(self, text, save=save, position=position)

    def probe(
        self,
        texts: list[str],
        labels: list[int],
        *,
        at: str,
    ) -> dict[str, Any]:
        """Train a linear probe on activations at module *at*.

        Returns accuracy, top features by weight magnitude.
        Requires scikit-learn (``pip install interpkit[probe]``), falls back to
        a torch-based probe otherwise.
        """
        from interpkit.ops.probe import run_probe

        return run_probe(self, texts, labels, at=at)

    def features(
        self,
        input_data: str | torch.Tensor | Any,
        *,
        at: str,
        sae: str | Any,
        top_k: int = 20,
    ) -> dict[str, Any]:
        """Decompose activations at *at* through a Sparse Autoencoder.

        Parameters
        ----------
        sae:
            Either a HuggingFace repo ID (``"jbloom/GPT2-Small-SAEs-Reformatted"``)
            or a pre-loaded :class:`interpkit.ops.sae.SAE` object.
        """
        from interpkit.ops.sae import SAE as SAEClass
        from interpkit.ops.sae import load_sae, run_features

        if isinstance(sae, str):
            sae = load_sae(sae, device=self._device)
        elif not isinstance(sae, SAEClass):
            raise TypeError(f"Expected SAE or HF repo ID string, got {type(sae).__name__}")

        return run_features(self, input_data, at=at, sae=sae, top_k=top_k)

    def attribute(
        self,
        input_data: str | torch.Tensor | Any,
        *,
        target: int | None = None,
        save: str | None = None,
        html: str | None = None,
    ) -> dict[str, Any]:
        """Gradient saliency over the input.

        For NLP: returns ``{"tokens", "scores", "target"}`` with per-token importance.
        For vision: returns ``{"grad", "target"}`` with the pixel-gradient tensor.
        Pass ``save="path.png"`` to export a matplotlib figure.
        Pass ``html="path.html"`` to export an interactive HTML page.
        """
        from interpkit.ops.attribute import run_attribute

        return run_attribute(self, input_data, target=target, save=save, html=html)

    def dla(
        self,
        input_data: str | torch.Tensor | Any,
        *,
        token: int | str | None = None,
        position: int = -1,
        top_k: int = 10,
        save: str | None = None,
    ) -> dict[str, Any]:
        """Direct Logit Attribution: decompose the output logit by component.

        For each layer, measures how much the attention block and MLP
        contribute to the logit of *token* by projecting their outputs
        through the unembedding matrix.  Also provides a per-head breakdown.

        Returns a dict with ``target_token``, ``target_id``,
        ``contributions`` (list sorted by magnitude),
        ``head_contributions`` (per-head breakdown), and ``total_logit``.
        """
        from interpkit.ops.dla import run_dla

        return run_dla(
            self, input_data, token=token, position=position,
            top_k=top_k, save=save,
        )

    # ------------------------------------------------------------------
    # Batch / dataset operations
    # ------------------------------------------------------------------

    def batch(
        self,
        operation: str,
        dataset: list[dict[str, Any]],
        *,
        op_kwargs: dict[str, Any] | None = None,
        aggregate: bool = True,
    ) -> dict[str, Any]:
        """Run any operation over a dataset of examples.

        Parameters
        ----------
        operation:
            Method name: ``"trace"``, ``"patch"``, ``"dla"``, ``"attribute"``, etc.
        dataset:
            List of dicts, each unpacked as kwargs to the operation.
        op_kwargs:
            Extra kwargs applied to every call.
        aggregate:
            Compute summary statistics across all results.
        """
        from interpkit.ops.batch import run_batch

        return run_batch(
            self, operation, dataset, op_kwargs=op_kwargs, aggregate=aggregate,
        )

    def trace_batch(
        self,
        dataset: list[dict[str, str]],
        *,
        clean_col: str = "clean",
        corrupted_col: str = "corrupted",
        top_k: int | None = 20,
        mode: str = "module",
    ) -> dict[str, Any]:
        """Run causal tracing over a dataset of (clean, corrupted) pairs."""
        from interpkit.ops.batch import run_trace_batch

        return run_trace_batch(
            self, dataset, clean_col=clean_col, corrupted_col=corrupted_col,
            top_k=top_k, mode=mode,
        )

    def dla_batch(
        self,
        texts: list[str],
        *,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Run Direct Logit Attribution over a list of texts."""
        from interpkit.ops.batch import run_dla_batch

        return run_dla_batch(self, texts, top_k=top_k)

    # ------------------------------------------------------------------
    # Scan — automated multi-analysis
    # ------------------------------------------------------------------

    def scan(
        self,
        input_data: str | torch.Tensor | Any,
        *,
        save: str | None = None,
    ) -> dict[str, Any]:
        """One-command model overview: runs DLA, logit lens, and attention analysis.

        Automatically surfaces the most interesting findings.  Pass
        ``save="prefix"`` to export figures (e.g. ``prefix_dla.png``).
        """
        from interpkit.ops.scan import run_scan

        return run_scan(self, input_data, save=save)

    # ------------------------------------------------------------------
    # Residual stream decomposition & circuit analysis
    # ------------------------------------------------------------------

    def decompose(
        self,
        input_data: str | torch.Tensor | Any,
        *,
        position: int = -1,
    ) -> dict[str, Any]:
        """Decompose the residual stream into per-component contributions.

        Returns a dict with ``components`` (list of per-component
        ``{name, layer, type, vector, norm}``), ``residual`` (final
        residual stream vector), and ``position``.
        """
        from interpkit.ops.circuits import run_decompose

        return run_decompose(self, input_data, position=position)

    def ov_scores(self, *, layer: int) -> dict[str, Any]:
        """Analyse OV circuits: compute W_OV = W_O @ W_V for each head.

        Returns per-head Frobenius norms, singular values, and approximate
        ranks of the effective OV matrix.
        """
        from interpkit.ops.circuits import run_ov_scores

        return run_ov_scores(self, layer=layer)

    def qk_scores(self, *, layer: int) -> dict[str, Any]:
        """Analyse QK circuits: compute W_QK = W_Q^T @ W_K for each head.

        Returns per-head Frobenius norms, singular values, and approximate
        ranks of the effective QK matrix.
        """
        from interpkit.ops.circuits import run_qk_scores

        return run_qk_scores(self, layer=layer)

    def composition(
        self,
        *,
        src_layer: int,
        dst_layer: int,
        comp_type: str = "q",
    ) -> dict[str, Any]:
        """Compute composition scores between heads in two layers.

        Parameters
        ----------
        comp_type:
            ``"q"`` for Q-composition, ``"k"`` for K-composition,
            ``"v"`` for V-composition.

        Returns a dict with ``scores`` (tensor ``dst_heads x src_heads``),
        ``src_layer``, ``dst_layer``, ``comp_type``.
        """
        from interpkit.ops.circuits import run_composition

        return run_composition(
            self, src_layer=src_layer, dst_layer=dst_layer, comp_type=comp_type,
        )

    def find_circuit(
        self,
        clean: str | torch.Tensor | Any,
        corrupted: str | torch.Tensor | Any,
        *,
        threshold: float = 0.01,
    ) -> dict[str, Any]:
        """Discover the minimal circuit that explains a behaviour.

        Identifies which attention heads and MLPs are necessary by
        individually ablating each component and keeping those whose
        ablation changes the output by more than *threshold*.

        Returns a dict with ``circuit`` (list of important components),
        ``excluded``, ``verification`` (faithfulness check), and
        ``threshold``.
        """
        from interpkit.ops.find_circuit import run_find_circuit

        return run_find_circuit(
            self, clean, corrupted, threshold=threshold,
        )


# ======================================================================
# Top-level loader
# ======================================================================


def load(
    model_or_name: str | nn.Module,
    *,
    tokenizer: Any | None = None,
    image_processor: Any | None = None,
    device: str | torch.device | None = None,
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
        Device to run on. Defaults to CUDA if available, else CPU.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    is_tl = False

    if isinstance(model_or_name, str):
        model, tokenizer, image_processor = _load_from_hf(
            model_or_name, tokenizer=tokenizer, image_processor=image_processor, device=device
        )
    else:
        model = model_or_name

        # Detect TransformerLens HookedTransformer
        if _is_hooked_transformer(model):
            is_tl = True
            if tokenizer is None:
                tl_tok = getattr(model, "tokenizer", None)
                if tl_tok is not None:
                    tokenizer = tl_tok
                    if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token

        model.to(device)

    model.eval()
    registration = get_registration(model)

    # Build a dummy input for shape enumeration
    # TL models accept a raw token tensor, not tokenizer dict kwargs
    if is_tl:
        dummy = torch.tensor([[0]], device=device)
    else:
        dummy = _make_dummy_input(model, tokenizer=tokenizer, image_processor=image_processor, device=device)
    arch_info = discover(model, dummy_input=dummy)
    arch_info.is_tl_model = is_tl

    # Merge manual registration into arch_info
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
        device=device,
    )


def _load_from_hf(
    name: str,
    *,
    tokenizer: Any | None,
    image_processor: Any | None,
    device: str | torch.device,
) -> tuple[nn.Module, Any | None, Any | None]:
    from transformers import AutoModel, AutoTokenizer

    # Try loading as a causal/seq2seq/masked LM first, then fall back to AutoModel
    model = None
    for auto_cls_name in (
        "AutoModelForCausalLM",
        "AutoModelForSeq2SeqLM",
        "AutoModelForMaskedLM",
        "AutoModelForImageClassification",
        "AutoModel",
    ):
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(name)
            import transformers

            auto_cls = getattr(transformers, auto_cls_name)
            model = auto_cls.from_pretrained(name, config=config)
            break
        except (ValueError, OSError, KeyError):
            continue

    if model is None:
        model = AutoModel.from_pretrained(name)

    model = model.to(device)

    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
        except Exception:
            pass

    if tokenizer is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if image_processor is None:
        try:
            from transformers import AutoImageProcessor

            image_processor = AutoImageProcessor.from_pretrained(name)
        except (OSError, KeyError, ImportError):
            pass

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
            return {k: v.to(device) for k, v in encoded.items()}
        except Exception:
            pass

    if image_processor is not None:
        try:
            from PIL import Image

            dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            processed = image_processor(images=dummy_img, return_tensors="pt")
            return {k: v.to(device) for k, v in processed.items()}
        except Exception:
            pass

    # Fallback: try a simple tensor
    config = getattr(model, "config", None)
    if config is not None:
        hidden = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
        if hidden:
            return torch.randn(1, 8, hidden, device=device)

    return None


def _hash_input(model_input: dict[str, torch.Tensor] | torch.Tensor) -> int:
    """Compute a hash of a model input for cache key comparison.

    Uses the raw byte content of tensors to avoid collisions from inputs
    that happen to share the same sum/shape.
    """
    import hashlib

    h = hashlib.sha256()
    if isinstance(model_input, dict):
        for k in sorted(model_input.keys()):
            v = model_input[k]
            h.update(k.encode())
            h.update(v.detach().cpu().float().contiguous().numpy().tobytes())
    else:
        h.update(model_input.detach().cpu().float().contiguous().numpy().tobytes())
    return int.from_bytes(h.digest()[:8], "little")


def _is_hooked_transformer(model: nn.Module) -> bool:
    """Detect a TransformerLens HookedTransformer without importing the library."""
    cls_name = type(model).__name__
    if cls_name in ("HookedTransformer", "HookedEncoder", "HookedEncoderDecoder"):
        return True
    if hasattr(model, "hook_dict") and hasattr(model, "cfg"):
        return True
    return False
