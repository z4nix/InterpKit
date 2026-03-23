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
        result = prepare_input(
            raw,
            tokenizer=self._tokenizer,
            image_processor=self._image_processor,
            device=self._device,
        )
        return self._inject_decoder_ids(result)

    def _prepare_pair(
        self, raw_a: str | torch.Tensor | Any, raw_b: str | torch.Tensor | Any,
    ) -> tuple[dict[str, torch.Tensor] | torch.Tensor, dict[str, torch.Tensor] | torch.Tensor]:
        a, b = prepare_pair(
            raw_a, raw_b,
            tokenizer=self._tokenizer,
            image_processor=self._image_processor,
            device=self._device,
        )
        return self._inject_decoder_ids(a), self._inject_decoder_ids(b)

    def _inject_decoder_ids(
        self, model_input: dict[str, torch.Tensor] | torch.Tensor,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """Add ``decoder_input_ids`` for encoder-decoder models when missing."""
        if not isinstance(model_input, dict):
            return model_input
        if "decoder_input_ids" in model_input:
            return model_input
        config = getattr(self._model, "config", None)
        if not getattr(config, "is_encoder_decoder", False):
            return model_input
        decoder_start = getattr(config, "decoder_start_token_id", 0) or 0
        model_input["decoder_input_ids"] = torch.tensor(
            [[decoder_start]], dtype=torch.long, device=self._device,
        )
        return model_input

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
            if len(out) == 0:
                raise TypeError("Model returned an empty tuple/list — expected tensor output.")
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

        if not at:
            self._cache = {}
            self._cache_input_hash = input_hash
            return self

        result = run_activations(self, input_data, at=at, print_stats=False)
        self._cache = result if isinstance(result, dict) else {at[0]: result}
        self._cache_input_hash = input_hash
        return self

    def clear_cache(self) -> None:
        """Free cached activation tensors and release device memory."""
        self._cache.clear()
        self._cache_input_hash = None
        _empty_device_cache(self._device)

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
        positive: str | torch.Tensor | list | Any,
        negative: str | torch.Tensor | list | Any,
        *,
        at: str,
    ) -> torch.Tensor:
        """Extract a steering vector: mean(act(positives)) - mean(act(negatives)).

        *positive* and *negative* may each be a single input or a list of
        inputs for more robust direction estimation.
        """
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
        causal: bool | None = None,
        save: str | None = None,
        html: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """Show attention patterns. Returns None for non-transformer models.

        Parameters
        ----------
        causal:
            Apply causal mask.  Auto-detected from config if *None*.

        Pass ``save="path.png"`` to export a matplotlib heatmap.
        Pass ``html="path.html"`` to export an interactive HTML page.
        """
        from interpkit.ops.attention import run_attention

        return run_attention(self, input_data, layer=layer, head=head, causal=causal, save=save, html=html)

    def ablate(
        self,
        input_data: str | torch.Tensor | Any,
        *,
        at: str,
        method: str = "zero",
        reference: str | torch.Tensor | Any | None = None,
    ) -> dict[str, Any]:
        """Ablate a module and measure effect on output.

        Parameters
        ----------
        method:
            ``"zero"``, ``"mean"``, or ``"resample"`` (replace with
            activations from *reference*).
        reference:
            Input whose activations replace the target module's output
            when ``method="resample"``.

        Returns a dict with ``effect`` (0 = no change, 1 = max change).
        """
        from interpkit.ops.ablate import run_ablate

        return run_ablate(self, input_data, at=at, method=method, reference=reference)

    def patch(
        self,
        clean: str | torch.Tensor | Any,
        corrupted: str | torch.Tensor | Any,
        *,
        at: str,
        head: int | None = None,
        positions: list[int] | None = None,
        metric: str = "logit_diff",
    ) -> dict[str, Any]:
        """Activation patching: swap a module's output from clean into corrupted.

        Parameters
        ----------
        head:
            Patch only this attention head (requires an attention module with
            a detectable output projection).
        positions:
            Patch only these token positions.
        metric:
            Effect metric: ``"logit_diff"`` (default), ``"kl_div"``,
            ``"target_prob"``, or ``"l2_prob"``.

        Returns a dict with ``clean_logits``, ``corrupted_logits``, ``patched_logits``,
        and ``effect``.
        """
        from interpkit.ops.patch import run_patch

        return run_patch(self, clean, corrupted, at=at, head=head, positions=positions, metric=metric)

    def trace(
        self,
        clean: str | torch.Tensor | Any,
        corrupted: str | torch.Tensor | Any,
        *,
        top_k: int | None = 20,
        mode: str = "module",
        metric: str = "logit_diff",
        save: str | None = None,
        html: str | None = None,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Causal tracing: rank modules by how much patching them restores clean output.

        Parameters
        ----------
        mode:
            ``"module"`` (default) — two-phase module-level tracing.
            ``"position"`` — Meng et al. style (layer x position) heatmap.
        metric:
            Effect metric: ``"logit_diff"`` (default), ``"kl_div"``,
            ``"target_prob"``, or ``"l2_prob"``.

        Pass ``save="path.png"`` to export a matplotlib figure.
        Pass ``html="path.html"`` to export an interactive HTML page.
        """
        from interpkit.ops.trace import run_trace

        return run_trace(self, clean, corrupted, top_k=top_k, mode=mode, metric=metric, save=save, html=html)

    def lens(
        self,
        text: str | torch.Tensor | Any,
        *,
        save: str | None = None,
        html: str | None = None,
        position: int | None = None,
    ) -> list[dict[str, Any]] | None:
        """Logit lens: project each layer's output to vocabulary space.

        Analyses all token positions by default, producing the classic
        (layers x positions) heatmap.  Pass ``position=-1`` to analyse only
        the last token (original behaviour) or ``position=N`` for any single
        position.

        Pass ``save="path.png"`` to export a matplotlib heatmap.
        Pass ``html="path.html"`` to export an interactive HTML page.
        """
        from interpkit.ops.lens import run_lens

        return run_lens(self, text, save=save, html=html, position=position)

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
        attribute: bool = False,
    ) -> dict[str, Any]:
        """Decompose activations at *at* through a Sparse Autoencoder.

        Parameters
        ----------
        sae:
            Either a HuggingFace repo ID (``"jbloom/GPT2-Small-SAEs-Reformatted"``)
            or a pre-loaded :class:`interpkit.ops.sae.SAE` object.
        attribute:
            When ``True``, compute each top feature's logit contribution
            through the decoder → unembedding path.
        """
        from interpkit.ops.sae import SAE as SAEClass
        from interpkit.ops.sae import load_sae, run_features

        if isinstance(sae, str):
            sae = load_sae(sae, device=self._device)
        elif not isinstance(sae, SAEClass):
            raise TypeError(f"Expected SAE or HF repo ID string, got {type(sae).__name__}")

        return run_features(self, input_data, at=at, sae=sae, top_k=top_k, attribute=attribute)

    def contrastive_features(
        self,
        positive_inputs: list[Any],
        negative_inputs: list[Any],
        *,
        at: str,
        sae: str | Any,
        top_k: int = 20,
    ) -> dict[str, Any]:
        """Compare SAE feature activations between positive and negative groups.

        Returns features ranked by absolute differential activation,
        surfacing features that distinguish the two concepts.
        """
        from interpkit.ops.sae import SAE as SAEClass
        from interpkit.ops.sae import load_sae, run_contrastive_features

        if isinstance(sae, str):
            sae = load_sae(sae, device=self._device)
        elif not isinstance(sae, SAEClass):
            raise TypeError(f"Expected SAE or HF repo ID string, got {type(sae).__name__}")

        return run_contrastive_features(
            self, positive_inputs, negative_inputs, at=at, sae=sae, top_k=top_k,
        )

    def attribute(
        self,
        input_data: str | torch.Tensor | Any,
        *,
        target: int | None = None,
        method: str = "integrated_gradients",
        n_steps: int = 50,
        save: str | None = None,
        html: str | None = None,
    ) -> dict[str, Any]:
        """Gradient-based attribution over the input.

        Parameters
        ----------
        method:
            ``"integrated_gradients"`` (default), ``"gradient"``, or
            ``"gradient_x_input"``.
        n_steps:
            Interpolation steps for integrated gradients (default 50).

        For NLP: returns ``{"tokens", "scores", "target"}`` with per-token importance.
        For vision: returns ``{"grad", "target"}`` with the pixel-gradient tensor.
        Pass ``save="path.png"`` to export a matplotlib figure.
        Pass ``html="path.html"`` to export an interactive HTML page.
        """
        from interpkit.ops.attribute import run_attribute

        return run_attribute(self, input_data, target=target, method=method, n_steps=n_steps, save=save, html=html)

    def dla(
        self,
        input_data: str | torch.Tensor | Any,
        *,
        token: int | str | None = None,
        position: int = -1,
        top_k: int = 10,
        save: str | None = None,
        html: str | None = None,
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
            top_k=top_k, save=save, html=html,
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
        clean: str | torch.Tensor | list | Any,
        corrupted: str | torch.Tensor | list | Any,
        *,
        threshold: float = 0.01,
        method: str = "mean",
        metric: str = "logit_diff",
    ) -> dict[str, Any]:
        """Discover the minimal circuit that explains a behaviour.

        Identifies which attention heads and MLPs are necessary by
        individually ablating each component and keeping those whose
        ablation changes the output by more than *threshold*.

        *clean* and *corrupted* may each be a single input or parallel
        lists for multi-pair circuit discovery (effects are averaged).

        Parameters
        ----------
        method:
            Ablation method: ``"mean"`` (default), ``"zero"``, or
            ``"resample"`` (uses corrupted activations).
        metric:
            Effect metric: ``"logit_diff"`` (default), ``"kl_div"``,
            ``"target_prob"``, or ``"l2_prob"``.

        Returns a dict with ``circuit`` (list of important components),
        ``excluded``, ``verification`` (faithfulness check), and
        ``threshold``.
        """
        from interpkit.ops.find_circuit import run_find_circuit

        return run_find_circuit(
            self, clean, corrupted, threshold=threshold, method=method, metric=metric,
        )


    def report(
        self,
        input_data: str | torch.Tensor | Any,
        *,
        save: str = "report.html",
    ) -> dict[str, Any]:
        """Generate a comprehensive HTML report: prediction, DLA, logit lens,
        attention, and attribution combined in a single interactive document.

        Returns a dict with section results and ``html_path``.
        """
        from interpkit.ops.report import run_report

        return run_report(self, input_data, save=save)


# ======================================================================
# Device helpers
# ======================================================================


def _resolve_device() -> str:
    """Auto-detect the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _empty_device_cache(device: torch.device | str) -> None:
    """Release cached memory for the given device backend."""
    dev = str(device)
    if dev.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif dev == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


# ======================================================================
# Top-level loader
# ======================================================================


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
    # Resolve dtype string shortcuts
    _dtype_map = {
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

    # MPS does not fully support bfloat16 in many PyTorch versions;
    # silently downgrade to float16 to avoid hard-to-debug errors.
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

        # Detect TransformerLens HookedTransformer
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

    # Build Auto class priority from config.architectures so we pick the
    # right task head (e.g. QuestionAnswering, not CausalLM) automatically.
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
    auto_order.extend([
        "AutoModelForCausalLM",
        "AutoModelForSeq2SeqLM",
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
            result = {k: v.to(device) for k, v in encoded.items()}
            config = getattr(model, "config", None)
            is_enc_dec = getattr(config, "is_encoder_decoder", False)
            if is_enc_dec and "decoder_input_ids" not in result:
                decoder_start = getattr(config, "decoder_start_token_id", 0) or 0
                result["decoder_input_ids"] = torch.tensor(
                    [[decoder_start]], dtype=torch.long, device=device,
                )
            return result
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
            if isinstance(v, torch.Tensor):
                h.update(v.detach().cpu().contiguous().numpy().tobytes())
            else:
                h.update(repr(v).encode())
    else:
        h.update(model_input.detach().cpu().contiguous().numpy().tobytes())
    return int.from_bytes(h.digest()[:8], "little")


def _is_hooked_transformer(model: nn.Module) -> bool:
    """Detect a TransformerLens HookedTransformer without importing the library."""
    cls_name = type(model).__name__
    if cls_name in ("HookedTransformer", "HookedEncoder", "HookedEncoderDecoder"):
        return True
    if hasattr(model, "hook_dict") and hasattr(model, "cfg"):
        return True
    return False
