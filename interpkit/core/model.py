"""Universal model wrapper — load any HF model or nn.Module and run mech interp ops."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from interpkit.core.arch import ArchInfo
from interpkit.core.cache import empty_device_cache, hash_input
from interpkit.core.inputs import _looks_like_image_path, prepare_input, prepare_pair
from interpkit.core.loader import (
    _is_hooked_transformer,
    _load_from_hf,
    _make_dummy_input,
    _resolve_device,
    load,
    load_module,
)
from interpkit.core.registry import Registration


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
        arch_info: ArchInfo,
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
        # Lazy-loaded eager-attention copy for ops that need real attention
        # weights on SDPA/FlashAttention models (Phase 2).
        self._eager_model: nn.Module | None = None

    # ------------------------------------------------------------------
    # Public properties — surface dtype / device for ergonomics (F-017)
    # ------------------------------------------------------------------

    @property
    def device(self) -> str:
        """Device string (``"cpu"``, ``"cuda"``, ``"mps"``, ``"cuda:0"``).

        Pre-1.0 this was a private ``_device`` attribute holding a
        ``torch.device``; users reflexively typing ``model.device`` got
        an ``AttributeError`` (F-017).
        """
        return str(self._device)

    @property
    def dtype(self) -> torch.dtype:
        """Resolved dtype of the underlying model parameters.

        Surfaces the actual dtype of the loaded weights so users can
        confirm precision (F-007 / F-017).
        """
        try:
            return next(self._model.parameters()).dtype
        except StopIteration:
            return torch.float32

    def __repr__(self) -> str:
        family = getattr(self.arch_info, "family", None)
        family_str = family.value if family is not None and hasattr(family, "value") else "unknown"
        arch = getattr(self.arch_info, "arch_family", None) or type(self._model).__name__
        return (
            f"Model({arch}, family={family_str!r}, "
            f"device={self.device!r}, dtype={self.dtype})"
        )

    # ------------------------------------------------------------------
    # Eager-attention reload cache (Phase 2: F-001 / F-002)
    # ------------------------------------------------------------------

    def unload_eager_attention(self) -> None:
        """Free the cached eager-attention model copy.

        :meth:`Model.attention` and :meth:`Model.head_activations` may
        lazily reload the underlying model with
        ``attn_implementation="eager"`` (saved on ``self._eager_model``)
        when the primary model uses SDPA or FlashAttention. On a large
        fp16 model this second copy roughly doubles VRAM usage.

        This method releases the eager copy. PyTorch can retain
        references to module parameters via hook handles, weakrefs, or
        the autograd graph, so a plain ``del self._eager_model`` does
        not always free the memory. We therefore:

        1. Set ``self._eager_model`` back to ``None``.
        2. Run ``gc.collect()`` to drop any temporary references.
        3. Call ``empty_device_cache(self._device)`` to release the
           backend's reserved memory.

        After this call the next ``attention()`` invocation triggers a
        fresh reload — pair with ``attention_impl="eager"`` at load
        time if you'll be calling attention ops repeatedly and want to
        avoid the reload cost.
        """
        import gc

        if self._eager_model is None:
            return
        # Drop the reference. If the eager model IS the primary model
        # (the early-return path in _ensure_eager_attention when the
        # primary is already eager), keep the primary intact.
        if self._eager_model is self._model:
            self._eager_model = None
            return
        self._eager_model = None
        gc.collect()
        empty_device_cache(self._device)

    def _ensure_eager_attention(self) -> nn.Module:
        """Return a model with eager attention; reload from HF if needed.

        Pre-1.0 interpkit's ``attention()`` op silently returned wrong
        weights when the primary model was loaded with SDPA / FlashAttention
        backends (now the HF default for transformers 5.x). The fix is
        to load a separate model copy with ``attn_implementation="eager"``
        and use it for any op that needs real attention weights.

        The eager model is cached on the instance, so the reload cost is
        amortised across all attention-weight ops in the session. Falls
        back to the primary model if it's already eager (zero cost).

        Raises
        ------
        AttentionBackendUnavailable
            If the primary model has no ``config`` attribute (custom
            non-PreTrainedModel modules) or the eager reload itself fails.
        """
        if self._eager_model is not None:
            return self._eager_model

        config = getattr(self._model, "config", None)
        attn_impl = getattr(config, "_attn_implementation", None) if config is not None else None

        # Primary model is already eager → no reload needed.
        if attn_impl == "eager":
            self._eager_model = self._model
            return self._model

        # Custom nn.Module without config — can't reload via HF.
        if config is None or not hasattr(config, "_name_or_path"):
            from interpkit.core.exceptions import AttentionBackendUnavailable
            raise AttentionBackendUnavailable(
                f"Cannot reload {type(self._model).__name__} with eager "
                "attention: model is not a HuggingFace PreTrainedModel "
                "loaded from a known repo. Use `interpkit.load(model_id)` "
                "with a HF model id to enable attention()."
            )

        # Reload the model with attn_implementation='eager'. This downloads
        # nothing if the weights are already cached locally.
        try:
            from interpkit.core.loader import _load_from_hf
            model_name = config._name_or_path
            eager, _, _ = _load_from_hf(
                model_name,
                tokenizer=None,
                image_processor=None,
                device=self._device,
                torch_dtype=next(self._model.parameters()).dtype,
                device_map=None,
            )
            # Force eager backend on the reloaded model's config.
            eager.config._attn_implementation = "eager"
            eager.eval()
            self._eager_model = eager
            return eager
        except Exception as exc:
            from interpkit.core.exceptions import AttentionBackendUnavailable
            raise AttentionBackendUnavailable(
                f"Could not reload {type(self._model).__name__} with eager "
                f"attention ({type(exc).__name__}: {exc}). Try loading the "
                f"model directly with attn_implementation='eager' or use "
                f"interpkit.load(...) on a different model id."
            ) from exc

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def _reject_wrong_input_type(self, raw: Any) -> None:
        """Fail loud when a vision model receives a text string (A2 / NR vision UX).

        The operation is supported for the family — the *input* is wrong.
        This lives on :class:`Model` (not in ``prepare_input``) because the
        ``inputs`` helpers have no ``arch_info``, and it is called from both
        :meth:`_prepare` and :meth:`_prepare_pair` so pair-based ops (diff /
        patch / ablate) are covered too, not just the single-input path.
        """
        arch = getattr(self, "arch_info", None)
        if arch is None or not getattr(arch, "spatial", False):
            return
        if not isinstance(raw, str):
            return
        # A .pt path loads a tensor (valid vision input); an image path is
        # the expected input. Everything else is a text string the vision
        # model cannot consume.
        if _looks_like_image_path(raw) or raw.endswith(".pt"):
            return
        from interpkit.core.exceptions import WrongInputType

        family = arch.family.value if hasattr(arch.family, "value") else str(arch.family)
        raise WrongInputType(
            f"This model is a vision model (family={family}). "
            f"Pass an image path, a (B, C, H, W) tensor, or call "
            f"interpkit.load(..., image_processor=...). "
            f"The string {raw!r} is not an image path."
        )

    def _prepare(self, raw: str | torch.Tensor | Any) -> dict[str, torch.Tensor] | torch.Tensor:
        self._reject_wrong_input_type(raw)
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
        self._reject_wrong_input_type(raw_a)
        self._reject_wrong_input_type(raw_b)
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
        # C2: single source of truth for the seq2seq decoder-id quirk.
        if not self.arch_info.needs_decoder_input_ids:
            return model_input
        config = getattr(self._model, "config", None)
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
            logits: torch.Tensor = out.logits
            return logits
        if hasattr(out, "start_logits"):
            return torch.stack([out.start_logits, out.end_logits], dim=-1)
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, (tuple, list)):
            if len(out) == 0:
                raise TypeError("Model returned an empty tuple/list — expected tensor output.")
            first: torch.Tensor = out[0]
            return first
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
    ) -> Model:
        """Run a forward pass and cache activations for reuse by other operations.

        The cache holds activations for one input at a time; calling
        ``cache(other_input)`` or any op that runs a forward on a
        different input invalidates the previous cache. Hashing the
        input is O(input bytes) — trivial for text inputs (typically
        4-8KB of ``int64`` token ids), but a measurable fraction of
        forward-pass time for vision tensors (a 3×224×224 fp32 tensor
        is ~588KB).

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
        input_hash = hash_input(model_input)

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
        empty_device_cache(self._device)

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
        input_hash = hash_input(model_input)

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
        and ``module``. For shared-weight architectures (ALBERT and
        similar), the dict also contains
        ``head_acts_per_invocation: list[Tensor]`` of length
        ``num_hidden_layers`` — one entry per logical-layer invocation
        of the shared physical block. ``head_acts`` for those models
        defaults to the FINAL invocation; iterate
        ``head_acts_per_invocation`` for per-logical-layer access.
        On non-shared models ``head_acts_per_invocation`` is ``None``.

        When *output_proj* is True (default), each head's output is
        projected through its slice of W_o so the result lives in
        residual-stream space.

        Note: this op needs real per-head attention weights and may
        trigger an eager-attention reload on models loaded with SDPA
        or FlashAttention. See :meth:`Model.attention` for VRAM
        implications and :meth:`Model.unload_eager_attention` for the
        cleanup helper.
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
        kind: str = "self",
        save: str | None = None,
        html: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """Show attention patterns. Returns None for non-transformer models.

        Parameters
        ----------
        causal:
            Apply causal mask.  Auto-detected from config if *None*.
        kind:
            For encoder-decoder models (T5/BART/Flan-T5), selects which
            attention tensor to return: ``"self"`` (decoder self-attention,
            default), ``"cross"`` (decoder→encoder cross-attention), or
            ``"encoder"`` (encoder self-attention). Ignored on causal-LM
            / MLM / vision models. Each result row carries
            ``attention_kind`` so callers can confirm what was returned.

        Pass ``save="path.png"`` to export a matplotlib heatmap.
        Pass ``html="path.html"`` to export an interactive HTML page.

        Note: when the underlying model was loaded with a non-eager
        attention backend (SDPA / FlashAttention — the modern HF
        default), the first call here triggers a second model copy
        with eager attention so real per-head weights can be observed.
        Pass ``attention_impl="eager"`` to :func:`interpkit.load` to
        preempt that reload, or call :meth:`Model.unload_eager_attention`
        afterwards to free the second copy. On a 1B-parameter fp16
        model the second copy is roughly an additional 2 GB of VRAM.
        """
        from interpkit.ops.attention import run_attention

        return run_attention(
            self, input_data,
            layer=layer, head=head, causal=causal, kind=kind,
            save=save, html=html,
        )

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
        top_k: int | None = 10,
        mode: str = "module",
        method: str = "auto",
        metric: str = "logit_diff",
        exhaustive_threshold: int = 500,
        top_k_search: int | None = None,
        pin_modules: list[str] | None = None,
        save: str | None = None,
        html: str | None = None,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Causal tracing with three-tier dispatcher (F-015).

        Pre-1.0 used a cheap activation-norm proxy to shortlist modules,
        which silently missed important modules like ``transformer.wte``
        (the audit found ``wte`` tied for top-1 by true effect but ranked
        low by the proxy, so was excluded from ``top_k=3``). The 1.0 fix
        replaces the proxy with Attribution Patching and adds an
        exhaustive-by-default mode for small models.

        Parameters
        ----------
        mode:
            ``"module"`` (default) — module-level tracing.
            ``"position"`` — Meng et al. style (layer x position) heatmap.
        method:
            ``"auto"`` (default), ``"exhaustive"``, ``"exhaustive_forced"``,
            or ``"approximate"``. See :func:`interpkit.ops.trace.run_trace`
            for the dispatch rules.
        exhaustive_threshold:
            For ``method="auto"``: candidates ≤ this run exhaustive.
        top_k_search:
            For ``method="approximate"``: number of AtP-shortlisted modules
            to confirm with full patching. Defaults to ``4 * top_k``.
        pin_modules:
            For ``method="approximate"``: modules to always include in the
            confirmation regardless of AtP score. Defaults to embed /
            unembed / final_norm / pos_embed when present.
        metric:
            ``"logit_diff"`` (default), ``"kl_div"``, ``"target_prob"``,
            ``"target_prob_effect"``, or ``"l2_prob"``.

        Returns
        -------
        list[dict] | dict
            For module mode: ``{"results": [...], "meta": {...}}`` with
            per-module provenance fields and a meta block describing the
            algorithm. (Position mode returns its own dict shape.)

        Pass ``save="path.png"`` to export a matplotlib figure.
        Pass ``html="path.html"`` to export an interactive HTML page.
        """
        from interpkit.ops.trace import run_trace

        return run_trace(
            self, clean, corrupted,
            top_k=top_k, mode=mode, method=method, metric=metric,
            exhaustive_threshold=exhaustive_threshold,
            top_k_search=top_k_search, pin_modules=pin_modules,
            save=save, html=html,
        )

    def lens(
        self,
        text: str | torch.Tensor | Any,
        *,
        save: str | None = None,
        html: str | None = None,
        position: int | None = None,
        kind: str = "logit",
        tuned_lens: Any = None,
    ) -> list[dict[str, Any]] | None:
        """Logit lens: project each block's output through the head pipeline.

        ``kind="tuned"`` applies trained per-block affine translators
        (Belrose et al. 2023) before the head projection — the unbiased
        readout for layers far from the output. Pass ``tuned_lens=`` a
        :class:`~interpkit.ops.tuned_lens.TunedLens` or a saved path;
        train one with :meth:`train_tuned_lens`.

        For language models: projects through ``pre_head`` (LayerNorm) →
        ``project_out`` (OPT only) → ``head`` (lm_head). For vision
        transformers and CNNs: spatially-pools then projects through the
        classifier head ("vision lens"; per-layer top-1 shows the model's
        evolving classification confidence).

        Encoder-decoder models (T5/BART) project decoder hidden states
        through the LM head. Use :meth:`encoder_lens` for explicit
        encoder-side projection.

        On first use per Model, runs a validation contract that asserts
        lens-at-last-block matches model logits — catches resolver bugs
        loudly via :class:`LensPipelineMismatch` rather than producing
        silent wrong results (Phase 0e).

        Note: lens may disagree with TransformerLens at the final layer
        for some architectures because TL folds ``unembed.b`` into
        ``ln_final`` (F-005). This is a known TL-side reformulation
        difference, not an interpkit bug.

        Pass ``save="path.png"`` for a matplotlib heatmap, or
        ``html="path.html"`` for an interactive HTML page.
        """
        from interpkit.ops.lens import run_lens

        return run_lens(
            self, text, save=save, html=html, position=position,
            kind=kind, tuned_lens=tuned_lens,
        )

    def train_tuned_lens(
        self,
        corpus: list[str],
        *,
        steps: int = 200,
        batch_size: int = 4,
        lr: float = 1e-3,
        max_length: int = 64,
        seed: int = 0,
        save: str | None = None,
    ) -> Any:
        """Train per-block tuned-lens translators (Belrose et al. 2023).

        The model stays frozen; only ``n_blocks × (hidden² + hidden)``
        affine parameters train, minimising KL between the model's final
        distribution and each block's translated readout. A few hundred
        diverse sentences is plenty for small models; expect a few
        minutes on CPU for gpt2 at the defaults, seconds on GPU.

        Returns a :class:`~interpkit.ops.tuned_lens.TunedLens` for use
        with ``lens(kind="tuned", tuned_lens=...)``. Pass ``save=`` a
        directory or ``.safetensors`` path to persist it.
        """
        from interpkit.ops.tuned_lens import train_tuned_lens

        return train_tuned_lens(
            self, corpus,
            steps=steps, batch_size=batch_size, lr=lr,
            max_length=max_length, seed=seed, save=save,
        )

    def encoder_lens(
        self,
        text: str | torch.Tensor | Any,
        *,
        position: int | None = None,
    ) -> list[dict[str, Any]] | None:
        """Encoder-side logit lens for encoder-decoder models (T5/BART).

        Projects each ENCODER block's output through the model's head
        (typically tied to the same ``lm_head`` as the decoder for
        T5/BART). Useful for analysing what the encoder encodes vs what
        the decoder generates. Raises
        :class:`OperationNotSupportedForArchitecture` for non-encoder-decoder
        models — use :meth:`lens` instead.

        N-002: pre-1.0 ``encoder_lens`` was a no-op alias for ``lens``;
        it now actually hooks the encoder block stack rather than the
        decoder.
        """
        from interpkit.core.support_matrix import check_op_supported
        from interpkit.ops.lens import run_encoder_lens

        check_op_supported("encoder_lens", self.arch_info)
        return run_encoder_lens(self, text, position=position)

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
        sae_subfolder: str | None = None,
    ) -> dict[str, Any]:
        """Decompose activations at *at* through a Sparse Autoencoder.

        Parameters
        ----------
        sae:
            Either a HuggingFace repo ID (``"jbloom/GPT2-Small-SAEs-Reformatted"``)
            or a pre-loaded :class:`interpkit.ops.sae.SAE` object.  The
            shorthand ``"<org>/<repo>/<subfolder>"`` is also accepted for
            repos that store weights in per-layer subdirectories.
        attribute:
            When ``True``, compute each top feature's logit contribution
            through the decoder → unembedding path.
        sae_subfolder:
            Explicit subfolder within the SAE repo (alternative to the
            shorthand).  Ignored when *sae* is already an :class:`SAE`.
        """
        from interpkit.ops.sae import SAE as SAEClass
        from interpkit.ops.sae import load_sae, run_features

        if isinstance(sae, str):
            sae = load_sae(sae, device=self._device, subfolder=sae_subfolder)
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
        sae_subfolder: str | None = None,
    ) -> dict[str, Any]:
        """Compare SAE feature activations between positive and negative groups.

        Returns features ranked by absolute differential activation,
        surfacing features that distinguish the two concepts.
        """
        from interpkit.ops.sae import SAE as SAEClass
        from interpkit.ops.sae import load_sae, run_contrastive_features

        if isinstance(sae, str):
            sae = load_sae(sae, device=self._device, subfolder=sae_subfolder)
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
        n_steps: int = 128,
        baseline: str | torch.Tensor = "pad",
        quadrature: str = "trapezoidal",
        auto_bump: bool = True,
        max_n_steps: int = 512,
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
            Interpolation steps for integrated gradients (default 128 in 1.0;
            was 50 pre-1.0). Higher values reduce completeness error.
        baseline:
            IG baseline embedding (F-011). One of ``"pad"`` (default),
            ``"zero"``, ``"mean"``, or a ``torch.Tensor``. The default
            switched from ``"zero"`` (out-of-distribution, ~17 nat
            completeness error at 50 steps) to ``"pad"`` (in-distribution,
            <1 nat error at 128 steps).
        quadrature:
            N-008 — IG integration scheme: ``"trapezoidal"`` (default,
            strictly more accurate than midpoint at the same n_steps),
            ``"riemann_midpoint"`` (legacy), or ``"gauss_legendre"``
            (faster convergence on smooth integrands; needs numpy).
        auto_bump:
            N-008 — when ``True`` (default), retry IG with double n_steps
            once if the completeness axiom fails on the first pass.
            Capped by ``max_n_steps``.
        max_n_steps:
            Cap on auto_bump retry (default 512).

        For text inputs: returns ``{"tokens", "scores", "target", "method",
        "ig_diagnostics"}`` with per-token importance and an IG diagnostics
        block reporting baseline / n_steps / completeness error / pass status.
        For vision: returns ``{"grad", "target"}`` with the pixel-gradient tensor.

        Pass ``save="path.png"`` to export a matplotlib figure.
        Pass ``html="path.html"`` to export an interactive HTML page.

        Note: ``gradient_x_input`` and ``integrated_gradients`` can disagree
        on the same input (F-012 — anti-correlated rankings on some models).
        IG satisfies the completeness axiom; gradient_x_input does not.
        """
        from interpkit.ops.attribute import run_attribute

        return run_attribute(
            self, input_data, target=target, method=method,
            n_steps=n_steps, baseline=baseline,
            quadrature=quadrature, auto_bump=auto_bump, max_n_steps=max_n_steps,
            save=save, html=html,
        )

    def dla(
        self,
        input_data: str | torch.Tensor | Any,
        *,
        token: int | str | None = None,
        position: int = -1,
        top_k: int = 10,
        save: str | None = None,
        html: str | None = None,
        sae: str | Any | None = None,
        sae_at: str | None = None,
        sae_subfolder: str | None = None,
    ) -> dict[str, Any]:
        """Direct Logit Attribution: decompose the output logit by component.

        For each layer, measures how much the attention block and MLP
        contribute to the logit of *token* by projecting their outputs
        through the unembedding matrix.  Also provides a per-head breakdown.

        Parameters
        ----------
        sae:
            Either a HuggingFace repo ID (``"jbloom/GPT2-Small-SAEs-Reformatted"``)
            or a pre-loaded :class:`interpkit.ops.sae.SAE` object.  When
            provided with *sae_at*, the specified component's contribution
            is further decomposed into per-feature logit attributions.
            The ``"<org>/<repo>/<subfolder>"`` shorthand is also accepted.
        sae_at:
            Module path of the component to decompose through the SAE
            (e.g. ``"transformer.h.11.attn"``).  Required when *sae* is
            provided.
        sae_subfolder:
            Explicit subfolder within the SAE repo (alternative to the
            shorthand).  Ignored when *sae* is already an :class:`SAE`.

        Returns a dict with ``target_token``, ``target_id``,
        ``contributions`` (list sorted by magnitude),
        ``head_contributions`` (per-head breakdown),
        ``total_logit_pre_ln`` (sum of per-component contributions),
        ``model_logit`` (actual model logit at target),
        ``ln_error`` (gap between the two — captures the LayerNorm
        non-linearity), and optionally ``feature_contributions`` when
        *sae* is provided.

        Pre-1.0 exposed a single ``total_logit`` field that was the sum
        of contributions but routinely deviated by 3.5–12.1 nats from
        the actual model logit (F-006). The 1.0 split makes the
        approximation explicit.
        """
        from interpkit.ops.dla import run_dla

        loaded_sae = None
        if sae is not None:
            from interpkit.ops.sae import SAE as SAEClass
            from interpkit.ops.sae import load_sae

            if isinstance(sae, str):
                loaded_sae = load_sae(sae, device=self._device, subfolder=sae_subfolder)
            elif isinstance(sae, SAEClass):
                loaded_sae = sae
            else:
                raise TypeError(f"Expected SAE or HF repo ID string, got {type(sae).__name__}")

        return run_dla(
            self, input_data, token=token, position=position,
            top_k=top_k, save=save, html=html,
            sae=loaded_sae, sae_at=sae_at,
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
        exact: bool = False,
    ) -> dict[str, Any]:
        """Decompose the residual stream into per-component contributions.

        Per-component contributions are accumulated in fp32 regardless of
        the model's underlying dtype (F-013). The residual stream itself
        accumulates in the model's native dtype, so bf16/fp16 models exhibit
        up to ~10% relative drift at attention-sink positions; pass
        ``exact=True`` to re-run the forward in fp32 for an exact
        reconstruction (doubles memory).

        Returns a dict with ``components`` (list of per-component
        ``{name, layer, type, vector, norm}``), ``residual`` (final
        residual stream vector), ``position``, and ``precision_note``
        describing the precision regime.
        """
        from interpkit.ops.circuits import run_decompose

        return run_decompose(self, input_data, position=position, exact=exact)

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

    def max_activating(
        self,
        dataset: list[str] | str,
        *,
        at: str,
        neuron: int | None = None,
        feature: int | None = None,
        head: int | None = None,
        sae: str | Any | None = None,
        top_k: int = 20,
        batch_size: int = 8,
        max_examples: int | None = None,
        max_length: int = 128,
        context: int = 12,
    ) -> dict[str, Any]:
        """Find the dataset examples that most activate one unit at *at*.

        The feature-browsing workflow: "what does this unit fire on?".
        Streams batched forwards over *dataset* and keeps the top-k
        (example, position) records by activation score — memory stays
        O(k) regardless of dataset size.

        Exactly one of ``neuron=`` (raw activation at that index),
        ``feature=`` (SAE feature activation; requires ``sae=``), or
        ``head=`` (L2 norm of the head's pre-projection output slice)
        selects the unit. *dataset* is a list of texts or an
        ``"hf:name[:split[:column]]"`` spec (requires the
        ``interpkit[data]`` extra and ``max_examples=``).

        Returns a dict with ``unit``, ``examples`` (each with the peak
        token and a ±``context``-token scored window), scan counters,
        and ``meta``.
        """
        from interpkit.ops.maxact import run_max_activating

        return run_max_activating(
            self, dataset, at=at,
            neuron=neuron, feature=feature, head=head, sae=sae,
            top_k=top_k, batch_size=batch_size, max_examples=max_examples,
            max_length=max_length, context=context,
        )

    def atp(
        self,
        clean: str | torch.Tensor | Any,
        corrupted: str | torch.Tensor | Any,
        *,
        top_k: int | None = 20,
        metric: str = "logit_diff",
    ) -> dict[str, Any]:
        """Attribution Patching: first-order patch-effect scores for all modules.

        Three model passes (clean forward, corrupted forward, one
        backward) score *every* module simultaneously — the fast first
        look before :meth:`trace`'s per-module full patching. Scores
        approximate the true patch effect (correlation typically
        0.85–0.95) but are first-order: confirm top candidates causally.

        Returns ``{"results": [{"module", "role", "score", "rank"}],
        "meta": {...}}`` sorted by absolute score.
        """
        from interpkit.ops.atp import run_atp

        return run_atp(self, clean, corrupted, top_k=top_k, metric=metric)

    def eap(
        self,
        clean: str | torch.Tensor | Any,
        corrupted: str | torch.Tensor | Any,
        *,
        ig_steps: int = 0,
        top_k_edges: int | None = 30,
        metric: str = "logit_diff",
    ) -> dict[str, Any]:
        """Edge Attribution Patching: gradient-based edge scores for circuits.

        Scores every (component → residual-stream) edge from one clean
        forward + one corrupted forward + one backward. ``ig_steps > 0``
        switches to EAP-IG (gradients averaged over embeddings
        interpolated from corrupted toward clean — more faithful in
        saturated regions; try 5).

        Requires token-aligned clean/corrupted pairs (same length).
        Returns ``{"edges": [...], "nodes": [...], "meta": {...}}``;
        see :func:`interpkit.ops.eap.run_eap` for edge semantics.
        """
        from interpkit.ops.eap import run_eap

        return run_eap(
            self, clean, corrupted,
            ig_steps=ig_steps, top_k_edges=top_k_edges, metric=metric,
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

    def intervene(self, *interventions: Any) -> Any:
        """Context manager applying interventions to every op run inside it.

        Pass :class:`~interpkit.core.interventions.Intervention` objects
        (``SteerIntervention``, ``AblateIntervention``,
        ``PatchIntervention``, ``FnIntervention``, ``CaptureProbe``)::

            with model.intervene(SteerIntervention("transformer.h.6", vector=v)):
                model.lens("The capital of France is")

        Hooks are registered on entry and always removed on exit (even on
        exception). Note: ops that internally reload an eager-attention
        copy (``attention`` on SDPA models) run that copy without these
        hooks.
        """
        from interpkit.core.interventions import apply_interventions
        from interpkit.core.support_matrix import check_op_supported

        check_op_supported("intervene", self.arch_info)
        return apply_interventions(self, list(interventions))

    def generate(
        self,
        input_data: str | torch.Tensor | Any,
        *,
        max_new_tokens: int = 64,
        interventions: list[Any] | None = None,
        capture: str | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> dict[str, Any]:
        """Generate text with interventions active across every decode step.

        The generation-time counterpart of the single-forward ops:
        steering / ablation / patching hooks stay registered for the
        prefill and all subsequent KV-cached decode steps, and
        ``capture`` records per-token analysis.

        Parameters
        ----------
        interventions:
            :class:`~interpkit.core.interventions.Intervention` objects.
            ``positions`` on an intervention are **absolute and
            prompt-indexed** — generated token *i* sits at position
            ``prompt_len + i``; a :class:`GenerationContext` maps them
            into each decode window. An intervened output at step *t*
            feeds the KV cache, so positional interventions influence
            all later steps by design.
        capture:
            ``"lens"`` — per-token logit-lens trajectory (each block's
            hidden state projected through the validated head pipeline);
            ``"logits"`` — per-step final logits.

        Greedy / sampling only (``num_beams=1`` semantics): beam search
        re-feeds tokens, which breaks position tracking.

        Returns a dict with ``prompt``, ``response``, ``input_ids``,
        ``output_ids``, ``interventions`` and (with *capture*) ``steps``.
        """
        from interpkit.ops.generate import run_generate

        return run_generate(
            self,
            input_data,
            max_new_tokens=max_new_tokens,
            interventions=interventions,
            capture=capture,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

    def chat(
        self,
        message: str | list[dict[str, str]],
        *,
        max_new_tokens: int = 128,
        system: str | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        interventions: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a chat response from the model.

        Builds a chat-templated prompt, runs ``model.generate``, and
        returns both the templated prompt (so it can be fed back into
        any other interpkit op like :meth:`dla` or :meth:`scan`) and
        the decoded response.

        Parameters
        ----------
        message:
            Either a plain user-message string (auto-wrapped as
            ``[{"role": "user", "content": message}]``), or a full
            list of message dicts.
        max_new_tokens:
            Maximum new tokens to generate (default 128).
        system:
            Optional system prompt.  Only valid when *message* is a
            string; for full conversations include the system role in
            the message list.
        do_sample, temperature, top_p:
            Standard HuggingFace ``generate`` sampling controls.  Default
            is greedy (``do_sample=False``).
        interventions:
            Optional :class:`~interpkit.core.interventions.Intervention`
            objects kept active during generation. Applied without
            position tracking — use :meth:`generate` for positional
            (``positions=...``) interventions.

        Returns
        -------
        dict
            ``{"prompt": str, "response": str, "messages": list,
            "input_ids": Tensor, "output_ids": Tensor}``.

        Raises
        ------
        ValueError
            If *system* is passed alongside a message list, or the
            message format is invalid.
        RuntimeError
            If the tokenizer is missing or has no chat template, or
            the underlying model lacks a ``generate`` method.
        """
        from interpkit.core.inputs import (
            NO_CHAT_TEMPLATE_MSG,
            _is_message_list,
            prepare_input,
        )

        if self._tokenizer is None:
            raise RuntimeError(
                "Model has no tokenizer — cannot run chat(). "
                "Pass tokenizer=... when loading the model."
            )

        template = getattr(self._tokenizer, "chat_template", None)
        if template is None and not getattr(self._tokenizer, "default_chat_template", None):
            raise RuntimeError(NO_CHAT_TEMPLATE_MSG)

        if not hasattr(self._model, "generate"):
            raise RuntimeError(
                f"Underlying model {type(self._model).__name__} has no "
                "generate() method — cannot run chat()."
            )

        if isinstance(message, str):
            messages: list[dict[str, str]] = []
            if system is not None:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": message})
        elif _is_message_list(message):
            if system is not None:
                raise ValueError(
                    "Pass `system` only with a string `message`; for a full "
                    "conversation include the system role in the list."
                )
            messages = list(message)
        else:
            raise ValueError(
                "chat(message=...) must be a string or a list of "
                "{'role', 'content'} dicts."
            )

        encoded = prepare_input(messages, tokenizer=self._tokenizer, device=self._device)
        if not isinstance(encoded, dict):
            raise RuntimeError("Chat template did not produce a dict input.")
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")

        prompt_text = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        pad_id = (
            getattr(self._tokenizer, "pad_token_id", None)
            or getattr(self._tokenizer, "eos_token_id", None)
        )

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(top_p)
        if pad_id is not None:
            gen_kwargs["pad_token_id"] = pad_id
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        generate_fn: Any = self._model.generate
        if interventions:
            from interpkit.core.interventions import apply_interventions

            with apply_interventions(self, list(interventions)), torch.no_grad():
                output_ids = generate_fn(input_ids=input_ids, **gen_kwargs)
        else:
            with torch.no_grad():
                output_ids = generate_fn(input_ids=input_ids, **gen_kwargs)

        input_len = input_ids.shape[-1]
        new_tokens = output_ids[0, input_len:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        return {
            "prompt": prompt_text,
            "response": response,
            "messages": messages,
            "input_ids": input_ids,
            "output_ids": output_ids,
        }

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


# Backward-compatible re-exports so existing ``from interpkit.core.model import ...``
# statements continue to work.
__all__ = [
    "Model",
    "load",
    "load_module",
    "_resolve_device",
    "_load_from_hf",
    "_make_dummy_input",
    "_is_hooked_transformer",
]

# Keep private names accessible for tests that patch them at this module path.
_hash_input = hash_input
_empty_device_cache = empty_device_cache
