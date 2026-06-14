# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Local web GUI** (`interpkit gui`) — a FastAPI server plus a no-build,
  no-CDN single-page app that runs every operation in the browser. Install with
  the new optional extra: `pip install "interpkit[gui]"`.
  - Models load into named **sessions**, each with its own single-worker queue
    so ops never run concurrently on the same model; every action is a polled
    **job** so request handlers never block on PyTorch.
  - One operation registry (`interpkit/gui/ops/`) drives both the API dispatch
    and the schema-generated forms, so the GUI always covers the full CLI
    surface (guarded by a parity test). Module / layer / head inputs are pickers
    populated from the detected architecture; unsupported ops are greyed out
    using the library's own support matrix.
  - Native result rendering (heatmaps, bar charts, token strips, attention
    explorer, chat thread) reusing the existing dark-theme visual language, with
    raw-JSON download and per-op run history on every panel.
- **`progress_callback`** parameter on `Model.trace()`, `Model.max_activating()`,
  and `Model.train_tuned_lens()` (and the underlying `ops.*` functions) — an
  optional `(done, total, message)` hook for programmatic progress reporting
  (used by the GUI job queue). The rich progress bar is unchanged when it is
  omitted.

## [0.6.0] - 2026-06-10

Generation-time interventions, gradient-based circuit discovery, and
feature-browsing workflows. Upgrading from **0.5.0** is additive 

### Added

- **`interpkit.core.interventions`** — declarative hook-write plumbing shared
  across single-forward and multi-token generation. Public types:
  `Intervention`, `SteerIntervention`, `AblateIntervention`,
  `PatchIntervention`, `FnIntervention`, `CaptureProbe`,
  `GenerationContext`, and `apply_interventions()`. Exported from
  `interpkit`.
- **`Model.generate()`** — text generation with interventions active across
  the prefill and every KV-cached decode step. `capture="lens"` records
  per-token logit-lens trajectories; `capture="logits"` records per-step
  final logits. Positional interventions use absolute, prompt-indexed
  positions (`prompt_len + i` for generated token *i*).
- **`Model.intervene()`** — context manager that applies
  `Intervention` objects to any op run inside the block (including
  `lens`, `dla`, `trace`, etc.).
- **`Model.atp()`** — Attribution Patching: first-order patch-effect scores
  for all modules from three passes (clean forward, corrupted forward, one
  backward). Public home of the machinery that previously lived in private
  `ops/_atp.py` (now a re-export shim for `trace(method="approximate")`).
- **`Model.eap()`** — Edge Attribution Patching: gradient-based
  component → residual-stream edge scores. `ig_steps > 0` switches to
  EAP-IG (interpolated embedding gradients; try 5). Requires token-aligned
  clean/corrupted pairs.
- **`Model.train_tuned_lens()`** and **`lens(kind="tuned")`** — train
  per-block affine translators (Belrose et al. 2023) so early-layer
  readouts match the model's final distribution under KL. Artifacts are
  safetensors weights + a JSON metadata sidecar.
- **`Model.max_activating()`** — scan a corpus for the examples that most
  activate a neuron, SAE feature, or attention head. Streams batched
  forwards with O(k) memory via `interpkit.core.topk.TopKTracker`.
  Accepts a `list[str]` or an `"hf:name[:split[:column]]"` dataset spec.
- **`find_circuit(method="eap")` / `method="eap-ig"`** — gradient-based
  component selection (a handful of passes) followed by causal
  verification via mean ablation of excluded components. EAP mode adds
  `edges` and `meta` keys to the result dict; the legacy `circuit` /
  `excluded` / `verification` schema is unchanged.
- **`Model.chat(..., interventions=[...])`** — apply `Intervention` objects
  during chat generation (without positional tracking; use `generate()` for
  `positions=...`).
- **CLI commands:** `generate`, `train-tuned-lens`, `atp`, `eap`, `maxact`.
  `find-circuit` gains `--method eap` / `eap-ig`; `lens` gains
  `--tuned-lens`.
- **`interpkit[data]` optional extra** — `datasets>=2.14` for HuggingFace
  dataset specs in `max_activating`.
- **`sentencepiece>=0.1.99`** added as a core dependency for
  SentencePiece-only tokenizers (Marian, XLM, some T5/ALBERT checkpoints).
- **Examples:** `examples/11_generation_interventions.ipynb`,
  `examples/12_circuit_discovery_and_lenses.ipynb`.

### Changed

- **`steer` / `ablate` / `patch` / `find_circuit`** — hook closures now
  compile from `Intervention` objects in `core.interventions`. External
  behaviour and return shapes are preserved; dtype/device writeback is
  unified (F-008 cast path).
- **`ops/atp.py`** promoted to the public AtP implementation;
  `ops/_atp.py` is a thin shim re-exporting `compute_atp_scores` for
  `trace`'s approximate shortlist.
- **README and `docs/cli.md`** — operations table, CLI examples, and
  notebook index updated for the new features.

### Tests

- `tests/test_interventions.py` — intervention types, context manager,
  generation-time hooks, position semantics.
- `tests/test_generate.py` — `generate()` with steer/ablate/capture.
- `tests/test_atp.py`, `tests/test_eap.py` — gradient patching scores.
- `tests/test_tuned_lens.py` — train/load round-trip and `lens(kind="tuned")`.
- `tests/test_maxact.py` — neuron / feature / head scoring over corpora.
- `tests/test_topk.py` — streaming top-k tracker.
- `tests/test_cli.py` — new CLI commands and options.

## [0.5.0] - 2026-06-09

Major correctness rewrite and architecture consolidation. Upgrading from **0.4.0**
is a strict break — see [Breaking changes](#breaking-changes) below. (This is a
0.x release: the public API is not yet stable and may change between minor
versions.)

Development was guided by two independent audit passes (the original
26-finding sweep and a follow-up 28-model stress harness). Regression
coverage lives in `tests/test_audit_regressions.py`, `tests/test_resolver.py`,
and `tests/test_resolver_golden.py` (28-model golden snapshots).

### Breaking changes

The library has no stable public API yet (0.x); fixes land at the
architectural layer rather than behind deprecations.

- `load()` defaults to `dtype="float32"` (was `"auto"`); `dtype=None` raises.
- `dla()` returns three logit fields instead of one (`total_logit` removed):
  `total_logit_pre_ln`, `model_logit`, `ln_error`.
- `trace()` returns `{"results": ..., "meta": ...}` instead of a bare list.
  New kwargs: `method`, `exhaustive_threshold`, `top_k_search`, `pin_modules`
  (all with defaults).
- `attribute(method="...")` validates the method string; typos raise
  `ValueError`.
- `attribute()` defaults `n_steps=128` (was 50) and `baseline="pad"` (was
  implicit zero). New kwargs: `quadrature`, `auto_bump`, `max_n_steps`.
- `ov_scores` / `qk_scores` / `composition` raise on out-of-range layers
  rather than silently redirecting.
- `to_native_name(...)` raises `KeyError` for TL-internal hooks rather than
  returning a bogus path.
- `attention()` is eager-only; raises `AttentionBackendUnavailable` when eager
  is unavailable. Returned dicts always have `source="eager"`.
- `patch()` returns `NaN` + `warnings=["degenerate_gap"]` for undefined ratio
  metrics (was silent 0).
- `decompose()` adds `precision_note` and `post_ln` fields; gains `exact=True`
  kwarg.
- SAE results report `dead_fraction` / `active_fraction` / `loss_ratio`
  instead of the ambiguous `sparsity` field.
- `head_acts_per_invocation` is always a list on shared-weight models (was
  `None` when only one invocation was captured).
- **`interpkit.core.discovery` removed.** Import architecture types and
  `resolve_arch` from `interpkit` or `interpkit.core.arch`.
- **`ModelArchInfo` removed.** Use the unified `ArchInfo` dataclass in
  `interpkit.core.arch.types`.

### Architecture

- **New `interpkit/core/arch/` package** — single home for model-structure
  resolution. Replaces the regex-based `core/discovery.py` and earlier
  resolver/residual/names splits. Modules: `types`, `names`, `tree`, `probe`,
  `family`, `blocks`, `layers`, `heads`, `resolve`, `residual`.
- **Unified `ArchInfo`** — one dataclass (no `ModelArchInfo` subclass).
  `is_language_model` is family-based, not `has_lm_head`-based. `LayerInfo` /
  `ModuleInfo` remain as sub-structures.
- **Three-layer resolver** in `arch/resolve.py` — overrides > conventions >
  walker + runtime hooks. Works on HuggingFace transformers, timm models, and
  arbitrary `nn.Module` instances.
- **Unified helpers** — `arch.module_at_path`, `arch.get_weight`,
  `arch.extract_proj_weight`.
- **Shared-layer synthesis** happens once in `resolve_arch` (removed duplicate
  `_synth_shared_lm_blocks` branch) with a post-resolve contract assertion.
- **New `ArchFamily` values:** `MLM`, `ENCODER_ONLY`, plus existing families.
- **New `ArchInfo` fields:** `decoder_blocks`, `mlm_head_module` /
  `mlm_head_path`, DistilBERT vocab paths, `is_shared_layers`,
  `has_disentangled_attention`, `needs_decoder_input_ids`.
- **New `LayerInfo` field:** `attn_inner_path` (pre-residual attention anchor).
- **Per-op `SUPPORT_MATRIX`** — unsupported families raise
  `OperationNotSupportedForArchitecture` with a helpful suggestion.
- **Lens-pipeline validation** — `load(..., validate_pipeline=True)` (default)
  runs the contract at load time for headed models and raises
  `LensPipelineMismatch` immediately (with top-3 `arch_override` suggestions).
  Pass `validate_pipeline=False` for inspect-only workflows.
- **`support_matrix.lens_blocks(arch)`** — single source of truth for which
  blocks `lens` / validation hook (decoder stack for seq2seq, `arch.blocks`
  otherwise).

### Added

- `interpkit.load_module(module, sample_input, ...)` — wrap any `nn.Module`
  (timm, custom research code) as an InterpKit `Model`.
- `arch_override=` on `load()` and `load_module()` — manual escape hatch when
  auto-detection is wrong.
- `Model.encoder_lens(text)` — encoder-side projection for seq2seq models
  (no longer an alias for `lens`).
- `Model.attention(..., kind=...)` — `"self"`, `"cross"`, or `"encoder"` for
  seq2seq routing.
- `Model.device`, `Model.dtype` — public properties.
- `interpkit.list_roundtrippable_hooks()` — enumerate TL hooks that round-trip
  to native names.
- `interpkit.WrongInputType` — raised when text is passed to a vision model.
- `attribute()` `result["interpretation"]` ∈ `{"quantitative", "ranking_only"}`.
- `prepare_input` / `prepare_pair` reject empty and whitespace-only strings
  before tokenization.

### Fixed — original audit (F-001 … F-026)

| Finding | Severity | Change |
|---|---|---|
| **F-001 / F-002** Attention silently fell back to RoPE/ALBi-less Q/K reconstruction | CRITICAL | `attention()` is eager-only. QK-reconstruction fallback deleted. `Model._ensure_eager_attention()` reloads with `attn_implementation="eager"` when needed. |
| **F-003** Encoder-decoder lens projected encoder hidden states through decoder head | HIGH | Lens hooks each block directly. T5/BART use decoder blocks; `encoder_lens()` for encoder-side projection. |
| **F-004** OPT lens disagreed with model logits at final layer | MEDIUM | Lens hooks last block output and applies family-aware projection (`pre_head` → `project_out` → `head`). |
| **F-005** GPT-2 lens disagreed with TransformerLens at final layer | LOW | Documented TL-side reformulation difference in `Model.lens` docstring. |
| **F-006** DLA `total_logit` deviated from actual model logit | MEDIUM | `total_logit` removed; replaced with `total_logit_pre_ln`, `model_logit`, `ln_error`. |
| **F-007** `load()` honoured HF default dtype | HIGH | `load()` defaults to `dtype="float32"`; `dtype="auto"` is explicit opt-in. |
| **F-008** Head/position patching crashed on fp16/bf16 | CRITICAL | Patch hooks cast back to module dtype before re-injection. |
| **F-009** `target_prob` was a raw probability, not a normalised effect | LOW | Added `target_prob_effect`; documented raw field. |
| **F-010** `logit_diff` returned 0 for degenerate gaps | LOW | Degenerate gaps return `NaN` + `warnings=["degenerate_gap"]`. |
| **F-011** IG completeness error with zero baseline | LOW | `baseline=` kwarg (default `"pad"`); `n_steps` default 128; `ig_diagnostics` block. |
| **F-012** IG and `gradient_x_input` anti-correlated on some models | MEDIUM | Documented in `Model.attribute` docstring. |
| **F-013** `decompose()` precision drift in low precision | LOW | Per-component accumulation in fp32; `precision_note`; `exact=True` kwarg. |
| **F-014** SAE ambiguous sparsity / wrong encode | MEDIUM | `dead_fraction`, `active_fraction`, `loss_ratio`; reads SAE config fields; `from_sae_lens()` shim. |
| **F-015** `trace(top_k=K)` excluded tied top-1 modules | MEDIUM | Dict return shape; three-tier `method=` dispatcher; ATP shortlist + provenance fields. |
| **F-016** `diff()` failed on HF model id strings | LOW | Auto-loads via `interpkit.load()`. |
| **F-017** `Model.device` was private | VERY LOW | Public `device` and `dtype` properties. |
| **F-018 / F-019** Unknown enum strings silently defaulted | MEDIUM | `VALID_*` frozensets; typos raise `ValueError`. |
| **F-020** `circuits` ops silently redirected OOR layers | LOW-MED | Raise `IndexError` / `ValueError` instead. |
| **F-021** `probe([], [])` opaque numpy error | LOW | Friendly entry guard. |
| **F-022** Bad module paths opaque | LOW | `validate_module_path()` with `difflib` suggestions. |
| **F-023** CLI JSON mode interleaved Rich output | MEDIUM | Console re-bound to stderr; progress bars silenced. |
| **F-024** Empty input opaque `RuntimeError` | LOW | Explicit post-tokenization empty check. |
| **F-025** `to_native_name` round-trip failed for TL hooks | MEDIUM | TL-internal hooks raise `KeyError`; `list_roundtrippable_hooks()`. |
| **F-026** BOS-handling mismatch with TransformerLens | LOW | Documented in `tl_compat.py`; `warn_bos_mismatch_once()`. |

### Fixed — audit2 stress harness (N-001 … N-010)

| Finding | Severity | Invariant restored |
|---|---|---|
| **N-001** `decompose()` dropped embedding contribution | HIGH | `L-1.embed` component prepended; `post_ln` flag and corrected `precision_note`. |
| **N-002** `lens()` failed on encoder-only and seq2seq models | HIGH | `ArchFamily.MLM` / `ENCODER_ONLY`; MLM head cascade; decoder block routing; updated `SUPPORT_MATRIX`. |
| **N-003** `attention()` failed on T5/BART | HIGH | `kind=` routing for `decoder_attentions` / `cross_attentions` / `encoder_attentions`. |
| **N-004** DeBERTa-v3 crashed 8 ops under hooks | HIGH | `has_disentangled_attention` gates affected ops at load time; clear warning. |
| **N-005** ALBERT shape mismatches; resample ablate ≠ identity | MEDIUM | Square `o_proj` guard; `_find_mlp_output_sibling`; shared-layer hook dedup. |
| **N-006** ELECTRA DLA shape mismatch | MEDIUM | MLM project-out fallback; pre_head parent scan. |
| **N-007** `head_activations` sum invariant on ALiBi models | MEDIUM | `attn_inner_path` anchor; provenance fields in result. |
| **N-008** IG completeness on Qwen at pad baseline | MEDIUM | `quadrature` parameter; `auto_bump` retry policy. |
| **N-009** Empty string handling inconsistent across ops | LOW | Uniform `ValueError` before tokenization / validation. |
| **N-010** Stale F-022 status in old audit notes | LOW | Confirmed fixed (no user-facing change). |

### Fixed — audit2 regressions (NR-001 … NR-008)

| Finding | Severity | Invariant restored |
|---|---|---|
| **NR-001** `lens()` silently `None` on fp16/bf16 LMs | HIGH | `_dtype_aware_apply` in `support_matrix.py`; narrowed exception handlers. |
| **NR-002** DeBERTa-v3 load `NameError: name 'name'` | CRITICAL | Warning label from `name_or_path` or class name in `_build_model`. |
| **NR-003** BERT lens regressed to `LensPipelineMismatch` | HIGH | Downstream fix from NR-001 dtype handling. |
| **NR-004** ELECTRA lens wrong top-1 tokens | HIGH | Narrowed MLM classification; ELECTRA family guard; MLM head cascade fix. |
| **NR-005** OPT-350m DLA size mismatch | HIGH | `_pick_directional_project_out` disambiguates sibling Linears. |
| **NR-006** ALBERT decompose sum invariant blew up | MEDIUM | Residual hook reads `[-1]` on shared-layer forwards. |
| **NR-007** `ov_scores` failed on Qwen3 / flan-t5 | MEDIUM | Non-square output projection allowed when output dim matches hidden size. |
| **NR-008** BART DLA tensor shape crash | HIGH | `project_out` requires a dimension bridge (`head.in_features != hidden_size`). |

Additional remediation from the June 2026 baseline re-audit:

- **ALBERT shared-weight lens** — resolver synthesises N logical blocks for
  shared-layer models so `arch.blocks` is never empty.
- **BART/T5/OPT-125m/GPT-2** — `project_out=None` when no genuine bridge
  exists; OPT-350m keeps its real `project_out`.

### Performance notes

- **fp32 default** doubles memory on non-fp32 checkpoints; use
  `dtype="auto"` for large models.
- **Eager attention reload** may cache a second model copy when SDPA /
  FlashAttention models need true weights.
- **Exhaustive trace** (default for ≤ 500 modules) is more compute-intensive
  than the previous proxy approach but guarantees correct rankings.

### Tests

- `tests/test_audit_regressions.py` — regression tests for audit findings.
- `tests/test_resolver.py` — synthetic resolver unit tests (all families,
  overrides, residual / pre-head detection).
- `tests/test_resolver_golden.py` — 28-model `ArchInfo` golden snapshots.
- Additional coverage: `test_capabilities.py`, `test_validation.py`,
  `test_seq2seq_contract.py`, `test_phase3_regressions.py`,
  `test_cache_invalidation.py`, `test_archinfo_serialization.py`.

## [0.4.0]

Prior releases. See git history on `main` before the 0.5.0 rewrite.
