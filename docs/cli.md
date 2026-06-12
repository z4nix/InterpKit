# CLI Reference

InterpKit ships with a fully-featured CLI that mirrors every method on
[`Model`](api/model.md). Every command takes a HuggingFace model ID as its first
positional argument, followed by any operation-specific arguments and options.

```bash
interpkit <command> <model> [args] [options]
```

If the `interpkit` console script isn't on your `PATH` (fresh environment,
sandboxed install, or running from a checkout without `pip install -e .`),
every command also works as `python -m interpkit ...`:

```bash
python -m interpkit scan gpt2 "The capital of France is"
```

For a styled in-terminal walkthrough of every command with examples and
options, run `interpkit --extensive`.

## Quick start

These commands are the fastest way to get oriented with a model.

### scan

One-command model overview — runs DLA, logit lens, attention, and gradient
attribution in a single pass and prints a combined summary.

```bash
interpkit scan gpt2 "The capital of France is"
interpkit scan gpt2 "The capital of France is" --save scan
```

`--save scan` exports each sub-figure as `scan_dla.png`, `scan_lens.png`, etc.

### report

Like `scan`, but bundles everything into a self-contained interactive HTML
file instead of printing to the terminal.

```bash
interpkit report gpt2 "The capital of France is" --save report.html
```

### chat

Send a message to an instruction-tuned chat model and print its reply. The
message is routed through the tokenizer's chat template (e.g. ChatML, Llama-2
Inst, Qwen, Gemma) with `add_generation_prompt=True` before generation, so any
HF chat model that ships a template just works.

```bash
interpkit chat HuggingFaceTB/SmolLM2-360M-Instruct "Write a haiku about cats."
interpkit chat HuggingFaceTB/SmolLM2-360M-Instruct "What is 2+2?" \
    --system "You are terse." --show-prompt
interpkit chat HuggingFaceTB/SmolLM2-360M-Instruct "Tell me a joke." \
    --sample --temperature 0.8 --top-p 0.9
```

Errors clearly when the model has no chat template (i.e. a base/non-instruct
model) — load an instruct variant or call any other command with a plain
string instead.

### generate

Generate text with interventions active across *every* decode step — the
generation-time counterpart of `steer` / `ablate`, which analyse a single
forward pass. A steering vector or ablation stays hooked for the prefill and
all KV-cached decode steps, so you can watch a nudged model write.

```bash
interpkit generate gpt2 "I feel" \
    --positive " joy" --negative " fear" --at transformer.h.6 --scale 8
interpkit generate gpt2 "My favorite place in the world is" \
    --sae jbloom/GPT2-Small-SAEs-Reformatted/blocks.8.hook_resid_pre \
    --feature 9752 --at transformer.h.7 --strength 50
interpkit generate gpt2 "The capital of France is" --capture lens
interpkit generate gpt2 "The weather is" --ablate-at transformer.h.4.mlp
```

`--capture lens` records each generated token's logit-lens trajectory: for
every step, each block's hidden state is projected through the validated head
pipeline, showing which layer first predicted the token the model emitted.
`--capture logits` records each step's final logits instead.

Steering accepts the same options as `steer`: contrastive (`--positive` /
`--negative` or `--positive-file` / `--negative-file`, plus `--at` and
`--scale`) or SAE-feature (`--sae` + `--feature` + `--at`, with
`--feature-mode add|clamp` and `--strength` — the Golden Gate manipulation,
applied at every decode step). Greedy by default;
`--sample --temperature 0.8 --top-p 0.9` for sampling.

## Core operations

### inspect

Print the model's module tree with types, parameter counts, and detected roles.
The module names printed here are what you pass to `--at` in other commands.

```bash
interpkit inspect gpt2
```

### dla

Direct Logit Attribution. Decomposes the predicted-token logit into per-component
contributions. Optionally decompose through an SAE for feature-level attribution.

```bash
interpkit dla gpt2 "The capital of France is"
interpkit dla gpt2 "The capital of France is" --token Paris --top-k 20
interpkit dla gpt2 "The capital of France is" \
    --sae jbloom/GPT2-Small-SAEs-Reformatted \
    --sae-at transformer.h.11.attn
interpkit dla gpt2 "The capital of France is" \
    --sae jbloom/GPT2-Small-SAEs-Reformatted \
    --sae-subfolder blocks.11.hook_resid_pre \
    --sae-at transformer.h.11.attn
```

### trace

Causal tracing (Meng et al. 2022). Patches each module's clean activation into
the corrupted run and ranks modules by how much output is recovered.

```bash
interpkit trace gpt2 \
    --clean "The Eiffel Tower is in Paris" \
    --corrupted "The Eiffel Tower is in Rome" --top-k 20
interpkit trace gpt2 --clean "...Paris..." --corrupted "...Rome..." \
    --mode position --save trace.png
```

### lens

Logit lens. Project each layer's hidden state into vocabulary space.

```bash
interpkit lens gpt2 "The capital of France is"
interpkit lens gpt2 "The capital of France is" --position -1
interpkit lens gpt2 "The capital of France is" --tuned-lens lens_dir/
```

The raw projection is biased for early layers (their basis isn't aligned with
the unembedding). `--tuned-lens <path>` applies trained per-layer translators
(Belrose et al. 2023) for an unbiased readout — train them once with
`train-tuned-lens` below.

### train-tuned-lens

Train tuned-lens translators for a model: one affine map per layer, trained so
each layer's translated readout matches the model's own final distribution
under KL. The model stays frozen; only the translators
(`n_layers × (hidden² + hidden)` parameters) train. A few hundred diverse
sentences is plenty for small models; expect a few minutes on CPU for gpt2 at
the defaults, seconds on GPU.

```bash
interpkit train-tuned-lens gpt2 --corpus-file texts.txt --save lens_dir/
interpkit train-tuned-lens gpt2 --corpus-file texts.txt \
    --steps 500 --batch-size 8 --lr 1e-3
```

Artifacts are safetensors weights + a JSON metadata sidecar; loading validates
the hidden size and layer structure against the live model. Omitting `--save`
writes to `~/.cache/interpkit/tuned_lens/<model>/`.

### attribute

Gradient-based input attribution. Returns per-token (or per-pixel for vision
models) saliency scores.

```bash
interpkit attribute gpt2 "The capital of France is"
interpkit attribute gpt2 "The capital of France is" --method integrated_gradients
interpkit attribute microsoft/resnet-50 cat.jpg --target 281
```

### patch

Activation patching. Swaps one module's clean output into the corrupted run
and measures recovery. Supports module-level, head-level, and position-level
patching.

```bash
interpkit patch gpt2 \
    --clean "The Eiffel Tower is in Paris" \
    --corrupted "The Eiffel Tower is in Rome" --at transformer.h.8.mlp
interpkit patch gpt2 --clean "..." --corrupted "..." \
    --at transformer.h.8 --head 3 --positions 3,4,5
```

### attention

Visualize attention weight heatmaps for transformer models.

```bash
interpkit attention gpt2 "The capital of France is" --layer 8 --save attn.png
interpkit attention gpt2 "Hello world" --html attention.html
```

## Analysis

### activations

Extract the raw activation tensor at one or more named modules and print
summary statistics.

```bash
interpkit activations gpt2 "Hello world" --at transformer.h.8.mlp
interpkit activations gpt2 "Hello world" --at "transformer.h.0,transformer.h.8.mlp"
```

### ablate

Replace a module's output with zeros, its mean activation, or a resampled
value, then report how much the prediction changed.

```bash
interpkit ablate gpt2 "Hello world" --at transformer.h.8.mlp
interpkit ablate gpt2 "Hello world" --at transformer.h.8.mlp \
    --method resample --reference "Goodbye"
```

### decompose

Break the residual stream at a token position into per-component contributions.

```bash
interpkit decompose gpt2 "The capital of France is"
interpkit decompose gpt2 "The capital of France is" --position -1
```

### steer

Activation steering. Computes a direction from contrasting examples (the
"steering vector") and adds a scaled copy to a module's activations during
inference.

> **BPE tokenization tip:** GPT-2 / Llama / Qwen tokenizers treat `" love"`
> and `"love"` as different tokens. Use the leading-space variants — they are
> what the model sees in normal text.

```bash
interpkit steer gpt2 "The weather today is" \
    --positive " love" --negative " hate" --at transformer.h.8 --scale 2.0
interpkit steer gpt2 "The weather today is" \
    --positive-file pos.txt --negative-file neg.txt \
    --at transformer.h.8 --scale 2.0
```

Alternatively steer along a single **SAE feature**'s decoder direction — the
Golden Gate Claude manipulation. `--feature-mode clamp` (default) pins the
feature's activation to `--strength` regardless of input; `add` injects
`strength · W_dec[feature]` unconditionally. The SAE must have been trained on
the module passed to `--at` (find feature indices with `features` or `maxact`;
useful strengths are a few times the feature's max activation — `maxact` tells
you that scale). Mutually exclusive with `--positive` / `--negative`.

```bash
interpkit steer gpt2 "My favorite place in the world is" \
    --sae jbloom/GPT2-Small-SAEs-Reformatted/blocks.8.hook_resid_pre \
    --feature 9752 --at transformer.h.7 --strength 50
```

### probe

Train a linear classifier on activations from labeled examples and report
accuracy.

```bash
# data.json: {"texts": [...], "labels": [...]}
interpkit probe gpt2 --at transformer.h.8 --data data.json
```

### diff

Run two models on the same input and compare activations layer by layer.

```bash
interpkit diff gpt2 my-finetuned-gpt2 "The capital of France is" --save diff.png
```

## Circuit analysis

### find-circuit

Automated circuit discovery. With the default ablation methods, iteratively
ablates components and keeps those whose removal changes the output above
`--threshold`. With `--method eap` (or `eap-ig`), components are selected by
Edge Attribution Patching scores instead — a handful of gradient passes
replaces the per-component ablation sweep, and the threshold becomes a
fraction of the top component's score. Either way, the discovered circuit is
verified *causally* by ablating the excluded components and measuring how
much of the clean-vs-corrupted distinction survives.

```bash
interpkit find-circuit gpt2 \
    --clean "The Eiffel Tower is in Paris" \
    --corrupted "The Eiffel Tower is in Rome" \
    --threshold 0.05
interpkit find-circuit gpt2 \
    --clean-file cleans.txt --corrupted-file corrupteds.txt
interpkit find-circuit gpt2 \
    --clean "..." --corrupted "..." --method eap --threshold 0.2
```

### atp

Attribution Patching (Syed et al. 2023). A first-order gradient approximation
of activation patching: one clean forward, one corrupted forward, and one
backward pass score *every* module simultaneously — versus one forward per
module for exhaustive tracing. Use it as the fast first look, then confirm
top candidates with `trace` or `patch`.

```bash
interpkit atp gpt2 \
    --clean "The capital of France is" \
    --corrupted "The capital of Germany is" --top-k 15
```

### eap

Edge Attribution Patching. Where `atp` scores modules, `eap` scores *edges*:
how much each component's clean-vs-corrupted delta matters as it flows into
each downstream residual-stream layer. The edge at a component's own layer is
its total effect; deeper edges show how the effect persists down the stream.
Inputs must tokenize to the same length.

```bash
interpkit eap gpt2 \
    --clean "The capital of France is" \
    --corrupted "The capital of Germany is"
interpkit eap gpt2 --clean "..." --corrupted "..." --ig-steps 5
```

`--ig-steps 5` switches to EAP-IG: gradients averaged over embeddings
interpolated from corrupted toward clean, which is more faithful when the
corrupted point sits in a saturated region.

### features

Sparse Autoencoder feature decomposition. Projects a module's activation
through a separately trained SAE to recover a sparse set of interpretable
features.

```bash
interpkit features gpt2 "The capital of France is" \
    --at transformer.h.8 --sae jbloom/GPT2-Small-SAEs-Reformatted
interpkit features gpt2 "The capital of France is" \
    --at transformer.h.8 --sae ./my_sae.safetensors

# Contrastive mode — find features that differentiate two groups of inputs
interpkit features gpt2 \
    --at transformer.h.8 --sae jbloom/GPT2-Small-SAEs-Reformatted \
    --positive-file pos.txt --negative-file neg.txt
```

`--sae` accepts a HuggingFace repo ID, a local `.safetensors` / `.pt` file, or
the `org/repo/subfolder` shorthand. Alternatively pass `--sae-subfolder` to
target a subfolder explicitly.

### maxact

Max-activating examples — the feature-browsing workflow: scan a corpus and
show the contexts where one unit fires hardest, with the peak token
highlighted. Works for raw neurons (`--neuron`), SAE features (`--feature` +
`--sae`), and attention heads (`--head`, scored by the head's pre-projection
output norm). Streams batched forwards and keeps only the top-k scored
contexts, so memory stays flat however large the corpus.

```bash
interpkit maxact gpt2 --at transformer.h.6.mlp --neuron 42 \
    --texts-file corpus.txt --top-k 20
interpkit maxact gpt2 --at transformer.h.6 --feature 1337 \
    --sae jbloom/GPT2-Small-SAEs-Reformatted --texts-file corpus.txt
interpkit maxact gpt2 --at transformer.h.6.mlp --neuron 42 \
    --dataset hf:imdb --max-examples 256
```

HF dataset specs (`hf:name[:split[:column]]`, default split `train`, column
`text`) require the `datasets` package: `pip install 'interpkit[data]'`.
`--max-examples` is mandatory for HF datasets so a typo can't start an
unbounded scan.

## Common options

Most commands accept the following options:

| Option | Description |
|--------|-------------|
| `--device` | Device to run on (`cpu`, `cuda`, `mps`). Auto-detected if omitted. |
| `--dtype` | Model dtype (`float16`, `bfloat16`, `float32`, `auto`). |
| `--device-map` | HuggingFace `device_map` for multi-GPU loading (e.g. `auto`). |
| `--save path.png` | Export a static figure (matplotlib). |
| `--html path.html` | Export an interactive HTML visualization. |

Pass `--format json` *before* the command name for machine-readable JSON output:

```bash
interpkit --format json dla gpt2 "The capital of France is"
```

## Output formats

For visual operations:

- **Static figures** — pass `--save figure.png` to export a matplotlib PNG.
- **Interactive HTML** — pass `--html figure.html` to export a self-contained
  HTML file with hover tooltips, filters, and sliders. Open it in any browser
  or share with collaborators.
- **JSON** — pass `--format json` (before the command) to print a JSON-encoded
  result dict, suitable for piping into other tools.

## Discoverability

- `interpkit` (no arguments) — styled command index grouped by category.
- `interpkit --extensive` — beginner-friendly walkthrough of every command.
- `interpkit <command> --help` — full option list for any command.
- `python -m interpkit ...` — works identically when the console script isn't on `PATH`.
