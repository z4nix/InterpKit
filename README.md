```
IIIII         tt                          KK  KK iii tt
 III  nn nnn  tt      eee  rr rr  pp pp   KK KK      tt
 III  nnn  nn tttt  ee   e rrr  r ppp  pp KKKK   iii tttt
 III  nn   nn tt    eeeee  rr     pppppp  KK KK  iii tt
IIIII nn   nn  tttt  eeeee rr     pp      KK  KK iii  tttt
                                  pp
```

> Mech interp for any HuggingFace model.

[![PyPI version](https://img.shields.io/pypi/v/interpkit.svg)](https://pypi.org/project/interpkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## Why InterpKit?

Mechanistic interpretability tooling today is fragmented. Each library supports a narrow set of architectures, and moving to a different model family usually means rewriting hook code from scratch.

InterpKit provides a single, consistent interface for mech interp operations across any HuggingFace model — transformers, SSMs, vision models, and more — with zero annotation required.

---

## Install

```bash
pip install interpkit

# For linear probe support:
pip install interpkit[probe]
```

Or install from source for development:

```bash
git clone https://github.com/davidezani/InterpKit.git
cd InterpKit
pip install -e ".[dev]"
```

---

## Quickstart

```python
import interpkit

model = interpkit.load("gpt2")

model.inspect()                    # module tree with roles, params, shapes
model.trace("...Paris...", "...Rome...", top_k=20)   # causal tracing
model.patch("...Paris...", "...Rome...", at="transformer.h.8.mlp")
model.lens("The capital of France is")               # logit lens
model.attribute("The capital of France is")          # gradient saliency
```

Works the same on any HF architecture:

```python
model = interpkit.load("state-spaces/mamba-370m")
model = interpkit.load("google/vit-base-patch16-224")
model = interpkit.load("bert-base-uncased")
```

---

## Operations

| Operation | What it does | Works on |
|-----------|-------------|----------|
| `inspect` | Module tree with types, param counts, shapes | Any model |
| `patch` | Activation patching at a named module | Any model |
| `trace` | Causal tracing across modules, ranked by effect | Any model |
| `attribute` | Gradient saliency over inputs | Any model |
| `lens` | Logit lens — project activations to vocabulary | LMs (auto-detected) |
| `activations` | Extract raw activation tensors at any module | Any model |
| `ablate` | Zero/mean ablate a component and measure effect | Any model |
| `attention` | Visualize attention patterns per layer/head | Transformers |
| `steer` | Extract and apply steering vectors | Any model |
| `probe` | Linear probe on activations | Any model |
| `diff` | Compare activations between two models | Any model |
| `features` | SAE feature decomposition | Any model |

---

## Activations, Ablation, Attention

```python
# Extract raw activations
act = model.activations("The capital of France is", at="transformer.h.8.mlp")
acts = model.activations("...", at=["transformer.h.0", "transformer.h.8.mlp"])

# Ablation — zero or mean
result = model.ablate("The capital of France is", at="transformer.h.8.mlp")
result = model.ablate("...", at="transformer.h.8.mlp", method="mean")

# Attention patterns
model.attention("The capital of France is")                   # all layers
model.attention("The capital of France is", layer=8, head=3)  # single head
```

## Steering

```python
# 1. Extract a steering vector
vector = model.steer_vector("Love", "Hate", at="transformer.h.8")

# 2. Apply during inference — side-by-side comparison
model.steer("The weather today is", vector=vector, at="transformer.h.8", scale=2.0)
```

## Linear Probe

```python
result = model.probe(
    texts=["The cat sat", "The dog ran", "A bird flew", "A fish swam"],
    labels=[0, 0, 1, 1],
    at="transformer.h.8",
)
print(result["accuracy"])
```

## Model Diff

```python
base = interpkit.load("gpt2")
finetuned = interpkit.load("my-finetuned-gpt2")
interpkit.diff(base, finetuned, "The capital of France is")
```

## SAE Features

Decompose activations into interpretable features using pre-trained Sparse Autoencoders from HuggingFace:

```python
model.features(
    "The capital of France is",
    at="transformer.h.8",
    sae="jbloom/GPT2-Small-SAEs-Reformatted",
)
```

No SAELens dependency — weights are loaded directly via `safetensors`.

## Activation Cache

Avoid redundant forward passes when exploring the same input with multiple operations:

```python
model.cache("The capital of France is")  # one forward pass, cache all layers
model.activations("The capital of France is", at="transformer.h.8.mlp")  # instant
model.activations("The capital of France is", at="transformer.h.0.mlp")  # instant

model.clear_cache()  # free memory
```

---

## Visualizations

Pass `save="path.png"` to export a static matplotlib figure, or `html="path.html"` for an interactive visualization:

```python
model.attention("hello world", layer=0, head=0, save="attention.png")
model.trace("...Paris...", "...Rome...", save="trace.png")
model.lens("The capital of France is", save="lens.png")
model.steer("The weather is", vector=vector, at="transformer.h.8", save="steer.png")
model.attribute("The capital of France is", save="attribution.png")
interpkit.diff(base, finetuned, "...", save="diff.png")

# Interactive HTML — self-contained files with hover tooltips, filters, and sliders
model.attention("hello world", html="attention.html")
model.trace("...Paris...", "...Rome...", html="trace.html")
model.attribute("The capital of France is", html="attribution.html")
```

---

## CLI

```bash
interpkit inspect gpt2
interpkit trace gpt2 --clean "...Paris..." --corrupted "...Rome..." --top-k 20
interpkit lens gpt2 "The capital of France is"
interpkit attention gpt2 "The capital of France is" --layer 8 --save attention.png
interpkit steer gpt2 "The weather is" --positive Love --negative Hate --at transformer.h.8
interpkit ablate gpt2 "The capital of France is" --at transformer.h.8.mlp
interpkit diff gpt2 my-finetuned-gpt2 "The capital of France is" --save diff.png
interpkit features gpt2 "The capital of France is" --at transformer.h.8 --sae jbloom/GPT2-Small-SAEs-Reformatted

# Interactive HTML output
interpkit attention gpt2 "hello world" --html attention.html
interpkit trace gpt2 --clean "...Paris..." --corrupted "...Rome..." --html trace.html
interpkit attribute gpt2 "The capital of France is" --html attribution.html

# Vision models — auto-preprocessed
interpkit attribute microsoft/resnet-50 cat.jpg --target 281
```

Run `interpkit` with no arguments for a full command reference.

---

## TransformerLens interop

Already using TransformerLens? Pass your `HookedTransformer` directly into InterpKit — it auto-detects the model and extracts the tokenizer:

```python
from transformer_lens import HookedTransformer
import interpkit

tl_model = HookedTransformer.from_pretrained("gpt2")
model = interpkit.load(tl_model)

# All InterpKit operations work on TL models
model.trace("The Eiffel Tower is in Paris", "The Eiffel Tower is in Rome", top_k=20)
model.attention("The capital of France is", save="attention.png")
model.steer("The weather is", vector=vector, at="blocks.8", scale=2.0)
```

Translate between native and TL hook point names:

```python
interpkit.to_tl_name("transformer.h.8.mlp")       # -> "blocks.8.mlp"
interpkit.to_native_name("blocks.8.attn", model.arch_info)  # -> "transformer.h.8.attn"
interpkit.list_tl_hooks(tl_model)                  # -> ["blocks.0.hook_resid_pre", ...]
```

---

## Local models

```python
import torch.nn as nn
import interpkit

my_model = MyCustomModel()
interpkit.register(my_model, layers=["blocks.0", "blocks.1"], output_head="head")
model = interpkit.load(my_model, tokenizer=my_tokenizer)
model.trace(input_a, input_b, top_k=10)
```

---

## Examples

See the [`examples/`](examples/) directory for Jupyter notebooks:

| Notebook | Topics |
|----------|--------|
| `01_quickstart` | Inspect, trace, lens, attribution, patching, ablation |
| `02_attention_patterns` | Per-head heatmaps, layer filtering, HTML export |
| `03_steering_vectors` | Extract and apply steering vectors at different layers/scales |
| `04_sae_features` | Sparse Autoencoder feature decomposition |
| `05_caching_and_probing` | Activation cache, linear probes across layers |
| `06_model_comparison` | Diff two models, side-by-side tracing and logit lens |
| `07_vision_models` | ResNet/ViT attribution, ablation, activations |

---

## License

MIT
