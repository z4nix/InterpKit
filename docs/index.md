# InterpKit

> Mech interp for any HuggingFace model.

InterpKit provides a single, consistent interface for mechanistic interpretability operations across any HuggingFace model — transformers, SSMs, vision models, and more — with zero annotation required.

## Features

- **Universal model support** — works with any HuggingFace model via automatic architecture discovery
- **Complete operation set** — causal tracing, activation patching, logit lens, DLA, attribution, SAE features, circuit finding, and more
- **Chat-model aware** — pass message lists (`[{"role": "user", "content": "..."}]`) to any op; the tokenizer's chat template is applied automatically
- **Rich CLI** — run any operation from the command line with `interpkit` (or `python -m interpkit`)
- **Interactive output** — HTML reports, matplotlib plots, and Rich console tables

## Install

```bash
pip install interpkit
```

For development:

```bash
git clone https://github.com/z4nix/interpkit.git
cd interpkit
pip install -e ".[dev]"
```

## Quick example

```python
import interpkit

model = interpkit.load("gpt2")
model.trace("The Eiffel Tower is in Paris", "The Eiffel Tower is in Rome")
```

See the [Quickstart](quickstart.md) for more examples, or dive into the [API Reference](api/model.md).
