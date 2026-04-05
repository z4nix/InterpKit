# CLI Reference

InterpKit provides a command-line interface for running interpretability
operations without writing Python code.

## Usage

```bash
interpkit <command> [options]
```

## Commands

### inspect

Print the model's module tree with types, param counts, and detected roles.

```bash
interpkit inspect gpt2
```

### trace

Run causal tracing between clean and corrupted inputs.

```bash
interpkit trace gpt2 --clean "The Eiffel Tower is in Paris" --corrupted "The Eiffel Tower is in Rome"
```

### patch

Activation patching at a specific module.

```bash
interpkit patch gpt2 --clean "The Eiffel Tower is in Paris" --corrupted "The Eiffel Tower is in Rome" --at transformer.h.8.mlp
```

### lens

Run logit lens analysis.

```bash
interpkit lens gpt2 --text "The capital of France is"
```

### dla

Direct Logit Attribution.

```bash
interpkit dla gpt2 --text "The capital of France is"
```

### attribute

Gradient-based input attribution.

```bash
interpkit attribute gpt2 --text "The capital of France is"
```

### scan

One-command model overview (DLA + lens + attention).

```bash
interpkit scan gpt2 --text "The capital of France is"
```

### attention

Visualize attention patterns.

```bash
interpkit attention gpt2 --text "The capital of France is"
```

## Common options

Most commands accept:

- `--device` — Device to run on (cpu, cuda, mps)
- `--dtype` — Model dtype (float16, bfloat16, float32, auto)
- `--device-map` — HuggingFace device_map for multi-GPU
- `--save` — Save output to a file (PNG)
- `--html` — Save interactive HTML output
- `--json` — Output results as JSON
