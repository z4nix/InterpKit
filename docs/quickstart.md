# Quickstart

## Loading a model

```python
import interpkit

model = interpkit.load("gpt2")
```

Any HuggingFace model ID works. For vision models:

```python
model = interpkit.load("google/vit-base-patch16-224")
```

For chat / instruction-tuned models, every op accepts message lists in addition
to plain strings; the tokenizer's chat template is applied automatically:

```python
chat = interpkit.load("HuggingFaceTB/SmolLM2-360M-Instruct")

result = chat.chat("Write a haiku about cats.", max_new_tokens=64)
print(result["response"])

# Pass a chat-message list to any op
chat.dla([{"role": "user", "content": "Capital of France?"}])
```

See [`examples/10_chat_models.ipynb`](https://github.com/z4nix/interpkit/blob/main/examples/10_chat_models.ipynb)
for a full walkthrough including chat-style steering.

## Core operations

### Causal tracing

```python
result = model.trace(
    "The Eiffel Tower is in Paris",
    "The Eiffel Tower is in Rome",
)
```

### Activation patching

```python
result = model.patch(
    "The Eiffel Tower is in Paris",
    "The Eiffel Tower is in Rome",
    at="transformer.h.8.mlp",
)
```

### Logit lens

```python
result = model.lens("The capital of France is")
```

### Direct Logit Attribution

```python
result = model.dla("The capital of France is")
```

### Attribution

```python
result = model.attribute("The capital of France is")
```

### One-command scan

```python
result = model.scan("The capital of France is")
```

## Saving output

Most operations accept `save` and `html` parameters:

```python
model.trace(clean, corrupted, save="trace.png", html="trace.html")
model.attention("Hello world", save="attn.png", html="attn.html")
```

## Custom models

For models that auto-discovery can't handle, use `register()`:

```python
interpkit.register(
    my_model,
    layers=["blocks.0", "blocks.1", "blocks.2"],
    attention_modules=["blocks.0.attn", "blocks.1.attn", "blocks.2.attn"],
    mlp_modules=["blocks.0.ffn", "blocks.1.ffn", "blocks.2.ffn"],
    output_head="head",
)

model = interpkit.load(my_model, tokenizer=my_tokenizer)
```

See the full [API Reference](api/model.md) for all available operations and parameters.
