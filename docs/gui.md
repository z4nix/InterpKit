# Web GUI

InterpKit ships a clean, local web GUI that exposes every operation through the
browser. It runs a small [FastAPI](https://fastapi.tiangolo.com/) server on your
machine and serves a no-build single-page app — there is no JavaScript toolchain
to install and nothing leaves your computer.

## Install & launch

The GUI lives behind the optional `gui` extra:

```bash
pip install "interpkit[gui]"
interpkit gui
```

This starts the server and opens [http://127.0.0.1:7860](http://127.0.0.1:7860)
in your default browser.

| Option | Default | Meaning |
|--------|---------|---------|
| `--host` | `127.0.0.1` | Interface to bind to. |
| `--port` | `7860` | Port to serve on. |
| `--no-browser` | off | Don't open the browser automatically. |

If the optional dependencies are missing, `interpkit gui` prints a one-line
install hint instead of a traceback.

## Loading a model

The first screen is a single card: type a HuggingFace model ID (or pick a
suggestion chip), choose a device and dtype, and hit **Load model**. Loading
runs as a background job with a live status indicator; the first load of a model
downloads its weights.

Once loaded, the sidebar appears and the model is shown as a chip in the header
(with an **unload** button that frees the memory). The architecture is detected
automatically — you never specify layer counts or module paths by hand.

## Running operations

Operations are grouped in the sidebar by category:

- **Overview** — `scan`, `inspect`, `report`
- **Analysis** — `lens`, `dla`, `attribute`, `attention`, `activations`,
  `trace`, `patch`, `ablate`, `decompose`, `diff`, `probe`
- **Steering & Generation** — `steer`, `generate`, `chat`, `features`
- **Circuits** — `find-circuit`, `atp`, `eap`, `maxact`
- **Advanced** — `train-tuned-lens`

Each panel is a form generated from the operation's parameters. Module, layer,
and head inputs are pickers populated from the detected architecture, so you
don't need to remember paths like `transformer.h.8.mlp`. Sensible defaults are
pre-filled, and rarely-used fields are tucked under **Advanced options**.

Operations that don't apply to the loaded model (for example, `attention` on a
CNN, or `chat` on a base model) are greyed out in the sidebar with the reason
shown on hover — the same support checks the library enforces.

### Results

Results are rendered natively: logit-lens and attention heatmaps, DLA and
trace bar charts, attribution token strips, generation text, an interactive
chat thread, and more. Every result also offers a collapsible **Raw JSON** view
and a **Download JSON** button. Each panel keeps a short history of recent runs
you can click back to.

### Long-running operations

`scan`, `trace`, `maxact`, `train-tuned-lens`, `generate`, and `report` can take
a while. They run as background jobs with a progress bar (where the underlying
op reports progress) and a **Cancel** button. Operations on a single model run
one at a time; loading a second model into another session runs independently.

## How it works

- Each loaded model is a **session** with its own single-worker queue, so two
  operations never run concurrently on the same model (forward hooks and the
  activation cache are not reentrant).
- Every action is a **job**; the browser polls for its status and result. The
  server's request handlers never block on PyTorch.
- The same operation registry that drives the API also generates the forms, so
  the GUI always covers the full CLI surface.

## Troubleshooting

- **`The GUI requires the optional [gui] extra`** — run
  `pip install "interpkit[gui]"`.
- **Port already in use** — pass `--port <n>` (e.g. `interpkit gui --port 8000`).
- **An op is greyed out** — it isn't supported for the loaded architecture;
  hover the sidebar entry for the reason.
- **File / corpus inputs** (`probe`, `maxact`, `train-tuned-lens`) — paste
  examples directly into the textarea (one per line), or point at a server-side
  file path. File uploads from the browser are not supported yet.
