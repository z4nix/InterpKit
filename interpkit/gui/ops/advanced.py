"""Advanced ops: train-tuned-lens."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from interpkit.gui.ops.base import JobContext, OpSpec, lines_list, ui


class TrainTunedLensParams(BaseModel):
    corpus: str | None = Field(
        None,
        description="Training sentences, one per line",
        json_schema_extra=ui(widget="textarea", rows=6),
    )
    corpus_path: str | None = Field(
        None,
        description="Alternative: server-side text file with one sentence per line",
        json_schema_extra=ui(widget="path"),
    )
    steps: int = Field(200, ge=1, description="Training steps")
    batch_size: int = Field(4, ge=1, description="Batch size")
    lr: float = Field(1e-3, gt=0, title="learning rate", description="Adam learning rate")
    max_length: int = Field(64, ge=1, description="Token truncation length", json_schema_extra=ui(advanced=True))
    seed: int = Field(0, description="Random seed", json_schema_extra=ui(advanced=True))
    save: str | None = Field(
        None,
        description="Output directory or .safetensors path (default: ~/.cache/interpkit/tuned_lens/<model>/)",
        json_schema_extra=ui(widget="path", advanced=True),
    )


def _run_train_tuned_lens(model: Any, p: TrainTunedLensParams, ctx: JobContext) -> Any:
    from interpkit.ops.tuned_lens import default_tuned_lens_dir

    if bool(p.corpus) == bool(p.corpus_path):
        raise ValueError("train-tuned-lens: provide exactly one of corpus or corpus_path.")
    if p.corpus:
        corpus = lines_list(p.corpus)
    else:
        from interpkit.core.inputs import read_examples_file

        corpus = read_examples_file(p.corpus_path)

    out = p.save if p.save else str(default_tuned_lens_dir(ctx.model_id or "model"))
    lens_obj = model.train_tuned_lens(
        corpus,
        steps=p.steps,
        batch_size=p.batch_size,
        lr=p.lr,
        max_length=p.max_length,
        seed=p.seed,
        save=out,
        progress_callback=ctx.progress_callback,
    )
    return {"saved_to": out, "meta": lens_obj.meta}


SPECS: list[OpSpec] = [
    OpSpec(
        name="train-tuned-lens",
        category="advanced",
        title="Train tuned lens",
        description="Train per-layer affine translators (Belrose et al. 2023) for unbiased early-layer lens readouts. Use the saved path in the Logit lens panel.",
        params=TrainTunedLensParams,
        run=_run_train_tuned_lens,
        long_running=True,
        support_key="train_tuned_lens",
    ),
]
