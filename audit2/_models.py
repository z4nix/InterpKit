"""Model catalogue for audit2 — small + mid (~28 models, ≤2B, no gated)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelMeta:
    model_id: str
    family: str           # short tag for grouping
    task: str             # "lm_decoder" | "lm_encoder" | "lm_enc_dec" | "vision"
    qkv_style: str = "unknown"
    has_chat: bool = False
    text: str = "The capital of France is"
    skip_reason: str = ""           # non-empty → don't even try to load


# ---------------------------------------------------------------------------
# In-scope models (28)
# ---------------------------------------------------------------------------

LM_DECODER: tuple[ModelMeta, ...] = (
    ModelMeta("gpt2",                                  "gpt2",     "lm_decoder", "conv1d_concat"),
    ModelMeta("distilgpt2",                            "gpt2",     "lm_decoder", "conv1d_concat"),
    ModelMeta("EleutherAI/pythia-70m",                 "pythia",   "lm_decoder", "gptneox_interleaved"),
    ModelMeta("EleutherAI/pythia-160m",                "pythia",   "lm_decoder", "gptneox_interleaved"),
    ModelMeta("EleutherAI/gpt-neo-125m",               "gpt_neo",  "lm_decoder", "linear_separate"),
    ModelMeta("facebook/opt-125m",                     "opt",      "lm_decoder", "linear_separate"),
    ModelMeta("facebook/opt-350m",                     "opt",      "lm_decoder", "linear_separate"),
    ModelMeta("bigscience/bloom-560m",                 "bloom",    "lm_decoder", "bloom_interleaved"),
    ModelMeta("Qwen/Qwen2-0.5B",                       "qwen2",    "lm_decoder", "gqa_separate"),
    ModelMeta("Qwen/Qwen2.5-0.5B",                     "qwen2",    "lm_decoder", "gqa_separate"),
    ModelMeta("Qwen/Qwen2.5-0.5B-Instruct",            "qwen2",    "lm_decoder", "gqa_separate", has_chat=True),
    ModelMeta("Qwen/Qwen3-0.6B",                       "qwen3",    "lm_decoder", "gqa_separate"),
    ModelMeta("HuggingFaceTB/SmolLM-135M",             "smollm",   "lm_decoder", "gqa_separate"),
    ModelMeta("HuggingFaceTB/SmolLM2-360M-Instruct",   "smollm",   "lm_decoder", "gqa_separate", has_chat=True),
    ModelMeta("TinyLlama/TinyLlama-1.1B-Chat-v1.0",    "llama",    "lm_decoder", "gqa_separate", has_chat=True),
    ModelMeta("Felladrin/Llama-160M-Chat-v1",          "llama",    "lm_decoder", "linear_separate", has_chat=True),
)

LM_ENCODER: tuple[ModelMeta, ...] = (
    ModelMeta("distilbert-base-uncased",               "distilbert", "lm_encoder", "linear_separate", text="The capital of France is [MASK]."),
    ModelMeta("bert-base-uncased",                     "bert",       "lm_encoder", "linear_separate", text="The capital of France is [MASK]."),
    ModelMeta("albert-base-v2",                        "albert",     "lm_encoder", "linear_separate", text="The capital of France is [MASK]."),
    ModelMeta("roberta-base",                          "roberta",    "lm_encoder", "linear_separate", text="The capital of France is <mask>."),
    ModelMeta("microsoft/deberta-v3-small",            "deberta_v3", "lm_encoder", "linear_separate", text="The capital of France is [MASK]."),
    ModelMeta("google/electra-small-discriminator",    "electra",    "lm_encoder", "linear_separate", text="The capital of France is."),
)

LM_ENC_DEC: tuple[ModelMeta, ...] = (
    ModelMeta("t5-small",                              "t5",     "lm_enc_dec", "linear_separate",
              text="translate English to German: Hello world."),
    ModelMeta("google/flan-t5-small",                  "t5",     "lm_enc_dec", "linear_separate",
              text="Summarize: The quick brown fox jumps."),
    ModelMeta("facebook/bart-base",                    "bart",   "lm_enc_dec", "linear_separate",
              text="The capital of France is."),
)

VISION: tuple[ModelMeta, ...] = (
    ModelMeta("google/vit-base-patch16-224",                "vit",   "vision", "linear_separate"),
    ModelMeta("microsoft/swin-tiny-patch4-window7-224",     "swin",  "vision", "linear_separate"),
    ModelMeta("microsoft/resnet-18",                        "resnet","vision", "none"),
)


ALL_MODELS: tuple[ModelMeta, ...] = (
    *LM_DECODER, *LM_ENCODER, *LM_ENC_DEC, *VISION,
)


# Models intentionally excluded from this audit:
DROP_LIST = {
    "google/gemma-2b":                "gated by Google",
    "google/recurrentgemma-2b":       "gated, also recurrent (Griffin) — out of scope",
    "allenai/longformer-base-4096":   "memory-heavy windowed attention; deferred",
    "facebook/dinov2-small":          "DINOv2 not in scope this round",
    "facebook/convnext-tiny-224":     "deferred",
    "openai/clip-vit-base-patch32":   "CLIP dual-encoder out of scope",
    "microsoft/phi-1_5":              "MIT license but slow MPS; deferred",
    "Helsinki-NLP/opus-mt-en-de":     "Marian translation; deferred",
    "deepset/roberta-base-squad2":    "QA head; out of scope",
}


def by_id(model_id: str) -> ModelMeta | None:
    for m in ALL_MODELS:
        if m.model_id == model_id:
            return m
    return None


def of_task(task_prefix: str) -> list[ModelMeta]:
    return [m for m in ALL_MODELS if m.task.startswith(task_prefix)]
