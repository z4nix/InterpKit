"""Parametrized smoke tests across 25+ HuggingFace model architectures.

Every test in this module is gated by ``@pytest.mark.slow`` — they only
run when ``pytest --slow`` is passed.  A dedicated CI job runs them on
merges to ``main`` only.

Each model family exercises:
  - Architecture discovery (layers, attention, LM head detection)
  - ``inspect()``
  - ``activations()`` on the first detected layer
  - Family-appropriate ops (lens/dla for LMs, attention for transformers, etc.)
"""

from __future__ import annotations

import os
import tempfile

import pytest

from tests.conftest import load_or_skip

pytestmark = pytest.mark.slow

# ──────────────────────────────────────────────────────────────────
# Model lists
# ──────────────────────────────────────────────────────────────────

CAUSAL_LM_MODELS = [
    "openai-community/gpt2",
    "EleutherAI/pythia-160m",
    "facebook/opt-125m",
    "EleutherAI/gpt-neo-125m",
    "HuggingFaceTB/SmolLM-135M",
    "Qwen/Qwen2-0.5B",
]

ENCODER_ONLY_MODELS = [
    "distilbert-base-uncased",
    "google-bert/bert-base-uncased",
    "FacebookAI/roberta-base",
    "google/electra-small-discriminator",
]

ENCODER_DECODER_MODELS = [
    "google-t5/t5-small",
    "facebook/bart-base",
    "Helsinki-NLP/opus-mt-en-de",
]

VISION_MODELS = [
    "microsoft/resnet-18",
    "google/vit-base-patch16-224",
    "facebook/dinov2-small",
    "microsoft/swin-tiny-patch4-window7-224",
    "facebook/convnext-tiny-224",
]

MULTIMODAL_MODELS = [
    "openai/clip-vit-base-patch32",
]

SSM_MODELS = [
    "state-spaces/mamba-130m",
]


# ──────────────────────────────────────────────────────────────────
# Fixtures — session-scoped, parametrized per family
# ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session", params=CAUSAL_LM_MODELS)
def causal_lm(request):
    return load_or_skip(request.param)


@pytest.fixture(scope="session", params=ENCODER_ONLY_MODELS)
def encoder_model(request):
    return load_or_skip(request.param)


@pytest.fixture(scope="session", params=ENCODER_DECODER_MODELS)
def enc_dec_model(request):
    return load_or_skip(request.param)


@pytest.fixture(scope="session", params=VISION_MODELS)
def vision_model(request):
    return load_or_skip(request.param)


@pytest.fixture(scope="session", params=MULTIMODAL_MODELS)
def multimodal_model(request):
    return load_or_skip(request.param)


@pytest.fixture(scope="session", params=SSM_MODELS)
def ssm_model(request):
    return load_or_skip(request.param)


@pytest.fixture(scope="session")
def test_image():
    from PIL import Image

    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    path = os.path.join(tempfile.gettempdir(), "interpkit_arch_test.jpg")
    img.save(path)
    return path


# ──────────────────────────────────────────────────────────────────
# Causal LM tests
# ──────────────────────────────────────────────────────────────────


class TestCausalLM:
    def test_has_layers(self, causal_lm):
        assert len(causal_lm.arch_info.layer_names) > 0

    def test_has_lm_head(self, causal_lm):
        assert causal_lm.arch_info.has_lm_head

    def test_is_language_model(self, causal_lm):
        assert causal_lm.arch_info.is_language_model

    def test_attention_resolved(self, causal_lm):
        attn_layers = [li for li in causal_lm.arch_info.layer_infos if li.attn_path]
        assert len(attn_layers) > 0

    def test_mlp_resolved(self, causal_lm):
        mlp_layers = [li for li in causal_lm.arch_info.layer_infos if li.mlp_path]
        assert len(mlp_layers) > 0

    def test_inspect(self, causal_lm):
        causal_lm.inspect()

    def test_activations(self, causal_lm):
        layer = causal_lm.arch_info.layer_names[0]
        act = causal_lm.activations("Hello world", at=layer)
        assert act is not None
        assert act.dim() >= 2

    def test_lens(self, causal_lm):
        result = causal_lm.lens("The capital of France is")
        assert result is not None
        assert len(result) > 0

    def test_dla(self, causal_lm):
        result = causal_lm.dla("The capital of France is")
        assert "contributions" in result
        assert "target_token" in result

    def test_attribute(self, causal_lm):
        result = causal_lm.attribute("Hello world")
        assert "tokens" in result
        assert "scores" in result

    def test_scan(self, causal_lm):
        result = causal_lm.scan("Hello world")
        assert isinstance(result, dict)

    def test_decompose(self, causal_lm):
        result = causal_lm.decompose("Hello world")
        assert "components" in result
        assert len(result["components"]) > 0

    def test_qkv_style_detected(self, causal_lm):
        for li in causal_lm.arch_info.layer_infos:
            if li.attn_path:
                assert li.qkv_style in ("separate", "fused", None)
                break


# ──────────────────────────────────────────────────────────────────
# Encoder-only tests
# ──────────────────────────────────────────────────────────────────


class TestEncoderOnly:
    def test_has_layers(self, encoder_model):
        assert len(encoder_model.arch_info.layer_names) > 0

    def test_attention_resolved(self, encoder_model):
        attn_layers = [li for li in encoder_model.arch_info.layer_infos if li.attn_path]
        assert len(attn_layers) > 0

    def test_inspect(self, encoder_model):
        encoder_model.inspect()

    def test_activations(self, encoder_model):
        layer = encoder_model.arch_info.layer_names[0]
        act = encoder_model.activations("Hello world", at=layer)
        assert act is not None

    def test_attribute(self, encoder_model):
        result = encoder_model.attribute("Hello world")
        assert "tokens" in result
        assert "scores" in result


# ──────────────────────────────────────────────────────────────────
# Encoder-decoder tests
# ──────────────────────────────────────────────────────────────────


class TestEncoderDecoder:
    def test_has_layers(self, enc_dec_model):
        assert len(enc_dec_model.arch_info.layer_names) > 0

    def test_is_encoder_decoder(self, enc_dec_model):
        assert enc_dec_model.arch_info.is_encoder_decoder

    def test_inspect(self, enc_dec_model):
        enc_dec_model.inspect()

    def test_activations(self, enc_dec_model):
        layer = enc_dec_model.arch_info.layer_names[0]
        act = enc_dec_model.activations("Hello world", at=layer)
        assert act is not None


# ──────────────────────────────────────────────────────────────────
# Vision tests
# ──────────────────────────────────────────────────────────────────


class TestVision:
    def test_has_layers(self, vision_model):
        assert len(vision_model.arch_info.layer_names) > 0

    def test_not_language_model(self, vision_model):
        assert not vision_model.arch_info.is_language_model

    def test_inspect(self, vision_model):
        vision_model.inspect()

    def test_activations(self, vision_model, test_image):
        layer = vision_model.arch_info.layer_names[0]
        act = vision_model.activations(test_image, at=layer)
        assert act is not None

    def test_attribute(self, vision_model, test_image):
        result = vision_model.attribute(test_image, target=0)
        assert "grad" in result


# ──────────────────────────────────────────────────────────────────
# Multimodal tests
# ──────────────────────────────────────────────────────────────────


class TestMultimodal:
    def test_has_layers(self, multimodal_model):
        assert len(multimodal_model.arch_info.layer_names) > 0

    def test_inspect(self, multimodal_model):
        multimodal_model.inspect()

    def test_activations(self, multimodal_model, test_image):
        layer = multimodal_model.arch_info.layer_names[0]
        act = multimodal_model.activations(test_image, at=layer)
        assert act is not None


# ──────────────────────────────────────────────────────────────────
# SSM / hybrid tests
# ──────────────────────────────────────────────────────────────────


class TestSSM:
    def test_has_layers(self, ssm_model):
        assert len(ssm_model.arch_info.layer_names) > 0

    def test_inspect(self, ssm_model):
        ssm_model.inspect()

    def test_activations(self, ssm_model):
        layer = ssm_model.arch_info.layer_names[0]
        act = ssm_model.activations("Hello world", at=layer)
        assert act is not None

    def test_no_attention_expected(self, ssm_model):
        attn_layers = [li for li in ssm_model.arch_info.layer_infos if li.attn_path]
        assert len(attn_layers) == 0, "Pure SSM should have no attention layers"
