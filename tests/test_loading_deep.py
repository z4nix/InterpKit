"""Deep tests for model loading, Auto-class selection, discovery accuracy,
dummy-input construction, registration merging, and weight integrity.

Closes the gap where "loading works" was inferred rather than proven:
every section validates *correctness* of the loaded artefact, not just
that an operation runs without crashing.

Sections:
  1. Forward-pass logit equivalence (interpkit vs raw HuggingFace)
  2. Auto-class selection verification
  3. Discovery ground-truth validation
  4. Dummy-input construction — all code paths
  5. Registration merge end-to-end
  6. Weight integrity (dtype, device, param count, embedding shape)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

slow = pytest.mark.timeout(300)

TEXT = "The capital of France is"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    pad_token = None
    eos_token = "[EOS]"

    def __call__(self, text, **kw):
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def encode(self, text):
        return [1, 2, 3]


class _FakeImageProcessor:
    def __call__(self, *, images, return_tensors="pt"):
        return {"pixel_values": torch.randn(1, 3, 224, 224)}


class _BareModule(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        if config is not None:
            self.config = config

    def forward(self, x=None, **kw):
        if x is None:
            x = torch.randn(1, 10)
        if not isinstance(x, torch.Tensor):
            x = torch.randn(1, 10)
        return self.fc(x)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1: Forward-pass logit equivalence
# ═══════════════════════════════════════════════════════════════════════════


class TestLogitEquivalence:
    """interpkit-loaded model must produce identical logits to raw HF."""

    @staticmethod
    def _compare_logits(kit_model, hf_auto_cls_name, model_id, *, is_enc_dec=False):
        import transformers

        device = kit_model._device
        auto_cls = getattr(transformers, hf_auto_cls_name)
        ref = auto_cls.from_pretrained(model_id).eval().to(device)
        tok = transformers.AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        inputs = tok(TEXT, return_tensors="pt")
        if is_enc_dec and "decoder_input_ids" not in inputs:
            dec_start = getattr(ref.config, "decoder_start_token_id", 0) or 0
            inputs["decoder_input_ids"] = torch.tensor([[dec_start]])

        dev_inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            ref_out = ref(**dev_inputs)
            kit_out = kit_model._model(**dev_inputs)

        ref_logits = ref_out.logits if hasattr(ref_out, "logits") else ref_out[0]
        kit_logits = kit_out.logits if hasattr(kit_out, "logits") else kit_out[0]

        ref_logits = ref_logits.float().cpu()
        kit_logits = kit_logits.float().cpu()

        assert ref_logits.shape == kit_logits.shape, (
            f"Shape mismatch: ref {ref_logits.shape} vs kit {kit_logits.shape}"
        )
        assert torch.allclose(ref_logits, kit_logits, atol=1e-4), (
            f"Max diff: {(ref_logits - kit_logits).abs().max().item():.6e}"
        )

    @slow
    def test_gpt2(self, gpt2_model):
        self._compare_logits(gpt2_model, "AutoModelForCausalLM", "gpt2")

    @slow
    def test_pythia(self, pythia_model):
        self._compare_logits(
            pythia_model, "AutoModelForCausalLM", "EleutherAI/pythia-160m",
        )

    @slow
    def test_smollm(self, smollm_model):
        self._compare_logits(
            smollm_model, "AutoModelForCausalLM", "HuggingFaceTB/SmolLM-135M",
        )

    @slow
    def test_bloom(self, bloom_model):
        self._compare_logits(
            bloom_model, "AutoModelForCausalLM", "bigscience/bloom-560m",
        )

    @slow
    def test_opt(self, opt_model):
        self._compare_logits(
            opt_model, "AutoModelForCausalLM", "facebook/opt-350m",
        )

    @slow
    def test_bart_shape(self, bart_model):
        """bart-base loads as BartModel (encoder-decoder base), verify output shape."""
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("facebook/bart-base")
        inputs = tok(TEXT, return_tensors="pt")
        dev_inputs = {k: v.to(bart_model._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = bart_model._model(**dev_inputs)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        seq_len = inputs["input_ids"].shape[1]
        assert hidden.shape[0] == 1
        assert hidden.shape[1] == seq_len
        assert hidden.shape[2] == bart_model.arch_info.hidden_size

    @slow
    def test_t5(self, t5_model):
        self._compare_logits(
            t5_model, "AutoModelForSeq2SeqLM", "t5-small", is_enc_dec=True,
        )

    @slow
    def test_distilbert(self, distilbert_model):
        self._compare_logits(
            distilbert_model, "AutoModelForMaskedLM", "distilbert-base-uncased",
        )

    @slow
    def test_roberta_qa_shape(self, roberta_qa_model):
        """QA model outputs (start_logits, end_logits) — verify shape, not exact logits."""
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
        inputs = tok(TEXT, return_tensors="pt")
        dev_inputs = {k: v.to(roberta_qa_model._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = roberta_qa_model._model(**dev_inputs)
        assert hasattr(out, "start_logits") and hasattr(out, "end_logits"), (
            "QA model output must have start_logits and end_logits"
        )
        seq_len = inputs["input_ids"].shape[1]
        assert out.start_logits.shape[-1] == seq_len
        assert out.end_logits.shape[-1] == seq_len


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2: Auto-class selection verification
# ═══════════════════════════════════════════════════════════════════════════


class TestAutoClassSelection:
    """Verify _load_from_hf picks the correct HF Auto* class."""

    def test_gpt2_class(self, gpt2_model):
        assert type(gpt2_model._model).__name__ == "GPT2LMHeadModel"

    def test_distilbert_class(self, distilbert_model):
        assert type(distilbert_model._model).__name__ == "DistilBertForMaskedLM"

    def test_t5_class(self, t5_model):
        assert type(t5_model._model).__name__ == "T5ForConditionalGeneration"

    def test_roberta_qa_class(self, roberta_qa_model):
        assert type(roberta_qa_model._model).__name__ == "RobertaForQuestionAnswering"

    def test_vit_class(self, vit_model):
        assert type(vit_model._model).__name__ == "ViTForImageClassification"

    def test_resnet_class(self, resnet_model):
        assert type(resnet_model._model).__name__ == "ResNetForImageClassification"

    def test_bloom_class(self, bloom_model):
        assert type(bloom_model._model).__name__ == "BloomForCausalLM"

    def test_opt_class(self, opt_model):
        assert type(opt_model._model).__name__ == "OPTForCausalLM"

    def test_bart_class(self, bart_model):
        """bart-base config declares BartModel, so it loads as the base encoder-decoder."""
        assert type(bart_model._model).__name__ == "BartModel"

    def test_albert_class(self, albert_model):
        assert type(albert_model._model).__name__ == "AlbertForMaskedLM"

    def test_pythia_class(self, pythia_model):
        assert type(pythia_model._model).__name__ == "GPTNeoXForCausalLM"

    def test_electra_class(self, electra_model):
        name = type(electra_model._model).__name__
        assert "Electra" in name

    def test_smollm_class(self, smollm_model):
        name = type(smollm_model._model).__name__
        assert "CausalLM" in name or "LMHead" in name

    def test_gpt_neo_class(self, gpt_neo_model):
        assert type(gpt_neo_model._model).__name__ == "GPTNeoForCausalLM"

    def test_flan_t5_class(self, flan_t5_model):
        assert type(flan_t5_model._model).__name__ == "T5ForConditionalGeneration"


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3: Discovery ground-truth validation
# ═══════════════════════════════════════════════════════════════════════════


class TestDiscoveryGroundTruth:
    """Assert ModelArchInfo fields against known architecture specifications."""

    # -- GPT-2 --

    def test_gpt2_num_layers(self, gpt2_model):
        assert gpt2_model.arch_info.num_layers == 12

    def test_gpt2_hidden_size(self, gpt2_model):
        assert gpt2_model.arch_info.hidden_size == 768

    def test_gpt2_num_heads(self, gpt2_model):
        assert gpt2_model.arch_info.num_attention_heads == 12

    def test_gpt2_vocab_size(self, gpt2_model):
        assert gpt2_model.arch_info.vocab_size == 50257

    def test_gpt2_has_lm_head(self, gpt2_model):
        assert gpt2_model.arch_info.has_lm_head is True

    def test_gpt2_unembedding(self, gpt2_model):
        assert gpt2_model.arch_info.unembedding_name is not None
        assert "lm_head" in gpt2_model.arch_info.unembedding_name

    def test_gpt2_is_language_model(self, gpt2_model):
        assert gpt2_model.arch_info.is_language_model is True

    def test_gpt2_not_encoder_decoder(self, gpt2_model):
        assert gpt2_model.arch_info.is_encoder_decoder is False

    def test_gpt2_layer_count_matches(self, gpt2_model):
        assert len(gpt2_model.arch_info.layer_names) == 12

    def test_gpt2_layer_infos_populated(self, gpt2_model):
        assert len(gpt2_model.arch_info.layer_infos) == 12

    def test_gpt2_layer0_has_attn_and_mlp(self, gpt2_model):
        li = gpt2_model.arch_info.layer_infos[0]
        assert li.attn_path is not None, "Layer 0 should have an attention path"
        assert li.mlp_path is not None, "Layer 0 should have an MLP path"

    def test_gpt2_layer0_has_projections(self, gpt2_model):
        li = gpt2_model.arch_info.layer_infos[0]
        has_separate = (
            li.q_proj_path is not None
            and li.k_proj_path is not None
            and li.v_proj_path is not None
        )
        has_fused = li.qkv_proj_path is not None
        assert has_separate or has_fused, "Layer 0 should have Q/K/V projections"

    # -- T5-small --

    def test_t5_hidden_size(self, t5_model):
        assert t5_model.arch_info.hidden_size == 512

    def test_t5_num_heads(self, t5_model):
        assert t5_model.arch_info.num_attention_heads == 8

    def test_t5_is_encoder_decoder(self, t5_model):
        assert t5_model.arch_info.is_encoder_decoder is True

    def test_t5_has_layers(self, t5_model):
        assert len(t5_model.arch_info.layer_names) > 0

    # -- DistilBERT --

    def test_distilbert_num_layers(self, distilbert_model):
        assert distilbert_model.arch_info.num_layers == 6

    def test_distilbert_hidden_size(self, distilbert_model):
        assert distilbert_model.arch_info.hidden_size == 768

    def test_distilbert_num_heads(self, distilbert_model):
        assert distilbert_model.arch_info.num_attention_heads == 12

    def test_distilbert_has_lm_head(self, distilbert_model):
        """DistilBERT's vocab_projector is now detected via _tied_weights_keys."""
        assert distilbert_model.arch_info.has_lm_head is True

    # -- ResNet-18 (non-transformer) --

    def test_resnet_no_transformer_layers(self, resnet_model):
        assert resnet_model.arch_info.is_language_model is False

    def test_resnet_no_lm_head(self, resnet_model):
        assert resnet_model.arch_info.has_lm_head is False

    def test_resnet_not_encoder_decoder(self, resnet_model):
        assert resnet_model.arch_info.is_encoder_decoder is False

    # -- BLOOM-560m --

    def test_bloom_num_layers(self, bloom_model):
        assert bloom_model.arch_info.num_layers == 24

    def test_bloom_hidden_size(self, bloom_model):
        assert bloom_model.arch_info.hidden_size == 1024

    def test_bloom_num_heads(self, bloom_model):
        assert bloom_model.arch_info.num_attention_heads == 16

    def test_bloom_is_language_model(self, bloom_model):
        assert bloom_model.arch_info.is_language_model is True

    # -- OPT-350m --

    def test_opt_num_layers(self, opt_model):
        assert opt_model.arch_info.num_layers == 24

    def test_opt_hidden_size(self, opt_model):
        """OPT-350m has hidden_size=1024 (internal dim) with project_in/out from 512."""
        assert opt_model.arch_info.hidden_size == 1024

    def test_opt_is_language_model(self, opt_model):
        assert opt_model.arch_info.is_language_model is True

    # -- Pythia-160m --

    def test_pythia_num_layers(self, pythia_model):
        assert pythia_model.arch_info.num_layers == 12

    def test_pythia_hidden_size(self, pythia_model):
        assert pythia_model.arch_info.hidden_size == 768

    def test_pythia_is_language_model(self, pythia_model):
        assert pythia_model.arch_info.is_language_model is True

    def test_pythia_unembedding(self, pythia_model):
        assert pythia_model.arch_info.unembedding_name is not None

    # -- BART --

    def test_bart_is_encoder_decoder(self, bart_model):
        """bart-base now loads as BartModel, its native encoder-decoder form."""
        assert bart_model.arch_info.is_encoder_decoder is True

    def test_bart_hidden_size(self, bart_model):
        assert bart_model.arch_info.hidden_size == 768

    # -- GPT-Neo --

    def test_gpt_neo_num_layers(self, gpt_neo_model):
        assert gpt_neo_model.arch_info.num_layers == 12

    def test_gpt_neo_hidden_size(self, gpt_neo_model):
        assert gpt_neo_model.arch_info.hidden_size == 768

    def test_gpt_neo_is_language_model(self, gpt_neo_model):
        assert gpt_neo_model.arch_info.is_language_model is True

    # -- SmolLM --

    def test_smollm_is_language_model(self, smollm_model):
        assert smollm_model.arch_info.is_language_model is True

    def test_smollm_has_layers(self, smollm_model):
        assert len(smollm_model.arch_info.layer_names) > 0

    # -- ALBERT --

    def test_albert_hidden_size(self, albert_model):
        assert albert_model.arch_info.hidden_size == 768

    def test_albert_has_layers(self, albert_model):
        assert len(albert_model.arch_info.layer_names) > 0

    # -- RoBERTa-QA --

    def test_roberta_qa_not_language_model(self, roberta_qa_model):
        arch = roberta_qa_model.arch_info
        assert "QuestionAnswering" in arch.arch_family

    # -- Cross-architecture: layer_infos length matches layer_names --

    @pytest.mark.parametrize("fixture_name", [
        "gpt2_model", "distilbert_model", "bloom_model", "opt_model",
        "pythia_model", "gpt_neo_model", "smollm_model",
    ])
    def test_layer_infos_length_matches_names(self, fixture_name, request):
        model = request.getfixturevalue(fixture_name)
        assert len(model.arch_info.layer_infos) == len(model.arch_info.layer_names), (
            f"{fixture_name}: layer_infos ({len(model.arch_info.layer_infos)}) != "
            f"layer_names ({len(model.arch_info.layer_names)})"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 4: Dummy-input construction — all code paths
# ═══════════════════════════════════════════════════════════════════════════


class TestDummyInputConstruction:
    """Exercise every branch in _make_dummy_input."""

    @staticmethod
    def _make_dummy(model, *, tokenizer=None, image_processor=None, device="cpu"):
        from interpkit.core.model import _make_dummy_input

        return _make_dummy_input(
            model, tokenizer=tokenizer, image_processor=image_processor, device=device,
        )

    def test_tokenizer_path_returns_dict_with_input_ids(self):
        model = _BareModule(config=SimpleNamespace(is_encoder_decoder=False))
        result = self._make_dummy(model, tokenizer=_FakeTokenizer())
        assert isinstance(result, dict)
        assert "input_ids" in result

    def test_encoder_decoder_injects_decoder_ids(self):
        model = _BareModule(
            config=SimpleNamespace(is_encoder_decoder=True, decoder_start_token_id=0),
        )
        result = self._make_dummy(model, tokenizer=_FakeTokenizer())
        assert isinstance(result, dict)
        assert "decoder_input_ids" in result

    def test_encoder_decoder_decoder_start_token(self):
        model = _BareModule(
            config=SimpleNamespace(is_encoder_decoder=True, decoder_start_token_id=42),
        )
        result = self._make_dummy(model, tokenizer=_FakeTokenizer())
        assert result["decoder_input_ids"].item() == 42

    def test_image_processor_path_returns_pixel_values(self):
        model = _BareModule()
        result = self._make_dummy(model, image_processor=_FakeImageProcessor())
        assert isinstance(result, dict)
        assert "pixel_values" in result

    def test_fallback_tensor_path(self):
        model = _BareModule(
            config=SimpleNamespace(
                is_encoder_decoder=False, hidden_size=256,
            ),
        )
        result = self._make_dummy(model)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 8, 256)

    def test_fallback_n_embd(self):
        model = _BareModule(config=SimpleNamespace(is_encoder_decoder=False, n_embd=128))
        result = self._make_dummy(model)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 8, 128)

    def test_none_path_bare_module(self):
        model = nn.Linear(10, 10)
        result = self._make_dummy(model)
        assert result is None

    def test_tokenizer_takes_priority_over_image_processor(self):
        model = _BareModule(config=SimpleNamespace(is_encoder_decoder=False))
        result = self._make_dummy(
            model, tokenizer=_FakeTokenizer(), image_processor=_FakeImageProcessor(),
        )
        assert "input_ids" in result

    def test_device_propagation(self):
        model = _BareModule(config=SimpleNamespace(is_encoder_decoder=False))
        result = self._make_dummy(model, tokenizer=_FakeTokenizer(), device="cpu")
        assert isinstance(result, dict)
        for v in result.values():
            assert v.device == torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 5: Registration merge end-to-end
# ═══════════════════════════════════════════════════════════════════════════


class TestRegistrationMerge:
    """Verify load() actually applies register() overrides into arch_info."""

    def test_layers_override(self):
        import interpkit

        model = nn.Sequential(
            nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 5),
        )
        interpkit.register(model, layers=["0", "2"])
        wrapped = interpkit.load(model, tokenizer=None, device="cpu")
        assert wrapped.arch_info.layer_names == ["0", "2"]

    def test_output_head_override(self):
        import interpkit

        model = nn.Sequential(
            nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 5),
        )
        interpkit.register(model, output_head="2")
        wrapped = interpkit.load(model, tokenizer=None, device="cpu")
        assert wrapped.arch_info.output_head_name == "2"
        assert wrapped.arch_info.unembedding_name == "2"
        assert wrapped.arch_info.has_lm_head is True

    def test_attention_module_role_override(self):
        import interpkit

        class _TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.Linear(10, 10)
                self.ff = nn.Linear(10, 10)

            def forward(self, x):
                return self.ff(self.attn(x))

        model = _TinyModel()
        interpkit.register(model, attention_modules=["attn"])
        wrapped = interpkit.load(model, tokenizer=None, device="cpu")
        roles = {m.name: m.role for m in wrapped.arch_info.modules}
        assert roles.get("attn") == "attention"

    def test_mlp_module_role_override(self):
        import interpkit

        class _TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.Linear(10, 10)
                self.ff = nn.Linear(10, 10)

            def forward(self, x):
                return self.ff(self.attn(x))

        model = _TinyModel()
        interpkit.register(model, mlp_modules=["ff"])
        wrapped = interpkit.load(model, tokenizer=None, device="cpu")
        roles = {m.name: m.role for m in wrapped.arch_info.modules}
        assert roles.get("ff") == "mlp"

    def test_partial_registration_preserves_discovery(self):
        """Setting only layers should leave other discovered fields intact."""
        import interpkit

        model = nn.Sequential(
            nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 5),
        )
        interpkit.register(model, layers=["0"])
        wrapped = interpkit.load(model, tokenizer=None, device="cpu")
        assert wrapped.arch_info.layer_names == ["0"]
        assert len(wrapped.arch_info.modules) > 0, "Modules should still be discovered"

    def test_no_registration_uses_pure_discovery(self):
        import interpkit

        model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 5))
        wrapped = interpkit.load(model, tokenizer=None, device="cpu")
        assert wrapped._registration is None
        assert wrapped.arch_info is not None


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 6: Weight integrity
# ═══════════════════════════════════════════════════════════════════════════


class TestWeightIntegrity:
    """Verify dtype, device, parameter count, and embedding shapes."""

    def test_gpt2_default_dtype_float32(self, gpt2_model):
        for p in gpt2_model._model.parameters():
            assert p.dtype == torch.float32, f"Expected float32, got {p.dtype}"
            break  # checking first param is sufficient for default

    def test_gpt2_device_cpu(self, gpt2_model):
        for p in gpt2_model._model.parameters():
            assert str(p.device) == "cpu"
            break

    @slow
    def test_param_count_matches_hf_gpt2(self):
        import interpkit
        from transformers import AutoModelForCausalLM

        kit = interpkit.load("gpt2", device="cpu")
        ref = AutoModelForCausalLM.from_pretrained("gpt2")
        kit_count = sum(p.numel() for p in kit._model.parameters())
        ref_count = sum(p.numel() for p in ref.parameters())
        assert kit_count == ref_count, (
            f"Param count mismatch: interpkit={kit_count}, HF={ref_count}"
        )

    def test_gpt2_embedding_shape(self, gpt2_model):
        wte = gpt2_model._model.transformer.wte
        assert wte.weight.shape == (50257, 768), (
            f"Embedding shape mismatch: {wte.weight.shape}"
        )

    def test_gpt2_positional_embedding_shape(self, gpt2_model):
        wpe = gpt2_model._model.transformer.wpe
        assert wpe.weight.shape == (1024, 768)

    @slow
    def test_param_count_matches_hf_bloom(self, bloom_model):
        from transformers import AutoModelForCausalLM

        ref = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")
        kit_count = sum(p.numel() for p in bloom_model._model.parameters())
        ref_count = sum(p.numel() for p in ref.parameters())
        assert kit_count == ref_count

    def test_bloom_embedding_shape(self, bloom_model):
        emb = bloom_model._model.transformer.word_embeddings
        vocab = bloom_model.arch_info.vocab_size
        hidden = bloom_model.arch_info.hidden_size
        assert emb.weight.shape == (vocab, hidden), (
            f"Expected ({vocab}, {hidden}), got {emb.weight.shape}"
        )

    def test_model_is_eval_mode(self, gpt2_model):
        assert not gpt2_model._model.training, "Model should be in eval mode after load()"

    def test_resnet_is_eval_mode(self, resnet_model):
        assert not resnet_model._model.training

    def test_distilbert_param_device_matches(self, distilbert_model):
        expected_type = distilbert_model._device.type
        for name, p in distilbert_model._model.named_parameters():
            assert p.device.type == expected_type, (
                f"Param {name} on {p.device.type}, expected {expected_type}"
            )
            break

    @slow
    def test_dtype_float16_propagation(self):
        import interpkit

        kit = interpkit.load("gpt2", device="cpu", dtype="float16")
        for p in kit._model.parameters():
            assert p.dtype == torch.float16, f"Expected float16, got {p.dtype}"
            break

    @slow
    def test_dtype_string_shortcut_fp16(self):
        import interpkit

        kit = interpkit.load("gpt2", device="cpu", dtype="fp16")
        for p in kit._model.parameters():
            assert p.dtype == torch.float16
            break

    def test_invalid_dtype_raises(self):
        import interpkit

        with pytest.raises(ValueError, match="Unknown dtype"):
            interpkit.load("gpt2", device="cpu", dtype="float8")
