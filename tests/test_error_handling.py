"""Error handling and edge case tests — zero prior coverage.

Covers gap 4 from the reliability audit: empty inputs, invalid layer/module
names, invalid head/position values, mismatched token lengths, empty datasets,
invalid batch operations, and out-of-range positions.

All tests use the GPT-2 fixture (fastest, already session-scoped).
"""

from __future__ import annotations

import pytest

TEXT = "The capital of France is"
PAIR_CLEAN = "The Eiffel Tower is in Paris"
PAIR_CORRUPT = "The Eiffel Tower is in Rome"


def _first_layer(model):
    layers = model.arch_info.layer_names
    if not layers:
        pytest.skip("No layers detected")
    return layers[0]


# ═══════════════════════════════════════════════════════════════════════════
#  Empty string inputs
# ═══════════════════════════════════════════════════════════════════════════


def test_empty_string_activations(gpt2_model):
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        gpt2_model.activations("", at=_first_layer(gpt2_model))


def test_empty_string_attention(gpt2_model):
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        gpt2_model.attention("")


def test_empty_string_trace(gpt2_model):
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        gpt2_model.trace("", "")


def test_empty_string_attribute(gpt2_model):
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        gpt2_model.attribute("")


def test_empty_string_ablate(gpt2_model):
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        gpt2_model.ablate("", at=_first_layer(gpt2_model))


def test_empty_string_decompose(gpt2_model):
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        gpt2_model.decompose("")


# ═══════════════════════════════════════════════════════════════════════════
#  Invalid module / layer names
# ═══════════════════════════════════════════════════════════════════════════


def test_invalid_layer_name_activations(gpt2_model):
    with pytest.raises((KeyError, ValueError, AttributeError, RuntimeError)):
        gpt2_model.activations(TEXT, at="nonexistent.module.xyz")


def test_invalid_layer_name_ablate(gpt2_model):
    with pytest.raises((KeyError, ValueError, AttributeError, RuntimeError)):
        gpt2_model.ablate(TEXT, at="fake_layer_that_does_not_exist")


def test_invalid_layer_name_patch(gpt2_model):
    with pytest.raises((KeyError, ValueError, AttributeError, RuntimeError)):
        gpt2_model.patch(PAIR_CLEAN, PAIR_CORRUPT, at="fake_layer_xyz")


def test_invalid_layer_name_head_activations(gpt2_model):
    with pytest.raises((KeyError, ValueError, AttributeError, RuntimeError)):
        gpt2_model.head_activations(TEXT, at="nonexistent.attn.module")


# ═══════════════════════════════════════════════════════════════════════════
#  Invalid head / positions values
# ═══════════════════════════════════════════════════════════════════════════


def test_invalid_head_index_attention(gpt2_model):
    results = gpt2_model.attention(TEXT, head=9999)
    assert results is None or results == [] or isinstance(results, list)


def test_invalid_head_index_patch(gpt2_model):
    try:
        result = gpt2_model.patch(
            PAIR_CLEAN, PAIR_CORRUPT,
            at=_first_layer(gpt2_model),
            head=9999,
        )
        assert "effect" in result
    except (IndexError, ValueError, RuntimeError):
        pass


def test_invalid_positions_patch(gpt2_model):
    try:
        result = gpt2_model.patch(
            PAIR_CLEAN, PAIR_CORRUPT,
            at=_first_layer(gpt2_model),
            positions=[9999],
        )
        assert "effect" in result
    except (IndexError, ValueError, RuntimeError):
        pass


def test_negative_out_of_range_position_decompose(gpt2_model):
    with pytest.raises((IndexError, ValueError, RuntimeError)):
        gpt2_model.decompose(TEXT, position=-100)


# ═══════════════════════════════════════════════════════════════════════════
#  Mismatched clean/corrupted token lengths
# ═══════════════════════════════════════════════════════════════════════════


def test_mismatched_lengths_patch(gpt2_model):
    """Patch with very different length inputs should raise or handle gracefully."""
    try:
        result = gpt2_model.patch(
            "short", "a very long sentence with many many tokens in it",
            at=_first_layer(gpt2_model),
        )
        assert "effect" in result
    except (ValueError, RuntimeError):
        pass


def test_mismatched_lengths_trace(gpt2_model):
    try:
        result = gpt2_model.trace(
            "short", "a very long sentence with many many tokens in it",
            top_k=3,
        )
        assert isinstance(result, (list, dict))
    except (ValueError, RuntimeError):
        pass


def test_mismatched_lengths_find_circuit(gpt2_model):
    try:
        result = gpt2_model.find_circuit(
            "short", "a very long sentence with many many tokens in it",
            threshold=0.1,
        )
        assert "circuit" in result
    except (ValueError, RuntimeError):
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  Batch edge cases
# ═══════════════════════════════════════════════════════════════════════════


def test_empty_dataset_batch(gpt2_model):
    result = gpt2_model.batch("attention", [])
    assert result["count"] == 0
    assert result["results"] == []


def test_invalid_operation_batch(gpt2_model):
    dataset = [{"input_data": "Hello world"}]
    with pytest.raises((ValueError, AttributeError)):
        gpt2_model.batch("nonexistent_op_xyz", dataset)


def test_empty_texts_dla_batch(gpt2_model):
    result = gpt2_model.dla_batch([])
    assert result["count"] == 0


def test_empty_dataset_trace_batch(gpt2_model):
    result = gpt2_model.trace_batch([])
    assert result["count"] == 0


# ═══════════════════════════════════════════════════════════════════════════
#  Out-of-range positions
# ═══════════════════════════════════════════════════════════════════════════


def test_out_of_range_position_lens(gpt2_model):
    try:
        results = gpt2_model.lens(TEXT, position=9999)
        assert results is None or isinstance(results, list)
    except (IndexError, ValueError, RuntimeError):
        pass


def test_out_of_range_position_dla(gpt2_model):
    with pytest.raises((IndexError, ValueError, RuntimeError)):
        gpt2_model.dla(TEXT, position=9999)
