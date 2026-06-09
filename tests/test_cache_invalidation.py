"""P4e: Model._cache invalidate-on-different-input policy.

This file pins the existing invalidation behavior (no LRU; the cache
holds activations for the most recent input only). The behavior was
already implemented at ``model.py:cache()`` / ``_get_cached()`` via
``interpkit.core.cache.hash_input``; this test makes sure a future
refactor does not silently introduce LRU semantics or skip the hash
check.
"""
from __future__ import annotations

import warnings

import pytest

import interpkit

TINY_GPT2 = "hf-internal-testing/tiny-random-GPT2LMHeadModel"


@pytest.fixture(scope="module")
def tiny_gpt2():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return interpkit.load(TINY_GPT2, device="cpu")


def test_cache_invalidates_on_different_input(tiny_gpt2):
    """``_get_cached(text_b, ...)`` after ``cache(text_a)`` returns None."""
    m = tiny_gpt2
    m.cache("hello world")
    assert m.cached is True

    module_names = list(m._cache.keys())
    assert module_names, "cache should be populated after cache() call"

    cached_a = m._get_cached("hello world", module_names)
    assert cached_a is not None, "Same input should hit the cache"

    cached_b = m._get_cached("a different input", module_names)
    assert cached_b is None, "Different input must miss the cache (invalidate-on-different-input policy)"


def test_cache_repopulates_on_new_input(tiny_gpt2):
    """A second ``cache(text_b)`` call replaces the cache with text_b's activations."""
    m = tiny_gpt2
    m.cache("input one")
    first_hash = m._cache_input_hash
    m.cache("input two")
    second_hash = m._cache_input_hash
    assert first_hash != second_hash

    # input_one no longer satisfies the hash check.
    module_names = list(m._cache.keys())
    assert m._get_cached("input one", module_names) is None
    assert m._get_cached("input two", module_names) is not None


def test_clear_cache_resets_hash(tiny_gpt2):
    """``clear_cache()`` empties the dict AND resets the hash sentinel."""
    m = tiny_gpt2
    m.cache("anything")
    assert m.cached is True
    m.clear_cache()
    assert m.cached is False
    assert m._cache_input_hash is None
