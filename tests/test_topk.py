"""Pure unit tests for core.topk.TopKTracker (no model)."""

from __future__ import annotations

import pytest

from interpkit.core.topk import TopKTracker


def test_keeps_top_k_sorted_desc():
    t = TopKTracker(3)
    for score in [1.0, 5.0, 3.0, 2.0, 4.0]:
        t.push(score, f"p{score}")
    items = t.items()
    assert [s for s, _ in items] == [5.0, 4.0, 3.0]
    assert [p for _, p in items] == ["p5.0", "p4.0", "p3.0"]


def test_k_larger_than_n():
    t = TopKTracker(10)
    t.push(2.0, "a")
    t.push(1.0, "b")
    assert len(t) == 2
    assert [s for s, _ in t.items()] == [2.0, 1.0]


def test_ties_keep_earlier_insertion():
    t = TopKTracker(2)
    assert t.push(1.0, "first")
    assert t.push(1.0, "second")
    # A tying third must NOT evict an incumbent.
    assert not t.push(1.0, "third")
    payloads = [p for _, p in t.items()]
    assert payloads == ["first", "second"]


def test_threshold_tracks_min_retained():
    t = TopKTracker(2)
    assert t.threshold == float("-inf")
    t.push(3.0, "a")
    assert t.threshold == float("-inf")  # not full yet
    t.push(5.0, "b")
    assert t.threshold == 3.0
    t.push(4.0, "c")
    assert t.threshold == 4.0


def test_nan_dropped_and_payload_integrity():
    t = TopKTracker(2)
    assert not t.push(float("nan"), "bad")
    payload = {"pos": 3, "ids": [1, 2, 3]}
    t.push(1.0, payload)
    assert t.items()[0][1] is payload


def test_invalid_k_rejected():
    with pytest.raises(ValueError, match="k must be > 0"):
        TopKTracker(0)
