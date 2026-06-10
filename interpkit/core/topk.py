"""Streaming top-k tracker — model-free utility for dataset-scale scans.

Used by :mod:`interpkit.ops.maxact` to retain only the k best-scoring
(example, position) records while scanning an arbitrarily large dataset:
memory stays O(k) regardless of how many scores are pushed.
"""

from __future__ import annotations

import heapq
import math
from typing import Any

__all__ = ["TopKTracker"]


class TopKTracker:
    """Keep the *k* highest-scoring payloads seen so far.

    Scores are compared as plain floats; NaN scores are dropped. Ties are
    broken by insertion order (earlier pushes win), keeping results
    deterministic for repeated scans of the same data.
    """

    def __init__(self, k: int) -> None:
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}.")
        self.k = k
        # Min-heap of (score, -insertion_index, payload): the smallest
        # retained score sits at heap[0] for O(1) admission checks. The
        # negated index makes earlier insertions sort *higher* on ties,
        # so a tying newcomer never evicts an incumbent.
        self._heap: list[tuple[float, int, Any]] = []
        self._counter = 0

    def __len__(self) -> int:
        return len(self._heap)

    @property
    def threshold(self) -> float:
        """Smallest retained score (``-inf`` until the tracker is full)."""
        if len(self._heap) < self.k:
            return float("-inf")
        return self._heap[0][0]

    def push(self, score: float, payload: Any) -> bool:
        """Offer one record; returns True if it was retained."""
        if math.isnan(score):
            return False
        entry = (score, -self._counter, payload)
        self._counter += 1
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, entry)
            return True
        if entry > self._heap[0]:
            heapq.heapreplace(self._heap, entry)
            return True
        return False

    def items(self) -> list[tuple[float, Any]]:
        """Retained ``(score, payload)`` pairs, best first."""
        ordered = sorted(self._heap, reverse=True)
        return [(score, payload) for score, _idx, payload in ordered]
