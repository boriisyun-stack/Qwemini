"""Small, dependency-free primitives for experimenting with routed-expert paging.

This does not load model weights.  It provides the two pieces we can test on a
16 GB Mac before wiring in Qwen3-Next's full forward pass: top-k reduction with
renormalized gates, and a byte-bounded LRU cache for per-layer experts.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Hashable, Iterable, Sequence, TypeVar

T = TypeVar("T")


def reduce_topk(
    scores: Sequence[float],
    k: int = 10,
) -> list[tuple[int, float]]:
    """Return the top-k expert indices and renormalized softmax gates.

    Qwen3-Next was trained with top-10.  Reducing k at inference time must
    renormalize the surviving probabilities; otherwise every MoE layer is
    unintentionally attenuated.
    """
    if not 1 <= k <= len(scores):
        raise ValueError(f"k must be in [1, {len(scores)}], got {k}")
    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)[:k]
    # Stable softmax on only the selected logits.
    max_score = max(score for _, score in ranked)
    weights = [(index, __import__("math").exp(score - max_score)) for index, score in ranked]
    normalizer = sum(weight for _, weight in weights)
    return [(index, weight / normalizer) for (index, _), (_, weight) in zip(ranked, weights)]


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    loaded_bytes: int = 0


class ExpertLRU:
    """Byte-bounded LRU cache; loader/evictor own actual CPU/GPU transfers."""

    def __init__(self, capacity_bytes: int):
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        self.capacity_bytes = capacity_bytes
        self._items: OrderedDict[Hashable, tuple[T, int]] = OrderedDict()
        self.stats = CacheStats()

    def get(self, key: Hashable, loader: Callable[[Hashable], tuple[T, int]]) -> T:
        if key in self._items:
            value, size = self._items.pop(key)
            self._items[key] = (value, size)
            self.stats.hits += 1
            return value
        value, size = loader(key)
        if size > self.capacity_bytes:
            raise MemoryError(f"expert {key!r} is {size} bytes, larger than cache")
        while self.stats.loaded_bytes + size > self.capacity_bytes and self._items:
            _, (_, old_size) = self._items.popitem(last=False)
            self.stats.loaded_bytes -= old_size
            self.stats.evictions += 1
        self._items[key] = (value, size)
        self.stats.loaded_bytes += size
        self.stats.misses += 1
        return value

    def clear(self) -> None:
        self._items.clear()
        self.stats.loaded_bytes = 0


def page_plan(router_scores: Iterable[Sequence[float]], k: int = 10) -> list[list[int]]:
    """Return selected expert IDs for a token across MoE layers."""
    return [[index for index, _ in reduce_topk(scores, k)] for scores in router_scores]
