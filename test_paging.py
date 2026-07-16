import unittest

from qwen_next_paging import ExpertLRU, reduce_topk


class PagingTests(unittest.TestCase):
    def test_top7_is_renormalized(self):
        selected = reduce_topk(list(range(10)), 7)
        self.assertEqual([i for i, _ in selected], [9, 8, 7, 6, 5, 4, 3])
        self.assertAlmostEqual(sum(w for _, w in selected), 1.0)


    def test_lru_evicts_by_bytes(self):
        cache = ExpertLRU(10)
        loads = 0

        def load(key):
            nonlocal loads
            loads += 1
            return key, 6

        cache.get("a", load)
        cache.get("b", load)
        self.assertEqual(cache.stats.evictions, 1)
        cache.get("a", load)
        self.assertEqual(loads, 3)
