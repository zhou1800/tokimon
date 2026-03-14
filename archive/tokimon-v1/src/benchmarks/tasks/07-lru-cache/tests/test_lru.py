from lru import LRUCache
from scoring import score


def test_eviction_order():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")
    cache.put("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3


def test_score_bounds():
    assert 0.0 <= score() <= 1.0
