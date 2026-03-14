import threading

from counter import Counter
from scoring import score


def test_thread_safety():
    counter = Counter()
    threads = [threading.Thread(target=lambda: [counter.increment() for _ in range(200)]) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert counter.get() == 2000


def test_score_bounds():
    assert 0.0 <= score() <= 1.0
