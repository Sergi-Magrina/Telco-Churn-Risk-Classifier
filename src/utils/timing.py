from __future__ import annotations

import time
from contextlib import contextmanager
from statistics import mean
from typing import Callable, Iterable


@contextmanager
def timed_block(name: str):
    """
    context manager to time a code block.
    """

    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000
        print(f"[timing] {name}: {elapsed_ms:.2f} ms")


def benchmark_inference(
    func: Callable[[Iterable], object],
    X,
    n_runs: int = 5,
) -> float:


    times_ms = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = func(X)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)
    return mean(times_ms)

