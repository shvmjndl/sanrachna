"""Thread-pool helpers for parallel per-page OCR work."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def map_parallel(
    items: list[T],
    fn: Callable[[T], R],
    *,
    max_workers: int = 4,
) -> list[R]:
    """Run fn on each item in parallel; results preserve input order."""
    if not items:
        return []
    workers = max(1, min(max_workers, len(items)))
    if workers == 1:
        return [fn(item) for item in items]

    indexed = list(enumerate(items))
    results: list[R | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): idx for idx, item in indexed}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
    return results  # type: ignore[return-value]
