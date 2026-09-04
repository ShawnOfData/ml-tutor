"""Loose LRU cache with TTL for RAG query results.

Used to avoid re-running the full vector retrieval for identical or
near-identical queries within a short window. This is deliberately a small,
dependency-free structure (built-in ``dict`` + ``collections.OrderedDict``)
rather than a bespoke backend.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any


# Merge runs of whitespace so "gradient  descent" == "gradient descent".
_WS_RE = re.compile(r"\s+")


def normalize_query(text: str) -> str:
    """Normalize a query string for cache-key stability."""
    return _WS_RE.sub(" ", (text or "").strip().lower())


class QueryResultCache:
    """Thread-safe LRU cache with per-entry TTL.

    ``get`` returns ``None`` on miss *and* on expiry, so callers can build a
    straight ``cached or fresh`` path.
    """

    def __init__(self, max_entries: int = 256, default_ttl: float = 300.0) -> None:
        self._data: "dict[str, tuple[float, Any]]" = {}
        self._max_entries = max(1, int(max_entries))
        self._default_ttl = float(default_ttl)
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if now >= expires_at:
                del self._data[key]
                return None
            # refresh recency (move to front)
            self._data.pop(key)
            self._data[key] = (expires_at, value)
            return value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        ttl = self._default_ttl if ttl is None else float(ttl)
        expires_at = time.monotonic() + ttl
        with self._lock:
            if len(self._data) >= self._max_entries:
                # evict oldest (first inserted key)
                try:
                    self._data.pop(next(iter(self._data)))
                except (StopIteration, KeyError):
                    pass
            self._data[key] = (expires_at, value)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


# Process-wide shared cache. Retrieval is stateless so a single instance is
# safe to reuse across all knowledge bases.
_query_cache = QueryResultCache()


def get_query_cache() -> QueryResultCache:
    return _query_cache