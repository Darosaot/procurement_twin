"""
Thread-safe LRU cache for simulation results.

Two-level cache key: sha256(sorted JSON of params) → result dict.
Items expire after CACHE_TTL_SECONDS (default 3600).
Maximum CACHE_MAXSIZE entries are kept (LRU eviction, default 500).

Environment overrides
---------------------
  CACHE_MAXSIZE   int   Maximum entries                (default 500)
  CACHE_TTL       int   Entry TTL in seconds           (default 3600)
  CACHE_ENABLED   0|1   Disable cache entirely         (default 1)
"""

import os
import hashlib
import json
import time
import logging
import threading
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)

_MAXSIZE = int(os.environ.get("CACHE_MAXSIZE", "500"))
_TTL     = int(os.environ.get("CACHE_TTL",     "3600"))
_ENABLED = os.environ.get("CACHE_ENABLED", "1").strip() not in ("0", "false", "no")


def _make_key(params: dict) -> str:
    return hashlib.sha256(
        json.dumps(params, sort_keys=True, default=str).encode()
    ).hexdigest()


class SimulationCache:
    """
    Bounded, TTL-aware LRU cache safe for concurrent FastAPI workers.

    get()  → None on miss, cached dict on hit (TTL-valid entries only)
    set()  → store result; evicts LRU entry when maxsize is exceeded
    stats  → property returning hit/miss/size counters
    """

    def __init__(self, maxsize: int = _MAXSIZE, ttl: int = _TTL):
        self._maxsize = maxsize
        self._ttl = ttl
        self._cache: OrderedDict[str, tuple[dict, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, (_, ts) in self._cache.items() if now - ts > self._ttl]
        for k in expired:
            del self._cache[k]

    def get(self, params: dict) -> Optional[dict]:
        if not _ENABLED:
            return None
        key = _make_key(params)
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            result, ts = self._cache[key]
            if time.monotonic() - ts > self._ttl:
                del self._cache[key]
                self._misses += 1
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            return result

    def set(self, params: dict, result: dict) -> None:
        if not _ENABLED:
            return
        key = _make_key(params)
        with self._lock:
            self._cache[key] = (result, time.monotonic())
            self._cache.move_to_end(key)
            # Evict LRU entries until within maxsize
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def invalidate(self, params: dict) -> bool:
        key = _make_key(params)
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        with self._lock:
            size = len(self._cache)
        return {
            "enabled":    _ENABLED,
            "hits":       self._hits,
            "misses":     self._misses,
            "hit_rate":   round(self._hits / total, 4) if total else 0.0,
            "size":       size,
            "maxsize":    self._maxsize,
            "ttl_seconds": self._ttl,
        }


# Module-level singleton used by the API
simulation_cache = SimulationCache()

logger.info(
    "Simulation cache initialised (enabled=%s, maxsize=%d, ttl=%ds).",
    _ENABLED, _MAXSIZE, _TTL,
)
