"""Runtime monkey-patches for CityLearn 1.0.2 — fix the memory leak that causes
the offline RL data collector to OOM at ~step 16000 on 15-min full-year (35040 step)
episodes.

Root cause (empirically confirmed via tracemalloc snapshot diffing; see Bug 7
investigation):
  CityLearnEntityInterfaceService._action_feedback_series_cache is a plain dict
  keyed by `id(values)`. Each access to the `electricity_consumption` property
  on energy-model objects returns a fresh ndarray (because the property does
  `self.__electricity_consumption * self.time_step_ratio` which materialises a
  new array). New array → new id → new cache entry. Entries are never pruned.
  Each entry also has three lists (`sum_prefix`, `last_nonzero`, `value_snapshot`)
  that grow per step.

  Net: memory growth is roughly quadratic in episode_steps (number of entries
  × size per entry), which makes 35040-step episodes infeasible on commodity
  hardware (~16 MB/step measured on 15-min schema with 17 buildings).

Fix:
  Wrap `_action_feedback_series_summary` so that after each call it evicts the
  oldest entries until the cache holds at most `ACTION_FEEDBACK_CACHE_MAX`
  entries. Python dicts preserve insertion order in 3.7+, giving us FIFO
  eviction semantics for free. FIFO is correct here because:
    1. Each call's cache hit/miss is determined by `id(values)`.
    2. Old `values` arrays are not re-used by upstream callers — once they fall
       out of the dict, evicting them just costs a cache miss on a hypothetical
       re-access, which simply rebuilds the entry. Correctness preserved.

Apply via `apply_citylearn_patches()` at the top of any entry script that builds
CityLearn environments. Safe to call multiple times (idempotent).
"""

from __future__ import annotations

from typing import Any, Dict


__all__ = ["apply_citylearn_patches", "ACTION_FEEDBACK_CACHE_MAX"]


# Bound on cache entries. 17 buildings × ~5 source arrays each ≈ 85 typical
# entries during a single env step; 128 leaves headroom for future schemas.
ACTION_FEEDBACK_CACHE_MAX: int = 128

# Module-level guard so `apply_citylearn_patches()` is idempotent.
_PATCHED: bool = False


def _bound_dict_size(d: Dict[Any, Any], maxsize: int) -> None:
    """Evict oldest entries from ``d`` until ``len(d) <= maxsize``.

    Python dicts iterate in insertion order (3.7+), so `next(iter(d))` returns
    the oldest key. We pop in a loop because callers may have inserted multiple
    entries since the last call — we want FIFO eviction down to the bound.

    No-op if ``maxsize`` is greater than or equal to current size.
    """
    while len(d) > maxsize:
        oldest_key = next(iter(d))
        d.pop(oldest_key)


def apply_citylearn_patches() -> None:
    """Apply all runtime monkey-patches to CityLearn 1.0.2.

    Currently patches:
    - ``CityLearnEntityInterfaceService._action_feedback_series_summary`` —
      bounds the per-step cache to prevent unbounded growth (OOM at ~16000
      steps).

    Idempotent: calling multiple times has no additional effect.
    """
    global _PATCHED
    if _PATCHED:
        return

    _patch_action_feedback_series_cache()
    _PATCHED = True


def _patch_action_feedback_series_cache() -> None:
    """Wrap ``_action_feedback_series_summary`` to enforce the cache bound."""
    from citylearn.internal.entity_interface import CityLearnEntityInterfaceService

    original = CityLearnEntityInterfaceService._action_feedback_series_summary

    def patched_summary(self, values, index):
        result = original(self, values, index)
        _bound_dict_size(
            self._action_feedback_series_cache,
            ACTION_FEEDBACK_CACHE_MAX,
        )
        return result

    CityLearnEntityInterfaceService._action_feedback_series_summary = patched_summary
