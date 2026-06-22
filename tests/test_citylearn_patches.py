"""Tests for utils/citylearn_patches.py — runtime patches for CityLearn 1.0.2 OOM bug.

Bug 7 root cause: CityLearn's CityLearnEntityInterfaceService caches per-step
`_action_feedback_series_*` data keyed by `id(values)`. Each `electricity_consumption`
property access returns a fresh ndarray → new id → new cache entry. Entries never
prune → unbounded memory growth → OOM at ~step 16000 on 15-min full-year episodes.

Fix: bound the cache size via runtime monkey-patch. Validate here.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Unit tests: _bound_dict_size helper
# ---------------------------------------------------------------------------


class TestBoundDictSize:
    """Eviction helper used by the cache patch. Dict iteration order is insertion order
    in Python 3.7+, so evicting from the front gives FIFO semantics."""

    def test_under_limit_keeps_all_entries(self):
        from utils.citylearn_patches import _bound_dict_size

        d = {1: "a", 2: "b"}
        _bound_dict_size(d, maxsize=5)
        assert d == {1: "a", 2: "b"}

    def test_at_limit_keeps_all_entries(self):
        from utils.citylearn_patches import _bound_dict_size

        d = {1: "a", 2: "b", 3: "c"}
        _bound_dict_size(d, maxsize=3)
        assert d == {1: "a", 2: "b", 3: "c"}

    def test_above_limit_evicts_oldest_until_bounded(self):
        from utils.citylearn_patches import _bound_dict_size

        d = {1: "a", 2: "b", 3: "c", 4: "d", 5: "e"}
        _bound_dict_size(d, maxsize=2)
        # Should keep the 2 most-recently-inserted (4 and 5)
        assert len(d) == 2
        assert 4 in d
        assert 5 in d
        assert 1 not in d
        assert 2 not in d
        assert 3 not in d

    def test_zero_maxsize_evicts_all(self):
        from utils.citylearn_patches import _bound_dict_size

        d = {1: "a", 2: "b"}
        _bound_dict_size(d, maxsize=0)
        assert d == {}

    def test_empty_dict_no_op(self):
        from utils.citylearn_patches import _bound_dict_size

        d = {}
        _bound_dict_size(d, maxsize=10)
        assert d == {}


# ---------------------------------------------------------------------------
# Patch installation tests
# ---------------------------------------------------------------------------


@pytest.fixture
def _restore_citylearn_patches():
    """Save and restore the original `_action_feedback_series_summary` and the patch flag.

    Without this, tests pollute each other via the module-level `_PATCHED` flag
    and the class-level method replacement.
    """
    from citylearn.internal.entity_interface import CityLearnEntityInterfaceService

    import utils.citylearn_patches as patches

    # Snapshot before
    original_method = CityLearnEntityInterfaceService._action_feedback_series_summary
    original_patched_flag = getattr(patches, "_PATCHED", False)

    yield

    # Restore after
    CityLearnEntityInterfaceService._action_feedback_series_summary = original_method
    patches._PATCHED = original_patched_flag


class TestApplyCityLearnPatches:
    """Verify `apply_citylearn_patches()` correctly installs the runtime monkey-patch."""

    def test_apply_patches_replaces_summary_method(self, _restore_citylearn_patches):
        from citylearn.internal.entity_interface import CityLearnEntityInterfaceService

        # Make sure we start unpatched
        import utils.citylearn_patches as patches

        patches._PATCHED = False

        before = CityLearnEntityInterfaceService._action_feedback_series_summary

        patches.apply_citylearn_patches()

        after = CityLearnEntityInterfaceService._action_feedback_series_summary
        assert after is not before, "Expected `apply_citylearn_patches` to replace the method."

    def test_apply_patches_is_idempotent(self, _restore_citylearn_patches):
        """Calling apply twice must not double-wrap (would create call-stack overhead)."""
        from citylearn.internal.entity_interface import CityLearnEntityInterfaceService

        import utils.citylearn_patches as patches

        patches._PATCHED = False

        patches.apply_citylearn_patches()
        first = CityLearnEntityInterfaceService._action_feedback_series_summary

        patches.apply_citylearn_patches()
        second = CityLearnEntityInterfaceService._action_feedback_series_summary

        assert first is second, "Second `apply_citylearn_patches` call double-wrapped the method."

    def test_patched_method_bounds_cache(self, _restore_citylearn_patches):
        """End-to-end behavior: after patch, cache stays bounded across many unique calls.

        We build a tiny fake `self` that mimics the parts of CityLearnEntityInterfaceService
        the patched wrapper touches, AND we stub the original method to just populate the
        cache with a fresh entry per call. The wrapper should then evict oldest entries
        once the cache exceeds the configured max.
        """
        from citylearn.internal.entity_interface import CityLearnEntityInterfaceService

        import utils.citylearn_patches as patches

        # Reset and re-apply
        patches._PATCHED = False
        original = CityLearnEntityInterfaceService._action_feedback_series_summary

        # Replace original with a controlled stub BEFORE patching
        def stub_original(self, values, index):
            cache = self._action_feedback_series_cache
            cache_key = id(values)
            cache[cache_key] = {"values": values, "upto": index}
            return cache[cache_key], index

        CityLearnEntityInterfaceService._action_feedback_series_summary = stub_original
        try:
            patches.apply_citylearn_patches()  # wraps our stub

            # Fake instance with just the cache attribute
            class _Fake:
                pass

            fake = _Fake()
            fake._action_feedback_series_cache = {}

            # Call enough times to exceed the cache bound
            max_cache = patches.ACTION_FEEDBACK_CACHE_MAX
            n_calls = max_cache + 50

            patched_summary = CityLearnEntityInterfaceService._action_feedback_series_summary
            for i in range(n_calls):
                # Unique array per call (fresh id each time)
                values = [float(i)]
                patched_summary(fake, values, i)

            assert len(fake._action_feedback_series_cache) <= max_cache, (
                f"Cache size {len(fake._action_feedback_series_cache)} exceeds "
                f"max {max_cache} after {n_calls} unique calls."
            )
        finally:
            CityLearnEntityInterfaceService._action_feedback_series_summary = original
