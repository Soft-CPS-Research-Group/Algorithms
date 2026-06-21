"""Unit tests for ``scripts.collect_rbcsmart_dataset._compute_safety_limit``.

Regression guard for the production crash observed on 2026-06-21 where a
hard-coded ``SAFETY_MAX_STEPS = 6000`` global constant aborted the 15-min
full-year rollout (episode_steps = 35040) at step 6000 before the env
naturally truncated.

The fix introduces a per-call helper:

    _compute_safety_limit(episode_steps) -> max(SAFETY_MAX_STEPS, episode_steps + buffer)

so that:

* short episodes (e.g. 15-sec daily = 5760) still get the historical 6000
  floor as a defensive backstop, and
* long episodes (e.g. 15-min annual = 35040) get a per-episode limit with a
  small margin above the configured episode length.
"""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.collect_rbcsmart_dataset as collect_module


def test_compute_safety_limit_short_episode_uses_floor():
    """Tiny episode counts must still get the historical 6000-step floor.

    Preserves backward-compatible behavior: callers passing e.g. a smoke
    rollout of 200 steps still get the same defensive backstop as before
    the bug fix.
    """
    assert collect_module._compute_safety_limit(200) == 6000


def test_compute_safety_limit_at_15s_daily_episode_uses_floor():
    """The historic 15-sec-daily episode length (5760) lies under the floor.

    5760 + buffer = 5860, which is below the 6000 floor, so the floor wins.
    """
    assert collect_module._compute_safety_limit(5760) == 6000


def test_compute_safety_limit_15min_annual_episode_exceeds_floor():
    """15-min annual (35040 steps) must yield a per-episode limit above
    episode_steps, not the static 6000 floor.

    Direct regression test for the 2026-06-21 production crash.
    """
    limit = collect_module._compute_safety_limit(35040)
    assert limit > 35040
    # Sanity: should not be ridiculously larger either.
    assert limit < 35040 + 10_000


def test_compute_safety_limit_15sec_annual_episode():
    """15-sec annual (~ 2.1M steps) — pathological but must still scale."""
    limit = collect_module._compute_safety_limit(2_102_400)
    assert limit > 2_102_400


def test_compute_safety_limit_zero_episode_uses_floor():
    """Defensive: zero or negative episode_steps still gets the floor."""
    assert collect_module._compute_safety_limit(0) == 6000


def test_compute_safety_limit_exposes_safety_max_steps_floor():
    """The helper's floor must match the module's SAFETY_MAX_STEPS constant.

    Guards against silent drift between the constant and the helper logic.
    """
    floor = collect_module.SAFETY_MAX_STEPS
    assert collect_module._compute_safety_limit(0) == floor
    assert collect_module._compute_safety_limit(floor - 100) == floor
