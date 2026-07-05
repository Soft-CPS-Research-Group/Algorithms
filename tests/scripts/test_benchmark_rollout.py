# tests/scripts/test_benchmark_rollout.py
"""Behavioral tests for scripts.benchmark_entity_agents rollout and main.

These tests cover Bug 9 (Phase 13):
- entity_rollout() must not silently cap step count when max_steps is None.
  Previously, entity_rollout hardcoded max_steps=6000, silently truncating
  the intended full-year (35,040-step) IQL/CQL evaluations to ~62 days
  while RBCSmart (via rbc_rollout, which had no cap) ran the full year.
  Result: horizon-mismatched CityLearn normalized KPIs.
- main() must respect --skip-rbc + --merge-existing, so we can rerun only
  IQL and CQL (which need the fix) and splice in the pre-existing full-year
  RBC block without redoing that expensive rollout.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeEnv:
    """Minimal CityLearn-shaped env stub for driving entity_rollout.

    Terminates after `terminate_at` step() calls so tests can distinguish
    env-driven termination from cap-driven termination.
    """

    def __init__(self, *, terminate_at: int) -> None:
        self._terminate_at = terminate_at
        self._step = 0
        self.buildings: List[Any] = []
        self.action_names = [["a1"]]
        self.flat_action_space: List[Any] = []
        self.seconds_per_time_step = 15

    def reset(self):
        self._step = 0
        return ({"tables": {}, "edges": {}, "meta": {}}, {})

    def step(self, actions):
        self._step += 1
        terminated = self._step >= self._terminate_at
        return ({"tables": {}, "edges": {}, "meta": {}}, {}, terminated, False, {})

    def evaluate(self):
        import pandas as pd
        return pd.DataFrame(
            [
                {"level": "district", "cost_function": "cost_total", "value": 1.0},
                {"level": "district", "cost_function": "carbon_emissions_total", "value": 1.0},
            ]
        )


class _FakeAdapter:
    def to_agent_encoded_observations(self, payload):
        return (
            [np.zeros(3, dtype=np.float32)],
            [["obs1", "obs2", "obs3"]],
            {},
        )

    def to_entity_actions(self, actions, action_names):
        return {}


class _FakeAgent:
    def attach_environment(self, **kwargs):
        return None

    def predict(self, obs_list, deterministic=None):
        return [np.zeros(1, dtype=np.float32) for _ in obs_list]


def _install_fakes(monkeypatch, *, terminate_at: int) -> _FakeEnv:
    """Swap _make_env/_make_adapter for lightweight fakes; return the env
    so the test can assert on internal step counts if needed."""
    import scripts.benchmark_entity_agents as m
    env = _FakeEnv(terminate_at=terminate_at)
    monkeypatch.setattr(m, "_make_env", lambda seed, **kw: env)
    monkeypatch.setattr(m, "_make_adapter", lambda env_: _FakeAdapter())
    return env


# ---------------------------------------------------------------------------
# entity_rollout: default is now None (no silent cap)
# ---------------------------------------------------------------------------


def test_entity_rollout_max_steps_default_is_none():
    """Regression: entity_rollout() must default max_steps to None so it
    respects env-driven termination (matches rbc_rollout behavior). The
    previous default of 6000 silently truncated full-year IQL/CQL rollouts
    to ~62 days (15-min resolution) while RBC ran the full 35,039 steps —
    producing horizon-mismatched CityLearn normalized KPIs (Bug 9).
    """
    import scripts.benchmark_entity_agents as m
    sig = inspect.signature(m.entity_rollout)
    assert "max_steps" in sig.parameters
    assert sig.parameters["max_steps"].default is None, (
        "Bug 9 regression: entity_rollout.max_steps must default to None."
    )


def test_entity_rollout_no_cap_when_max_steps_none(monkeypatch):
    """When max_steps=None, entity_rollout must run until the env signals
    termination — even if that exceeds the old 6000 hardcoded cap."""
    import scripts.benchmark_entity_agents as m
    _install_fakes(monkeypatch, terminate_at=8_000)
    result = m.entity_rollout(
        _FakeAgent(), env_seed=0, label="TEST", max_steps=None
    )
    assert result["steps"] == 8_000, (
        f"Expected 8000 steps (env-driven), got {result['steps']}. "
        "Bug 9 regression: entity_rollout is silently capping again."
    )


def test_entity_rollout_respects_explicit_max_steps(monkeypatch):
    """When callers explicitly pass max_steps=N, entity_rollout must
    honor the cap (used for smoke-test-style bounded rollouts)."""
    import scripts.benchmark_entity_agents as m
    _install_fakes(monkeypatch, terminate_at=10_000)
    result = m.entity_rollout(
        _FakeAgent(), env_seed=0, label="TEST", max_steps=500
    )
    assert result["steps"] == 500, (
        f"Expected explicit cap at 500, got {result['steps']}."
    )


# ---------------------------------------------------------------------------
# main: --skip-rbc + --merge-existing splice
# ---------------------------------------------------------------------------


def _fake_rbc_block(seeds: List[int], *, steps: int) -> Dict[str, Any]:
    kpis = {
        "cost_total": 1.4672,
        "carbon_emissions_total": 1.4920,
        "daily_peak_average": 1.3149,
        "ramping_average": 1.4774,
        "annual_normalized_unserved_energy_total": 0.0,
        "electricity_consumption_total": 1.4067,
        "zero_net_energy": -2.2167,
    }
    return {
        "runs": [
            {"env_seed": s, "label": "RBCSmart", "district": dict(kpis), "steps": steps}
            for s in seeds
        ],
        "aggregate": {},
    }


def test_main_skip_rbc_does_not_call_rbc_rollout(monkeypatch, tmp_path):
    """--skip-rbc must suppress the rbc_rollout call loop entirely so
    reruns for IQL+CQL don't pay the ~7h RBC cost."""
    import scripts.benchmark_entity_agents as m

    calls: List[Dict[str, Any]] = []

    def _no_rbc(**kwargs):
        calls.append(kwargs)
        return {"env_seed": kwargs["env_seed"], "label": "RBCSmart", "district": {}, "steps": 0}

    monkeypatch.setattr(m, "rbc_rollout", _no_rbc)

    out = tmp_path / "out.json"
    rc = m.main(
        [
            "--skip-rbc",
            "--eval-seeds", "200,201",
            "--no-iql", "--no-cql",
            "--output", str(out),
        ]
    )
    assert rc == 0
    assert calls == [], f"rbc_rollout must not be invoked with --skip-rbc; got {len(calls)} calls"


def test_main_merge_existing_splices_rbc_block(monkeypatch, tmp_path):
    """--skip-rbc + --merge-existing PATH copies the RBCSmart block from
    PATH into the fresh output, preserving eval_seeds and per-run steps.
    This is Phase 13's splice mechanism: rerun IQL+CQL against the Bug 9
    fix without redoing the untainted RBC full-year rollouts."""
    import scripts.benchmark_entity_agents as m

    prev = tmp_path / "prev.json"
    prev.write_text(
        json.dumps(
            {
                "eval_seeds": [200, 201],
                "iql_root": None,
                "cql_root": None,
                "RBCSmart": _fake_rbc_block([200, 201], steps=35_039),
            }
        )
    )

    # Belt-and-suspenders: any accidental rbc_rollout call also gets caught
    def _no_rbc(**kwargs):
        raise AssertionError("rbc_rollout must not be called with --skip-rbc")

    monkeypatch.setattr(m, "rbc_rollout", _no_rbc)

    out = tmp_path / "out.json"
    rc = m.main(
        [
            "--skip-rbc",
            "--merge-existing", str(prev),
            "--eval-seeds", "200,201",
            "--no-iql", "--no-cql",
            "--output", str(out),
        ]
    )
    assert rc == 0
    saved = json.loads(out.read_text())
    assert "RBCSmart" in saved, "merged RBCSmart block missing from output"
    runs = saved["RBCSmart"]["runs"]
    assert len(runs) == 2
    assert [r["env_seed"] for r in runs] == [200, 201]
    assert all(r["steps"] == 35_039 for r in runs), (
        "merged RBC runs must retain their original step counts (full-year 35039)"
    )
    assert all(r["district"]["cost_total"] == pytest.approx(1.4672) for r in runs)
