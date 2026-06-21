"""Integration tests for scripts/collect_rbcsmart_dataset.collect_episode().

These exercise the real CityLearn env + EntityContractAdapter + RBCSmartPolicy
to verify that:

1. RBCSmartPolicy receives the inputs it was designed for (raw kW values,
   since the class declares ``_use_raw_observations = True``).
2. With a reasonable episode window covering PV-active hours, the rule-based
   policy DOES emit non-zero ``electrical_storage`` actions on the 15-min
   schema (i.e. the dataset is non-degenerate).

The 15-min schema is chosen because it's the slowest control resolution and
therefore the most sensitive to RBC misconfiguration. If 15-min works, 15-sec
also works.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_15MIN = (
    REPO_ROOT
    / "datasets"
    / "citylearn_three_phase_electrical_service_demo_15min_parquet"
    / "schema.json"
)


@pytest.fixture(scope="module")
def collect_module():
    """Import the script module once."""
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import scripts.collect_rbcsmart_dataset as m
    return m


@pytest.mark.slow
def test_collect_episode_15min_produces_nonzero_storage_actions(collect_module):
    """RBCSmartPolicy must produce at least one non-zero ``electrical_storage``
    action when rolled out over a daytime window on the 15-min schema.

    Regression guard: before this test, collect_episode bypassed the wrapper
    and fed the encoder's minmax-normalised observations to RBCSmartPolicy
    (which expects raw kW values via ``_use_raw_observations = True``). That
    caused the encoder to collapse all kW features to the 0.5 midpoint, so
    the RBC's ``_pv_surplus_kw`` always returned 0 and no charge branch ever
    fired.

    Window: 60 steps starting at step 40 covers ~10:00-01:00 with PV active.
    """
    if not SCHEMA_15MIN.exists():
        pytest.skip(f"15-min schema not present at {SCHEMA_15MIN}")

    result = collect_module.collect_episode(
        seed=22,
        episode_idx=0,
        start_step=40,
        episode_steps=60,
        schema_path=str(SCHEMA_15MIN),
        offline=True,
    )

    rows = result["rows"]
    assert rows, "collect_episode returned no rows"

    storage_values = [
        row["action__electrical_storage"]
        for row in rows
        if "action__electrical_storage" in row
        and row["action__electrical_storage"] is not None
    ]
    assert storage_values, "no agents recorded an action__electrical_storage column"

    nonzero = [v for v in storage_values if abs(float(v)) > 1e-8]
    assert nonzero, (
        f"all {len(storage_values)} electrical_storage actions are zero - "
        "RBC is being fed encoded observations instead of raw, or the storage "
        "policy itself is misconfigured."
    )
