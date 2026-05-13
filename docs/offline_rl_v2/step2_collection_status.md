# v2 Step 2 — RBC Dataset Collection (Status)

> Step 2 of the v2 sequencing in `OFFLINE_RL_AGENTS.md`. Done.
> Next: Step 3 (calibrate `reward_v2`) → Step 4 (BC-v2) → Step 5 (IQL-v2).

---

## What was done

- Built the v2 collector: `scripts/collect_rbc_dataset_v2.py`.
- Built the schema module: `algorithms/offline_rl_v2/schema_v2.py` (single
  source of truth for column names, dtypes, validation, hash).
- Built the v2 RBC wrapper: `algorithms/offline_rl_v2/rbc_v2.py`
  (`RuleBasedPolicyV2`).
- Collected 10 RBC rollouts: `datasets/offline_rl_v2/rbc/seed_{22..31}.parquet`.
- Wrote manifest, KPI summary, sample CSV.

## Numbers

| Quantity | Value |
|----------|-------|
| Seeds | 10 (`22, 23, …, 31`) |
| Steps per rollout | 8 759 (one rollout shorter than 8 760 by env convention) |
| Total transitions | **87 590** |
| Schema fields | 80 |
| Disk footprint | **11.0 MB** Parquet (~50 MB equivalent CSV) |
| Schema hash | `7c57d2e40b66f0ea…` |

### Action statistics (across seeds)

| Action column | mean | std | min | max | mode breakdown |
|---|---:|---:|---:|---:|---|
| `action_electrical_storage` | 0.000 | **0.000** | 0.000 | 0.000 | constant 0 (RBC v2 does not control this) |
| `action_electric_vehicle_storage_charger_5_1` | ~0.114 | **~0.246** | 0.000 | 1.000 | 81% at 0.0, 18% at 0.6 (PV bonus), 1% at 1.0 (emergency) |

The EV column has real, structured variance — three modes that map
1-to-1 with the RBC card's logic. **No silent zero-action bug.**

### Reward statistics

`reward_env` (per-step, per-building from `V2GPenaltyReward`):

| | mean of means | mean of stds |
|---|---:|---:|
| Across 10 seeds | **−39.37** | **95.76** |

`reward_v2` is `NaN` in the collected dataset by design — it's filled in
during step 3 by the reward calibration script.

### KPI summary (district level, 10 seeds, mean ± std)

| KPI | RBC v2 | Direction |
|---|---:|---|
| `cost_total` | 2.228 ± 0.057 | ↓ better |
| `carbon_emissions_total` | 2.217 ± 0.055 | ↓ better |
| `electricity_consumption_total` | 2.226 ± 0.060 | ↓ better |
| `daily_peak_average` | 2.143 ± 0.022 | ↓ better |
| `all_time_peak_average` | 1.825 ± 0.023 | ↓ better |
| `ramping_average` | 1.579 ± 0.050 | ↓ better |
| `daily_one_minus_load_factor_average` | 0.852 ± 0.008 | ↓ better |
| `annual_normalized_unserved_energy_total` | 0.000 ± 0.000 | ↓ better |
| `zero_net_energy` | −5.558 ± 0.289 | (signed) |

Reading: numbers > 1.0 mean **worse than the no-control baseline** (where
the agent does nothing). RBC v2 is roughly **2× worse than no-op on
cost/peak** because it charges aggressively and creates grid-side stress
without offsetting via storage arbitrage. This is consistent with the
RBC card's predicted blind spots and gives offline RL a clear improvement
target. Critically: `unserved = 0` — the RBC reliably meets EV
deadlines, so any v2 agent must keep this near 0 too.

### Per-seed cross-validation

| seed | cost_total | daily_peak_average | ramping_average |
|---:|---:|---:|---:|
| 22 | 2.149 | 2.115 | 1.503 |
| 23 | 2.214 | 2.133 | 1.566 |
| 24 | 2.305 | 2.176 | 1.647 |
| 25 | 2.288 | 2.165 | 1.631 |
| 26 | 2.170 | 2.123 | 1.529 |
| 27 | 2.194 | 2.126 | 1.554 |
| 28 | 2.259 | 2.157 | 1.608 |
| 29 | 2.287 | 2.166 | 1.629 |
| 30 | 2.242 | 2.151 | 1.593 |
| 31 | 2.167 | 2.122 | 1.527 |

Spread is healthy: enough seed-to-seed variation (~3% rsd on cost) that
weight calibration in step 3 will see real cross-rollout signal, but
small enough that the agent isn't training against pure noise.

---

## Bug discovered & fixed during this step

### The v1 RBC was emitting constant zero on every step

**Symptom:** First smoke run of the collector tripped the v2 fail-fast
check: `std=0` on **both** action columns of Building 5.

**Root cause:** `algorithms/agents/rbc_agent.py::RuleBasedPolicy._compute_ev_action`
reads EV observations using bare names:

```python
self._get_value(obs, obs_map, "electric_vehicle_charger_state", default=0.0)
self._get_value(obs, obs_map, "electric_vehicle_soc", default=0.0)
self._get_value(obs, obs_map, "electric_vehicle_required_soc_departure", ...)
self._get_value(obs, obs_map, "electric_vehicle_departure_time", ...)
self._get_value(obs, obs_map, "electric_vehicle_is_flexible", default=1.0)
```

The CityLearn env exposes these fields **namespaced by charger ID**:

| RBC requested | Env exposes |
|---|---|
| `electric_vehicle_charger_state` | `electric_vehicle_charger_charger_5_1_connected_state` |
| `electric_vehicle_soc` | `connected_electric_vehicle_at_charger_charger_5_1_soc` |
| `electric_vehicle_required_soc_departure` | `connected_electric_vehicle_at_charger_charger_5_1_required_soc_departure` |
| `electric_vehicle_departure_time` | `connected_electric_vehicle_at_charger_charger_5_1_departure_time` |
| `electric_vehicle_is_flexible` | (not exposed) |

Every lookup falls back to its default. `charger_state` defaults to
`0.0`. `_compute_ev_action` short-circuits at `if charger_state <= 0.0:
return 0.0`. **The v1 RBC has never actually charged any EV in any
benchmark.**

**Implications for v1 results:**

- The "RBC" rows in `docs/offline_rl/m4/iql_vs_rbc_vs_bc_vs_random_benchmark_m4.md`
  and the M3 BC benchmark were a do-nothing baseline.
- BC and IQL "tying with RBC" simply means all three did nothing, and
  the env scored similarly for all three. That is a much weaker
  conclusion than the M3/M4 reports claimed.
- The M4 reward proxy was calibrated against rollouts from this dead
  RBC, so its λ constants were also unreliable.

**Fix (v2-only, v1 untouched):**
`algorithms/offline_rl_v2/rbc_v2.py::RuleBasedPolicyV2` overrides
`_compute_ev_action` to remap the v1 bare names to the namespaced obs
keys via the agent's known charger IDs, then defers to the parent
implementation. Behaviour and hyperparameters are otherwise identical.

**Evidence the wrapper works:**

- EV action is now non-trivial: 18% PV-bonus rate (0.6), 1% emergency
  rate (1.0).
- `reward_env` is no longer near-zero; mean ≈ −39.4 / step.
- Building-level `unserved_energy = 0` confirms the RBC actually
  delivers the required EV charge before departure (it never did in v1).
- District peak/ramping/cost are *worse* than no-op — also consistent
  with an active RBC that charges hard without grid-side awareness.

This bug was never about RL hyperparameters, network capacity, or
algorithm choice. It was a silent observation-key mismatch. The v2
fail-fast guard caught it on the first rollout.

---

## What's *still* a known limitation

| Item | Why | Mitigation |
|------|-----|-----------|
| `action_electrical_storage` is constant 0 | v1 RBC hard-codes the non-EV branch to 0; v2 wrapper inherits this. | Documented in `RBC_V2_EXPECTED_CONSTANT_ACTIONS`. BC will trivially imitate this dim; IQL has no in-distribution alternative for it from this dataset alone. The next behaviour-policy iteration (step 7) is where this dim gets real coverage. |
| All KPIs > 1.0 (worse than no-op on cost/peak) | The RBC charges aggressively to ensure unserved=0. | Exactly the gap offline RL should exploit. **BC-v2 is expected to score ≈ RBC; the case for v2 is whether IQL-v2 can do better.** |
| Single-year rollouts (no episode subsampling/windowing) | Approved option in plan: multi-seed only, N=10. | If IQL data-hungry behaviour shows up, step 3 of the *next* iteration can add windowing. |

---

## What enables RBC vs BC vs IQL comparison from here

1. **Coverage.** EV-action distribution has 3 modes with non-trivial
   mass on each → BC has something to imitate, IQL has higher- and
   lower-return subsets to re-weight.
2. **Reward signal.** `reward_env` has std ≈ 96 → strong per-step
   variability. After step 3 we'll have `reward_v2` aligned with
   the KPIs we'll evaluate on, so IQL training and KPI evaluation
   measure the same thing.
3. **Stable evaluation harness.** `scripts/_benchmark_common.py`
   already has the env builder, action-clip, and KPI extractor used by
   v1 — we'll reuse them for the v2 benchmarks (no new env code).
4. **Per-seed variability.** ~3% RSD on cost across seeds means the
   weight regression in step 3 has real signal, and the eventual BC vs
   IQL benchmark can compute meaningful mean ± std.

---

## Files added in step 2

- `algorithms/offline_rl_v2/__init__.py`
- `algorithms/offline_rl_v2/schema_v2.py`
- `algorithms/offline_rl_v2/rbc_v2.py`
- `scripts/collect_rbc_dataset_v2.py`
- `datasets/offline_rl_v2/rbc/seed_{22..31}.parquet` (10 files, 11 MB total)
- `datasets/offline_rl_v2/rbc/manifest.json`
- `datasets/offline_rl_v2/rbc/kpi_summary.csv`
- `datasets/offline_rl_v2/rbc/sample_first_1000.csv`

## Files updated in step 2

- `OFFLINE_RL_AGENTS.md` — v1 retrospective gained item 5 (the RBC
  bug); "Why v2" gained item 2 (`RuleBasedPolicyV2`); behaviour-policy
  class quick-fact updated.

## Files **not** touched in step 2

- All v1 code, datasets, and docs under `algorithms/`,
  `algorithms/agents/`, `algorithms/offline/`, `datasets/offline_rl/`,
  `docs/offline_rl/`, `runs/offline_*`. v1 remains frozen.
