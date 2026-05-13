# Reward Design

> The reward function is the **only** way the agent learns what "good"
> means. v1's reward (`reward_b5_raw = -(grid_cost + λ_carbon · carbon
> + λ_unserved · unserved)`) covered three KPIs and ignored three
> others. this redesign fixes that with a **KPI-aligned weighted sum** whose
> weights are calibrated against RBC rollouts.

---

## 1. Design principle

The agent's actions move the building's **net electricity consumption**
trajectory `e_t`. Every CityLearn KPI is some functional of that
trajectory (and of exogenous signals like price, carbon). So the reward
should expose, at every step, *every* term that the KPIs penalise at
the end of the episode — otherwise the agent has no signal along that
dimension and any improvement on it would be incidental.

**Plain version:** if we want the agent to reduce daily peaks, the
reward must mention peaks. If we want it to reduce ramping, the reward
must mention ramping. v1 didn't; this design does.

---

## 2. The reward, term by term

At step `t`, given the transition `(obs_t, action_t, obs_{t+1})`, we
compute five non-negative *cost* terms (all are "the more, the worse")
and combine them as

> `reward_t = − ( w_cost · C_t + w_carbon · G_t + w_peak · P_t + w_ramp · R_t + w_unserved · U_t )`

| Symbol | Definition | Source signal | Units |
|--------|-----------|--------------|-------|
| `C_t` | `p_t · max(0, e_t)` | grid price `p_t`, net consumption `e_t` | $ |
| `G_t` | `c_t · max(0, e_t)` | carbon intensity `c_t`, net consumption `e_t` | kg CO₂ |
| `P_t` | `max(0, e_t − μ_t)` where `μ_t` is a 24-hour rolling mean | net consumption | kWh |
| `R_t` | `|e_t − e_{t−1}|` | net consumption | kWh |
| `U_t` | unmet EV energy at this step (if EV departs at `t+1` and `soc < required`, the gap; else 0) | EV obs fields | kWh |

Each term has a clear physical interpretation and a 1-to-1 mapping to a
KPI:

| Term | KPI it targets |
|------|----------------|
| `C_t` | `cost_total` |
| `G_t` | `carbon_emissions_total` |
| `P_t` | `daily_peak_average`, `all_time_peak_average` |
| `R_t` | `ramping_average`, `daily_one_minus_load_factor_average` |
| `U_t` | `annual_normalized_unserved_energy_total` |

> `electricity_consumption_total` and `zero_net_energy` are not given
> their own term because they are largely co-moved by `C_t` (which
> already penalises positive `e_t`). We will verify this empirically
> during calibration; if correlation is weak we'll add a term
> `S_t = max(0, e_t)` with its own weight.

---

## 3. Why "weighted sum" and not "use the env reward"

The env reward (`V2GPenaltyReward`, see
`reward_function/V2G_Reward.py`) bundles cost/carbon **with** EV-shaped
shaping bonuses (close-to-required-SoC reward, no-car-charging
penalty, etc.) and a community-level peak/ramping term that mixes all
buildings. Two problems for our setup:

1. **Per-building proxy is fuzzy.** The community term is shared, so
 per-building credit assignment is ambiguous. v1 absorbed this into
 `λ` constants fitted post-hoc.
2. **Hard-coded weights.** The shaping constants
 (`PENALTY_NO_CAR_CHARGING = −5`, `REWARD_CLOSE_SOC = 10`, …) were
 designed for a different research question and are not motivated by
 the CityLearn KPIs we are scored on.

 keeps the env reward in the dataset as `reward_env` for traceability
but trains on `reward`, which is **(i) per-building, (ii) aligned
to the KPIs we care about, and (iii) weight-calibrated against RBC
data**.

---

## 4. Weight calibration

The five cost terms have different units and different magnitudes.
Naively summing them would let one term (probably `C_t`, dollars)
dominate. We pick weights once, against RBC rollouts, with a simple
procedure.

### 4.1 Procedure

1. Run RBC for `K = 10` seeds (the collector already produces
 these). For each rollout `k`:
 - Record per-step terms `(C^k_t, G^k_t, P^k_t, R^k_t, U^k_t)`.
 - Record episode-end KPIs `(K^k_cost, K^k_carbon, K^k_peak,
 K^k_ramp, K^k_unserved)`.
2. Standardise each cost term to unit variance per rollout (so weights
 end up in comparable units):
 - `Ĉ^k_t = C^k_t / σ(C^k)`, etc.
3. Form per-rollout episode aggregates `S^k_x = Σ_t X̂^k_t` for each
 term `x`.
4. Fit non-negative weights `w` such that
 `w · (S^k_cost, S^k_carbon, S^k_peak, S^k_ramp, S^k_unserved)`
 correlates with `K^k_cost + K^k_carbon + K^k_peak + K^k_ramp +
 K^k_unserved` across the `K` rollouts.
5. Round weights to 2 significant figures, freeze into
 `datasets/offline_rl/derived/reward_weights.json`.
6. Sanity-check: the resulting `reward` summed across each rollout
 must rank-correlate (Spearman) with the per-rollout sum of KPI
 improvements, ρ ≥ 0.9.

> If `K = 10` proves too few for a stable fit, we increase to `K = 30`
> (cheap — RBC is fast). The same calibration is *not* re-run when we
> swap behaviour policies; the weights stay frozen, so reward
> comparisons across iterations are apples-to-apples.

### 4.2 Initial guess (used only if calibration fails)

Should the procedure produce degenerate weights (e.g. all variance
absorbed by one term), we fall back to:

| Weight | Value | Rationale |
|--------|-------|-----------|
| `w_cost` | 1.0 | Anchor, matches dollar cost |
| `w_carbon` | 1.0 | Comparable scale once standardised |
| `w_peak` | 2.0 | Peaks are penalised quadratically by KPI averaging |
| `w_ramp` | 1.0 | One-to-one with ramping KPI |
| `w_unserved` | 50.0 | Safety: large to keep agent from stranding EV |

These are starting points only. The real numbers come from the
calibration above and are recorded in `reward_weights.json`.

---

## 5. Implementation contract

### 5.1 Pure function

`algorithms/offline_rl/reward.py` exports

```python
def compute_reward(
 obs: dict, # raw obs dict at step t
 action: np.ndarray, # action taken at step t
 next_obs: dict, # raw obs dict at step t+1
 *,
 weights: dict, # loaded from reward_weights.json
 state: dict, # mutable; carries 24h rolling window, last e_t
) -> tuple[float, dict]:
 """Returns (reward, term_breakdown). Pure given (obs, action, next_obs, weights, state)."""
```

The breakdown (`{"cost": C, "carbon": G, "peak": P, "ramp": R,
"unserved": U}`) is written to the dataset in a sidecar Parquet
(`reward_breakdown.parquet`) for analysis. We don't add the
breakdown columns to the main dataset to keep the schema lean.

### 5.2 Unit tests (`tests/offline_rl/test_reward.py`)

| Test | Expectation |
|------|-------------|
| `test_finite_on_zero_action` | Reward is finite when action=0. |
| `test_sign_cost_negative` | Importing energy at any positive price → reward decreases. |
| `test_sign_carbon_negative` | Importing energy at positive carbon → reward decreases. |
| `test_peak_only_above_mean` | `e_t < μ_t` → peak term = 0. |
| `test_ramp_zero_on_constant_load` | Constant `e` for 24 h → ramp term ≈ 0. |
| `test_unserved_only_at_departure` | EV not at required SoC, but car not leaving → unserved term = 0. |
| `test_monotonic_in_each_term` | Holding others fixed, increasing any term → reward decreases. |
| `test_reward_outperforms_v1_on_peaks` | On a synthetic "peak day",  penalises the peak hour and v1's reward does not. |

### 5.3 Calibration script

`scripts/calibrate_reward.py`:

- Reads `datasets/offline_rl/rbc/seed_*.parquet`.
- Computes per-step terms with `compute_reward(..., weights=ones)`.
- Computes per-rollout KPIs by replaying the env (or by reading
 `kpi_summary.csv` if already cached).
- Fits weights as in §4.1.
- Writes `datasets/offline_rl/derived/reward_weights.json`.
- Prints the rank-correlation sanity check.

---

## 6. Traceability

| Quantity | Where stored |
|----------|-------------|
| Frozen weights | `datasets/offline_rl/derived/reward_weights.json` |
| Per-step `reward` values | `derived/rbc_with_reward.parquet` (column) |
| Per-step term breakdown | `derived/reward_breakdown.parquet` |
| Calibration log | `derived/reward_calibration.log` (procedure, regression diagnostics, final weights, sanity-check ρ) |

When a future iteration swaps the behaviour policy (e.g. RBC → BC),
the weights file is **not** regenerated — same yardstick, different
ruler-bearer.

---

## 7. Failure modes we explicitly guard against

1. **One term dominates.** Standardisation in §4.1 step 2 prevents the
 dollar-cost term from drowning the others.
2. **Reward-KPI sign flip.** §4.1 step 6 asserts ρ ≥ 0.9; if it fails
 we stop and re-think before training.
3. **Reward leaks future info.** `compute_reward` only consumes
 `(obs_t, action_t, next_obs_{t+1}, state)` — no look-ahead.
4. **Reward is unbounded.** All five terms are non-negative and
 bounded by physical quantities (price × max load, etc.); no
 division, no exp.
5. **Reward changes mid-experiment.** Frozen weights file is hashed
 into the dataset's `manifest.json`; loading code asserts the hash
 matches.
