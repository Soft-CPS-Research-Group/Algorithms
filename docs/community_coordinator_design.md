# Community Coordinator — Design Document

> Living document. Input/output schemas are defined here before implementation.
> The results section is a template — fill it in as experiments complete.

---

## Architecture Overview

The system uses a two-level hierarchy. The **Community Coordinator (CC)** runs at a coarse timescale (~15 min intervals) and observes the community as a whole. It outputs coordination signals that are consumed by **Local Controllers** — one per building — which run at every environment step and control the actual assets (battery, EV charger, etc.).

The CC does not touch assets directly. Its job is to decide *what each building should aim for*, not *how each asset should act*.

```
                    ┌──────────────────────────────────┐
                    │      Community Coordinator        │
                    │   (single agent, ~15 min step)    │
                    │                                   │
                    │  community state ──► GRU + PPO   │
                    │                         │         │
                    │              target signals out   │
                    └─────────────┬────────────┬────────┘
                                  │            │
               ┌──────────────────┘            └──────────────────┐
               ▼                                                   ▼
   ┌───────────────────────┐                         ┌───────────────────────┐
   │   Local Controller 1  │         . . .           │   Local Controller N  │
   │  (building 1, 1-min)  │                         │  (building N, 1-min)  │
   │  battery / EV / load  │                         │  battery / EV / load  │
   └───────────────────────┘                         └───────────────────────┘
```

**Key design decisions** (open — to be resolved through experiments):
1. What information does the CC observe? → Input Schema (Section 1)
2. What signal does the CC output? → Output Schema (Section 2)

---

## Section 1 — Input Schema: Community State

What aggregated information flows from the buildings up to the CC.

### Path A — Aggregate Only

The CC sees only community-wide totals and global context. No per-building breakdown.

**Feature list:**

| Group | Feature | Source | Notes |
|---|---|---|---|
| Time | `hour`, `day_type`, `month` | CityLearn obs | Periodic encoding |
| Pricing | `current_tariff`, `tariff_next_k` | `pricing.csv` | Import cost signal + short lookahead |
| Weather | `outdoor_dry_bulb_temperature` | `weather.csv` | Current temperature |
| Solar | `diffuse_solar_irradiance`, `_predicted_6h`, `_predicted_12h` | `weather.csv` | PV generation proxy + forecast |
| Energy (agg) | `total_net_electricity_consumption` | Sum over buildings | Total community import/export |
| Energy (agg) | `total_solar_generation` | Sum over buildings | Total on-site PV |
| Energy (agg) | `total_non_shiftable_load` | Sum over buildings | Baseline inflexible demand |
| EV (agg) | `total_evs_connected` | Sum over chargers | Available charging capacity |
| EV (agg) | `total_ev_flexibility_kw` | Sum of SoC headroom × connected | Aggregate shiftable EV load |
| History | `net_import_history_k` | Rolling window (last K steps) | Community import/export trend |
| Feedback | `agg_target_tracking_error` | Computed externally | How well buildings followed last CC signal |

**Approximate input dimension:** ~20–30 scalars (depending on K and forecast horizon)

| Pros | Cons |
|---|---|
| Scalable — input size independent of N buildings | No per-building differentiation |
| Privacy-preserving — no individual data exposed | CC blind to outlier buildings |
| Generalizable across RECs with different N | Fairness must be handled at local level |
| Simpler to train and interpret | |

**Open questions:**
- What value of K (history length) is most informative?
- Should tracking error be included from step 1 or added later?

---

### Path B — Aggregate + Compact Building Summaries

Path A features, plus a compact fixed-size summary per building.

**Additional features per building (on top of Path A):**

| Feature | Source | Notes |
|---|---|---|
| `net_electricity_consumption_i` | Building i obs | Individual import/export |
| `ev_flexibility_available_i` | SoC headroom × `connected_state` | Shiftable capacity for building i |
| `tracking_error_i` | Computed externally | How well building i followed last target |

**Input dimension:** Path A dim + 3 × N (e.g., 3 × 17 = 51 extra features for this dataset)

| Pros | Cons |
|---|---|
| Enables fairness-aware output decisions | Input size grows with N — less generalizable |
| CC can identify stressed or over-consuming buildings | More complex input representation |
| Richer signal for coordination | Risk of overfitting to specific community topology |

**Open questions:**
- Should per-building features be normalized relative to community average?
- Does adding per-building info actually improve coordination, or just add noise?

---

### Path C — Building-Aware Set Encoder

Full per-building observation vectors passed through a permutation-invariant encoder (e.g., mean/max pooling, or attention over building embeddings). Most expressive option.

**Input:** N × `obs_dim` matrix → encoded to fixed-size context vector fed to CC policy

| Pros | Cons |
|---|---|
| Maximum information available | Most architecturally complex |
| Permutation-invariant — handles variable N naturally | Harder to train and interpret |
| Can capture inter-building relationships | Encoder adds parameters and training cost |

**Open questions:**
- Simple pooling (mean/max) vs. learned attention — does the difference matter here?
- Is the added complexity justified vs. Path B?

---

## Section 2 — Output Schema: Community Target

What signal the CC sends down to the local controllers.

### O1 — Single Global Community Target

The CC outputs one scalar (or small fixed vector) representing a community-level goal.

**Example signals:**
- `community_net_import_target_kw` — target net import for the community over the next interval
- `community_flexibility_budget_kw` — how much flexible load the community should shift

Local controllers scale it proportionally based on their local state.

| Pros | Cons |
|---|---|
| Output size is fixed, regardless of N | No per-building differentiation |
| Simplest to train and interpret | Local controllers must infer fair share |
| Most scalable and REC-agnostic | Fairness not guaranteed |

---

### O2 — Per-Building Import/Export Targets

The CC outputs N targets — one per building.

**Example:** `[target_1, target_2, ..., target_N]` where `target_i` = desired net import for building i (kW)

| Pros | Cons |
|---|---|
| Fine-grained per-building coordination | Output size scales with N — topology-specific |
| Enables explicit fairness distribution | CC must be retrained if buildings join/leave |
| Direct and interpretable signal | |

---

### O3 — Abstract Coordination Signal Vector (Global, Fixed-Size)

The CC outputs a fixed-size interpretable vector of coordination signals, regardless of N.

**Example:** `[flexibility_budget, grid_stress, urgency, fairness_weight]`

Local controllers interpret the signal given their own state.

| Pros | Cons |
|---|---|
| Fixed-size output — fully scalable | Signal semantics must be carefully designed |
| Expressive but community-agnostic | Harder to debug and verify |
| Buildings can specialize their interpretation | May require careful reward shaping |

---

### O4 — Receding Horizon Envelope *(modifier on O1 / O2 / O3)*

Any output type above extended to a T-step sequence. The CC reasons over future steps internally and outputs a short target trajectory (e.g., T=8 steps → 2 hours at 15-min resolution).

| Pros | Cons |
|---|---|
| Gives local controllers predictability over short horizon | Output space grows by factor T |
| Smoother, more stable coordination signals | More complex training objective |
| Closer to real-world MPC-style operation | Harder to evaluate and debug |

**Note:** O4 is an orthogonal axis — it can be combined with O1, O2, or O3.

---

## Section 3 — Experiment Matrix

Each experiment is identified by: **Input Path × Output Type × Horizon × Memory**

| Exp ID | Input | Output | Horizon | Memory | Priority | Status |
|---|---|---|---|---|---|---|
| A-O1-S-MLP | A | O1 | Single step | MLP only | 1 | Planned |
| A-O1-S-GRU | A | O1 | Single step | GRU + MLP | 1 | Planned |
| A-O1-R-GRU | A | O1 | Receding (T=8) | GRU + MLP | 2 | Planned |
| A-O3-S-MLP | A | O3 | Single step | MLP only | 2 | Planned |
| A-O3-S-GRU | A | O3 | Single step | GRU + MLP | 2 | Planned |
| B-O1-S-GRU | B | O1 | Single step | GRU + MLP | 3 | Planned |
| B-O2-S-GRU | B | O2 | Single step | GRU + MLP | 3 | Planned |
| B-O3-S-GRU | B | O3 | Single step | GRU + MLP | 3 | Planned |
| C-O1-S-GRU | C | O1 | Single step | GRU + MLP | 4 | Planned |
| C-O3-S-GRU | C | O3 | Single step | GRU + MLP | 4 | Planned |

**Ablation axes covered:**
- With vs. without recurrence: `*-MLP` vs. `*-GRU`
- With vs. without per-building info: Path A vs. B vs. C
- With vs. without receding horizon: Single vs. Receding
- Simple vs. abstract output: O1 vs. O3

**Recommended first experiment:** `A-O1-S-MLP` — simplest possible baseline (aggregate state, global target, single step, no memory). All subsequent experiments are measured against this.

---

## Section 4 — Results Template

> Fill in as experiments complete. All metrics are community-level unless noted.

| Exp ID | Peak Shaving (%) | Self-Consumption (%) | Total Cost (↓) | Fairness (Jain) | Stable Targets | Scalability Notes | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| A-O1-S-MLP | — | — | — | — | — | — | Planned | Baseline |
| A-O1-S-GRU | — | — | — | — | — | — | Planned | |
| A-O1-R-GRU | — | — | — | — | — | — | Planned | |
| A-O3-S-MLP | — | — | — | — | — | — | Planned | |
| A-O3-S-GRU | — | — | — | — | — | — | Planned | |
| B-O1-S-GRU | — | — | — | — | — | — | Planned | |
| B-O2-S-GRU | — | — | — | — | — | — | Planned | |
| B-O3-S-GRU | — | — | — | — | — | — | Planned | |
| C-O1-S-GRU | — | — | — | — | — | — | Planned | |
| C-O3-S-GRU | — | — | — | — | — | — | Planned | |

**Metric definitions:**
- **Peak Shaving (%):** reduction in community peak import vs. baseline (no CC)
- **Self-Consumption (%):** fraction of local PV generation consumed locally
- **Total Cost:** sum of electricity bills across all buildings over the episode
- **Fairness (Jain):** Jain's fairness index over per-building cost reduction
- **Stable Targets:** qualitative — do CC output signals vary smoothly over time?

---

## Open Decisions (to resolve with supervisor)

1. **Output granularity:** Should O1/O3 be in kW (power) or kWh (energy over the 15-min interval)?
2. **Tracking error signal:** Should the CC receive feedback on how well buildings followed the last target, and if so from step 1 or added in a later ablation?
3. **Local controller type:** Is the local controller a fixed heuristic (for CC evaluation isolation) or a co-trained RL agent? This affects reward design significantly.
4. **Reward for CC:** What community-level reward signal does the CC optimize? (cost reduction, peak shaving, self-consumption, or composite?)
5. **Receding horizon T:** Is T=8 (2h at 15min) the right window, or should T be tunable?
