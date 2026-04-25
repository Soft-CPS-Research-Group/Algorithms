# Evaluation of the BC Offline RL Pipeline

## Executive Summary

You built a complete, well-engineered Milestone-1 + Milestone-2 pipeline: an RBC-driven dataset collector (`EVDataCollectionRBC`), an offline BC trainer (`BCPolicy` + `bc_trainer.py`), and an apples-to-apples in-simulator benchmark (`scripts/benchmark_bc_vs_rbc.py`). The BC policy successfully imitates the RBC and even *marginally* outperforms it on most CityLearn district KPIs (e.g., −0.93% all-time peak, −0.20% cost, −0.84% Building 5 consumption) without introducing any constraint violations (`unserved_energy_total = 0.0` for both).

That said, **the result is "BC successfully clones an RBC and slightly smooths it"** — which validates the *infrastructure* but produces a model with **little meaningful policy improvement** and several methodological gaps worth addressing before the next milestone.

---

## What Was Done — Conclusions

### 1. Infrastructure quality (strong)

- Clean separation of concerns: data-collection agent (`algorithms/agents/ev_data_collection_agent.py`), offline trainer (`algorithms/offline/`), and inference agent (`offline_bc_agent.py`) all sit naturally inside the existing `BaseAgent` contract and registry.
- Episode-level train/val split (`data_loader.py:75-91`) — correct choice; row-shuffled splits would have leaked information between adjacent timesteps.
- Train-only normalization stats (`data_loader.py:120-127`) — correct; no validation leakage. Stats are persisted to `normalization_stats.json` and re-applied at inference (`offline_bc_agent.py:58-63`). This is textbook ML hygiene.
- Best-validation checkpoint selection (`bc_trainer.py:181-216`) acts as soft early stopping.
- The benchmark script masks non-target buildings to zero in *both* runs, guaranteeing a fair RBC vs BC comparison.

### 2. Behavioural fidelity (validated)

- BC closely tracks RBC across all headline KPIs (|Δ| ≤ 1.3% on Building 5; ≤ 1% on district).
- No KPI gets meaningfully worse — `annual_normalized_unserved_energy_total = 0.0` for both, and `ev_departure_success_rate` is identical (0.7813).
- Reward sums: RBC = −17,378 vs BC = −17,232 (target building); +0.84% improvement, marginal but consistent.

### 3. Notable behavioural drift — `bess_*` metrics drop ~9.3%

This is the most interesting finding in the benchmark, and worth highlighting:

| KPI | RBC | BC | Δ |
|---|---:|---:|---:|
| `bess_throughput_total_kwh` | 3,300.9 | 2,992.8 | −9.33% |
| `bess_equivalent_full_cycles` | 257.9 | 233.8 | −9.33% |
| `bess_capacity_fade_ratio` | 0.0026 | 0.0023 | −9.34% |

BC uses the home battery ~9% less aggressively than the RBC. This is **not** imitation — it's distributional smoothing introduced by MSE+tanh regression on a discrete two-value action map (RBC outputs only +0.091 / −0.08 for `electrical_storage`). The tanh head softens the bang-bang behaviour into something less extreme. In this case it happens to slightly *improve* most KPIs (less battery degradation, similar consumption), but it's a side effect of architecture, not learned optimization.

### 4. The fundamental ceiling

BC, by construction, **cannot exceed the behaviour policy's competence** on the criteria the RBC was tuned for. The reward signal stored in the dataset is **completely unused** by training (`bc_trainer.py` only uses `(obs, action)` pairs). Any improvement here is incidental, not earned through value learning. The Q&A doc itself (`offline_rl_qa.md` §7) acknowledges this — BC is the floor, not the ceiling.

---

## Improvement Points

Grouped by priority. I'll call out specific code locations.

### A. High-priority methodological gaps

1. **The dataset is too narrow for serious offline RL** (`offline_rl_m1_plan.md` D3).
   - 3 episodes × 8,759 steps = 26,277 transitions of a *purely deterministic* RBC is essentially **the same trajectory three times** — CityLearn's only stochasticity is EV arrival timing. State-action coverage is extremely thin.
   - The plan doc (M1 §"Further Considerations" point 2) already acknowledges this: "noise injection for dataset diversity" was deferred. It's now blocking. Without coverage, IQL/CQL/TD3+BC won't have anywhere meaningful to extrapolate to.
   - **Fix:** Implement the deferred `noise_fraction` hyperparameter (e.g., 70% clean RBC + 30% Gaussian-perturbed actions). Also collect from at least one alternate behaviour policy (random, untrained MADDPG, or a perturbed RBC) to widen the action manifold.

2. **The reward column is never used.**
   - `data_loader.py:65-72` only extracts `obs_*` and `action_*`. The CSV stores `reward`, `next_obs_*`, `done` — *all* the ingredients for value-based offline RL — and BC throws them away. This is appropriate for BC but means the dataset's value is currently underutilized.
   - **Fix:** With M2 done, M3 should be **TD3+BC or IQL** (per `offline_rl_qa.md` §5). The dataset is already in the right shape (the `_split_columns` function would just need to also return reward/next_obs/done arrays). This is the single biggest unlock.

3. **Single-building specialist is a strong limitation.**
   - `OfflineBCAgent.predict()` (lines 141-164) zeros all 16 other buildings. The benchmark masks RBC actions to match — fair, but that means **the comparison only validates Building 5's controller**, not a deployable district controller.
   - The 17-building district KPIs in the benchmark are mostly inherited from "do nothing on 16 buildings" — they are not meaningful as a district-level evaluation.
   - **Fix:** Either (a) train per-building BC models (independent learners — `offline_rl_qa.md` §9.4), or (b) restrict the benchmark and the headline narrative explicitly to Building 5. Currently the report mixes both framings.

### B. Medium-priority technical issues

4. **No regularization at all** (`bc_policy.py`, `bc_trainer.py:129`).
   - No dropout, no weight decay, no batch/layer norm, no LR schedule, no gradient clipping. With 26k samples and a 35→256→256→2 net (~75k parameters), overfitting risk on the deterministic RBC trajectory is real — the model can memorize hour→action mappings rather than generalize.
   - **Fix:** Add `weight_decay=1e-5` to Adam, plus dropout (`p=0.1`) between hidden layers. Worth comparing val loss curves before/after.

5. **`done` collapses `terminated` and `truncated`.**
   - `ev_data_collection_agent.py:259` writes `int(terminated or truncated)`. For BC this doesn't matter, but for downstream value-based methods (IQL/CQL), the bootstrapping of $V(s_{t+1})$ should differ between true terminals and time-limit truncations. Recovering this from a CSV will require re-collecting data.
   - **Fix:** Store `terminated` and `truncated` as separate columns now. Cheap forward-compatibility for M3.

6. **Validation episode is the *last* episode** (`data_loader.py:75-91`).
   - Only 3 episodes exist, so episode 2 is held out. Because CityLearn cycles the same year, episodes 0/1/2 differ only in EV-arrival randomness — there's no temporal/seasonal split. The "validation" loss is therefore not really measuring generalization to unseen distributions.
   - **Fix:** With more episodes (point 1), use proper k-fold over episode index, or split by stochastic seed. Report variance across folds.

7. **No multiple-seed evaluation.**
   - The benchmark runs **a single rollout** (`benchmark_bc_vs_rbc.py` line 117 — `env.evaluate()` after one episode). The Q&A doc itself recommends "at least 3-5 seeds, report mean ± std" (`offline_rl_qa.md` §7).
   - All current "BC > RBC by 0.2-0.9%" deltas could easily be within evaluation-seed noise. With 8,759 steps the within-episode averaging helps, but inter-episode variance in EV arrivals is not characterized.
   - **Fix:** Run 5 episodes per agent, report mean ± std on each KPI. Without this, the "BC slightly improves" conclusion is statistically unsupported.

8. **No comparison to obvious baselines.**
   - The Q&A doc (§7) lists the recommended baselines: behaviour policy ✅, BC ✅, **random policy ✗, no-action baseline ✗**.
   - The CityLearn KPIs are *already* normalized to no-control = 1.0, so the "no-action" baseline is implicit — but a random policy is missing. Without it, you can't claim BC "learned" anything beyond producing valid actions.
   - **Fix:** Add a 30-line random-action rollout to the benchmark.

### C. Lower-priority polish

9. **The `_split_columns` regex is fragile** (`data_loader.py:65-66`).
   - `c.startswith("obs_") and not c.startswith("obs_next")` works because the CSV uses `next_obs_` (not `obs_next_`), so the second clause is dead code. The intent was probably `not c.startswith("next_obs_")`. Currently it works by accident. Rewrite for clarity.

10. **BC is structurally deterministic — `deterministic` flag is dead.**
    - `OfflineBCAgent.predict()` accepts `deterministic` and ignores it. For BC this is fine, but if you later add stochastic policies (Gaussian-head BC, BC-Q for CQL), the agent contract should be honoured. Document or implement.

11. **No feature analysis / interpretability** (`offline_rl_qa.md` §4 explicitly recommends this for the thesis).
    - 37 features were fed to the model with no analysis of which actually drive its decisions. SHAP or permutation importance on the trained BC would be a cheap, high-value thesis contribution and would tell you whether the model is using the EV-charger features (the actual decision-relevant ones) or just `hour` (which would mean it's a 24-bucket lookup table).

12. **Evaluation reward mismatch with training objective.**
    - The reward function used during data collection is the simplest possible (`RewardFunction` = −net consumption). The KPIs you actually care about (`cost_total`, `carbon_emissions_total`, `peak_average`, `ev_departure_success_rate`) are *not* what the RBC was optimizing for, and not what BC is imitating signal-wise. The doc (`offline_rl_m1_plan.md` D2) already flags `V2GPenaltyReward` as a future upgrade.
    - For BC alone this doesn't matter (BC ignores reward), but for the next algorithm (IQL/TD3+BC), reward shaping will dominate results. Worth re-collecting the dataset with a richer reward before M3.

---

## Recommended Next Steps (Prioritized)

1. **Re-collect the dataset** with: (a) noise injection for action diversity, (b) more episodes (10-30), (c) separate `terminated`/`truncated`, (d) `V2GPenaltyReward` (or a multi-component reward that matches the KPIs you report).
2. **Add multi-seed evaluation + random baseline** to the benchmark — this is a 1-hour fix that strengthens every claim.
3. **Implement TD3+BC or IQL** as Milestone 3 — this is where offline RL actually starts earning its name. Your data pipeline already supports it.
4. **Add light regularization** to BC (`weight_decay`, `dropout`) and report the impact on val loss + KPIs to demonstrate methodological rigor.
5. **Add a feature importance section** — strong thesis material for low cost.
6. **Decide on framing:** single-building specialist vs district controller. Right now the report mixes both; pick one and be consistent.

The infrastructure you've built is the right foundation. The current BC result validates the plumbing but is not yet a research contribution — it's a baseline. The improvements above are how you turn it into one.
