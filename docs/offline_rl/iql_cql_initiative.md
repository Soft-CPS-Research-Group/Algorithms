# IQL + CQL 15-min Initiative — Engineering Note

> **Status:** Implementation note (production in flight; numbers marked
> `<!-- TBD: production -->` will be filled in during Phase 12).
> **Branch:** `feature/offline-agents-implementation`.
> **Parent plan:** [`iql_cql_initiative_plan.md`](iql_cql_initiative_plan.md).
> **Spec:** [`phase11_consolidated_doc_design.md`](phase11_consolidated_doc_design.md).
> **Date drafted:** 2026-06-22.

This document is a self-contained engineering note for the offline-RL
initiative that takes the Building-5-only iteration (recorded in
[`thesis_notes.md`](thesis_notes.md)) to all 17 buildings on the 15-min
CityLearn schema, using IQL and CQL trained from RBCSmart rollouts.
It is the repo deliverable; the thesis treatment lives in Ch4-Ch6.

---

## Table of contents

1. [Motivation & scope](#1-motivation--scope)
2. [Dataset](#2-dataset)
3. [Algorithms (IQL + CQL)](#3-algorithms-iql--cql)
4. [Resume & status visibility](#4-resume--status-visibility)
5. [Engineering — the CityLearn OOM](#5-engineering--the-citylearn-oom)
6. [Training setup](#6-training-setup)
7. [Benchmark results](#7-benchmark-results)
8. [Feature analysis highlights](#8-feature-analysis-highlights)
9. [Reproducing](#9-reproducing)
10. [Limitations](#10-limitations)
11. [References](#11-references)

---

## 1. Motivation & scope

The Building-5-only iteration recorded in [`thesis_notes.md`](thesis_notes.md) worked: BC beat its RBC teacher by ~3% on Building-5 `cost_total`, and IQL pushed further while keeping `unserved_energy` at zero. The catch was scope. With one learned building inside a 17-building district, any local gain was diluted ~17× before it reached district KPIs, so the only honest evaluation target was Building-5 performance. This initiative closes that gap by training across all 17 buildings, which makes district-level KPIs the meaningful comparison surface for the first time.

**Why offline RL.** Training never touches the live environment. The cost, carbon, peak, ramp, and unserved-energy constraints baked into the reward are precisely the ones that make exploratory online learning unsafe on real infrastructure; offline RL sidesteps that risk entirely. The data we need already exists — RBCSmart rollouts, the same controller class used in the Building-5 study — and the deployment story matches the training story: a trained policy hot-swaps into the dispatcher without disturbing the live grid.

**Why IQL and CQL.** Both algorithms address the same failure mode (querying the Q-function on out-of-distribution actions) with structurally different defences. IQL avoids OOD queries by construction: its expectile-regressed V-function is only ever evaluated on dataset actions, and the policy is extracted by advantage-weighted regression on the same support (*Offline Reinforcement Learning with Implicit Q-Learning*, Kostrikov et al. 2021). CQL keeps the standard actor-critic loop but adds a conservative penalty that pushes Q-values down on OOD actions (*Conservative Q-Learning for Offline Reinforcement Learning*, Kumar et al. 2020). Running both on the same data lets us compare these two OOD defences head to head.

**Why 15-min, not 15-sec.** The CityLearn schema ships in two cadences. The 15-sec variant gives 5760 steps/day × 365 = ~2.1M steps per seed-year — finer-grained, but intractable for CPU-only training over 10 seeds. The 15-min variant gives 96 steps/day × 365 = 35040 steps per seed-year, roughly 17× smaller. EV charging dynamics live at the ~10s to ~1h scale, so 15-min still resolves the control loop adequately; the trade-off buys hours-per-seed instead of days-per-seed for collection, and keeps the full multi-seed pipeline within an overnight budget.

**Why this initiative now.** The full multi-building run only became feasible after Phases 1-8 of the parent plan landed: per-stage sentinels, atomic checkpoints, `status.json` progress reporting, and a single resumable orchestrator that survives crashes mid-collection or mid-training. Without that scaffolding, a 10-seed × 17-building run would be one fragile process away from restarting from scratch.

<!-- TBD: production -->
![End-to-end pipeline: collect (RBCSmart, 10 seeds × 35040 steps) → train-iql + train-cql (150k steps, 4 groups × 9 seeds) → benchmark (10 eval seeds) → feature-analysis → curated figures.](iql_cql_figures/01_pipeline_overview.png)

The rest of this note walks the pipeline from data to results.

---

## 2. Dataset

The corpus is a single RBCSmart rollout per seed, ten seeds (22-31), 35040 steps each — one full year of 15-min control. Every per-step row carries all 17 buildings' observations, the actions issued by RBCSmart to each charger and battery, and a per-agent reward computed live (rather than reconstructed after the fact). Across the ten seeds that is roughly 350k transitions, written to `runs/offline_iql_cql_initiative_15min/data/seed_<N>.parquet`. The collection run is the longest single stage in the pipeline; everything downstream reads parquet only.

**Schema and provenance.** The simulation schema is `datasets/citylearn_three_phase_electrical_service_demo_15min_parquet/schema.json`; the column layout (observation names, action keys, per-building groupings, reward column) is documented in [`dataset_schema.md`](dataset_schema.md). Each seed parquet is stamped with a `schema_hash` field that pins the data to the exact schema it was generated from, so a downstream training run that loads a parquet with a mismatched hash fails fast. When the collect stage finishes a seed it appends an entry to `manifest.json` (per-seed row counts, mean and tail KPIs, schema hash, RBCSmart variant) and updates `kpi_summary.csv`; when all ten seeds complete, it writes a `.collect.done` sentinel that the orchestrator uses to skip the stage on resume.

**Entity interface and agent groups.** CityLearn's *entity interface* returns per-building observations as a dict-of-arrays keyed by building, replacing the legacy flat-vector layout where every building's features were concatenated. This matters because the 17 production buildings do not share a common observation shape: they have different EV configurations, different battery setups, and different metering, which collapses into four unique `(obs_dim, action_dim)` cohorts.

| Group key       | Buildings | obs_dim | action_dim | Notes                                  |
|-----------------|----------:|--------:|-----------:|----------------------------------------|
| `obs627_act1`   | 10        | 627     | 1          | 10-building cohort, headline cohort     |
| `obs706_act2`   | 5         | 706     | 2          | 5-building cohort                       |
| `obs749_act3`   | 1         | 749     | 3          | singleton                               |
| `obs785_act3`   | 1         | 785     | 3          | singleton                               |

Each group is trained as its own model. This matches CityLearn's per-building agent contract and avoids the alternative — zero-padding heterogeneous shapes into a single wide tensor — which would let the network learn around padded slots in ways that do not transfer at deployment.

**Reward.** The per-agent reward column stores `CostServiceCommunityFeasiblePrecisionRewardV46` captured **live** during collection, not synthesised from KPIs after the rollout. This preserves the exact reward signal that any subsequent online comparison would see and removes a class of reconstruction bugs. The term-by-term breakdown, the design rationale, and the calibration history (the Building-5 iteration's NNLS fit plus the hybrid-floor rule that handles collinear KPI terms) are recorded in [`reward_design.md`](reward_design.md); both still apply unchanged for the multi-building data, with no per-building re-tuning.

<!-- TBD: production -->
![Dataset stats: per-seed row counts, total transitions, disk size, schema hash.](iql_cql_figures/02_dataset_stats.png)

<!-- TBD: production -->
![Action coverage for the 10-building `obs627_act1` cohort: the RBCSmart policy concentrates on a narrow action regime, motivating CQL's pessimism penalty on OOD actions.](iql_cql_figures/03_action_coverage_group_a.png)

<!-- TBD: production -->
![Reward distribution segmented by RBCSmart action regime (charge / idle / discharge × peak / off-peak).](iql_cql_figures/04_reward_by_regime.png)

<!-- TBD: production -->
![Temporal patterns: mean reward and action by hour-of-day, day-of-week.](iql_cql_figures/06_temporal_patterns.png)

For the full exploratory analysis see [`feature_analysis/feature_analysis.md`](feature_analysis/feature_analysis.md); §8 of this note summarises the three insights from that EDA that matter for IQL and CQL training.

---

## 3. Algorithms (IQL + CQL)

**IQL recap.** Implicit Q-Learning splits the offline value problem across three networks. A value function `V(s)` is trained with an asymmetric expectile loss at `τ=0.7`, which biases `V` toward the higher-return actions actually present in the data — without ever asking the network to score an action it has not seen. A twin Q is trained against `V(s')` as its bootstrap target, so the standard `max_a Q(s', a)` step (the classic source of OOD over-estimation in offline RL) never executes. The policy is then extracted by advantage-weighted regression with temperature `β=3.0` and an advantage clip of 100, upweighting transitions where the dataset action did better than `V`. For the full derivation see [`iql_reference.md`](iql_reference.md).

**CQL recap.** Conservative Q-Learning keeps a standard actor-critic loop and adds a regulariser — `α · ( E_{s∼D, a∼OOD}[Q(s, a)] − E_{(s,a)∼D}[Q(s, a)] )` — that pushes Q down on OOD actions and up on dataset actions. The penalty weight `α=0.2` is the same value the Building-5 iteration settled on; small enough to avoid over-pessimism (which would collapse the policy onto a narrow band of high-conviction actions and undo CQL's purpose) but large enough to noticeably re-shape Q on the OOD side.

| Parameter | IQL | CQL | Source |
|-----------|----:|----:|--------|
| Hidden layers | [256, 256] | [256, 256] | `--hidden-layers 256,256` |
| Dropout | 0.1 | 0.1 | trainer default |
| Activation | ReLU | ReLU | trainer default (hardcoded in `iql_networks.py`) |
| Optimiser | Adam | Adam | trainer default (`torch.optim.Adam`) |
| Learning rate | 3e-4 | 3e-4 | trainer default |
| Batch size | 256 | 256 | trainer default |
| γ (discount) | 0.99 | 0.99 | trainer default |
| Target soft-update τ | 0.005 | 0.005 | trainer default |
| Gradient clip | 1.0 | 1.0 | trainer default |
| Expectile τ (V loss) | 0.7 | — | IQL only |
| β (AWR temp) | 3.0 | — | IQL only |
| Advantage clip | 100 | — | IQL only |
| CQL α | — | 0.2 | `--cql-alpha 0.2` |
| CQL random actions / state | — | 10 | trainer default (`cql_n_random_actions`) |
| Gradient steps | 150,000 | 150,000 | `--gradient-steps 150000` |
| Checkpoint every | 5,000 | 5,000 | `--checkpoint-every 5000` |

The IQL config dataclass:

```python
@dataclasses.dataclass
class IQLTrainingConfig:
    # Architecture
    hidden_layers: List[int] = dataclasses.field(default_factory=lambda: [256, 256])
    dropout: float = 0.1
    log_std_init: float = math.log(0.1)
    # IQL hyperparameters
    tau_expectile: float = 0.7
    beta_advantage: float = 3.0
    advantage_clip: float = 100.0
    gamma: float = 0.99
    tau_target: float = 0.005
    # Optimisation
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    batch_size: int = 256
    gradient_steps: int = 150_000
    eval_every_n_steps: int = 2_500
    checkpoint_every_n_steps: int = 5_000
    val_fraction: float = 0.1
    device: str = "cpu"
# from algorithms/offline_rl/iql_trainer.py:81-103
```

CQL extends the IQL config, inheriting all fields and adding the conservative penalty knobs:

```python
@dataclasses.dataclass
class CQLTrainingConfig(IQLTrainingConfig):
    """Inherits all IQL fields; adds conservative Q-learning params."""
    cql_alpha: float = 0.2
    """Weight of the conservative Q penalty."""
    cql_n_random_actions: int = 10
    """Random actions sampled per state for the logsumexp approximation."""
    cql_min_q_weight: float = 0.0
    """Optional action-gap target; 0.0 disables (standard CQL)."""
# from algorithms/offline_rl/cql_entity_trainer.py:66-82
```

**No online fine-tuning.** This initiative stays purely offline by design: the constraint set forbids live env interaction during training, and any online refinement is a separate downstream stage outside the scope of this note.

<!-- TBD: production -->
![CQL penalty trace over training, all four agent groups: the penalty rises as Q estimates drift OOD and is bounded by `α=0.2`.](iql_cql_figures/09_training_cql_penalty.png)

---

## 4. Resume & status visibility

<!-- task 5 writes this section -->

---

## 5. Engineering — the CityLearn OOM

<!-- task 6 writes this section -->

---

## 6. Training setup

<!-- task 7 writes this section -->

---

## 7. Benchmark results

<!-- task 8 writes this section -->

---

## 8. Feature analysis highlights

<!-- task 9 writes this section -->

---

## 9. Reproducing

<!-- task 10 writes this section -->

---

## 10. Limitations

<!-- task 10 writes this section -->

---

## 11. References

<!-- task 10 writes this section -->
