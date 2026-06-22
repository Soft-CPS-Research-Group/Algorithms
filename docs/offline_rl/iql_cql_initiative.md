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

<!-- task 3 writes this section -->

---

## 3. Algorithms (IQL + CQL)

<!-- task 4 writes this section -->

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
