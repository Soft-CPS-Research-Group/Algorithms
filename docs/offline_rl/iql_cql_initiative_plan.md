# IQL+CQL 15-min Initiative — Implementation Plan

> **Status:** Approved, in execution.
> **Owner:** offline-RL initiative branch (`feature/offline-agents-implementation`).
> **Date approved:** 2026-06-20.

---

## Goal

End-to-end, **resumable**, single-command offline-RL pipeline
(`collect → train-iql → train-cql → benchmark → feature-analysis`) on the
**15-min `electrical_service_demo_15min_parquet` schema** covering all 17
buildings, with thesis-grade dataset analysis and one consolidated markdown
deliverable.

---

## Constraints (locked)

- **Schema**: `datasets/citylearn_three_phase_electrical_service_demo_15min_parquet/schema.json`
- **Buildings**: all 17 (4 agent groups: `obs627_act1`, `obs706_act2`,
  `obs749_act3`, `obs785_act3`)
- **Reward**: `CostServiceCommunityFeasiblePrecisionRewardV46`, captured live
- **Seeds**: collect 22–31, train 22–30, val 31, eval 200–209
- **Training**: 150 000 gradient steps, hidden `[256, 256]`,
  `--cql-alpha 0.2`, checkpoint every 5 000 steps
- **Output root**: `runs/offline_iql_cql_initiative_15min/`
- **venv**: `/Users/guilherme.desousa/MEIA/Thesis/Project/repos/Algorithms/.venv/bin/python`

---

## Phases

### Phase 0 — Smoke validation (no commit)

Confirm the 15-min schema works end-to-end before kicking off ~11 h training
per (group, seed).

1. Run `collect_rbcsmart_dataset.py` with `--seeds 22 --episodes 1
   --schema <15min>` to `/tmp/`; verify parquet shape matches expected
   (1 day × 96 steps × ~2150 cols).
2. Run `train_iql_entity.py` for `--gradient-steps 1000` on that data;
   verify no crash.
3. Verify `episode_steps_for_schema()` derives `86400 // 900 = 96` from
   `seconds_per_time_step: 900` (not hardcoded `5760`).

**Gate**: both smoke runs succeed → proceed.

---

### Phase 1 — Pipeline orchestrator defaults

**File**: `scripts/run_entity_pipeline.py`

| Constant | From | To |
|----------|------|------|
| `DEFAULT_SCHEMA` | 15-sec parquet | **15-min parquet** |
| `DEFAULT_GRADIENT_STEPS` | `50_000` | **`150_000`** |
| `ALL_STEPS` | `collect,train-iql,train-cql,benchmark` | **`collect,train-iql,train-cql,benchmark,feature-analysis`** |
| `--output` default | — | **`runs/offline_iql_cql_initiative_15min/`** |
| New: `--cql-alpha` | — | `0.2` |
| New: `--hidden-layers` | — | `256,256` |
| New: `--checkpoint-every` | — | `5000` |
| New: `--force STAGE[,STAGE…]` | — | bypass `.done` sentinels |

**Tests** (update `tests/scripts/test_run_entity_pipeline.py`): defaults,
new flags forwarded, sentinel-based skipping, `--force` override.

---

### Phase 2 — Trainer resume machinery

**Files**: `algorithms/offline_rl/iql_entity_trainer.py`, then mirrored in
`cql_entity_trainer.py`.

**Optional new helper module**:
`algorithms/offline_rl/checkpoint_utils.py`
- `atomic_save(obj, path)` — write to `path.tmp`, `os.replace(path.tmp, path)`.
- `write_status(path, payload)` — atomic JSON write for `status.json`.

**Per-(group, seed) checkpoint format**: `checkpoint_latest.pt`
```python
{
    "step": int,                       # last completed gradient step
    "policy_state": ...,
    "qf1_state": ..., "qf2_state": ...,
    "qf1_target_state": ..., "qf2_target_state": ...,
    "vf_state": ...,
    "policy_opt_state": ...,
    "qf_opt_state": ...,
    "vf_opt_state": ...,
    "rng_state": ...,                  # torch + numpy + python RNG
    "best_val_mse": float,
    "best_step": int,
    "wall_clock_seconds": float,
}
```

**Trainer changes**
1. Add `--checkpoint-every N` arg (default `5000`).
2. At trainer start: if `checkpoint_latest.pt` exists → load, resume from `step+1`.
3. In gradient loop: every `N` steps → `atomic_save(checkpoint_latest.pt)`.
4. On val improvement: `atomic_save(best_policy.pt)`.
5. On seed completion: write `seed.done` sentinel.
6. Skip seed if `seed.done` exists (unless `--force`).
7. Status writer: after each checkpoint, update
   `runs/<output>/status.json` with `{group, seed, step, val_mse, eta_seconds}`.

**Tests** (new):
- `tests/offline_rl/test_atomic_save.py` — kill mid-write, no corruption.
- `tests/offline_rl/test_iql_entity_resume.py` — train 200 steps, kill,
  resume, identical final weights vs uninterrupted 400-step run
  (deterministic seed).

---

### Phase 3 — Collector idempotency

**File**: `scripts/collect_rbcsmart_dataset.py`
- Add `--skip-existing` (default ON).
- For each seed: if `<dataset_dir>/seed_<N>.parquet` exists AND non-empty
  → skip with INFO log.
- After all seeds complete: write `.collect.done` sentinel.
- Tests: `tests/scripts/test_collect_skip_existing.py`.

---

### Phase 4 — Thesis-grade dataset analyzer

**New file**: `scripts/analyze_entity_dataset.py`

Sections in generated `summary.md` (~30–40 figures total):

1. **Dataset stats**: rows per seed, total transitions, disk size, schema link.
2. **Per-group observation distributions**: histograms for top-20 features per
   group (KDE + box).
3. **Action coverage**: 2-D scatter + marginal hists per agent group;
   **CQL motivation overlay** showing concentration → conservatism rationale.
4. **Reward distribution**: by per-RBC-action regime
   (charge/idle/discharge × peak/off-peak).
5. **Feature × reward correlations**: heatmap per group; top-10 features
   ranked by `|Spearman ρ|`.
6. **Per-building summary table**: episode count, mean reward, action entropy,
   obs PCA explained-var.
7. **Temporal patterns**: reward/action by hour-of-day, day-of-week.

**Output structure**

```
runs/<output>/feature_analysis/
├── summary.md
├── figures/
│   ├── 01_dataset_stats_table.png
│   ├── 02_obs_distributions_<group>.png  × 4
│   ├── 03_action_coverage_<group>.png    × 4
│   ├── 04_reward_by_regime.png
│   ├── 05_correlations_<group>.png       × 4
│   ├── 06_per_building_table.png
│   └── 07_temporal_patterns.png
└── .feature_analysis.done
```

**Tests**: `tests/scripts/test_analyze_entity_dataset.py` (synthetic
50-row parquet, verify all figures + summary.md generated).

---

### Phase 5 — Orchestrator stage wiring

**File**: `scripts/run_entity_pipeline.py`
- Add `feature-analysis` to `ALL_STEPS` (after `benchmark`).
- Per stage: check `{stage}.done` sentinel → skip with INFO log
  (unless `--force`).
- After each stage: update `runs/<output>/status.json` with
  `{stage, status, timestamp, duration_seconds}`.

---

### Phase 6 — Status viewer CLI

**New file**: `scripts/show_pipeline_status.py`
- Read `runs/<output>/status.json`.
- If absent → scan disk for sentinels + checkpoints → derive status.
- Output: pretty unicode table (no deps beyond stdlib + `rich` if present).

Example
```
┌─────────────────┬──────────┬─────────────────┬──────────┐
│ Stage           │ Status   │ Progress        │ ETA      │
├─────────────────┼──────────┼─────────────────┼──────────┤
│ collect         │ ✓ done   │ 10/10 seeds     │ —        │
│ train-iql       │ ▶ running│ group 2/4, 67k  │ 4h 12m   │
│ train-cql       │ pending  │                 │          │
└─────────────────┴──────────┴─────────────────┴──────────┘
```

- Tests: `tests/scripts/test_show_pipeline_status.py`.

---

### Phase 7 — End-to-end resume integration test

**New file**: `tests/scripts/test_run_entity_pipeline_resume.py`
- Tiny config: 1 seed, 2 episodes, 500 grad steps, checkpoint-every 100.
- Run orchestrator → kill mid-train-iql → re-launch → verify completion +
  identical final weights to uninterrupted baseline.

---

### Phase 8 — Cleanup

- `git rm run_full_pipeline.sh` (legacy bash wrapper, no longer used).

---

### Phase 9 — Production launch

```bash
nohup .venv/bin/python -m scripts.run_entity_pipeline \
    --output runs/offline_iql_cql_initiative_15min/ \
    > runs/offline_iql_cql_initiative_15min/pipeline.log 2>&1 &
```

Monitor:

```bash
watch -n 30 .venv/bin/python -m scripts.show_pipeline_status \
    runs/offline_iql_cql_initiative_15min/
```

On crash: re-launch same command; resume kicks in automatically.

**Estimated wall-clock**: collect ~3 h + (train-iql 11 h × 4 groups × 9 seeds)
+ (train-cql similar) + bench 2 h + analysis 30 min ≈ **~3–4 days CPU**.

---

### Phase 10 — Figure curation

Cherry-pick **10–12 figures** from `feature_analysis/figures/` + `benchmark/`
to `docs/offline_rl/iql_cql_figures/`, renaming with semantic prefixes
(e.g., `01_cost_reduction.png`).

---

### Phase 11 — Consolidated doc

**New file**: `docs/offline_rl/iql_cql_initiative.md`

Sections:
1. **Motivation & scope** — why offline RL, why 17 buildings, why IQL + CQL.
2. **Dataset** — 15-min schema, RBCSmart collection, 10 seeds × 96-step
   episodes, link to `feature_analysis/summary.md`.
3. **Algorithms** — IQL + CQL recap, hyperparams table, hidden layers,
   `α=0.2` rationale.
4. **Resume & status visibility** ← new — 3-level idempotency, atomic save,
   `status.json` schema, viewer CLI.
5. **Training setup** — 150k steps, 4 groups × 9 train seeds, val seed 31.
6. **Benchmark results** — table + 10 eval seeds (mean ± std), cost / carbon
   / peak / ramping vs RBCSmart baseline.
7. **Feature analysis highlights** — top-3 insights from §4 (e.g., action
   concentration → CQL motivation).
8. **Reproducing** — single command, expected outputs, status viewer usage.
9. **Limitations** — single schema, no online fine-tuning, etc.
10. **References** — link to thesis Ch4/5/6, status doc, registry entries.

Update `docs/offline_rl/README.md` to link the new doc.

---

### Phase 12 — Final verification & commit

- Run full test suite: `.venv/bin/python -m pytest -q` → all green.
- `git status` clean (only intended new/modified files staged).
- Commit message follows repo style:
  `Add resumable 15-min IQL+CQL pipeline with feature analysis`.

---

## Deliverables checklist

- [ ] `runs/offline_iql_cql_initiative_15min/` populated
      (data, models, benchmark, feature_analysis)
- [ ] `runs/offline_iql_cql_initiative_15min/status.json` + `pipeline.log`
- [ ] `docs/offline_rl/iql_cql_initiative.md` (single self-contained doc)
- [ ] `docs/offline_rl/iql_cql_figures/` (10–12 curated)
- [ ] `scripts/analyze_entity_dataset.py` + tests
- [ ] `scripts/show_pipeline_status.py` + tests
- [ ] Resume machinery in IQL+CQL trainers + tests
- [ ] `run_full_pipeline.sh` deleted
- [ ] All 588+ existing tests + new tests passing

---

## Risk register

| Risk | Mitigation |
|------|------------|
| 150k-step training too slow → wall-clock blows out | Phase 0 smoke validates per-step time; can lower to 100k if needed |
| Atomic save races on slow disk | `os.replace` is POSIX-atomic; test covers kill mid-write |
| Status JSON corruption | Same atomic-write pattern |
| Schema mismatch breaks `analyze_entity_dataset.py` | Reuse `entity_adapter` for schema-agnostic obs extraction |
| 15-min episode shape breaks downstream consumers | Phase 0 smoke covers this |
