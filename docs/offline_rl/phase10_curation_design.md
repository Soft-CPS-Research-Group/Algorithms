# Phase 10 — Initiative Figure Curation: Design Spec

> **Status:** Approved (design); awaits implementation.
> **Parent plan:** `docs/offline_rl/iql_cql_initiative_plan.md` (Phase 10).
> **Date approved:** 2026-06-21.
> **Owner:** offline-RL initiative branch (`feature/offline-agents-implementation`).

---

## Goal

After the production pipeline at `runs/offline_iql_cql_initiative_15min/`
completes, distill its outputs into a curated set of **11 thesis-grade
figures** at `docs/offline_rl/iql_cql_figures/` for the consolidated
initiative doc (Phase 11).

This is an offline, deterministic transformation — given the same inputs
it produces the same outputs, no model inference needed.

---

## Scope

In scope:

- Cherry-pick 5 figures from `<run_dir>/feature_analysis/figures/`
  (dataset-stats, action-coverage, reward-by-regime, correlations,
  temporal-patterns) and copy with semantic prefixes.
- Render 3 training-curve figures from per-seed `metrics.jsonl` files
  (loss, val-MSE, CQL-penalty) using `generate_training_figures.py`
  helpers.
- Render 2 benchmark figures **new** from `<run_dir>/benchmark/results.json`
  (KPI bar chart, IQL-vs-CQL scatter).
- Render 1 pipeline architecture diagram using
  `generate_architecture_figures.py` helpers.
- Write `.curation.done` sentinel on success.

Out of scope:

- Modifying the benchmark stage to emit figures (would require touching
  the running production pipeline).
- Adding a new pipeline stage to the orchestrator (Phase 10 is a
  one-shot post-processing step, not a pipeline stage).
- Per-building or per-group exhaustive figures (only the showcase group
  `obs627_act1` is rendered for per-group figures; aggregate figures
  cover all groups).
- Updating the orchestrator's `ALL_STEPS` or `status.json` schema.

---

## Inputs

```
runs/<run_dir>/
├── feature_analysis/figures/                     # 16 PNGs produced by analyze_entity_dataset.py
│   ├── 01_dataset_stats_table.png                ← copy
│   ├── 03_action_coverage_<showcase_group>.png   ← copy
│   ├── 04_reward_by_regime.png                   ← copy
│   ├── 05_correlations_<showcase_group>.png      ← copy
│   └── 07_temporal_patterns.png                  ← copy
├── benchmark/results.json                        ← KPI tables for bars + scatter
├── models-iql/<group>/seed_*/metrics.jsonl       ← training curves
└── models-cql/<group>/seed_*/metrics.jsonl       ← training curves + CQL penalty
```

Showcase group default: `obs627_act1` (10-building Group A — largest,
drives headline metrics). Overridable via `--showcase-group`.

---

## Outputs

```
docs/offline_rl/iql_cql_figures/
├── 01_pipeline_overview.png             # from generate_architecture_figures
├── 02_dataset_stats.png                 # copy
├── 03_action_coverage_group_a.png       # copy (showcase group)
├── 04_reward_by_regime.png              # copy
├── 05_correlations_group_a.png          # copy (showcase group)
├── 06_temporal_patterns.png             # copy
├── 07_training_loss_group_a.png         # rendered
├── 08_training_valmse_all.png           # rendered (all groups)
├── 09_training_cql_penalty.png          # rendered (all groups, CQL only)
├── 10_benchmark_kpi_bars.png            # NEW: bars (cost, carbon, peak, ramping)
├── 11_iql_vs_cql_scatter.png            # NEW: per-eval-seed cost scatter
└── .curation.done                       # sentinel
```

---

## Component breakdown

Single file: `scripts/curate_initiative_figures.py` (~400-450 lines).

Test file: `tests/scripts/test_curate_initiative_figures.py`.

Functions:

| Function | Responsibility |
|----------|---------------|
| `_copy_feature_analysis_figures(run_dir, showcase_group, output_dir)` | Copy 5 figures from `feature_analysis/figures/` with semantic renames. |
| `_render_pipeline_diagram(output_dir)` | Import `generate_architecture_figures._fig_pipeline()`; save as `01_pipeline_overview.png`. |
| `_render_training_curves(run_dir, showcase_group, groups, output_dir)` | Import helpers from `generate_training_figures`; produce figs 07-09. |
| `_render_benchmark_kpi_bars(results_json, output_dir)` | Matplotlib 2×2 grid: cost / carbon / peak / ramping. 3 bars per panel (RBCSmart, IQL, CQL). Error bars from `aggregate.std`. |
| `_render_iql_vs_cql_scatter(results_json, output_dir)` | Per-eval-seed scatter: IQL cost (x) vs CQL cost (y). y=x reference line. Annotate paired-Wilcoxon p-value. |
| `_write_sentinel(output_dir)` | Atomic write of `.curation.done` with `{generated_at, run_dir, n_figures}`. |
| `main(argv)` | CLI: `--run-dir`, `--output-dir` (default `docs/offline_rl/iql_cql_figures/`), `--showcase-group` (default `obs627_act1`), `--groups` (default 4 production groups). |

---

## Data flow

```
runs/<run_dir>/
├── feature_analysis/figures/ ──[shutil.copy + rename]──┐
├── benchmark/results.json ───[matplotlib]──────────────┤
└── models-{iql,cql}/<group>/seed_*/metrics.jsonl ──┐   │
                                                    └──[matplotlib]──> docs/offline_rl/iql_cql_figures/
                                                                       └── .curation.done
+ generate_architecture_figures._fig_pipeline() ─────[matplotlib]─────^
```

---

## Error handling

- **Missing benchmark/results.json**: log WARNING, skip figs 10-11, continue
  with the other 9 figures.
- **Missing feature_analysis/figures/**: log WARNING, skip figs 02-06,
  continue with the other 6.
- **Missing metrics.jsonl** for a group: that group is dropped from the
  training-curve figure; other groups still rendered. If ALL groups
  missing, skip figs 07-09.
- **Architecture script import fails**: log WARNING, skip fig 01.
- **Sentinel write fails**: error out with a clear message (the curation
  is meaningless without the sentinel for downstream Phase 11/12 to detect
  success).

Each skip emits exactly one WARNING-level log line via `logging.warning`.
The final summary prints which figures were produced vs skipped, and the
exit code is 0 iff at least one figure was produced AND `.curation.done`
was written. The sentinel's ``n_figures`` field records the actual count
of figures produced (range: 1-11), not the target count.

---

## Testing strategy

Fixture: `runs/smoke_pipeline_phase9/` (validated end-to-end smoke output,
all input artifacts present).

Tests in `tests/scripts/test_curate_initiative_figures.py`:

1. **End-to-end on smoke artifacts** — run curator with
   `--showcase-group obs163_act1` (smoke uses hourly schema's group keys).
   Assert all 11 PNGs are created, each non-zero size (> 5 KB),
   `.curation.done` sentinel present with `n_figures == 11`.
2. **Missing benchmark/results.json** — temporarily rename file, run
   curator; assert other 9 figures still produced, warning emitted,
   sentinel records `n_figures == 9`.
3. **Missing one feature_analysis figure** — same pattern; missing fig
   logged as skipped, other 10 produced, sentinel records `n_figures == 10`.
4. **Showcase-group override** — `--showcase-group obs225_act2` produces
   `03_action_coverage_group_a.png` from the Group B source.
5. **Pipeline diagram renders standalone** — no external font/data
   dependency.
6. **Sentinel write is atomic** — kill mid-write doesn't corrupt the
   JSON (use the existing `atomic_save` helper from
   `algorithms/offline_rl/checkpoint_utils.py`).

Smoke artifacts use hourly group keys (`obs163_act1` …) instead of
production's 15-min keys (`obs627_act1` …). The `--showcase-group` flag
must accept any string and the test asserts at least one figure renders
per group label.

---

## CLI

```
python -m scripts.curate_initiative_figures \
    --run-dir runs/offline_iql_cql_initiative_15min \
    [--output-dir docs/offline_rl/iql_cql_figures] \
    [--showcase-group obs627_act1] \
    [--groups obs627_act1 obs706_act2 obs749_act3 obs785_act3]
```

Idempotent: re-running with the same inputs overwrites the same outputs;
sentinel updates its `generated_at` timestamp.

---

## Dependencies

- Existing scripts:
  - `scripts/generate_architecture_figures.py` (for pipeline diagram)
  - `scripts/generate_training_figures.py` (for loss / valmse / cql_penalty)
- Existing module:
  - `algorithms/offline_rl/checkpoint_utils.py` (atomic_save helper)
- Stdlib: `argparse`, `json`, `shutil`, `pathlib`, `dataclasses`.
- Third-party: `numpy`, `matplotlib`, `scipy.stats` (for Wilcoxon).

No new pip dependencies.

---

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Importing `generate_*_figures` modules pulls in module-level constants pointing to stale paths (`thesis/ch4/assets`) | Import only specific functions/helpers via `from … import …`; pass `output_dir` explicitly. |
| Production output uses 15-min group keys not present in smoke | `--showcase-group` defaults work for production; smoke tests pass an override. |
| Wilcoxon p-value undefined with only 1 eval seed (smoke) | Annotate "n=1, p=N/A" when n < 2; full annotation when n >= 2 (production has 10 eval seeds). |
| Benchmark figures need IQL/CQL aggregate; current `results.json` may not have CQL section | Inspect smoke `results.json` structure: if a section is missing, log warning and skip that algorithm in the bar chart. |

---

## Acceptance criteria

- [ ] `scripts/curate_initiative_figures.py` exists and has CLI matching above.
- [ ] `tests/scripts/test_curate_initiative_figures.py` exists; all tests pass.
- [ ] Run against `runs/smoke_pipeline_phase9/` produces 11 PNGs + sentinel
      in a target directory.
- [ ] Each PNG > 5 KB (sanity check for non-empty plots).
- [ ] No regressions in existing test suite.
- [ ] No new pip deps added.
- [ ] When production at `runs/offline_iql_cql_initiative_15min/` finishes,
      the curator can be run as the final step before Phase 11 doc writing.

---

## Open questions

None. Design ready for writing-plans transition.
