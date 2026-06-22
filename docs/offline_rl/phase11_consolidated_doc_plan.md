# Phase 11 — Consolidated Doc Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `docs/offline_rl/iql_cql_initiative.md` (~620 lines, engineering-portfolio depth, narrative thesis-style), embed all 11 curated figures from Phase 10, and add a one-line cross-link from `docs/offline_rl/README.md`.

**Architecture:** Single markdown deliverable plus a one-line README edit. 11 sections totalling ~620 lines, written by the same author in coherent passes for voice consistency. Production-dependent numbers use a `<!-- TBD: production -->` marker so a single grep finds every fill-in site post-completion.

**Tech Stack:** Markdown (GitHub-flavored, CommonMark). Verification via `bash` + `grep` + `pytest -q` (no Python source modified).

**Spec:** `docs/offline_rl/phase11_consolidated_doc_design.md`

**Deliverable file:** `docs/offline_rl/iql_cql_initiative.md`
**Modified file:** `docs/offline_rl/README.md` (one line added)

**Voice reference:** `docs/offline_rl/thesis_notes.md`

---

## File Responsibility Map

| File | Responsibility | Lines |
|------|---------------|------:|
| `docs/offline_rl/iql_cql_initiative.md` | New — consolidated initiative narrative, 11 sections, embeds 11 curated figures from `iql_cql_figures/`. | ~620 |
| `docs/offline_rl/README.md` | Modified — add a single bullet under "Where things stand" cross-linking the new doc. | +1 |
| `docs/offline_rl/iql_cql_figures/*.png` | Read-only — 11 figures produced by Phase 10 (`scripts/curate_initiative_figures.py`), embedded via relative paths. | — |
| All other paths in the design spec's cross-link inventory | Read-only — referenced by relative path; existence is verified per task. | — |

---

## Section ordering (locked)

The deliverable's section headers in order:

```
1. Motivation & scope
2. Dataset
3. Algorithms (IQL + CQL)
4. Resume & status visibility
5. Engineering — the CityLearn OOM      ← NEW (Bug 7)
6. Training setup
7. Benchmark results
8. Feature analysis highlights
9. Reproducing
10. Limitations
11. References
```

Frontmatter precedes §1; TOC sits inside frontmatter.

---

## Conventions

- **Heading levels**: `#` for the doc title, `##` for sections (§1-§11), `###` for subsections (only used in §5).
- **Figure embeds**: `![<one-sentence caption>](iql_cql_figures/<file>.png)`.
- **TBD marker**: `<!-- TBD: production -->` on the line preceding any value or table that depends on production output. Numeric placeholders inside tables use `**TBD**`; mean±std uses `TBD ± TBD`.
- **Cross-link form**: relative repo path, e.g. `[dataset schema](dataset_schema.md)` (same-directory), `[show_pipeline_status.py](../../scripts/show_pipeline_status.py)` (other dir).
- **Code blocks**: triple-backtick with language hint (` ```python `, ` ```bash `, ` ```json `). End with a one-line trailing comment `# from path/to/file.py:Lstart-Lend` when excerpted from source.
- **No emojis. No exclamation marks in body prose.** First-person plural ("we", "the initiative") for decisions; passive for the system.
- **Per-section commit message form**: `Add Phase 11 \u00a7N <short title>`. The final commit is `Wire Phase 11 doc into README + final review`.

---

## Common verification steps

Two verifications recur across tasks. Define once here, reference by name in each task.

**[verify-readback]**: Read the section back to confirm it landed correctly.

```bash
.venv/bin/python -c "
from pathlib import Path
text = Path('docs/offline_rl/iql_cql_initiative.md').read_text()
import re
m = re.search(r'^## __SECTION__$.*?(?=^## |\\Z)', text, re.M | re.S)
assert m, 'section header not found'
print(m.group(0)[:400])
print('...')
print(m.group(0)[-200:])
print(f'section bytes: {len(m.group(0))}')
"
```

(Replace `__SECTION__` with the literal section header, e.g. `2. Dataset`.)

**[verify-suite]**: Confirm no Python regressions.

```bash
.venv/bin/python -m pytest -q 2>&1 | tail -3
```

Expected: same pass count as last green run; this task touches no Python.

---

## Task 1: Frontmatter + TOC + section skeleton

**Files:**
- Create: `docs/offline_rl/iql_cql_initiative.md`

**Goal:** Establish the file with frontmatter, table of contents, and 11 empty section headers. No section bodies yet. This lets later tasks each fill exactly one section without merge conflicts.

- [ ] **Step 1: Verify the spec exists and the figures directory will be a valid relative path**

```bash
ls docs/offline_rl/phase11_consolidated_doc_design.md
ls -d docs/offline_rl/iql_cql_figures/ 2>&1 || echo "iql_cql_figures/ not present yet (expected pre-Phase 10 run; embeds will use the path anyway)"
ls docs/offline_rl/thesis_notes.md
ls docs/offline_rl/iql_cql_initiative_plan.md
```

Expected: all three concrete files exist; figures dir may or may not.

- [ ] **Step 2: Create the file with frontmatter, TOC, and 11 empty section headers**

Use the Write tool to create `docs/offline_rl/iql_cql_initiative.md` with this content verbatim:

```markdown
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

<!-- task 2 writes this section -->

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
```

- [ ] **Step 3: Verify the file shape**

```bash
wc -l docs/offline_rl/iql_cql_initiative.md
grep -c "^## " docs/offline_rl/iql_cql_initiative.md
grep -nE "^## " docs/offline_rl/iql_cql_initiative.md
```

Expected: ~75 lines; 12 `## ` headers (1 TOC + 11 sections); section ordering 1..11.

- [ ] **Step 4: [verify-suite]**

Expected: same pass count as last green run.

- [ ] **Step 5: Commit**

```bash
git add docs/offline_rl/iql_cql_initiative.md
git commit -m "Add Phase 11 doc frontmatter and section skeleton"
```

---

## Task 2: §1 Motivation & scope

**Files:**
- Modify: `docs/offline_rl/iql_cql_initiative.md` — replace `<!-- task 2 writes this section -->` under `## 1. Motivation & scope` with the full §1 body.

**Goal:** ~50 lines. Open the doc by answering "why offline RL, why 17 buildings, why IQL + CQL, why 15-min, why now". Embeds `01_pipeline_overview.png`.

**Required content elements** (every point must appear; voice is narrative thesis-style):

1. **Opening paragraph (~80 words)**: Lead with the gap the initiative closes — Building-5 BC + IQL (in `thesis_notes.md`) worked but was diluted ~17× at district level; this initiative trains on all 17 buildings to make district KPIs meaningful evaluation targets. Reference `thesis_notes.md` for the prior iteration.
2. **Why offline RL (~70 words)**: No live env interaction needed for training; safer for thermal/grid constraints; data already exists (RBCSmart rollouts); aligns with deployment reality (trained policies hot-swap into the dispatcher without disturbing the live grid).
3. **Why IQL + CQL (~80 words)**: Both avoid OOD action queries differently — IQL via expectile-V (only evaluates dataset actions), CQL via conservative penalty on OOD actions. Running both lets us compare two distinct OOD-defences on the same data. Cite Kostrikov 2021 (IQL) and Kumar 2020 (CQL) inline (paper titles only, links in §11).
4. **Why 15-min, not 15-sec (~70 words)**: Trade-off — 15-sec is finer-grained but yields 5760 steps/day × 365 = 2.1M steps/year (intractable on CPU); 15-min is 96 steps/day × 365 = 35040 steps/year (~17× smaller). EV charging dynamics live at ~10s-1h scale, so 15-min is adequate. Wall-clock per seed is hours vs days.
5. **Why this initiative now (~50 words)**: Resumable single-command pipeline (Phase 1-8 prerequisites already shipped) lets the multi-building run survive crashes; this is the engineering enabler.
6. **Figure embed at end**:

```markdown
![End-to-end pipeline: collect (RBCSmart, 10 seeds × 35040 steps) → train-iql + train-cql (150k steps, 4 groups × 9 seeds) → benchmark (10 eval seeds) → feature-analysis → curated figures.](iql_cql_figures/01_pipeline_overview.png)
```

7. **Closing line**: "The rest of this note walks the pipeline from data to results."

- [ ] **Step 1: Verify the figure embed target exists OR plan a TBD marker**

```bash
ls docs/offline_rl/iql_cql_figures/01_pipeline_overview.png 2>&1
```

If the file is missing, prepend `<!-- TBD: production -->` to the figure-embed line.

- [ ] **Step 2: Use Edit to replace the placeholder with the §1 body**

Replace exactly `<!-- task 2 writes this section -->` (under `## 1. Motivation & scope`) with prose meeting the 7 required elements above. Target ~50 lines (~350 words).

- [ ] **Step 3: [verify-readback] for `1. Motivation & scope`**

Confirm the section is 40-60 lines, contains the 6 prose paragraphs + figure embed, contains zero literal "TBD" inside the prose (TBD is only acceptable inside HTML comments preceding figure embeds).

- [ ] **Step 4: Verify cross-link targets**

```bash
ls docs/offline_rl/thesis_notes.md
```

- [ ] **Step 5: [verify-suite]**

- [ ] **Step 6: Commit**

```bash
git add docs/offline_rl/iql_cql_initiative.md
git commit -m "Add Phase 11 \u00a71 Motivation & scope"
```

---

## Task 3: §2 Dataset

**Files:**
- Modify: `docs/offline_rl/iql_cql_initiative.md` — replace `<!-- task 3 writes this section -->` under `## 2. Dataset` with the full §2 body.

**Goal:** ~70 lines. Cover RBCSmart collection on the 15-min schema, dataset shape, entity-interface contract, agent groups, reward capture, provenance. Embeds 4 figures.

**Required content elements:**

1. **Opening paragraph (~80 words)**: One RBCSmart rollout per seed, 10 seeds (22-31), 35040 steps each (1 full year of 15-min data). Per-step row contains all 17 buildings' observations + actions + per-agent reward via the entity interface. Total ~350k transitions.
2. **Schema + provenance subsection (~80 words)**: Schema = `datasets/citylearn_three_phase_electrical_service_demo_15min_parquet/schema.json`; cross-link `dataset_schema.md` for column layout. Each seed parquet has a `schema_hash` provenance field; collect stage writes a `.collect.done` sentinel + `manifest.json` listing per-seed row counts, KPI summary, and `kpi_summary.csv`.
3. **Entity interface + agent groups subsection (~100 words)**: CityLearn's entity interface (vs legacy flat-vector) returns per-building observations grouped by (obs_dim, action_dim). The 17 production buildings collapse into 4 unique shapes:

   | Group key       | Buildings | obs_dim | action_dim | Notes                                  |
   |-----------------|----------:|--------:|-----------:|----------------------------------------|
   | `obs627_act1`   | 10        | 627     | 1          | 10-building cohort, headline cohort     |
   | `obs706_act2`   | 5         | 706     | 2          | 5-building cohort                       |
   | `obs749_act3`   | 1         | 749     | 3          | singleton                               |
   | `obs785_act3`   | 1         | 785     | 3          | singleton                               |

   Each group is trained separately; this matches CityLearn's per-building agent contract and avoids zero-padding heterogeneous shapes.

4. **Reward subsection (~70 words)**: Reward is `CostServiceCommunityFeasiblePrecisionRewardV46` captured **live** (stored in each transition's reward column at collection time), not synthesised from KPIs afterwards. Cross-link `reward_design.md` for the term-by-term breakdown and the calibration history (the Building-5 iteration's NNLS + hybrid-floor rule still applies).
5. **Figure embeds, with one-sentence prose captions each** (~10 lines):

   ```markdown
   ![Dataset stats: per-seed row counts, total transitions, disk size, schema hash.](iql_cql_figures/02_dataset_stats.png)
   ```

   ```markdown
   ![Action coverage for the 10-building `obs627_act1` cohort: the RBCSmart policy concentrates on a narrow action regime, motivating CQL's pessimism penalty on OOD actions.](iql_cql_figures/03_action_coverage_group_a.png)
   ```

   ```markdown
   ![Reward distribution segmented by RBCSmart action regime (charge / idle / discharge × peak / off-peak).](iql_cql_figures/04_reward_by_regime.png)
   ```

   ```markdown
   ![Temporal patterns: mean reward and action by hour-of-day, day-of-week.](iql_cql_figures/06_temporal_patterns.png)
   ```

   Each prepended with `<!-- TBD: production -->` if the corresponding PNG is missing.

6. **Closing pointer (~30 words)**: For the full EDA, see [`feature_analysis/feature_analysis.md`](feature_analysis/feature_analysis.md). §8 of this doc summarises the three insights that matter for IQL/CQL training.

- [ ] **Step 1: Verify cross-link targets exist**

```bash
ls docs/offline_rl/dataset_schema.md docs/offline_rl/reward_design.md docs/offline_rl/feature_analysis/feature_analysis.md
ls datasets/citylearn_three_phase_electrical_service_demo_15min_parquet/schema.json
```

- [ ] **Step 2: Check which curated figures exist; mark missing ones TBD**

```bash
for f in 02_dataset_stats 03_action_coverage_group_a 04_reward_by_regime 06_temporal_patterns; do
  if [ -f "docs/offline_rl/iql_cql_figures/${f}.png" ]; then
    echo "OK ${f}"
  else
    echo "MISSING ${f} \u2014 prepend <!-- TBD: production --> to its embed"
  fi
done
```

- [ ] **Step 3: Use Edit to replace the placeholder with the §2 body**

Replace exactly `<!-- task 3 writes this section -->` (under `## 2. Dataset`) with the full §2 content built from the 6 required elements above. Use the agent-groups table verbatim (numbers locked from the spec).

- [ ] **Step 4: [verify-readback] for `2. Dataset`**

Confirm: 60-80 lines; 4 figure embeds; 1 agent-groups table; cross-links to `dataset_schema.md`, `reward_design.md`, `feature_analysis/feature_analysis.md`.

- [ ] **Step 5: [verify-suite]**

- [ ] **Step 6: Commit**

```bash
git add docs/offline_rl/iql_cql_initiative.md
git commit -m "Add Phase 11 \u00a72 Dataset"
```

---

## Task 4: §3 Algorithms (IQL + CQL)

**Files:**
- Modify: `docs/offline_rl/iql_cql_initiative.md` — replace `<!-- task 4 writes this section -->` under `## 3. Algorithms (IQL + CQL)` with the §3 body.

**Goal:** ~70 lines. IQL + CQL mechanism recap, hyperparameter table, code excerpts of both trainer configs, embed `09_training_cql_penalty.png`.

**Required content elements:**

1. **IQL recap (~100 words)**: Three networks — value V (expectile loss, τ=0.7 → optimistic), twin Q (bootstraps with V instead of max over actions → no OOD queries), policy (advantage-weighted regression, β=3.0, clip 100). The expectile bias is what biases the extracted policy toward higher-return actions seen in the data. Cross-link [`iql_reference.md`](iql_reference.md) for the deeper derivation.
2. **CQL recap (~80 words)**: Standard Q-learning + a conservative penalty term `α · E_{s,a~OOD}[Q(s,a)] - E_{s,a~D}[Q(s,a)]` that pushes Q-values down on OOD actions and up on dataset actions. We use `α=0.2` — small enough to avoid over-pessimism (which would collapse the policy onto a few high-conviction actions) but large enough to be informative; the choice is empirical from the Building-5 iteration carry-over.
3. **Hyperparameter table** (verbatim):

   | Parameter | IQL | CQL | Source |
   |-----------|----:|----:|--------|
   | Hidden layers | [256, 256] | [256, 256] | `--hidden-layers 256,256` |
   | Dropout | 0.1 | 0.1 | trainer default |
   | Activation | ReLU | ReLU | trainer default |
   | Optimiser | Adam | Adam | trainer default |
   | Learning rate | 3e-4 | 3e-4 | trainer default |
   | Batch size | 256 | 256 | trainer default |
   | γ (discount) | 0.99 | 0.99 | trainer default |
   | Target soft-update τ | 0.005 | 0.005 | trainer default |
   | Gradient clip | 1.0 | 1.0 | trainer default |
   | Expectile τ (V loss) | 0.7 | — | IQL only |
   | β (AWR temp) | 3.0 | — | IQL only |
   | Advantage clip | 100 | — | IQL only |
   | CQL α | — | 0.2 | `--cql-alpha 0.2` |
   | Gradient steps | 150,000 | 150,000 | `--gradient-steps 150000` |
   | Checkpoint every | 5,000 | 5,000 | `--checkpoint-every 5000` |

4. **Code snippet 1 — `IQLTrainingConfig` excerpt** (from `algorithms/offline_rl/iql_entity_trainer.py`): ~12 lines showing dataclass fields with their actual defaults. Use a snippet of the form:

   ```python
   @dataclass
   class IQLTrainingConfig:
       hidden_layers: list[int] = field(default_factory=lambda: [256, 256])
       dropout: float = 0.1
       log_std_init: float = -2.3025850929940455
       tau_expectile: float = 0.7
       beta_advantage: float = 3.0
       advantage_clip: float = 100.0
       gamma: float = 0.99
       tau_target: float = 0.005
       learning_rate: float = 3e-4
       weight_decay: float = 1e-5
       gradient_clip_norm: float = 1.0
       batch_size: int = 256
       gradient_steps: int = 150_000
       checkpoint_every_n_steps: int = 5000
   # from algorithms/offline_rl/iql_entity_trainer.py (excerpt)
   ```

   The excerpt MUST match HEAD's actual field order and defaults — verify with the bash command in Step 1.

5. **Code snippet 2 — `CQLTrainingConfig` excerpt highlighting `cql_alpha`** (from `algorithms/offline_rl/cql_entity_trainer.py`): ~8 lines.

   ```python
   @dataclass
   class CQLTrainingConfig:
       hidden_layers: list[int] = field(default_factory=lambda: [256, 256])
       gamma: float = 0.99
       tau_target: float = 0.005
       learning_rate: float = 3e-4
       batch_size: int = 256
       cql_alpha: float = 0.2          # conservative penalty weight
       cql_n_samples: int = 10         # OOD action samples per state
       gradient_steps: int = 150_000
   # from algorithms/offline_rl/cql_entity_trainer.py (excerpt)
   ```

6. **Why no online fine-tuning (~40 words)**: Pure offline mandate from the constraint set (no live env interaction during training); online fine-tuning is a separate downstream stage and outside this initiative.
7. **Figure embed at end**:

   ```markdown
   ![CQL penalty trace over training, all four agent groups: the penalty rises as Q estimates drift OOD and is bounded by `α=0.2`.](iql_cql_figures/09_training_cql_penalty.png)
   ```

- [ ] **Step 1: Verify code snippets match source**

```bash
grep -nE "^(class|\\s+(hidden_layers|tau_expectile|beta_advantage|gamma|cql_alpha|gradient_steps|checkpoint_every_n_steps)):" algorithms/offline_rl/iql_entity_trainer.py algorithms/offline_rl/cql_entity_trainer.py 2>&1 | head -30
```

Compare field names and defaults against the snippets in elements 4 and 5. If anything differs, update the snippet to match HEAD before writing.

- [ ] **Step 2: Verify figure exists or mark TBD**

```bash
ls docs/offline_rl/iql_cql_figures/09_training_cql_penalty.png 2>&1 || echo "MISSING \u2014 prepend TBD marker"
```

- [ ] **Step 3: Verify cross-link**

```bash
ls docs/offline_rl/iql_reference.md
```

- [ ] **Step 4: Use Edit to replace placeholder with §3 body**

- [ ] **Step 5: [verify-readback] for `3. Algorithms (IQL + CQL)`**

Confirm: 60-80 lines; 1 hyperparameter table (15+ rows); 2 code blocks (python); 1 figure embed.

- [ ] **Step 6: [verify-suite]**

- [ ] **Step 7: Commit**

```bash
git add docs/offline_rl/iql_cql_initiative.md
git commit -m "Add Phase 11 \u00a73 Algorithms (IQL + CQL)"
```

---

## Task 5: §4 Resume & status visibility

**Files:**
- Modify: `docs/offline_rl/iql_cql_initiative.md` — replace `<!-- task 5 writes this section -->` under `## 4. Resume & status visibility` with the §4 body.

**Goal:** ~80 lines. Cover the three-level idempotency design (stage sentinels, seed sentinels, gradient-step checkpoints), atomic save invariant, status.json schema, the `show_pipeline_status.py` CLI. No figure; carried by code blocks + tables.

**Required content elements:**

1. **Opening paragraph (~80 words)**: A multi-day pipeline on a single workstation cannot rely on never crashing. Phase 2 of the initiative built three nested idempotency layers so a kill at any point resumes deterministically: (1) per-stage sentinels (`{stage}.done`), (2) per-seed sentinels inside training (`seed.done`), (3) per-checkpoint atomic saves of optimizer + RNG state every N gradient steps. Crash recovery = re-run the same command.
2. **Atomic save invariant (~50 words)**: `atomic_save(obj, path)` writes to `path.tmp` then calls `os.replace(path.tmp, path)` — POSIX-atomic, so a kill mid-write either leaves the old file or commits the new one, never a half-written file. Covered by `tests/offline_rl/test_atomic_save.py` (kill-mid-write doesn't corrupt).
3. **Code snippet — `checkpoint_latest.pt` schema** (~12 lines), from the parent plan's Phase 2:

   ```python
   {
       "step": int,                       # last completed gradient step
       "policy_state": dict,
       "qf1_state": dict, "qf2_state": dict,
       "qf1_target_state": dict, "qf2_target_state": dict,
       "vf_state": dict,
       "policy_opt_state": dict, "qf_opt_state": dict, "vf_opt_state": dict,
       "rng_state": dict,                 # torch + numpy + python RNGs
       "best_val_mse": float,
       "best_step": int,
       "wall_clock_seconds": float,
   }
   # from algorithms/offline_rl/checkpoint_utils.py (schema)
   ```

4. **status.json subsection + code block** (~25 lines): The orchestrator merges per-stage updates atomically into `runs/<output>/status.json`. The trainer pushes per-checkpoint progress into the same file. Show the schema:

   ```json
   {
     "stages": {
       "collect":         {"status": "done",   "started_at": "...", "duration_seconds": 17500.0},
       "train-iql":       {"status": "running", "group": "obs627_act1", "seed": 24, "step": 67000, "best_val_mse": 0.000158, "eta_seconds": 14400},
       "train-cql":       {"status": "pending"},
       "benchmark":       {"status": "pending"},
       "feature-analysis":{"status": "pending"}
     }
   }
   ```

   Mention that updates are atomic via the same `.tmp → os.replace` pattern.

5. **`show_pipeline_status.py` CLI subsection (~30 words + code block)**: Read-only viewer that prints the table the orchestrator computes. Falls back to disk scanning of sentinels + checkpoint files if `status.json` is missing.

   ```text
   ┌──────────────────┬──────────┬─────────────────┬──────────┐
   │ Stage            │ Status   │ Progress        │ ETA      │
   ├──────────────────┼──────────┼─────────────────┼──────────┤
   │ collect          │ \u2713 done   │ 10/10 seeds     │ \u2014        │
   │ train-iql        │ \u25b6 running│ group 2/4, 67k  │ 4h 12m   │
   │ train-cql        │ pending  │                 │          │
   │ benchmark        │ pending  │                 │          │
   │ feature-analysis │ pending  │                 │          │
   └──────────────────┴──────────┴─────────────────┴──────────┘
   # python -m scripts.show_pipeline_status runs/offline_iql_cql_initiative_15min/
   ```

6. **Resume semantics in one paragraph (~50 words)**: Re-running the same single command is the recovery procedure. Each stage checks its `.done` sentinel and skips if present; the trainer loads `checkpoint_latest.pt` and resumes from `step+1`. `--force STAGE[,STAGE...]` bypasses sentinels for explicit re-runs.

- [ ] **Step 1: Verify cross-link / source paths**

```bash
ls algorithms/offline_rl/checkpoint_utils.py scripts/show_pipeline_status.py tests/offline_rl/test_atomic_save.py
```

- [ ] **Step 2: Verify the status.json shape matches a live example (if available)**

```bash
.venv/bin/python -c "
import json, pathlib
p = pathlib.Path('runs/offline_iql_cql_initiative_15min/status.json')
print(p.read_text() if p.exists() else 'status.json absent yet')
"
```

- [ ] **Step 3: Use Edit to replace placeholder with §4 body**

- [ ] **Step 4: [verify-readback] for `4. Resume & status visibility`**

Confirm: 70-90 lines; 3 code blocks (1 python schema, 1 json status, 1 text table); cross-link to `checkpoint_utils.py`, `show_pipeline_status.py`, `test_atomic_save.py`.

- [ ] **Step 5: [verify-suite]**

- [ ] **Step 6: Commit**

```bash
git add docs/offline_rl/iql_cql_initiative.md
git commit -m "Add Phase 11 \u00a74 Resume & status visibility"
```

---

## Task 6: §5 Engineering — the CityLearn OOM (NEW)

**Files:**
- Modify: `docs/offline_rl/iql_cql_initiative.md` — replace `<!-- task 6 writes this section -->` under `## 5. Engineering — the CityLearn OOM` with the §5 body.

**Goal:** ~80 lines. Seven `###` subsections. Narrative-thesis voice; the engineering anecdote of the initiative. Anchored on commits `98f7944` (collect) + `f5238be` (benchmark).

**Required content elements:**

### Subsection 5.1 The crash (~10 lines)

Opening sentence: "Production launched 2026-06-21 at 10:50 UTC and was dead by 13:01 UTC."

Cover: 7906 s wall-clock, exit -9 (SIGKILL, kernel OOM-killer), step ~16000 of 35040, no traceback, partial `seed_22.parquet` (307 MB), `collect` status flipped to `failed` in `status.json`. Note why 15-min full-year made it observable while the 15-sec smoke (5760 steps) never tripped it.

### Subsection 5.2 Probing memory (~20 lines)

Three probes run outside the production loop (all scripted to `/tmp/`):

- **Probe v1 (RSS sampler)**: built env + RBC + adapter once, ran `env.step()` in a loop, sampled `psutil.Process().memory_info().rss` every 200 steps. Result: steady ~8 MB/step growth from step 200 onward; RSS at step 2000 = 16 GB.
- **Probe v2 (`tracemalloc` snapshot diff)**: took snapshots at step 0 and step 200 then diffed. Top culprit: `citylearn/energy_model.py:119` (a property returning `self.__electricity_consumption * self.time_step_ratio` — allocates a new ndarray every call). 465 MB allocated across 200 steps from that line.
- **Probe v3 (reset effectiveness)**: looped env.reset() between short rollouts to see if reset releases RSS. Per-cycle growth dropped from 4.4 to 0.3 MB/step but RSS never returned to baseline. Conclusion: the leak is class-level (not episode-local), and reset alone cannot recover.

Reference paths: `/var/folders/.../opencode/probe_oom_memory.py`, `probe_oom_tracemalloc.py`, `probe_oom_reset.py`.

### Subsection 5.3 Root cause (~15 lines)

`CityLearnEntityInterfaceService._action_feedback_series_summary` (in `.venv/lib/python3.10/site-packages/citylearn/internal/entity_interface.py:1716`) memoises a per-step feedback summary in `self._action_feedback_series_cache` (initialised at line 352, cleared in `reset()` at 358 and `invalidate()` at 370).

The cache key is `id(values)` for each source array. New ndarrays are allocated every step (see `energy_model.py:119` from Probe v2), so the key never matches a prior entry — the cache grows by ~85 entries per step (17 buildings × ~5 source arrays). After 16,000 steps that's 1.36M dict entries pinning the corresponding ndarrays. RSS climbs ~8 MB per step, kernel OOM at ~20 GB.

`env.reset()` does call `self._action_feedback_series_cache.clear()`, but Python's allocator does not return the freed pages to the OS in this case, and the next episode's accumulation starts allocating fresh.

### Subsection 5.4 Decision matrix (~15 lines)

Three options considered:

| Option | Invasiveness | Data-continuity risk | Performance cost | Complexity |
|--------|--------------|----------------------|------------------|-----------|
| (a) Subprocess chunking — split collect into N short jobs, reset RSS via process boundary | Medium (refactor orchestrator + collector to stitch chunks) | High (chunk boundary state-loss between rollouts; would need careful resume) | Higher (process spawn + JSON serialisation cost) | High |
| (b) Monkey-patch CityLearn cache — wrap `_action_feedback_series_summary` to FIFO-evict at 128 entries | Low (~90-line new module) | None (worst case = cache miss + rebuild) | Negligible (~1 µs per step) | Low |
| (c) Hybrid — patch + chunking belt-and-braces | High (both costs) | Same as (a) | Same as (a) | High |

We chose **(b)**. The patch is small, idempotent, and only modifies one method; (a) and (c) restructure the orchestrator for what is a downstream-library defect.

### Subsection 5.5 The fix (~15 lines)

`utils/citylearn_patches.apply_citylearn_patches()` wraps the original `_action_feedback_series_summary` so that, after every successful computation, the cache is bounded to `ACTION_FEEDBACK_CACHE_MAX = 128` entries via FIFO eviction (Python dict insertion order). Code excerpt:

```python
ACTION_FEEDBACK_CACHE_MAX = 128

def _bound_dict_size(d: dict, maxsize: int) -> None:
    """Evict oldest entries until len(d) <= maxsize. FIFO via insertion order."""
    while len(d) > maxsize:
        d.pop(next(iter(d)))

def apply_citylearn_patches() -> None:
    """Idempotent. Safe to call once per process (module import time)."""
    global _PATCHED
    if _PATCHED:
        return
    _patch_action_feedback_series_cache()
    _PATCHED = True
# from utils/citylearn_patches.py (excerpt)
```

The wrapper is installed at module-import time by `scripts/collect_rbcsmart_dataset.py` and `scripts/benchmark_entity_agents.py` (both call `apply_citylearn_patches()` before importing `CityLearnEnv`). Idempotency is guarded by a module-level `_PATCHED` flag.

Test coverage: `tests/test_citylearn_patches.py` — 8 tests (5 unit on `_bound_dict_size`, 2 idempotency, 1 end-to-end with stubbed original method).

### Subsection 5.6 Validation (~10 lines)

Probe v1 was rerun under the patch (PID 91328 in run logs); growth dropped from ~8 MB/step to ~0.1 MB/step (80× reduction). RSS comparison at sampled steps:

| Step  | Pre-patch RSS | Post-patch RSS | Ratio |
|------:|--------------:|---------------:|------:|
| 200   | 1286 MB       | 804 MB         | 1.6×  |
| 600   | 3017 MB       | 827 MB         | 3.6×  |
| 2000  | ~16336 MB     | 959 MB         | 17×   |
| 4000  | (OOM imminent)| 1148 MB        | —     |
| 35040 (projected, linear) | (OOM long since) | ~4269 MB | — |

Empirical full-year collect under the patch confirmed RSS plateaus around 4 GB per seed — well within the 32 GB Mac envelope.

### Subsection 5.7 Lesson (~5 lines)

Downstream library invariants matter. Memoisation tables that look harmless can OOM when called at fine temporal granularity, especially when keyed on object identity (which never collides for short-lived ndarrays). The patch is small (~90 lines) because we only had to bound one dict; the diagnostic effort (three probes) was the real work. When a library you depend on caches without bounds, expect to discover it under exactly the workload the library wasn't tested against.

- [ ] **Step 1: Verify commits referenced by §5 exist on HEAD**

```bash
git log --oneline 98f7944 f5238be 2>&1 | head -5
```

Expected: both commits printed; no "unknown revision" errors.

- [ ] **Step 2: Verify source paths cited in §5 exist**

```bash
ls utils/citylearn_patches.py tests/test_citylearn_patches.py
ls .venv/lib/python3.10/site-packages/citylearn/internal/entity_interface.py
ls .venv/lib/python3.10/site-packages/citylearn/energy_model.py
```

- [ ] **Step 3: Verify the `_bound_dict_size` code excerpt matches source**

```bash
grep -nE "(ACTION_FEEDBACK_CACHE_MAX|def _bound_dict_size|def apply_citylearn_patches)" utils/citylearn_patches.py
```

Update the snippet in element 5 if the signature differs from HEAD.

- [ ] **Step 4: Use Edit to replace placeholder with §5 body**

Replace the placeholder line exactly. Insert all 7 subsections with `### 5.N <title>` headers. Tables 5.4 and 5.6 are verbatim from this plan.

- [ ] **Step 5: [verify-readback] for `5. Engineering \u2014 the CityLearn OOM`**

Confirm: 70-90 lines; 7 `### 5.N` subsection headers; 1 python code block (`_bound_dict_size`); 2 tables (decision matrix + before/after RSS).

- [ ] **Step 6: [verify-suite]**

- [ ] **Step 7: Commit**

```bash
git add docs/offline_rl/iql_cql_initiative.md
git commit -m "Add Phase 11 \u00a75 Engineering \u2014 the CityLearn OOM"
```

---

## Task 7: §6 Training setup

**Files:**
- Modify: `docs/offline_rl/iql_cql_initiative.md` — replace `<!-- task 7 writes this section -->` under `## 6. Training setup` with the §6 body.

**Goal:** ~60 lines. Multi-group multi-seed training matrix, val/eval seed protocol, wall-clock estimate, 2 figures.

**Required content elements:**

1. **Opening paragraph (~60 words)**: 4 agent groups × 9 train seeds × 2 algorithms (IQL + CQL) = 72 independent training runs of 150,000 gradient steps each. Each run trains on seeds 22-30, validates on seed 31, and is evaluated separately (§7) on disjoint env seeds 200-209.
2. **Training matrix table** (verbatim):

   | Aspect | Value | Source |
   |--------|-------|--------|
   | Groups | 4 (`obs627_act1`, `obs706_act2`, `obs749_act3`, `obs785_act3`) | derived from schema |
   | Train seeds (per group, per algorithm) | 22, 23, 24, 25, 26, 27, 28, 29, 30 (9 seeds) | `--train-seeds 22,23,24,25,26,27,28,29,30` |
   | Val seed | 31 | `--val-seeds 31` |
   | Eval seeds | 200..209 (10 seeds, disjoint from train+val+collect) | `--eval-seeds 200,...,209` |
   | Gradient steps | 150,000 | `--gradient-steps 150000` |
   | Algorithms | IQL + CQL | `--algorithm both` |
   | Total runs | 4 × 9 × 2 = 72 | — |
   | Best-checkpoint policy | per-(group, seed): lowest validation MSE | trainer default |

3. **Wall-clock estimate (~50 words)**: On CPU, ~11 hours per (group, seed) for IQL or CQL → ~99 hours per (group, algorithm) → ~792 hours total if fully serial. The orchestrator runs groups serially; seeds within a group are also serial; trainer is single-threaded. <!-- TBD: production --> Update with actual wall-clock from `status.json` after production completes.
4. **Validation protocol paragraph (~40 words)**: Validation MSE on seed 31 is the model-selection signal. Every `--checkpoint-every 5000` steps, the trainer writes `checkpoint_latest.pt` (resume target) and, if val improved, `best_policy.pt` (eval target). The benchmark stage loads `best_policy.pt` per (group, seed).
5. **Figure embeds with captions**:

   ```markdown
   ![Training loss curve, IQL on `obs627_act1` showcase group, seed 22 \u2014 stable convergence over 150k gradient steps.](iql_cql_figures/07_training_loss_group_a.png)
   ```

   ```markdown
   ![Validation MSE across all four agent groups for both algorithms; shaded band = 1\u03c3 over the 9 train seeds.](iql_cql_figures/08_training_valmse_all.png)
   ```

- [ ] **Step 1: Verify figures exist or mark TBD**

```bash
for f in 07_training_loss_group_a 08_training_valmse_all; do
  [ -f "docs/offline_rl/iql_cql_figures/${f}.png" ] && echo "OK ${f}" || echo "MISSING ${f} \u2014 prepend TBD marker"
done
```

- [ ] **Step 2: Use Edit to replace placeholder with §6 body**

- [ ] **Step 3: [verify-readback] for `6. Training setup`**

Confirm: 50-70 lines; 1 training-matrix table; 2 figure embeds.

- [ ] **Step 4: [verify-suite]**

- [ ] **Step 5: Commit**

```bash
git add docs/offline_rl/iql_cql_initiative.md
git commit -m "Add Phase 11 \u00a76 Training setup"
```

---

## Task 8: §7 Benchmark results

**Files:**
- Modify: `docs/offline_rl/iql_cql_initiative.md` — replace `<!-- task 8 writes this section -->` under `## 7. Benchmark results` with the §7 body.

**Goal:** ~70 lines. Results table (TBD-marked), interpretation prose, statistical-test note, 2 figures. This section is the most TBD-heavy by design.

**Required content elements:**

1. **Opening paragraph (~50 words)**: Each (group, algorithm, train seed) is evaluated on env seeds 200..209 (10 seeds disjoint from train + val + collect). KPIs are CityLearn's normalised values — lower is better, 1.0 = no-control baseline. Headline KPIs: `cost_total`, `carbon_emissions_total`, `electricity_consumption_peak`, `ramping`. Unserved energy is monitored as a hard constraint.
2. **Headline results table** (TBD-marked, structure locked):

   ```markdown
   <!-- TBD: production -->

   | KPI                              | RBCSmart      | IQL           | CQL           | \u0394 (best \u2212 RBC) |
   |----------------------------------|--------------:|--------------:|--------------:|------------------:|
   | `cost_total` (district)          | TBD \u00b1 TBD     | TBD \u00b1 TBD     | TBD \u00b1 TBD     | TBD               |
   | `carbon_emissions_total`         | TBD \u00b1 TBD     | TBD \u00b1 TBD     | TBD \u00b1 TBD     | TBD               |
   | `electricity_consumption_peak`   | TBD \u00b1 TBD     | TBD \u00b1 TBD     | TBD \u00b1 TBD     | TBD               |
   | `ramping`                        | TBD \u00b1 TBD     | TBD \u00b1 TBD     | TBD \u00b1 TBD     | TBD               |
   | `unserved_energy`                | TBD           | TBD           | TBD           | TBD               |
   ```

3. **Per-group breakdown subsection (~60 words)**: For each of the 4 agent groups, report the same KPIs aggregated over the group's buildings. Cross-reference the JSON: `runs/offline_iql_cql_initiative_15min/benchmark/results.json` is the source of truth.
4. **Statistical test note (~50 words)**: We report paired-Wilcoxon p-values per (group, KPI) on the 10 eval seeds, IQL vs CQL paired by seed. With n=10, p < 0.05 is interpretable but underpowered for small effects. <!-- TBD: production --> Tabulate the four group-level p-values once production lands.
5. **Interpretation prose** (~80 words, TBD-bracketed): Lead with the headline — does either method beat RBCSmart on the four headline KPIs at district level, and by how much? Note unserved-energy constraint compliance. <!-- TBD: production --> Replace placeholder prose with concrete findings after running.

   Prose template to seed the section:

   > <!-- TBD: production -->
   > _On the headline cost KPI, [IQL/CQL] reduces district cost by **TBD%** relative to RBCSmart (paired-Wilcoxon p = **TBD**). Carbon follows the same direction. Peak demand and ramping show **TBD** \u2014 expected given the reward weights (peak:cost \u2248 2:1 in standardised space). Unserved energy stays at zero across all 50 (= 5 train \u00d7 10 eval) rollouts, matching the Building-5 iteration's safety result._

6. **Figure embeds with captions**:

   ```markdown
   ![Benchmark KPI bar chart \u2014 cost, carbon, peak, ramping for RBCSmart vs IQL vs CQL at district level. Error bars = \u00b11\u03c3 over the 10 eval seeds.](iql_cql_figures/10_benchmark_kpi_bars.png)
   ```

   ```markdown
   ![IQL vs CQL per-eval-seed cost scatter (district). y = x reference line. Points below the line: CQL beats IQL on that seed.](iql_cql_figures/11_iql_vs_cql_scatter.png)
   ```

- [ ] **Step 1: Verify figures exist or mark TBD**

```bash
for f in 10_benchmark_kpi_bars 11_iql_vs_cql_scatter; do
  [ -f "docs/offline_rl/iql_cql_figures/${f}.png" ] && echo "OK ${f}" || echo "MISSING ${f} \u2014 prepend TBD marker"
done
```

- [ ] **Step 2: Verify results.json path (may not exist yet)**

```bash
ls runs/offline_iql_cql_initiative_15min/benchmark/results.json 2>&1 || echo "expected absent pre-completion"
```

- [ ] **Step 3: Use Edit to replace placeholder with §7 body**

The headline table and interpretation prose carry literal `TBD` markers — those are intentional and must remain until Phase 12 fills them in.

- [ ] **Step 4: [verify-readback] for `7. Benchmark results`**

Confirm: 60-80 lines; 1 headline-KPI table; 2 figure embeds; at least 3 `<!-- TBD: production -->` markers.

- [ ] **Step 5: [verify-suite]**

- [ ] **Step 6: Commit**

```bash
git add docs/offline_rl/iql_cql_initiative.md
git commit -m "Add Phase 11 \u00a77 Benchmark results (TBD-marked)"
```

---

## Task 9: §8 Feature analysis highlights

**Files:**
- Modify: `docs/offline_rl/iql_cql_initiative.md` — replace `<!-- task 9 writes this section -->` under `## 8. Feature analysis highlights` with the §8 body.

**Goal:** ~50 lines. Three key insights from `feature_analysis/feature_analysis.md`; one figure.

**Required content elements:**

1. **Opening pointer (~30 words)**: §2 introduced the dataset; this section pulls out the three findings that most directly motivate the algorithm choices in §3. The full EDA lives in [`feature_analysis/feature_analysis.md`](feature_analysis/feature_analysis.md); we summarise rather than reproduce.
2. **Insight 1 — Action concentration (~80 words)**: RBCSmart's action distribution is tightly concentrated (mostly idle, narrow EV-charge band). Translated to offline RL: the dataset covers a small slice of (s, a) space, and learned Q-values would be over-confident on the unsampled regions if we used vanilla Q-learning. This is the textbook motivation for CQL's penalty on OOD actions. Figure 03 (in §2) shows the action coverage; IQL handles the same problem differently (expectile-V never queries OOD actions in the first place).
3. **Insight 2 — Feature×reward correlations (~70 words)**: A handful of features dominate predictive power for reward: net electricity consumption, carbon intensity, non-shiftable load (consistent with the Building-5 iteration). The correlation matrix per group (figure embed below) shows the same pattern across cohorts — a single feature-engineering pass would lift training MSE more than hyperparameter tuning at this point. Pointer to `feature_analysis/feature_analysis.md` §"What actually matters" for the per-group breakdown.
4. **Insight 3 — Temporal structure (~60 words)**: Reward and EV-action distributions cycle on hour-of-day (figure 06 in §2). Reward dips during evening peaks and bounces back overnight as EV charging dominates the action signal. Training never sees a transition with timestamp explicitly, but the proxy features (hour, day-of-week one-hot) encode the cycle adequately.
5. **Figure embed**:

   ```markdown
   ![Feature\u2013reward correlation matrix for the `obs627_act1` showcase group. The brightest off-diagonal cluster is the price/temperature forecast triplets \u2014 a redundancy worth flagging for future feature engineering.](iql_cql_figures/05_correlations_group_a.png)
   ```

- [ ] **Step 1: Verify figure exists or mark TBD**

```bash
ls docs/offline_rl/iql_cql_figures/05_correlations_group_a.png 2>&1 || echo "MISSING \u2014 prepend TBD marker"
```

- [ ] **Step 2: Verify cross-link**

```bash
ls docs/offline_rl/feature_analysis/feature_analysis.md
```

- [ ] **Step 3: Use Edit to replace placeholder with §8 body**

- [ ] **Step 4: [verify-readback] for `8. Feature analysis highlights`**

Confirm: 40-60 lines; 3 numbered insights; 1 figure embed.

- [ ] **Step 5: [verify-suite]**

- [ ] **Step 6: Commit**

```bash
git add docs/offline_rl/iql_cql_initiative.md
git commit -m "Add Phase 11 \u00a78 Feature analysis highlights"
```

---

## Task 10: §9 Reproducing + §10 Limitations + §11 References

**Files:**
- Modify: `docs/offline_rl/iql_cql_initiative.md` — replace placeholders in §9, §10, §11 with their bodies (grouped because they are small).

**Goal:** ~40 + ~30 + ~20 = ~90 lines total across three sections.

### §9 Reproducing (~40 lines)

1. **Opening (~30 words)**: One command runs the full pipeline; the same command resumes after a crash.
2. **Code block — launch**:

   ```bash
   nohup .venv/bin/python -m scripts.run_entity_pipeline \
       --episode-steps 35040 \
       > runs/offline_iql_cql_initiative_15min/pipeline.log 2>&1 &
   echo $! > runs/offline_iql_cql_initiative_15min/pipeline.pid
   ```

   Caption above it: "Launch (full pipeline). `--episode-steps 35040` is required for the 15-min full-year horizon; without it the collector defaults to one day (96 steps)."

3. **Code block — monitor**:

   ```bash
   .venv/bin/python -m scripts.show_pipeline_status \
       runs/offline_iql_cql_initiative_15min/
   ```

   Caption: "Read-only viewer; see §4 for the table format."

4. **Code block — resume on crash**: same as launch above; idempotency does the rest. Caption: "Resume after kill / crash. Sentinels and `checkpoint_latest.pt` files do the work."

5. **Expected output tree** (verbatim):

   ```text
   runs/offline_iql_cql_initiative_15min/
   ├── data/                              # seed_22.parquet \u2026 seed_31.parquet
   ├── models-iql/<group>/seed_<N>/       # best_policy.pt, checkpoint_latest.pt, metrics.jsonl
   ├── models-cql/<group>/seed_<N>/       # same shape
   ├── benchmark/results.json
   ├── feature_analysis/                  # summary.md + figures/
   ├── pipeline.log
   ├── status.json
   ├── .collect.done
   ├── .train-iql.done
   ├── .train-cql.done
   ├── .benchmark.done
   └── .feature-analysis.done
   ```

### §10 Limitations (~30 lines)

Bullet list, ~6 bullets, ~50 words each:

1. **Single schema (15-min)**: results don't generalise to 15-sec or hourly without re-training; schema choice constrains EV-charging dynamics resolution.
2. **No online fine-tuning**: pure offline mandate; safety-critical real-world deployment would benefit from a careful online refinement phase, out of scope here.
3. **CityLearn cache patch is a workaround**, not an upstream fix. We carry `utils/citylearn_patches.py` until the upstream addresses the unbounded `_action_feedback_series_cache`. The patch is idempotent and safe, but it is technical debt.
4. **CPU-only training**: GPU acceleration is straightforward (`--device cuda`) but not the default; wall-clock estimates in §6 assume CPU.
5. **Stale Building-5 docs**: `docs/offline_rl/README.md` and `thesis_notes.md` predate this initiative and still frame the work as Building-5-only. Out of scope to rewrite; left as historical record (this doc supersedes them).
6. **No hyperparameter sweep**: hyperparameters are carried over from the Building-5 iteration. A future iteration should sweep `cql_alpha`, `tau_expectile`, and `beta_advantage` on this larger dataset.

### §11 References (~20 lines)

Two subsections:

**Internal docs** (relative-path bullets):

- [`iql_cql_initiative_plan.md`](iql_cql_initiative_plan.md) — parent plan; phase definitions.
- [`phase10_curation_design.md`](phase10_curation_design.md) — figure curation spec.
- [`phase11_consolidated_doc_design.md`](phase11_consolidated_doc_design.md) — this doc's design spec.
- [`dataset_schema.md`](dataset_schema.md) — column-level dataset contract.
- [`kpi_reference.md`](kpi_reference.md) — KPI definitions and reward-term mapping.
- [`reward_design.md`](reward_design.md) — reward function structure + calibration history.
- [`iql_reference.md`](iql_reference.md) — IQL derivation + ablations.
- [`feature_analysis/feature_analysis.md`](feature_analysis/feature_analysis.md) — full EDA.
- [`thesis_notes.md`](thesis_notes.md) — Building-5 iteration narrative.

**External** (no clickable URLs — title + author + year):

- Kostrikov, I., Nair, A., & Levine, S. (2021). "Offline Reinforcement Learning with Implicit Q-Learning." ICLR 2022.
- Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). "Conservative Q-Learning for Offline Reinforcement Learning." NeurIPS 2020.
- Vázquez-Canteli, J. R., Dey, S., Henze, G., & Nagy, Z. (2020). "CityLearn: standardizing research in multi-agent reinforcement learning for demand response and urban energy management." (CityLearn paper.)

**Thesis cross-reference (~30 words)**: The academic treatment is in the thesis chapters at `/Users/guilherme.desousa/MEIA/Thesis/Project/meia-thesis-1211073/thesis/` (Ch4 Methodology, Ch5 Experiments, Ch6 Results).

- [ ] **Step 1: Verify all internal cross-link targets exist**

```bash
for f in iql_cql_initiative_plan phase10_curation_design phase11_consolidated_doc_design dataset_schema kpi_reference reward_design iql_reference thesis_notes; do
  [ -f "docs/offline_rl/${f}.md" ] && echo "OK ${f}.md" || echo "MISSING ${f}.md"
done
[ -f "docs/offline_rl/feature_analysis/feature_analysis.md" ] && echo "OK feature_analysis/feature_analysis.md" || echo "MISSING"
```

- [ ] **Step 2: Verify the thesis directory exists**

```bash
ls -d /Users/guilherme.desousa/MEIA/Thesis/Project/meia-thesis-1211073/thesis/ 2>&1
```

- [ ] **Step 3: Use Edit to replace each of §9, §10, §11 placeholders in turn**

Three Edit calls — one per section. Each replaces the literal `<!-- task 10 writes this section -->` line under its respective heading.

- [ ] **Step 4: [verify-readback] for `9. Reproducing`, `10. Limitations`, `11. References`**

Three readbacks. Confirm §9: ~30-50 lines, 3 bash blocks, 1 tree block; §10: ~25-35 lines, 6 bullet points; §11: ~15-25 lines, 2 subsections + thesis cross-ref.

- [ ] **Step 5: [verify-suite]**

- [ ] **Step 6: Commit**

```bash
git add docs/offline_rl/iql_cql_initiative.md
git commit -m "Add Phase 11 \u00a79 + \u00a710 + \u00a711 (Reproducing, Limitations, References)"
```

---

## Task 11: README cross-link + final self-review

**Files:**
- Modify: `docs/offline_rl/README.md` — add one bullet under "Where things stand".
- (Touch) `docs/offline_rl/iql_cql_initiative.md` for self-review fixes if any.

**Goal:** Wire the new doc into the directory's entry point; final voice + cross-link + TBD-count sanity pass on the new doc.

- [ ] **Step 1: Locate the insertion point in README.md**

```bash
grep -nE "(Where things stand|next stage is IQL)" docs/offline_rl/README.md
```

Expected: confirms "Where things stand" is around line 97 and the existing "next stage" sentence is around line 107.

- [ ] **Step 2: Use Edit to append the cross-link bullet**

Find the line:

```
The next stage is IQL on the same dataset and reward; the design is
frozen in `specs/iql_design.md` and the implementation plan goes
under `plans/`.
```

Replace with:

```
The next stage is IQL on the same dataset and reward; the design is
frozen in `specs/iql_design.md` and the implementation plan goes
under `plans/`.

The current stage extends offline RL to all 17 buildings with IQL + CQL
on the 15-min schema. See [`iql_cql_initiative.md`](iql_cql_initiative.md)
for that initiative.
```

- [ ] **Step 3: Verify the new README link resolves**

```bash
grep -nE "iql_cql_initiative.md" docs/offline_rl/README.md
ls docs/offline_rl/iql_cql_initiative.md
```

- [ ] **Step 4: Final voice + structure self-review**

Read the new doc end-to-end:

```bash
.venv/bin/python -c "
from pathlib import Path
print(Path('docs/offline_rl/iql_cql_initiative.md').read_text())
" | head -200
```

Then check:
- Section headers in order 1..11 with exactly one `## ` per section.
- All 11 curated figure filenames referenced exactly once (no duplicates, no missing).
- No literal "TBD" outside of `<!-- TBD: production -->` markers, table cells (`**TBD**`), or italic placeholder prose (`_(figure pending production run)_`).

```bash
grep -nE "^## " docs/offline_rl/iql_cql_initiative.md
grep -oE "iql_cql_figures/[0-9_a-z]+\.png" docs/offline_rl/iql_cql_initiative.md | sort | uniq -c
grep -nE "TBD" docs/offline_rl/iql_cql_initiative.md | wc -l
```

Expected: 12 `## ` headers (TOC + 11 sections); each figure referenced exactly once (count = 1); TBD count matches the expected open-items inventory (every TBD must be either in a `<!-- TBD: production -->` HTML comment, inside a table cell, or inside an italic figure-placeholder).

- [ ] **Step 5: Voice consistency check**

```bash
grep -nE "(\\bI \\b|\\bMe \\b|\\bmy \\b)" docs/offline_rl/iql_cql_initiative.md
```

Expected: zero matches (we use "we" / "the initiative", not first-person singular).

```bash
grep -nE "!" docs/offline_rl/iql_cql_initiative.md
```

Expected: zero exclamation marks outside of figure-embed markdown syntax (`![alt]`).

- [ ] **Step 6: Run pytest one more time**

```bash
.venv/bin/python -m pytest -q 2>&1 | tail -3
```

Expected: same green count as last run.

- [ ] **Step 7: Commit**

```bash
git add docs/offline_rl/README.md docs/offline_rl/iql_cql_initiative.md
git commit -m "Wire Phase 11 doc into README + final review"
```

---

## Self-review checklist (the planner runs this)

After writing the plan, walk through the spec at
`docs/offline_rl/phase11_consolidated_doc_design.md` and confirm every
section is covered by a task above.

| Spec section | Covered by task(s) |
|--------------|--------------------|
| Section structure (locked) — §1..§11 | Tasks 1 (skeleton), 2..10 (bodies), 11 (final review) |
| Figure binding (11/11) | Tasks 2 (§1), 3 (§2), 4 (§3), 7 (§6), 8 (§7), 9 (§8) — all 11 placed |
| Code snippet inventory (8 snippets) | Task 4 (2 snippets), Task 5 (3 snippets), Task 6 (1 snippet), Task 10 (2 launch/resume commands) |
| §5 detailed outline (5.1..5.7) | Task 6 — all 7 subsections itemised |
| Cross-link inventory | Tasks 2-11 — each task verifies the targets it cites |
| TBD-placeholder convention | Task 8 (most TBD-heavy), Task 6 §6 (training wall-clock), Task 11 (final grep audit) |
| README cross-link | Task 11 |
| Testing strategy (no pytest tests, only verification) | [verify-suite] runs at the end of every task |
| Voice & style guidelines | Task 11 step 5 (voice grep) + per-task verify-readback prose checks |
| Acceptance criteria | All 9 items mapped: structure (Task 1), figures (Tasks 2-9), §5 (Task 6), snippets-match-source (Task 4 step 1, Task 5 step 1, Task 6 step 3), cross-links (per-task verify), README link (Task 11), pytest green (every task), frontmatter (Task 1), final TBD grep (Task 11 step 4) |

No spec section is uncovered. No placeholders in any task ("TBD" appears only as the **literal marker** prescribed by the spec). Type / path consistency: every cited code file, snippet, table, and figure filename matches across tasks.
