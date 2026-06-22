# Phase 11 — Consolidated Initiative Doc: Design Spec

> **Status:** Approved (design); awaits implementation.
> **Parent plan:** `docs/offline_rl/iql_cql_initiative_plan.md` (Phase 11).
> **Date approved:** 2026-06-22.
> **Owner:** offline-RL initiative branch (`feature/offline-agents-implementation`).

---

## Goal

Produce a single, self-contained markdown deliverable at
`docs/offline_rl/iql_cql_initiative.md` that tells the engineering story
of the multi-building offline RL initiative end-to-end: motivation,
dataset, algorithms, resume machinery, the production-blocking OOM and
its fix, training setup, benchmark results, feature analysis insights,
reproducing instructions, limitations, and references.

The doc is written **before production completes** (~60 h from spec
approval). Numbers and KPI tables that depend on the production run
use TBD placeholders that are filled in during Phase 12.

---

## Scope

In scope:

- Write `docs/offline_rl/iql_cql_initiative.md` (~620 lines, engineering-
  portfolio depth, narrative thesis-style voice modelled on
  `docs/offline_rl/thesis_notes.md`).
- Embed all 11 curated figures from
  `docs/offline_rl/iql_cql_figures/` (produced by Phase 10).
- Include a NEW §5 "Engineering — the CityLearn OOM" covering Bug 7's
  symptom, root cause, fix, and validation (commits `98f7944` +
  `f5238be`).
- Add a one-line cross-link from `docs/offline_rl/README.md` to the new
  doc.
- TBD-placeholder convention so production-dependent numbers can be
  filled in mechanically post-completion.

Out of scope:

- Rewriting `docs/offline_rl/README.md` to drop its Building-5 framing
  (left as historical record).
- Updating `docs/offline_rl/thesis_notes.md` with multi-building content
  (it stays as the Building-5 iteration's notes).
- Re-running production, regenerating figures, or modifying Phase 10's
  curator.
- Touching the thesis itself
  (`/Users/guilherme.desousa/MEIA/Thesis/Project/meia-thesis-1211073/thesis/`).

---

## Inputs

```
docs/offline_rl/
├── iql_cql_initiative_plan.md             # parent plan (locks section list)
├── phase10_curation_design.md             # 11-figure list + provenance
├── thesis_notes.md                        # voice + tone reference
├── dataset_schema.md                      # cross-link target (§2)
├── kpi_reference.md                       # cross-link target (§7)
├── reward_design.md                       # cross-link target (§2)
├── iql_reference.md                       # cross-link target (§3)
├── feature_analysis/feature_analysis.md   # cross-link target (§8)
└── iql_cql_figures/                       # 11 PNGs embedded by §1-§8
    ├── 01_pipeline_overview.png
    ├── 02_dataset_stats.png
    ├── 03_action_coverage_group_a.png
    ├── 04_reward_by_regime.png
    ├── 05_correlations_group_a.png
    ├── 06_temporal_patterns.png
    ├── 07_training_loss_group_a.png
    ├── 08_training_valmse_all.png
    ├── 09_training_cql_penalty.png
    ├── 10_benchmark_kpi_bars.png
    └── 11_iql_vs_cql_scatter.png

algorithms/offline_rl/
├── iql_entity_trainer.py                  # IQLTrainingConfig snippet (§3)
├── cql_entity_trainer.py                  # CQLTrainingConfig snippet (§3)
└── checkpoint_utils.py                    # atomic_save + status writer (§4)

scripts/
└── show_pipeline_status.py                # CLI output sample (§4)

utils/
└── citylearn_patches.py                   # _bound_dict_size + wrapper (§5)

runs/offline_iql_cql_initiative_15min/     # production artifacts (§7, §9; TBD until complete)
```

---

## Outputs

```
docs/offline_rl/
├── iql_cql_initiative.md      # NEW — ~620 lines
└── README.md                  # touched — one cross-link line added
```

The new doc embeds all 11 curated figures by relative path
(`iql_cql_figures/<filename>.png`).

---

## Section structure (locked)

The parent plan locks 10 sections; Phase 11 design inserts a NEW §5
"Engineering — the CityLearn OOM" between §4 (Resume) and §6 (Training
setup), renumbering §5-§10 from the plan to §6-§11.

| §  | Title                              | Lines  | Voice    | Origin           |
|---:|------------------------------------|-------:|----------|------------------|
| 1  | Motivation & scope                 | ~50    | narrative| plan §1          |
| 2  | Dataset                            | ~70    | narrative+tables | plan §2  |
| 3  | Algorithms (IQL + CQL)             | ~70    | reference+narrative | plan §3 |
| 4  | Resume & status visibility         | ~80    | reference+narrative | plan §4 |
| 5  | **Engineering — the CityLearn OOM**| ~80    | narrative| **NEW (Bug 7)**  |
| 6  | Training setup                     | ~60    | reference+narrative | plan §5 |
| 7  | Benchmark results                  | ~70    | tables+narrative | plan §6 |
| 8  | Feature analysis highlights        | ~50    | narrative+figures | plan §7|
| 9  | Reproducing                        | ~40    | reference| plan §8          |
| 10 | Limitations                        | ~30    | narrative| plan §9          |
| 11 | References                         | ~20    | reference| plan §10         |

**Total**: ~620 lines plus ~15-line frontmatter (title + status + branch +
parent-plan link + TOC) = **~635 lines**, within the
500-700-line engineering-portfolio target.

---

## Figure binding

All 11 curated figures from Phase 10 are placed; each appears exactly
once. Sections without curated figures use tables, code blocks, or
inline bespoke artefacts.

| § | Figure(s)                                                                       |
|---|---------------------------------------------------------------------------------|
| 1 | `01_pipeline_overview.png`                                                       |
| 2 | `02_dataset_stats.png`, `03_action_coverage_group_a.png`, `04_reward_by_regime.png`, `06_temporal_patterns.png` |
| 3 | `09_training_cql_penalty.png` (motivates `α=0.2`)                                |
| 4 | — (code blocks + status.json schema carry it)                                    |
| 5 | bespoke before/after RSS table (no curated PNG)                                  |
| 6 | `07_training_loss_group_a.png`, `08_training_valmse_all.png`                     |
| 7 | `10_benchmark_kpi_bars.png`, `11_iql_vs_cql_scatter.png`                         |
| 8 | `05_correlations_group_a.png`                                                    |
| 9 | — (commands + tree)                                                              |
|10 | —                                                                                |
|11 | —                                                                                |

All embeds use the form `![<caption>](iql_cql_figures/<file>.png)`. When
a figure is missing at write time (Phase 10 not yet run), the embed is
preceded by `<!-- TBD: production -->`.

---

## Code snippet inventory

Eight code blocks total; each kept to 10-25 lines and excerpted (not
copy-pasted whole) from the source to avoid drift.

| § | Snippet purpose                              | Source file                                          |
|---|----------------------------------------------|------------------------------------------------------|
| 3 | `IQLTrainingConfig` field excerpt            | `algorithms/offline_rl/iql_entity_trainer.py`        |
| 3 | `CQLTrainingConfig` excerpt highlighting `cql_alpha=0.2` | `algorithms/offline_rl/cql_entity_trainer.py` |
| 4 | `checkpoint_latest.pt` dict schema           | `algorithms/offline_rl/checkpoint_utils.py`          |
| 4 | `status.json` payload example                | live `runs/.../status.json` (sanitised)              |
| 4 | `show_pipeline_status` sample output         | `scripts/show_pipeline_status.py`                    |
| 5 | `_bound_dict_size` + wrapper signature       | `utils/citylearn_patches.py`                         |
| 9 | Launch command (`nohup … run_entity_pipeline …`) | this initiative                                  |
| 9 | Resume command (same single command)         | this initiative                                      |

Snippets must be re-runnable verbatim where applicable (commands), and
must match current source (verified at write time, re-verified at
Phase 12).

---

## §5 "Engineering — the CityLearn OOM" — detailed outline

The new section follows the narrative-thesis pattern from
`docs/offline_rl/thesis_notes.md:36-38` ("The namespace bug"). Roughly
~3500 chars across 7 subsections.

| # | Subsection            | Lines | Content                                                                                                                                          |
|---|-----------------------|------:|--------------------------------------------------------------------------------------------------------------------------------------------------|
|5.1| The crash             | ~10   | 2026-06-21 launch, killed at step ~16000 of 35040 after 7906s. Exit -9 (SIGKILL). No traceback. RSS ~20 GB before kernel intervened. Why 15-min full-year made it observable. |
|5.2| Probing memory        | ~20   | Three probes outside production: (a) RSS sampler — ~8 MB/step from step 200; (b) `tracemalloc` snapshot diff — `citylearn/energy_model.py:119`; (c) `env.reset()` between cycles — no RSS release. Class-level not episode-local. |
|5.3| Root cause            | ~15   | `CityLearnEntityInterfaceService._action_feedback_series_summary` (in `citylearn/internal/entity_interface.py:1716`) memoises per-step summary keyed on `id()` of source array. New ndarrays each step → cache key never matches → unbounded growth (~85 entries × #steps). |
|5.4| Decision matrix       | ~15   | Three options: (a) subprocess chunking, (b) monkey-patch CityLearn cache, (c) hybrid. Trade-off table (invasiveness, data-continuity risk, performance). Chose (b) — idempotent, no data risk, ~90 lines. |
|5.5| The fix               | ~15   | `utils/citylearn_patches.apply_citylearn_patches()` wraps original method with `_bound_dict_size(cache, 128)` (FIFO via dict insertion order). Module-level call at import time of every script that constructs `CityLearnEnv` (`collect_rbcsmart_dataset.py`, `benchmark_entity_agents.py`). Code block. |
|5.6| Validation            | ~10   | Probe v1 rerun: growth 8 MB/step → 0.1 MB/step (80×). RSS at step 2000: 16 GB → 959 MB (17×). Projected at step 35040: ~4 GB (within 32 GB envelope). Before/after RSS table at steps 200, 2000, 35040(proj). |
|5.7| Lesson                | ~5    | "Downstream library invariants matter. Memoisation tables that look harmless can OOM when called at fine temporal granularity. The patch is small because we only had to bound one dict." |

Tables in §5:
- 5.4: 3-option decision matrix (option × invasiveness, data risk, perf impact, complexity).
- 5.6: before/after RSS at three sample steps (step, pre-patch RSS, post-patch RSS, ratio).

Cross-references: commits `98f7944` + `f5238be`, files
`utils/citylearn_patches.py:1-90`, `tests/test_citylearn_patches.py`.

---

## Cross-link inventory

Every internal cross-reference uses a relative repo path (and a line
number when pointing to a specific symbol):

| Target                                                            | Where it's cited           |
|-------------------------------------------------------------------|----------------------------|
| `docs/offline_rl/dataset_schema.md`                               | §2                         |
| `docs/offline_rl/kpi_reference.md`                                | §7                         |
| `docs/offline_rl/reward_design.md`                                | §2                         |
| `docs/offline_rl/iql_reference.md`                                | §3                         |
| `docs/offline_rl/feature_analysis/feature_analysis.md`            | §8                         |
| `docs/offline_rl/iql_cql_initiative_plan.md`                      | frontmatter, §11           |
| `docs/offline_rl/phase10_curation_design.md`                      | §11                        |
| `docs/offline_rl/phase11_consolidated_doc_design.md` (this file)  | §11                        |
| `runs/offline_iql_cql_initiative_15min/`                          | §9 (tree), §7 (numbers)    |
| `algorithms/offline_rl/iql_entity_trainer.py`                     | §3 (config), §4 (resume)   |
| `algorithms/offline_rl/cql_entity_trainer.py`                     | §3 (config)                |
| `algorithms/offline_rl/checkpoint_utils.py`                       | §4                         |
| `scripts/show_pipeline_status.py`                                 | §4, §9                     |
| `utils/citylearn_patches.py`                                      | §5                         |
| Thesis Ch4/5/6 at `/Users/.../meia-thesis-1211073/thesis/`        | §11                        |
| IQL paper (Kostrikov et al., 2021)                                | §3                         |
| CQL paper (Kumar et al., 2020)                                    | §3                         |
| CityLearn paper (Vázquez-Canteli et al., 2020)                    | §11                        |

---

## TBD-placeholder convention

Production-dependent content is marked so a single grep finds every
fill-in site post-Phase 9 completion.

- **HTML comment marker** preceding TBD content:
  ```
  <!-- TBD: production -->
  ```
- **Numeric placeholders** in tables: `**TBD**`; mean±std as
  `TBD ± TBD`; counts as `TBD`.
- **Figure references** where the figure hasn't been rendered yet:
  italic prose `_(figure pending production run)_` followed by the
  intended embed line in the same paragraph (kept consistent so the
  embed activates once the file appears).
- **Numbers known from probes / smoke / Bug 7 validation** (e.g. RSS
  table in §5) are filled in immediately — no TBD needed.

Phase 12 acceptance check: `grep -nE "TBD( |:|\\*)" docs/offline_rl/iql_cql_initiative.md`
returns zero matches before final commit.

---

## README cross-link

Single line added to `docs/offline_rl/README.md` under "Where things
stand":

> The next stage extends offline RL to all 17 buildings with IQL + CQL
> on the 15-min schema. See
> [`iql_cql_initiative.md`](iql_cql_initiative.md) for that initiative.

No other content in `README.md` changes; the Building-5 framing stays as
historical record.

---

## Testing strategy

This is a markdown writing task, not code, so the test strategy is
verification rather than unit testing.

1. **Markdown lint** (optional, if `mdformat`/`mdl` available): run
   over the new file and confirm zero parse errors.
2. **Cross-link sanity**: every relative path referenced exists at
   write time (figures are the only exception — they're TBD until
   Phase 10 runs).
3. **Snippet drift check**: every code excerpt in the doc matches its
   source file (manual diff during writing, re-checked at Phase 12).
4. **Existing test suite**: `pytest -q` remains green (the doc work
   touches no Python code).
5. **Phase 12 grep**: `grep -nE "TBD" docs/offline_rl/iql_cql_initiative.md`
   returns zero before final commit.

No new pytest tests are required for this phase.

---

## Voice & style guidelines

- Narrative thesis-style, first-person plural ("we", "the initiative")
  when describing decisions; passive when describing the system. Model:
  `docs/offline_rl/thesis_notes.md`.
- Each section opens with 2-3 sentences of context before any table /
  figure / code block.
- "The X bug / problem / finding" titled subsections in §5 echo the
  thesis-notes style ("The namespace bug", "The collinearity problem").
- Tables get a header row and a footnote when a column needs
  explanation; no header colspans.
- Code blocks include language hints (` ```python ` etc.) and end with
  a comment `# from path/to/file.py:Lstart-Lend` for traceability.
- Figures get a one-sentence prose caption above the embed, not Markdown
  alt-text-only.
- No emojis. No exclamation marks in body text.

---

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Production produces unexpected numbers that contradict prose direction (e.g., CQL underperforms IQL) | Voice is fact-based, not advocacy. Prose describes the methodology and expected behaviour; numbers are filled in literally during Phase 12 with whatever they actually are. |
| Phase 10 (figure curation) hasn't run yet when we write the doc | Embed lines use relative paths that will resolve once curator runs; figures are TBD-marked until then. |
| Stale code snippet drift (e.g., `IQLTrainingConfig` fields change after we excerpt them) | Parent plan's Phase 12 ("Final verification & commit") re-verifies every snippet against source before commit. |
| Bug 7 fix is reverted or amended before doc is finalised | §5 is anchored on commits `98f7944` + `f5238be`. If those change, §5 is updated to match HEAD. |
| Reader expects a thesis-chapter (book-length) and the doc reads as a repo deliverable | Frontmatter explicitly states scope ("self-contained engineering note", not a chapter); §11 cross-links to thesis Ch4/5/6 for the academic treatment. |
| Voice inconsistency across 11 sections | Produced by the same author in coherent passes (no multi-author handoff); a final self-review pass before commit enforces consistency. |
| `README.md` cross-link is added but never visited because the README still says "Building 5" | Out of scope per user decision; flagged in §10 Limitations of the new doc as a known stale upstream. |

---

## Acceptance criteria

- [ ] `docs/offline_rl/iql_cql_initiative.md` exists, ~500-700 lines,
      11 sections as locked above.
- [ ] All 11 curated figures referenced exactly once each.
- [ ] §5 "Engineering — the CityLearn OOM" covers all 7 subsections.
- [ ] Every code snippet matches its source at write time.
- [ ] Every cross-link target exists (figures TBD-OK; everything else
      hard).
- [ ] `docs/offline_rl/README.md` has the one-line cross-link added,
      nothing else changed.
- [ ] `pytest -q` remains green (no Python code modified).
- [ ] Frontmatter status is honest: "implementation note (production in
      flight)" while TBDs remain; updated to "complete" in Phase 12.
- [ ] After Phase 9 (production) completes and Phase 10 (curator) runs:
      `grep -nE "TBD" docs/offline_rl/iql_cql_initiative.md` returns
      zero matches.

---

## Open questions

None at design-spec time. The four design decisions
(production-data timing, voice, Bug 7 placement, scope) were resolved
during brainstorming on 2026-06-22 and locked in this document.
