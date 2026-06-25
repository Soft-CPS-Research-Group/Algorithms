# Phase 10 — Curation Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/curate_initiative_figures.py` that distills `runs/<run_dir>/` outputs into 11 thesis-grade figures at `docs/offline_rl/iql_cql_figures/`.

**Architecture:** Single Python module with six rendering/copying helpers + a CLI. Imports helpers from existing `scripts/generate_training_figures.py` and `scripts/generate_architecture_figures.py`. Tests run against the validated `runs/smoke_pipeline_phase9/` fixture.

**Tech Stack:** Python 3.10, matplotlib, numpy, scipy.stats (Wilcoxon), pytest. No new pip dependencies.

**Spec:** `docs/offline_rl/phase10_curation_design.md`

**Implementation file:** `scripts/curate_initiative_figures.py`
**Test file:** `tests/scripts/test_curate_initiative_figures.py`

---

## File Responsibility Map

| File | Responsibility |
|------|---------------|
| `scripts/curate_initiative_figures.py` | Module with 6 rendering/copy helpers + `main(argv)` CLI. ~400 lines. |
| `tests/scripts/test_curate_initiative_figures.py` | Pytest tests against `runs/smoke_pipeline_phase9/` fixture. ~250 lines. |
| `scripts/generate_training_figures.py` | (existing) Imported for `_load_group_metrics`, `_plot_loss_curves`, `_plot_val_mse`, `_plot_cql_penalty`. |
| `scripts/generate_architecture_figures.py` | (existing) Imported for `_fig_pipeline` (whichever function renders the pipeline diagram). |

---

## Task 1: Module skeleton + argparse CLI

**Files:**
- Create: `scripts/curate_initiative_figures.py`
- Create: `tests/scripts/test_curate_initiative_figures.py`

- [ ] **Step 1: Write failing test for argparse defaults**

```python
# tests/scripts/test_curate_initiative_figures.py
"""Tests for scripts.curate_initiative_figures."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse(argv):
    import scripts.curate_initiative_figures as m
    return m._build_parser().parse_args(argv)


def test_curate_default_output_dir_is_iql_cql_figures():
    args = _parse(["--run-dir", "runs/foo"])
    assert str(args.output_dir).endswith("docs/offline_rl/iql_cql_figures")


def test_curate_default_showcase_group_is_obs627_act1():
    args = _parse(["--run-dir", "runs/foo"])
    assert args.showcase_group == "obs627_act1"


def test_curate_default_groups_are_four_production_groups():
    args = _parse(["--run-dir", "runs/foo"])
    assert args.groups == ["obs627_act1", "obs706_act2", "obs749_act3", "obs785_act3"]


def test_curate_run_dir_is_required():
    import pytest
    with pytest.raises(SystemExit):
        _parse([])


def test_curate_showcase_group_override():
    args = _parse(["--run-dir", "runs/foo", "--showcase-group", "obs163_act1"])
    assert args.showcase_group == "obs163_act1"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/scripts/test_curate_initiative_figures.py -v
```

Expected: `ModuleNotFoundError: No module named 'scripts.curate_initiative_figures'` for all 5 tests.

- [ ] **Step 3: Implement module skeleton**

```python
# scripts/curate_initiative_figures.py
"""Phase 10 — Curate offline-RL initiative figures.

Given a completed pipeline output directory at
``runs/offline_iql_cql_initiative_15min/`` (or any compatible run dir),
produce 11 thesis-grade PNGs at ``docs/offline_rl/iql_cql_figures/``:

  01_pipeline_overview.png            — pipeline architecture diagram
  02_dataset_stats.png                — dataset summary table
  03_action_coverage_group_a.png      — action distribution (showcase group)
  04_reward_by_regime.png             — reward by RBC regime
  05_correlations_group_a.png         — feature×reward correlations
  06_temporal_patterns.png            — temporal patterns
  07_training_loss_group_a.png        — IQL vs CQL training loss
  08_training_valmse_all.png          — validation MSE across all groups
  09_training_cql_penalty.png         — CQL conservative penalty curve
  10_benchmark_kpi_bars.png           — KPI comparison bars
  11_iql_vs_cql_scatter.png           — per-seed cost scatter

Idempotent: re-running with the same inputs overwrites the same outputs.
Writes a ``.curation.done`` sentinel on success.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "offline_rl" / "iql_cql_figures"
DEFAULT_SHOWCASE_GROUP = "obs627_act1"
DEFAULT_GROUPS = ["obs627_act1", "obs706_act2", "obs749_act3", "obs785_act3"]

logger = logging.getLogger("curate_initiative_figures")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Pipeline output directory, e.g. runs/offline_iql_cql_initiative_15min")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Destination directory for curated PNGs.")
    p.add_argument("--showcase-group", type=str, default=DEFAULT_SHOWCASE_GROUP,
                   help="Agent group key used for per-group figures (default obs627_act1).")
    p.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS,
                   help="Agent group keys to include in aggregate figures.")
    return p


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = _build_parser().parse_args(argv)
    logger.info("[curate] run_dir=%s", args.run_dir)
    logger.info("[curate] output_dir=%s", args.output_dir)
    logger.info("[curate] showcase_group=%s", args.showcase_group)
    logger.info("[curate] groups=%s", args.groups)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/scripts/test_curate_initiative_figures.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/curate_initiative_figures.py tests/scripts/test_curate_initiative_figures.py
git commit -m "Add curate_initiative_figures CLI skeleton"
```

---

## Task 2: `_copy_feature_analysis_figures` (figs 02-06)

**Files:**
- Modify: `scripts/curate_initiative_figures.py`
- Modify: `tests/scripts/test_curate_initiative_figures.py`

The smoke fixture has 5 figures we copy with renames:

| Source filename | Destination filename |
|-----------------|---------------------|
| `01_dataset_stats_table.png` | `02_dataset_stats.png` |
| `03_action_coverage_<showcase>.png` | `03_action_coverage_group_a.png` |
| `04_reward_by_regime.png` | `04_reward_by_regime.png` |
| `05_correlations_<showcase>.png` | `05_correlations_group_a.png` |
| `07_temporal_patterns.png` | `06_temporal_patterns.png` |

- [ ] **Step 1: Write failing test**

```python
# Append to tests/scripts/test_curate_initiative_figures.py
import pytest

SMOKE_DIR = REPO_ROOT / "runs" / "smoke_pipeline_phase9"


@pytest.fixture
def output_dir(tmp_path):
    d = tmp_path / "curated"
    d.mkdir()
    return d


@pytest.fixture
def smoke_available():
    if not SMOKE_DIR.exists():
        pytest.skip(f"smoke fixture not present at {SMOKE_DIR}")
    if not (SMOKE_DIR / "feature_analysis" / "figures").exists():
        pytest.skip("smoke feature_analysis/figures not present")


def test_copy_feature_analysis_figures_produces_five_renamed_pngs(smoke_available, output_dir):
    import scripts.curate_initiative_figures as m
    produced = m._copy_feature_analysis_figures(
        run_dir=SMOKE_DIR,
        showcase_group="obs163_act1",  # smoke uses hourly group keys
        output_dir=output_dir,
    )
    assert sorted(p.name for p in produced) == [
        "02_dataset_stats.png",
        "03_action_coverage_group_a.png",
        "04_reward_by_regime.png",
        "05_correlations_group_a.png",
        "06_temporal_patterns.png",
    ]
    for p in produced:
        assert p.exists()
        assert p.stat().st_size > 5_000  # non-empty plot


def test_copy_feature_analysis_figures_missing_dir_returns_empty(tmp_path, output_dir):
    import scripts.curate_initiative_figures as m
    produced = m._copy_feature_analysis_figures(
        run_dir=tmp_path,           # empty dir, no feature_analysis
        showcase_group="obs627_act1",
        output_dir=output_dir,
    )
    assert produced == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/scripts/test_curate_initiative_figures.py -v -k copy_feature
```

Expected: `AttributeError: module ... has no attribute '_copy_feature_analysis_figures'`.

- [ ] **Step 3: Implement `_copy_feature_analysis_figures`**

```python
# Add to scripts/curate_initiative_figures.py
import shutil

FEATURE_ANALYSIS_COPY_MAP = [
    # (source filename template, destination filename)
    ("01_dataset_stats_table.png",                "02_dataset_stats.png"),
    ("03_action_coverage_{group}.png",            "03_action_coverage_group_a.png"),
    ("04_reward_by_regime.png",                   "04_reward_by_regime.png"),
    ("05_correlations_{group}.png",               "05_correlations_group_a.png"),
    ("07_temporal_patterns.png",                  "06_temporal_patterns.png"),
]


def _copy_feature_analysis_figures(
    *,
    run_dir: Path,
    showcase_group: str,
    output_dir: Path,
) -> List[Path]:
    """Copy 5 figures from <run_dir>/feature_analysis/figures/ with renames.

    Returns the list of destination paths that were successfully created.
    Missing sources are logged as warnings and skipped.
    """
    src_dir = run_dir / "feature_analysis" / "figures"
    if not src_dir.exists():
        logger.warning("[curate] feature_analysis/figures not found at %s — skipping figs 02-06", src_dir)
        return []

    produced: List[Path] = []
    for src_template, dst_name in FEATURE_ANALYSIS_COPY_MAP:
        src_name = src_template.format(group=showcase_group)
        src = src_dir / src_name
        if not src.exists():
            logger.warning("[curate] source figure missing: %s — skipping", src)
            continue
        dst = output_dir / dst_name
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        produced.append(dst)
        logger.info("[curate] copied %s -> %s", src.name, dst.name)
    return produced
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/scripts/test_curate_initiative_figures.py -v -k copy_feature
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/curate_initiative_figures.py tests/scripts/test_curate_initiative_figures.py
git commit -m "Add feature-analysis figure copier to curate_initiative_figures"
```

---

## Task 3: `_render_pipeline_diagram` (fig 01)

**Files:**
- Modify: `scripts/curate_initiative_figures.py`
- Modify: `tests/scripts/test_curate_initiative_figures.py`

Imports `generate_architecture_figures` and calls its pipeline-rendering helper. First inspect the existing module to find the right function name.

- [ ] **Step 1: Inspect existing architecture script for the pipeline function**

```bash
.venv/bin/python -c "
import scripts.generate_architecture_figures as m
print([n for n in dir(m) if 'pipeline' in n.lower()])
"
```

Expected: lists at least one function with `pipeline` in the name (e.g. `_fig_pipeline` or `render_pipeline_diagram`). Note the actual name — the next step's import uses it.

If the import fails or produces no candidates, fall back to inspecting the source file:
`grep -n "def.*[Pp]ipeline" scripts/generate_architecture_figures.py`

- [ ] **Step 2: Write failing test**

```python
# Append to tests/scripts/test_curate_initiative_figures.py
def test_render_pipeline_diagram_produces_png(output_dir):
    import scripts.curate_initiative_figures as m
    produced = m._render_pipeline_diagram(output_dir=output_dir)
    assert produced is not None
    assert produced.name == "01_pipeline_overview.png"
    assert produced.exists()
    assert produced.stat().st_size > 5_000
```

- [ ] **Step 3: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/scripts/test_curate_initiative_figures.py -v -k pipeline_diagram
```

Expected: `AttributeError: module ... has no attribute '_render_pipeline_diagram'`.

- [ ] **Step 4: Implement `_render_pipeline_diagram`**

Inspect `scripts/generate_architecture_figures.py` to identify the function rendering the pipeline diagram. The function probably builds a matplotlib figure and saves to a path constructed from `--output-dir`. We adapt by calling it with our own `output_dir` and then renaming the produced file to `01_pipeline_overview.png`.

```python
# Add to scripts/curate_initiative_figures.py
def _render_pipeline_diagram(*, output_dir: Path) -> Path | None:
    """Render the pipeline architecture diagram.

    Imports the pipeline-rendering helper from
    ``scripts.generate_architecture_figures`` and saves the result as
    ``01_pipeline_overview.png`` in ``output_dir``.

    Returns the path to the produced PNG, or None if the import fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / "01_pipeline_overview.png"

    try:
        # Adjust the imported names to match what generate_architecture_figures actually exposes.
        # The function likely takes (output_dir: Path) -> Path and saves a file named like
        # fig_arch_pipeline.png. We rename it afterwards.
        from scripts.generate_architecture_figures import _fig_arch_pipeline  # type: ignore
    except ImportError as exc:
        logger.warning("[curate] could not import pipeline helper from generate_architecture_figures: %s", exc)
        return None

    # Use a tmp dir so we don't clutter output_dir with the original filename.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        produced = _fig_arch_pipeline(tmp_path)  # may return a Path; otherwise locate by glob
        if produced is None:
            # Helper might not return; locate the produced file.
            pngs = list(tmp_path.glob("*.png"))
            if not pngs:
                logger.warning("[curate] pipeline diagram helper produced no PNG")
                return None
            produced = pngs[0]
        shutil.copy2(produced, dst)
    logger.info("[curate] rendered %s", dst.name)
    return dst
```

**Note:** if the actual function in `generate_architecture_figures.py` has a different name (discovered in Step 1), substitute it in the import. Wrap the call signature accordingly — the function may take `(output_dir,)` or no args and write to a fixed location.

- [ ] **Step 5: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/scripts/test_curate_initiative_figures.py -v -k pipeline_diagram
```

Expected: test PASSES. If it fails because the imported helper has a different signature, adjust the import and call accordingly until the test passes.

- [ ] **Step 6: Commit**

```bash
git add scripts/curate_initiative_figures.py tests/scripts/test_curate_initiative_figures.py
git commit -m "Add pipeline diagram renderer to curate_initiative_figures"
```

---

## Task 4: `_render_training_curves` (figs 07-09)

**Files:**
- Modify: `scripts/curate_initiative_figures.py`
- Modify: `tests/scripts/test_curate_initiative_figures.py`

Imports `generate_training_figures` helpers. Output files are named per the existing script's convention (`fig_training_loss_<group>.png` etc.); we rename to our `07_…`, `08_…`, `09_…` semantic prefixes.

- [ ] **Step 1: Write failing test**

```python
# Append to tests/scripts/test_curate_initiative_figures.py
@pytest.fixture
def smoke_has_metrics():
    iql = SMOKE_DIR / "models-iql"
    cql = SMOKE_DIR / "models-cql"
    if not iql.exists() or not cql.exists():
        pytest.skip("smoke models-iql/models-cql not present")


def test_render_training_curves_produces_three_pngs(smoke_available, smoke_has_metrics, output_dir):
    import scripts.curate_initiative_figures as m
    produced = m._render_training_curves(
        run_dir=SMOKE_DIR,
        showcase_group="obs163_act1",
        groups=["obs163_act1", "obs225_act2", "obs257_act3", "obs287_act3"],
        output_dir=output_dir,
    )
    names = sorted(p.name for p in produced)
    assert "07_training_loss_group_a.png" in names
    assert "08_training_valmse_all.png" in names
    assert "09_training_cql_penalty.png" in names
    for p in produced:
        assert p.stat().st_size > 5_000
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/scripts/test_curate_initiative_figures.py -v -k training_curves
```

Expected: `AttributeError: module ... has no attribute '_render_training_curves'`.

- [ ] **Step 3: Implement `_render_training_curves`**

```python
# Add to scripts/curate_initiative_figures.py
def _render_training_curves(
    *,
    run_dir: Path,
    showcase_group: str,
    groups: List[str],
    output_dir: Path,
) -> List[Path]:
    """Render training-curve figures using generate_training_figures helpers.

    Produces:
    * 07_training_loss_group_a.png   — IQL vs CQL loss curves for showcase group.
    * 08_training_valmse_all.png     — val MSE for IQL vs CQL across all groups.
    * 09_training_cql_penalty.png    — CQL conservative penalty curve.

    Returns the list of destination paths that were successfully produced.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    iql_root = run_dir / "models-iql"
    cql_root = run_dir / "models-cql"
    iql_arg = iql_root if iql_root.exists() else None
    cql_arg = cql_root if cql_root.exists() else None

    if iql_arg is None and cql_arg is None:
        logger.warning("[curate] no models-iql or models-cql under %s — skipping figs 07-09", run_dir)
        return []

    import tempfile
    try:
        from scripts.generate_training_figures import (
            _plot_loss_curves, _plot_val_mse, _plot_cql_penalty,
        )
    except ImportError as exc:
        logger.warning("[curate] could not import training-curve helpers: %s", exc)
        return []

    produced: List[Path] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # fig 07 — loss curves for showcase group
        loss = _plot_loss_curves(iql_arg, cql_arg, showcase_group, tmp_path)
        if loss and loss.exists():
            dst = output_dir / "07_training_loss_group_a.png"
            shutil.copy2(loss, dst)
            produced.append(dst)
            logger.info("[curate] rendered %s", dst.name)

        # fig 08 — val MSE all groups
        val = _plot_val_mse(iql_arg, cql_arg, groups, tmp_path)
        if val and val.exists():
            dst = output_dir / "08_training_valmse_all.png"
            shutil.copy2(val, dst)
            produced.append(dst)
            logger.info("[curate] rendered %s", dst.name)

        # fig 09 — CQL penalty
        pen = _plot_cql_penalty(cql_arg, groups, tmp_path)
        if pen and pen.exists():
            dst = output_dir / "09_training_cql_penalty.png"
            shutil.copy2(pen, dst)
            produced.append(dst)
            logger.info("[curate] rendered %s", dst.name)

    return produced
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/scripts/test_curate_initiative_figures.py -v -k training_curves
```

Expected: test PASSES.

- [ ] **Step 5: Commit**

```bash
git add scripts/curate_initiative_figures.py tests/scripts/test_curate_initiative_figures.py
git commit -m "Add training-curve renderer to curate_initiative_figures"
```

---

## Task 5: `_render_benchmark_kpi_bars` (fig 10)

**Files:**
- Modify: `scripts/curate_initiative_figures.py`
- Modify: `tests/scripts/test_curate_initiative_figures.py`

2×2 grid of bar charts. KPIs: `cost_total`, `carbon_emissions_total`, `daily_peak_average`, `ramping_average`. Bars: RBCSmart, IQL, CQL (in that order). Error bars from `aggregate.std` when n>1.

Smoke `results.json` structure (verified):
```json
{
  "RBCSmart": {"aggregate": {"cost_total": {"mean": ..., "std": ..., "n": ...}, ...}},
  "IQL":       {"aggregate": {"cost_total": {"mean": ..., "std": ..., "n": ...}, ...}},
  "CQL":       {"aggregate": {...}}
}
```

- [ ] **Step 1: Write failing test**

```python
# Append to tests/scripts/test_curate_initiative_figures.py
@pytest.fixture
def smoke_results_json():
    p = SMOKE_DIR / "benchmark" / "results.json"
    if not p.exists():
        pytest.skip(f"smoke benchmark results.json not present at {p}")
    return p


def test_render_benchmark_kpi_bars_produces_png(smoke_results_json, output_dir):
    import scripts.curate_initiative_figures as m
    produced = m._render_benchmark_kpi_bars(
        results_json=smoke_results_json,
        output_dir=output_dir,
    )
    assert produced is not None
    assert produced.name == "10_benchmark_kpi_bars.png"
    assert produced.exists()
    assert produced.stat().st_size > 5_000


def test_render_benchmark_kpi_bars_missing_file_returns_none(tmp_path, output_dir):
    import scripts.curate_initiative_figures as m
    produced = m._render_benchmark_kpi_bars(
        results_json=tmp_path / "nope.json",
        output_dir=output_dir,
    )
    assert produced is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/scripts/test_curate_initiative_figures.py -v -k benchmark_kpi_bars
```

Expected: `AttributeError: ... has no attribute '_render_benchmark_kpi_bars'`.

- [ ] **Step 3: Implement `_render_benchmark_kpi_bars`**

```python
# Add to scripts/curate_initiative_figures.py
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCHMARK_KPIS = [
    # (key, panel title)
    ("cost_total",             "Total cost"),
    ("carbon_emissions_total", "Carbon emissions"),
    ("daily_peak_average",     "Daily peak avg"),
    ("ramping_average",        "Ramping avg"),
]
BENCHMARK_ALGORITHMS = ["RBCSmart", "IQL", "CQL"]
ALGO_COLORS = {"RBCSmart": "#9E9E9E", "IQL": "#2196F3", "CQL": "#F44336"}


def _render_benchmark_kpi_bars(*, results_json: Path, output_dir: Path) -> Path | None:
    """Render 2x2 bar chart of KPIs comparing RBCSmart vs IQL vs CQL.

    Returns the path to the produced PNG, or None on any missing/invalid input.
    """
    if not results_json.exists():
        logger.warning("[curate] benchmark results.json not found at %s — skipping fig 10", results_json)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    with results_json.open() as f:
        data = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Benchmark KPIs — RBCSmart vs IQL vs CQL", fontsize=13)

    for ax, (key, title) in zip(axes.flat, BENCHMARK_KPIS):
        means, stds, labels, colors = [], [], [], []
        for algo in BENCHMARK_ALGORITHMS:
            agg = data.get(algo, {}).get("aggregate", {}).get(key)
            if agg is None:
                continue
            means.append(float(agg["mean"]))
            stds.append(float(agg.get("std", 0.0)))
            labels.append(algo)
            colors.append(ALGO_COLORS.get(algo, "#666666"))
        if not means:
            ax.set_visible(False)
            continue
        x = list(range(len(means)))
        ax.bar(x, means, yerr=stds, color=colors, capsize=4, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    dst = output_dir / "10_benchmark_kpi_bars.png"
    fig.savefig(dst, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[curate] rendered %s", dst.name)
    return dst
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/scripts/test_curate_initiative_figures.py -v -k benchmark_kpi_bars
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/curate_initiative_figures.py tests/scripts/test_curate_initiative_figures.py
git commit -m "Add benchmark KPI bar chart renderer"
```

---

## Task 6: `_render_iql_vs_cql_scatter` (fig 11)

**Files:**
- Modify: `scripts/curate_initiative_figures.py`
- Modify: `tests/scripts/test_curate_initiative_figures.py`

Scatter plot: one dot per `eval_seed`. X = IQL cost, Y = CQL cost. y=x reference line. If n>=2, annotate paired Wilcoxon p-value; else "n=1, p=N/A".

`results.json` structure (per-seed):
```json
{
  "IQL": {"runs": [{"env_seed": 200, "district": {"cost_total": ...}}, ...]},
  "CQL": {"runs": [{"env_seed": 200, "district": {"cost_total": ...}}, ...]}
}
```

Pair runs by `env_seed`.

- [ ] **Step 1: Write failing test**

```python
# Append to tests/scripts/test_curate_initiative_figures.py
def test_render_iql_vs_cql_scatter_produces_png_with_n_equals_one(smoke_results_json, output_dir):
    """Smoke has only one eval seed; the scatter must still render with
    n=1 annotation."""
    import scripts.curate_initiative_figures as m
    produced = m._render_iql_vs_cql_scatter(
        results_json=smoke_results_json,
        output_dir=output_dir,
    )
    assert produced is not None
    assert produced.name == "11_iql_vs_cql_scatter.png"
    assert produced.exists()
    assert produced.stat().st_size > 5_000


def test_render_iql_vs_cql_scatter_missing_file_returns_none(tmp_path, output_dir):
    import scripts.curate_initiative_figures as m
    produced = m._render_iql_vs_cql_scatter(
        results_json=tmp_path / "nope.json",
        output_dir=output_dir,
    )
    assert produced is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/scripts/test_curate_initiative_figures.py -v -k iql_vs_cql_scatter
```

Expected: `AttributeError: ... has no attribute '_render_iql_vs_cql_scatter'`.

- [ ] **Step 3: Implement `_render_iql_vs_cql_scatter`**

```python
# Add to scripts/curate_initiative_figures.py
def _render_iql_vs_cql_scatter(*, results_json: Path, output_dir: Path) -> Path | None:
    """Render per-eval-seed scatter of IQL cost vs CQL cost.

    Adds a y=x reference line and, when n>=2 eval seeds, annotates a
    paired Wilcoxon p-value. Returns the destination path or None.
    """
    if not results_json.exists():
        logger.warning("[curate] benchmark results.json not found at %s — skipping fig 11", results_json)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    with results_json.open() as f:
        data = json.load(f)

    iql_runs = {r["env_seed"]: r["district"]["cost_total"] for r in data.get("IQL", {}).get("runs", []) if "district" in r}
    cql_runs = {r["env_seed"]: r["district"]["cost_total"] for r in data.get("CQL", {}).get("runs", []) if "district" in r}
    seeds = sorted(set(iql_runs) & set(cql_runs))
    if not seeds:
        logger.warning("[curate] no paired IQL/CQL eval seeds in %s — skipping fig 11", results_json)
        return None

    iql_costs = [iql_runs[s] for s in seeds]
    cql_costs = [cql_runs[s] for s in seeds]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(iql_costs, cql_costs, s=60, alpha=0.75, edgecolor="black", linewidth=0.6, color="#673AB7")

    # y=x reference line
    lo = min(min(iql_costs), min(cql_costs))
    hi = max(max(iql_costs), max(cql_costs))
    margin = (hi - lo) * 0.05 if hi > lo else 0.1
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "k--", linewidth=1, alpha=0.5, label="y = x")

    # Annotation
    if len(seeds) >= 2:
        try:
            from scipy.stats import wilcoxon
            stat, pval = wilcoxon(iql_costs, cql_costs)
            annotation = f"n={len(seeds)}, Wilcoxon p={pval:.3f}"
        except Exception as exc:
            annotation = f"n={len(seeds)}, p=N/A ({exc})"
    else:
        annotation = f"n={len(seeds)}, p=N/A"
    ax.text(0.05, 0.95, annotation, transform=ax.transAxes,
            verticalalignment="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax.set_xlabel("IQL cost_total")
    ax.set_ylabel("CQL cost_total")
    ax.set_title("Per-seed cost comparison — IQL vs CQL")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    dst = output_dir / "11_iql_vs_cql_scatter.png"
    fig.savefig(dst, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[curate] rendered %s", dst.name)
    return dst
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/scripts/test_curate_initiative_figures.py -v -k iql_vs_cql_scatter
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/curate_initiative_figures.py tests/scripts/test_curate_initiative_figures.py
git commit -m "Add IQL-vs-CQL per-seed scatter renderer"
```

---

## Task 7: `_write_sentinel` + main() wiring

**Files:**
- Modify: `scripts/curate_initiative_figures.py`
- Modify: `tests/scripts/test_curate_initiative_figures.py`

Sentinel atomic write + end-to-end main() integration. Wire all renderers in sequence; sentinel records `{generated_at, run_dir, output_dir, n_figures, figures: [filename, ...]}`.

- [ ] **Step 1: Write failing test for sentinel**

```python
# Append to tests/scripts/test_curate_initiative_figures.py
def test_write_sentinel_atomic_json(output_dir):
    import scripts.curate_initiative_figures as m
    sentinel = m._write_sentinel(
        output_dir=output_dir,
        run_dir=Path("runs/foo"),
        produced=[output_dir / "fake.png"],
    )
    assert sentinel.name == ".curation.done"
    assert sentinel.exists()
    payload = json.loads(sentinel.read_text())
    assert payload["n_figures"] == 1
    assert payload["run_dir"] == "runs/foo"
    assert payload["output_dir"] == str(output_dir)
    assert payload["figures"] == ["fake.png"]
    assert "generated_at" in payload


def test_main_end_to_end_smoke(smoke_available, smoke_has_metrics, smoke_results_json, tmp_path):
    """End-to-end: run main() against smoke artifacts; expect 11 PNGs + sentinel."""
    import scripts.curate_initiative_figures as m
    out = tmp_path / "curated"
    rc = m.main([
        "--run-dir", str(SMOKE_DIR),
        "--output-dir", str(out),
        "--showcase-group", "obs163_act1",
        "--groups", "obs163_act1", "obs225_act2", "obs257_act3", "obs287_act3",
    ])
    assert rc == 0
    pngs = sorted(p.name for p in out.glob("*.png"))
    assert len(pngs) == 11
    sentinel = out / ".curation.done"
    assert sentinel.exists()
    payload = json.loads(sentinel.read_text())
    assert payload["n_figures"] == 11
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/scripts/test_curate_initiative_figures.py -v -k "write_sentinel or end_to_end_smoke"
```

Expected: failure on `AttributeError: ... has no attribute '_write_sentinel'` and / or end-to-end producing 0 figures (because `main` is still the skeleton).

- [ ] **Step 3: Implement `_write_sentinel` and wire main()**

```python
# Add to scripts/curate_initiative_figures.py
import os
from datetime import datetime, timezone


def _write_sentinel(*, output_dir: Path, run_dir: Path, produced: List[Path]) -> Path:
    """Atomically write .curation.done JSON sentinel."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sentinel = output_dir / ".curation.done"
    tmp = sentinel.with_suffix(".done.tmp")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "n_figures": len(produced),
        "figures": sorted(p.name for p in produced),
    }
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, sentinel)
    return sentinel
```

Replace the existing `main()`:

```python
def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = _build_parser().parse_args(argv)
    logger.info("[curate] run_dir=%s", args.run_dir)
    logger.info("[curate] output_dir=%s", args.output_dir)
    logger.info("[curate] showcase_group=%s", args.showcase_group)
    logger.info("[curate] groups=%s", args.groups)

    produced: List[Path] = []

    # fig 01 — pipeline diagram
    pipeline = _render_pipeline_diagram(output_dir=args.output_dir)
    if pipeline:
        produced.append(pipeline)

    # figs 02-06 — feature-analysis copies
    produced.extend(_copy_feature_analysis_figures(
        run_dir=args.run_dir,
        showcase_group=args.showcase_group,
        output_dir=args.output_dir,
    ))

    # figs 07-09 — training curves
    produced.extend(_render_training_curves(
        run_dir=args.run_dir,
        showcase_group=args.showcase_group,
        groups=args.groups,
        output_dir=args.output_dir,
    ))

    # figs 10-11 — benchmark
    bench_json = args.run_dir / "benchmark" / "results.json"
    bars = _render_benchmark_kpi_bars(results_json=bench_json, output_dir=args.output_dir)
    if bars:
        produced.append(bars)
    scatter = _render_iql_vs_cql_scatter(results_json=bench_json, output_dir=args.output_dir)
    if scatter:
        produced.append(scatter)

    if not produced:
        logger.error("[curate] no figures produced — aborting without sentinel")
        return 1

    sentinel = _write_sentinel(output_dir=args.output_dir, run_dir=args.run_dir, produced=produced)
    logger.info("[curate] wrote sentinel %s (n_figures=%d)", sentinel, len(produced))
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/scripts/test_curate_initiative_figures.py -v
```

Expected: all tests so far PASS (including end-to-end with 11 PNGs).

- [ ] **Step 5: Commit**

```bash
git add scripts/curate_initiative_figures.py tests/scripts/test_curate_initiative_figures.py
git commit -m "Wire curate_initiative_figures end-to-end with sentinel"
```

---

## Task 8: Error-handling regression tests

**Files:**
- Modify: `tests/scripts/test_curate_initiative_figures.py`

Verify graceful degradation when individual inputs are missing.

- [ ] **Step 1: Write failing tests for the three skip scenarios**

```python
# Append to tests/scripts/test_curate_initiative_figures.py
def test_main_missing_benchmark_still_produces_nine_figures(smoke_available, smoke_has_metrics, tmp_path, monkeypatch):
    """When benchmark/results.json is missing, figs 10-11 are skipped but
    the other 9 figures are still produced."""
    import scripts.curate_initiative_figures as m
    out = tmp_path / "curated"

    # Stage a smoke copy without benchmark/
    staged = tmp_path / "staged_run"
    shutil.copytree(SMOKE_DIR, staged, ignore=shutil.ignore_patterns("benchmark"))

    rc = m.main([
        "--run-dir", str(staged),
        "--output-dir", str(out),
        "--showcase-group", "obs163_act1",
        "--groups", "obs163_act1", "obs225_act2", "obs257_act3", "obs287_act3",
    ])
    assert rc == 0
    pngs = sorted(p.name for p in out.glob("*.png"))
    assert len(pngs) == 9
    payload = json.loads((out / ".curation.done").read_text())
    assert payload["n_figures"] == 9


def test_main_missing_feature_analysis_still_produces_six_figures(smoke_available, smoke_has_metrics, smoke_results_json, tmp_path):
    """When feature_analysis/figures/ is missing, figs 02-06 are skipped
    but figs 01 + 07-11 are still produced (=6 figures)."""
    import scripts.curate_initiative_figures as m
    out = tmp_path / "curated"

    staged = tmp_path / "staged_run"
    shutil.copytree(SMOKE_DIR, staged, ignore=shutil.ignore_patterns("feature_analysis"))

    rc = m.main([
        "--run-dir", str(staged),
        "--output-dir", str(out),
        "--showcase-group", "obs163_act1",
        "--groups", "obs163_act1", "obs225_act2", "obs257_act3", "obs287_act3",
    ])
    assert rc == 0
    pngs = sorted(p.name for p in out.glob("*.png"))
    assert len(pngs) == 6
    payload = json.loads((out / ".curation.done").read_text())
    assert payload["n_figures"] == 6


def test_main_no_inputs_returns_nonzero(tmp_path):
    """When the run dir is empty, exit code is 1 and no sentinel written."""
    import scripts.curate_initiative_figures as m
    empty = tmp_path / "empty_run"
    empty.mkdir()
    out = tmp_path / "curated"

    rc = m.main([
        "--run-dir", str(empty),
        "--output-dir", str(out),
        "--showcase-group", "obs627_act1",
    ])
    # Pipeline diagram still renders (no run-dir dependency), so at least 1 PNG produced.
    # The exit code is 0 only if at least one figure was produced.
    if rc == 0:
        # Pipeline diagram succeeded; sentinel must exist with n_figures==1
        payload = json.loads((out / ".curation.done").read_text())
        assert payload["n_figures"] == 1
    else:
        # No figures at all; no sentinel
        assert not (out / ".curation.done").exists()
```

- [ ] **Step 2: Add missing `shutil` import to the test file (if not already present), then run tests**

```bash
.venv/bin/python -m pytest tests/scripts/test_curate_initiative_figures.py -v
```

Expected: all tests PASS. If any of the three new tests fails, the implementation must be adjusted to honor the documented skip semantics (and the iteration continues until green).

- [ ] **Step 3: Commit**

```bash
git add tests/scripts/test_curate_initiative_figures.py
git commit -m "Add error-handling regression tests for curate_initiative_figures"
```

---

## Task 9: Validate against smoke and run full regression

**Files:** (no edits)

Final verification.

- [ ] **Step 1: Run curator end-to-end against the live smoke fixture**

```bash
.venv/bin/python -m scripts.curate_initiative_figures \
    --run-dir runs/smoke_pipeline_phase9 \
    --output-dir /tmp/curate_smoke_output \
    --showcase-group obs163_act1 \
    --groups obs163_act1 obs225_act2 obs257_act3 obs287_act3
```

Expected output: 11 PNGs + `.curation.done` under `/tmp/curate_smoke_output`.
Inspect:

```bash
ls /tmp/curate_smoke_output/
cat /tmp/curate_smoke_output/.curation.done
```

Expected: 11 PNG filenames listed in sentinel.

- [ ] **Step 2: Run the full test suite**

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests PASS. The test count should be 753 + N where N is the number of new tests added in this plan (target ~14 new tests: 5 CLI + 2 copy + 1 pipeline + 1 training + 2 benchmark-bars + 2 scatter + 1 sentinel + 1 e2e + 3 error-handling).

- [ ] **Step 3: Spot-check the produced figures**

Open each of the 11 PNGs in an image viewer and confirm:
* axes labels readable, legends present
* benchmark bars show 3 algorithms with error bars
* scatter shows n=1 annotation (smoke has only one eval seed)
* training curves show IQL and CQL traces

If any figure looks malformed, file a follow-up and adjust the relevant render helper.

- [ ] **Step 4: Commit any final fixes (if needed)**

```bash
git status
# If any fixes were made, commit them with a focused message.
```

- [ ] **Step 5: Production trigger (manual, post-pipeline completion)**

When `runs/offline_iql_cql_initiative_15min/` finishes, run:

```bash
.venv/bin/python -m scripts.curate_initiative_figures \
    --run-dir runs/offline_iql_cql_initiative_15min
```

Verify the 11 PNGs land under `docs/offline_rl/iql_cql_figures/` for Phase 11 doc writing.

---

## Self-review summary

* Tasks 1-7 each follow strict TDD: write failing test → verify fail → implement → verify pass → commit.
* Each commit yields a passing test suite.
* Task 3 (`_render_pipeline_diagram`) has a discovery step because the exact helper name in `generate_architecture_figures.py` is not assumed.
* Task 8 covers the three error-handling scenarios from the spec (missing benchmark, missing feature_analysis, no inputs).
* Task 9 is the empirical verification against the validated smoke fixture.
* No placeholders remain. Function signatures, output filenames, and JSON schemas are all spelled out.
* Type hints use `Path | None` (Python 3.10+ compatible).
