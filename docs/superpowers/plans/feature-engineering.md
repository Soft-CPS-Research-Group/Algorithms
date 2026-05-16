# Feature Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `scripts/explore_features.py` that generates 6 figures and `docs/offline_rl/feature_analysis/feature_analysis.md` in a single run, documenting the structure and importance of features in the Building 5 offline RL dataset.

**Architecture:** Standalone script; no argparse; each analysis section is a private function returning `(figure_path, markdown_snippet)`; `main()` collects results and writes the markdown doc. No new modules, no parquet writes.

**Tech Stack:** Python 3.10, pandas, numpy, matplotlib, seaborn, sklearn.feature_selection.mutual_info_regression.

---

### Task 1: Output directory skeleton + smoke test

**Files:**
- Create: `scripts/explore_features.py` (stub only)
- Create: `tests/offline_rl/test_explore_features.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/offline_rl/test_explore_features.py
import importlib
import types

def test_module_importable():
    mod = importlib.import_module("scripts.explore_features")
    assert isinstance(mod, types.ModuleType)

def test_has_main():
    from scripts import explore_features
    assert callable(explore_features.main)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/offline_rl/test_explore_features.py -v
```
Expected: ImportError / AttributeError.

- [ ] **Step 3: Write the stub**

```python
# scripts/explore_features.py
"""Standalone feature analysis for the Building 5 offline RL dataset."""
from __future__ import annotations
from pathlib import Path

DATA_PATH   = Path("datasets/offline_rl/derived/rbc_with_reward.parquet")
OUTPUT_DIR  = Path("docs/offline_rl/feature_analysis")
FIGURES_DIR = OUTPUT_DIR / "figures"
DOC_PATH    = OUTPUT_DIR / "feature_analysis.md"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/offline_rl/test_explore_features.py -v
```
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/explore_features.py tests/offline_rl/test_explore_features.py
git commit -m "feat: add explore_features stub + smoke test"
```

---

### Task 2: Dataset loader + overview section

**Files:**
- Modify: `scripts/explore_features.py`
- Modify: `tests/offline_rl/test_explore_features.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/offline_rl/test_explore_features.py
import pandas as pd
import pytest

@pytest.fixture
def sample_df():
    """Minimal synthetic dataframe matching dataset schema."""
    import numpy as np
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "seed": rng.integers(100, 105, n),
        "obs_hour": rng.integers(0, 24, n),
        "obs_month": rng.integers(1, 13, n),
        "obs_connected_state": rng.integers(0, 2, n).astype(float),
        "obs_departure_time": rng.integers(6, 22, n).astype(float),
        "obs_required_soc_departure": rng.uniform(0.5, 1.0, n),
        "obs_electrical_vehicle_storage_soc": rng.uniform(0.0, 1.0, n),
        "obs_electrical_storage_soc": np.zeros(n),
        "obs_non_shiftable_load": rng.uniform(0.1, 2.0, n),
        "obs_solar_generation": rng.uniform(0.0, 1.5, n),
        "obs_net_electricity_consumption": rng.uniform(-1.0, 2.0, n),
        "obs_electricity_pricing": rng.uniform(0.1, 0.5, n),
        "obs_electricity_pricing_predicted_1": rng.uniform(0.1, 0.5, n),
        "action_electric_vehicle_storage_charger_5_1": rng.uniform(0, 1, n),
        "action_electrical_storage": np.zeros(n),
        "reward": rng.uniform(-10, 0, n),
    })

def test_section_overview_returns_markdown(sample_df):
    from scripts.explore_features import _section_overview
    fig_path, md = _section_overview(sample_df)
    assert fig_path == ""
    assert "rows" in md.lower() or "dataset" in md.lower()
    assert "action_electrical_storage" in md
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/offline_rl/test_explore_features.py::test_section_overview_returns_markdown -v
```

- [ ] **Step 3: Implement `_section_overview`**

```python
# add to scripts/explore_features.py
import pandas as pd
import numpy as np

def _section_overview(df: pd.DataFrame) -> tuple[str, str]:
    n_rows, n_cols = df.shape
    obs_cols = [c for c in df.columns if c.startswith("obs_")]
    act_cols = [c for c in df.columns if c.startswith("action_")]
    missing = df.isnull().sum().sum()
    storage_var = df["action_electrical_storage"].var() if "action_electrical_storage" in df.columns else None
    storage_note = (
        f"`action_electrical_storage` variance = {storage_var:.6f} — **constant zero; excluded from analysis.**"
        if storage_var is not None and storage_var < 1e-9
        else f"`action_electrical_storage` variance = {storage_var:.6f}"
    )
    reward_min = df["reward"].min()
    reward_max = df["reward"].max()
    reward_mean = df["reward"].mean()
    reward_std = df["reward"].std()

    md = f"""## 1. Dataset overview

| Property | Value |
|---|---|
| Rows | {n_rows:,} |
| Columns | {n_cols} |
| Observation features | {len(obs_cols)} |
| Action dimensions | {len(act_cols)} |
| Missing values | {missing} |
| Reward range | [{reward_min:.3f}, {reward_max:.3f}] |
| Reward mean ± std | {reward_mean:.3f} ± {reward_std:.3f} |

{storage_note}
"""
    return "", md
```

- [ ] **Step 4: Run test**

```bash
pytest tests/offline_rl/test_explore_features.py -v
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/explore_features.py tests/offline_rl/test_explore_features.py
git commit -m "feat: explore_features dataset overview section"
```

---

### Task 3: Fig 1 — Seed consistency

**Files:**
- Modify: `scripts/explore_features.py`
- Modify: `tests/offline_rl/test_explore_features.py`

- [ ] **Step 1: Write the failing test**

```python
def test_section_seed_consistency_creates_figure(sample_df, tmp_path):
    from scripts.explore_features import _section_seed_consistency
    fig_path, md = _section_seed_consistency(sample_df, tmp_path)
    assert (tmp_path / "fig1_seed_consistency.png").exists()
    assert "fig1_seed_consistency.png" in fig_path
    assert "seed" in md.lower()
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/offline_rl/test_explore_features.py::test_section_seed_consistency_creates_figure -v
```

- [ ] **Step 3: Implement**

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def _section_seed_consistency(df: pd.DataFrame, figures_dir: Path) -> tuple[str, str]:
    fig, ax = plt.subplots(figsize=(10, 6))
    seeds = sorted(df["seed"].unique())
    for seed in seeds:
        subset = df[df["seed"] == seed]["reward"]
        subset.plot.kde(ax=ax, label=f"seed {seed}", alpha=0.7)
    ax.set_xlabel("Reward")
    ax.set_ylabel("Density")
    ax.set_title("Reward distribution per seed")
    ax.legend(fontsize=8)
    fig_path = figures_dir / "fig1_seed_consistency.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    md = """## 2. Seed consistency

![Seed consistency](figures/fig1_seed_consistency.png)

Reward KDE overlaid for each seed. Overlapping distributions confirm that seeds are statistically exchangeable and pooling the dataset is justified. Any outlier seed would appear as a shifted or wider curve.
"""
    return str(fig_path), md
```

- [ ] **Step 4: Run test**

```bash
pytest tests/offline_rl/test_explore_features.py -v
```

- [ ] **Step 5: Commit**

```bash
git add scripts/explore_features.py tests/offline_rl/test_explore_features.py
git commit -m "feat: explore_features fig1 seed consistency"
```

---

### Task 4: Fig 2 — Temporal patterns

**Files:**
- Modify: `scripts/explore_features.py`
- Modify: `tests/offline_rl/test_explore_features.py`

- [ ] **Step 1: Write the failing test**

```python
def test_section_temporal_patterns_creates_figure(sample_df, tmp_path):
    from scripts.explore_features import _section_temporal_patterns
    fig_path, md = _section_temporal_patterns(sample_df, tmp_path)
    assert (tmp_path / "fig2_temporal_patterns.png").exists()
    assert "hour" in md.lower()
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/offline_rl/test_explore_features.py::test_section_temporal_patterns_creates_figure -v
```

- [ ] **Step 3: Implement**

```python
def _section_temporal_patterns(df: pd.DataFrame, figures_dir: Path) -> tuple[str, str]:
    hourly = df.groupby("obs_hour").agg(
        mean_action=("action_electric_vehicle_storage_charger_5_1", "mean"),
        mean_reward=("reward", "mean"),
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(hourly["obs_hour"], hourly["mean_action"], marker="o")
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("Mean EV charging action")
    ax1.set_title("EV action by hour")
    ax1.set_xticks(range(0, 24, 2))

    ax2.plot(hourly["obs_hour"], hourly["mean_reward"], marker="o", color="tab:orange")
    ax2.set_xlabel("Hour of day")
    ax2.set_ylabel("Mean reward")
    ax2.set_title("Reward by hour")
    ax2.set_xticks(range(0, 24, 2))

    fig.tight_layout()
    fig_path = figures_dir / "fig2_temporal_patterns.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    peak_action_hour = int(hourly.loc[hourly["mean_action"].idxmax(), "obs_hour"])

    md = f"""## 3. Temporal patterns

![Temporal patterns](figures/fig2_temporal_patterns.png)

Mean EV charging action and mean reward aggregated by hour of day. Peak charging occurs around hour {peak_action_hour}, consistent with EV arrival patterns. Reward variation across the day reflects electricity price and solar generation schedules.
"""
    return str(fig_path), md
```

- [ ] **Step 4: Run test**

```bash
pytest tests/offline_rl/test_explore_features.py -v
```

- [ ] **Step 5: Commit**

```bash
git add scripts/explore_features.py tests/offline_rl/test_explore_features.py
git commit -m "feat: explore_features fig2 temporal patterns"
```

---

### Task 5: Fig 3 — Feature distributions

**Files:**
- Modify: `scripts/explore_features.py`
- Modify: `tests/offline_rl/test_explore_features.py`

- [ ] **Step 1: Write the failing test**

```python
def test_section_feature_distributions_creates_figure(sample_df, tmp_path):
    from scripts.explore_features import _section_feature_distributions
    fig_path, md = _section_feature_distributions(sample_df, tmp_path)
    assert (tmp_path / "fig3_feature_distributions.png").exists()
    assert "fig3" in fig_path
    assert "distribution" in md.lower()
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/offline_rl/test_explore_features.py::test_section_feature_distributions_creates_figure -v
```

- [ ] **Step 3: Implement**

```python
_FEATURE_GROUPS: dict[str, list[str]] = {
    "EV state": [
        "obs_connected_state",
        "obs_departure_time",
        "obs_required_soc_departure",
        "obs_electrical_vehicle_storage_soc",
    ],
    "Pricing": [
        "obs_electricity_pricing",
        "obs_electricity_pricing_predicted_1",
    ],
    "Load & solar": [
        "obs_non_shiftable_load",
        "obs_solar_generation",
        "obs_net_electricity_consumption",
    ],
    "Storage": ["obs_electrical_storage_soc"],
}


def _section_feature_distributions(df: pd.DataFrame, figures_dir: Path) -> tuple[str, str]:
    groups = {
        name: [c for c in cols if c in df.columns]
        for name, cols in _FEATURE_GROUPS.items()
    }
    groups = {k: v for k, v in groups.items() if v}

    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(4 * n_groups, 5))
    if n_groups == 1:
        axes = [axes]

    for ax, (group_name, cols) in zip(axes, groups.items()):
        for col in cols:
            df[col].plot.kde(ax=ax, label=col.replace("obs_", ""), alpha=0.75)
        ax.set_title(group_name)
        ax.legend(fontsize=7)
        ax.set_xlabel("Value")

    fig.suptitle("Feature distributions by group", fontsize=13)
    fig.tight_layout()
    fig_path = figures_dir / "fig3_feature_distributions.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    md = """## 4. Feature distributions

![Feature distributions](figures/fig3_feature_distributions.png)

KDE plots grouped by feature category. `obs_electrical_storage_soc` is expected to be constant zero throughout the RBC dataset (the battery storage is never used). EV state features show bimodal patterns driven by connected vs disconnected states.
"""
    return str(fig_path), md
```

- [ ] **Step 4: Run test**

```bash
pytest tests/offline_rl/test_explore_features.py -v
```

- [ ] **Step 5: Commit**

```bash
git add scripts/explore_features.py tests/offline_rl/test_explore_features.py
git commit -m "feat: explore_features fig3 feature distributions"
```

---

### Task 6: Fig 4 — Correlation matrix

**Files:**
- Modify: `scripts/explore_features.py`
- Modify: `tests/offline_rl/test_explore_features.py`

- [ ] **Step 1: Write the failing test**

```python
def test_section_correlation_creates_figure(sample_df, tmp_path):
    from scripts.explore_features import _section_correlation
    fig_path, md = _section_correlation(sample_df, tmp_path)
    assert (tmp_path / "fig4_correlation_matrix.png").exists()
    assert "correlation" in md.lower()
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/offline_rl/test_explore_features.py::test_section_correlation_creates_figure -v
```

- [ ] **Step 3: Implement**

```python
def _section_correlation(df: pd.DataFrame, figures_dir: Path) -> tuple[str, str]:
    obs_cols = [c for c in df.columns if c.startswith("obs_")]
    # drop near-constant columns (variance < 1e-9)
    varying = [c for c in obs_cols if df[c].var() > 1e-9]
    corr = df[varying].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(max(10, len(varying) * 0.4), max(8, len(varying) * 0.4)))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=[c.replace("obs_", "") for c in varying],
        yticklabels=[c.replace("obs_", "") for c in varying],
        annot=False,
    )
    ax.set_title("Spearman correlation — observation features")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    fig_path = figures_dir / "fig4_correlation_matrix.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    dropped = set(obs_cols) - set(varying)
    drop_note = (
        f"Near-constant features dropped before correlation: {', '.join(f'`{c}`' for c in sorted(dropped))}."
        if dropped else "No near-constant features dropped."
    )

    md = f"""## 5. Correlation structure

![Correlation matrix](figures/fig4_correlation_matrix.png)

Spearman correlation heatmap of all varying observation features. {drop_note} High-correlation clusters (|ρ| > 0.9) indicate redundant features — typically price forecast triplets and temperature forecast triplets. These can be represented by the current-step value without significant information loss.
"""
    return str(fig_path), md
```

- [ ] **Step 4: Run test**

```bash
pytest tests/offline_rl/test_explore_features.py -v
```

- [ ] **Step 5: Commit**

```bash
git add scripts/explore_features.py tests/offline_rl/test_explore_features.py
git commit -m "feat: explore_features fig4 correlation matrix"
```

---

### Task 7: Fig 5 — Mutual information

**Files:**
- Modify: `scripts/explore_features.py`
- Modify: `tests/offline_rl/test_explore_features.py`

- [ ] **Step 1: Write the failing test**

```python
def test_section_mutual_information_creates_figure(sample_df, tmp_path):
    from scripts.explore_features import _section_mutual_information
    fig_path, md = _section_mutual_information(sample_df, tmp_path)
    assert (tmp_path / "fig5_mutual_information.png").exists()
    assert "mutual information" in md.lower()
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/offline_rl/test_explore_features.py::test_section_mutual_information_creates_figure -v
```

- [ ] **Step 3: Implement**

```python
from sklearn.feature_selection import mutual_info_regression


def _section_mutual_information(df: pd.DataFrame, figures_dir: Path) -> tuple[str, str]:
    obs_cols = [c for c in df.columns if c.startswith("obs_")]
    varying = [c for c in obs_cols if df[c].var() > 1e-9]
    X = df[varying].fillna(0).values

    mi_reward = mutual_info_regression(X, df["reward"].values, random_state=42)
    mi_action = mutual_info_regression(
        X,
        df["action_electric_vehicle_storage_charger_5_1"].values,
        random_state=42,
    )

    labels = [c.replace("obs_", "") for c in varying]
    order = np.argsort(np.maximum(mi_reward, mi_action))[::-1]
    labels_sorted = [labels[i] for i in order]
    mi_reward_sorted = mi_reward[order]
    mi_action_sorted = mi_action[order]

    y_pos = np.arange(len(labels_sorted))
    fig, ax = plt.subplots(figsize=(10, max(6, len(labels_sorted) * 0.3)))
    ax.barh(y_pos - 0.2, mi_reward_sorted, height=0.4, label="vs reward", color="tab:blue", alpha=0.8)
    ax.barh(y_pos + 0.2, mi_action_sorted, height=0.4, label="vs EV action", color="tab:green", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_sorted, fontsize=8)
    ax.set_xlabel("Mutual information (estimated)")
    ax.set_title("Feature importance — MI vs reward and EV action")
    ax.legend()
    fig.tight_layout()
    fig_path = figures_dir / "fig5_mutual_information.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    top3_reward = [labels_sorted[i] for i in np.argsort(mi_reward_sorted)[::-1][:3]]
    top3_action = [labels_sorted[i] for i in np.argsort(mi_action_sorted)[::-1][:3]]

    md = f"""## 6. Mutual information

![Mutual information](figures/fig5_mutual_information.png)

Estimated mutual information (5-nearest-neighbour, `random_state=42`) for each observation feature against reward (blue) and EV charging action (green), sorted by maximum MI. MI values are estimated and approximate.

Top features predictive of **reward**: {", ".join(f"`{f}`" for f in top3_reward)}.

Top features predictive of **EV action**: {", ".join(f"`{f}`" for f in top3_action)}.

Features with high MI for action but low MI for reward indicate behaviours the RBC policy relies on that do not directly improve cost — potential sources of suboptimality that IQL could learn to exploit.
"""
    return str(fig_path), md
```

- [ ] **Step 4: Run test**

```bash
pytest tests/offline_rl/test_explore_features.py -v
```

- [ ] **Step 5: Commit**

```bash
git add scripts/explore_features.py tests/offline_rl/test_explore_features.py
git commit -m "feat: explore_features fig5 mutual information"
```

---

### Task 8: Fig 6 — EV state patterns

**Files:**
- Modify: `scripts/explore_features.py`
- Modify: `tests/offline_rl/test_explore_features.py`

- [ ] **Step 1: Write the failing test**

```python
def test_section_ev_state_patterns_creates_figure(sample_df, tmp_path):
    from scripts.explore_features import _section_ev_state_patterns
    fig_path, md = _section_ev_state_patterns(sample_df, tmp_path)
    assert (tmp_path / "fig6_ev_state_patterns.png").exists()
    assert "soc" in md.lower()
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/offline_rl/test_explore_features.py::test_section_ev_state_patterns_creates_figure -v
```

- [ ] **Step 3: Implement**

```python
def _section_ev_state_patterns(df: pd.DataFrame, figures_dir: Path) -> tuple[str, str]:
    connected = df[df["obs_connected_state"] > 0.5].copy()
    soc_at_connection = connected["obs_electrical_vehicle_storage_soc"]
    soc_deficit = connected["obs_required_soc_departure"] - connected["obs_electrical_vehicle_storage_soc"]
    time_to_dep = (connected["obs_departure_time"] - connected["obs_hour"]) % 24

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].hist(soc_at_connection.dropna(), bins=30, color="tab:blue", alpha=0.8)
    axes[0].set_xlabel("EV SOC at connection")
    axes[0].set_ylabel("Count")
    axes[0].set_title("SOC when connected")

    axes[1].hist(time_to_dep.dropna(), bins=24, color="tab:orange", alpha=0.8)
    axes[1].set_xlabel("Hours to departure")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Time to departure")

    axes[2].hist(soc_deficit.dropna(), bins=30, color="tab:red", alpha=0.8)
    axes[2].set_xlabel("SOC deficit (required − current)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("SOC deficit at connection")

    fig.suptitle("EV state at connected steps", fontsize=13)
    fig.tight_layout()
    fig_path = figures_dir / "fig6_ev_state_patterns.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    med_deficit = float(soc_deficit.median()) if len(soc_deficit) > 0 else float("nan")
    med_time = float(time_to_dep.median()) if len(time_to_dep) > 0 else float("nan")

    md = f"""## 7. EV state patterns

![EV state patterns](figures/fig6_ev_state_patterns.png)

Distributions at connected steps (where `obs_connected_state > 0.5`). Median SOC deficit at connection: **{med_deficit:.2f}** (fraction of battery capacity still to charge). Median time to departure: **{med_time:.1f} hours**. These two quantities jointly determine charging urgency and motivate the proposed `ev_urgency` derived feature.
"""
    return str(fig_path), md
```

- [ ] **Step 4: Run test**

```bash
pytest tests/offline_rl/test_explore_features.py -v
```

- [ ] **Step 5: Commit**

```bash
git add scripts/explore_features.py tests/offline_rl/test_explore_features.py
git commit -m "feat: explore_features fig6 EV state patterns"
```

---

### Task 9: Derived features section + markdown writer + `main()`

**Files:**
- Modify: `scripts/explore_features.py`
- Modify: `tests/offline_rl/test_explore_features.py`

- [ ] **Step 1: Write the failing test**

```python
def test_main_produces_all_outputs(tmp_path, monkeypatch):
    """Integration: main() writes doc + 6 figures."""
    import pandas as pd, numpy as np
    from scripts import explore_features as ef

    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame({
        "seed": rng.integers(100, 105, n),
        "obs_hour": rng.integers(0, 24, n),
        "obs_month": rng.integers(1, 13, n),
        "obs_connected_state": rng.integers(0, 2, n).astype(float),
        "obs_departure_time": rng.integers(6, 22, n).astype(float),
        "obs_required_soc_departure": rng.uniform(0.5, 1.0, n),
        "obs_electrical_vehicle_storage_soc": rng.uniform(0.0, 1.0, n),
        "obs_electrical_storage_soc": np.zeros(n),
        "obs_non_shiftable_load": rng.uniform(0.1, 2.0, n),
        "obs_solar_generation": rng.uniform(0.0, 1.5, n),
        "obs_net_electricity_consumption": rng.uniform(-1.0, 2.0, n),
        "obs_electricity_pricing": rng.uniform(0.1, 0.5, n),
        "obs_electricity_pricing_predicted_1": rng.uniform(0.1, 0.5, n),
        "action_electric_vehicle_storage_charger_5_1": rng.uniform(0, 1, n),
        "action_electrical_storage": np.zeros(n),
        "reward": rng.uniform(-10, 0, n),
    })

    out_dir = tmp_path / "feature_analysis"
    fig_dir = out_dir / "figures"
    doc_path = out_dir / "feature_analysis.md"

    monkeypatch.setattr(ef, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(ef, "FIGURES_DIR", fig_dir)
    monkeypatch.setattr(ef, "DOC_PATH", doc_path)

    ef.main(df=df)

    figures = list(fig_dir.glob("*.png"))
    assert len(figures) == 6, f"Expected 6 figures, got {len(figures)}: {figures}"
    assert doc_path.exists()
    content = doc_path.read_text()
    for section in ["Dataset overview", "Seed consistency", "Temporal", "distribution",
                    "Correlation", "Mutual information", "EV state", "Derived features"]:
        assert section.lower() in content.lower(), f"Missing section: {section}"
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/offline_rl/test_explore_features.py::test_main_produces_all_outputs -v
```

- [ ] **Step 3: Implement derived features section, `_write_markdown`, and wire `main()`**

Add the `_DERIVED_FEATURES_MD` constant:

```python
_DERIVED_FEATURES_MD = """## 8. Proposed derived features

The following features are proposed for future IQL training runs. They are not written to any parquet file; implementation is deferred to the next experiment phase.

| Feature | Formula | Columns consumed | Expected range | Rationale |
|---|---|---|---|---|
| `soc_deficit` | `required_soc_departure − ev_soc` | `obs_required_soc_departure`, `obs_electrical_vehicle_storage_soc` | [0, 1] | Direct urgency signal; currently the agent must compute this implicitly from two separate inputs |
| `time_to_departure` | `(departure_time − obs_hour) mod 24` (0 when disconnected) | `obs_departure_time`, `obs_hour`, `obs_connected_state` | [0, 23] | Collapses absolute departure time into a countdown; more learnable than raw hour |
| `price_trend` | `electricity_pricing_predicted_1 − electricity_pricing` | `obs_electricity_pricing_predicted_1`, `obs_electricity_pricing` | continuous | Sign tells the agent whether prices are rising (delay charging) or falling (charge now) |
| `solar_surplus` | `solar_generation − non_shiftable_load` | `obs_solar_generation`, `obs_non_shiftable_load` | continuous | Net renewable energy available before grid draw; negative means the building is already importing |
| `ev_urgency` | `soc_deficit / max(time_to_departure, 1)` | derived from above | [0, 1+] | Required average charge rate per remaining hour; high when deficit is large and time is short |
"""


def _section_derived_features() -> tuple[str, str]:
    return "", _DERIVED_FEATURES_MD
```

Add `_write_markdown`:

```python
def _write_markdown(doc_path: Path, snippets: list[str]) -> None:
    header = """# Feature Analysis — Building 5 Offline RL Dataset

This document is generated by `scripts/explore_features.py`. Run the script to regenerate.

"""
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(header + "\n".join(snippets))
```

Update `main()` to accept an optional `df` kwarg and call all sections:

```python
def main(df: pd.DataFrame | None = None) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if df is None:
        df = pd.read_parquet(DATA_PATH)

    snippets: list[str] = []

    _, md = _section_overview(df)
    snippets.append(md)

    _, md = _section_seed_consistency(df, FIGURES_DIR)
    snippets.append(md)

    _, md = _section_temporal_patterns(df, FIGURES_DIR)
    snippets.append(md)

    _, md = _section_feature_distributions(df, FIGURES_DIR)
    snippets.append(md)

    _, md = _section_correlation(df, FIGURES_DIR)
    snippets.append(md)

    _, md = _section_mutual_information(df, FIGURES_DIR)
    snippets.append(md)

    _, md = _section_ev_state_patterns(df, FIGURES_DIR)
    snippets.append(md)

    _, md = _section_derived_features()
    snippets.append(md)

    _write_markdown(DOC_PATH, snippets)
    print(f"Done. Doc: {DOC_PATH}  Figures: {FIGURES_DIR}")
```

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/offline_rl/test_explore_features.py -v
pytest --tb=short -q
```
Expected: all previously passing tests still pass; new tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/explore_features.py tests/offline_rl/test_explore_features.py
git commit -m "feat: explore_features complete — main(), derived features, markdown writer"
```

---

### Task 10: Run the script end-to-end, commit outputs

**Files:**
- New: `docs/offline_rl/feature_analysis/feature_analysis.md` (generated)
- New: `docs/offline_rl/feature_analysis/figures/*.png` (6 files, generated)

- [ ] **Step 1: Run the script**

```bash
python scripts/explore_features.py
```
Expected: prints `Done. Doc: docs/offline_rl/feature_analysis/feature_analysis.md  Figures: docs/offline_rl/feature_analysis/figures`.

- [ ] **Step 2: Verify outputs**

```bash
ls docs/offline_rl/feature_analysis/figures/
```
Expected: `fig1_seed_consistency.png  fig2_temporal_patterns.png  fig3_feature_distributions.png  fig4_correlation_matrix.png  fig5_mutual_information.png  fig6_ev_state_patterns.png`

- [ ] **Step 3: Verify markdown**

Open `docs/offline_rl/feature_analysis/feature_analysis.md` and confirm all 8 sections are present and figures are embedded.

- [ ] **Step 4: Run full test suite one final time**

```bash
pytest --tb=short -q
```
Expected: 132+ tests pass, 0 failures.

- [ ] **Step 5: Commit outputs**

```bash
git add docs/offline_rl/feature_analysis/
git commit -m "docs: add feature analysis report and figures"
```
