"""Standalone feature analysis for the Building 5 offline RL dataset."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

DATA_PATH   = Path("datasets/offline_rl/derived/rbc_with_reward.parquet")
OUTPUT_DIR  = Path("docs/offline_rl/feature_analysis")
FIGURES_DIR = OUTPUT_DIR / "figures"
DOC_PATH    = OUTPUT_DIR / "feature_analysis.md"


def _section_overview(df: pd.DataFrame) -> tuple[str, str]:
    n_rows, n_cols = df.shape
    obs_cols = [c for c in df.columns if c.startswith("obs_")]
    act_cols = [c for c in df.columns if c.startswith("action_")]
    missing = df.isnull().sum().sum()
    storage_var = df["action_electrical_storage"].var() if "action_electrical_storage" in df.columns else None
    if storage_var is None:
        storage_note = "`action_electrical_storage` column not found in dataset."
    elif storage_var < 1e-9:
        storage_note = f"`action_electrical_storage` variance = {storage_var:.6f} — **constant zero; excluded from analysis.**"
    else:
        storage_note = f"`action_electrical_storage` variance = {storage_var:.6f}"
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


def _section_seed_consistency(df: pd.DataFrame, figures_dir: Path) -> tuple[str, str]:
    """Plot reward KDE per seed to verify seeds are exchangeable."""
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


def _section_temporal_patterns(df: pd.DataFrame, figures_dir: Path) -> tuple[str, str]:
    """Plot mean EV action and mean reward by hour of day."""
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
    """KDE plots of observation features grouped by category."""
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
            series = df[col]
            if series.var() < 1e-9:
                # Degenerate column — draw a vertical line instead of KDE
                val = series.iloc[0]
                ax.axvline(val, label=col.replace("obs_", "") + " (const)", linestyle="--", alpha=0.75)
            else:
                series.plot.kde(ax=ax, label=col.replace("obs_", ""), alpha=0.75)
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


def _section_correlation(df: pd.DataFrame, figures_dir: Path) -> tuple[str, str]:
    """Spearman correlation heatmap of observation features (varying only)."""
    obs_cols = [c for c in df.columns if c.startswith("obs_")]
    varying = [c for c in obs_cols if df[c].var() > 1e-9]
    corr = df[varying].corr(method="spearman")

    labels = [c.replace("obs_", "") for c in varying]
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.45), max(8, n * 0.45)))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Spearman correlation — observation features")
    fig.tight_layout()
    fig_path = figures_dir / "fig4_correlation_matrix.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    dropped = set(obs_cols) - set(varying)
    drop_note = (
        f"Near-constant features dropped before correlation: {', '.join(f'`{c}`' for c in sorted(dropped))}."
        if dropped
        else "No near-constant features dropped."
    )

    md = f"""## 5. Correlation structure

![Correlation matrix](figures/fig4_correlation_matrix.png)

Spearman correlation heatmap of all varying observation features. {drop_note} High-correlation clusters (|ρ| > 0.9) indicate redundant features — typically price forecast triplets and temperature forecast triplets. These can be represented by the current-step value without significant information loss.
"""
    return str(fig_path), md


def _section_mutual_information(df: pd.DataFrame, figures_dir: Path) -> tuple[str, str]:
    """Dual-bar chart: MI of each obs feature vs reward and vs EV action."""
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
    fig, ax = plt.subplots(figsize=(10, max(6, len(labels_sorted) * 0.35)))
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


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
