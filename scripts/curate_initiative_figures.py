"""Phase 10 — Curate offline-RL initiative figures.

Given a completed pipeline output directory at
``runs/offline_iql_cql_initiative_15min/`` (or any compatible run dir),
produce 11 thesis-grade PNGs at ``docs/offline_rl/iql_cql_figures/``:

  01_pipeline_overview.png            — pipeline architecture diagram
  02_dataset_stats.png                — dataset summary table
  03_action_coverage_group_a.png      — action distribution (showcase group)
  04_reward_by_regime.png             — reward by RBC regime
  05_correlations_group_a.png         — feature x reward correlations
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
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Pipeline output directory, e.g. runs/offline_iql_cql_initiative_15min",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for curated PNGs.",
    )
    p.add_argument(
        "--showcase-group",
        type=str,
        default=DEFAULT_SHOWCASE_GROUP,
        help="Agent group key used for per-group figures (default obs627_act1).",
    )
    p.add_argument(
        "--groups",
        nargs="+",
        default=DEFAULT_GROUPS,
        help="Agent group keys to include in aggregate figures.",
    )
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
