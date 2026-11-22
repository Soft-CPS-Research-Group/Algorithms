"""Freeze training-only scales and derive the four declared IQL reward variants."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.indin_core30.audit import audit
from scripts.indin_core30.data import sha256


VARIANTS = {"lambda_000": 0.0, "lambda_010": 0.1, "lambda_030": 0.3, "lambda_100": 1.0}


def reward_scale(rewards) -> tuple[float, str]:
    values = np.abs(np.asarray(rewards, dtype=np.float64))
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Missing or non-finite training reward values")
    median = float(np.median(values))
    if median > 0:
        return median, "median_absolute"
    percentile = float(np.quantile(values, 0.75))
    if percentile > 0:
        return percentile, "p75_absolute_after_zero_median"
    positive = values[values > 0]
    if len(positive):
        return float(np.mean(positive)), "mean_positive_absolute_after_zero_median_and_p75"
    raise ValueError("Active cohort has no non-zero training reward; no arbitrary scale is allowed")


def training_scales(data_dir: Path) -> dict:
    rewards = {}
    physical = {}
    for file in sorted((data_dir / "train").glob("*.parquet")):
        frame = pq.read_table(file, columns=["behaviour_policy", "episode", "seed", "timestep",
            "obs_dim", "action_dim", "base_reward", "ramp_abs_kw", "has_previous_executed_interval"]).to_pandas()
        for (obs, act), group in frame.groupby(["obs_dim", "action_dim"]):
            if act > 0:
                rewards.setdefault(f"obs{obs}_act{act}", []).extend(group["base_reward"].tolist())
        unique = frame.drop_duplicates(["behaviour_policy", "episode", "seed", "timestep"])
        for row in unique.itertuples(index=False):
            key = (row.behaviour_policy, row.episode, row.seed, row.timestep)
            if key in physical:
                raise ValueError("Duplicate physical transition across training artifacts")
            physical[key] = float(row.ramp_abs_kw) if row.has_previous_executed_interval else 0.0
    positive = np.asarray([value for value in physical.values() if value > 0])
    if not len(positive) or not np.isfinite(positive).all():
        raise ValueError("Training collection contains no finite positive community ramp")
    cohort_scales = {}
    for name, values in sorted(rewards.items()):
        scale, rule = reward_scale(values)
        cohort_scales[name] = {"scale": scale, "rule": rule, "training_rows": len(values)}
    return {"community_ramp_scale_kw": float(np.median(positive)),
            "community_rule": "median_positive_executed_interval_ramp_kw_training_only",
            "unique_training_transitions": len(physical), "positive_ramp_transitions": len(positive),
            "cohorts": cohort_scales}


def derive(data_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite derived data: {output_dir}")
    raw_audit = audit(data_dir)
    scales = training_scales(data_dir)
    output_dir.mkdir(parents=True)
    (output_dir / "raw_audit.json").write_text(json.dumps(raw_audit, indent=2))
    manifest = {
        "schema_version": "indin.core30.reward_variants.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(data_dir.resolve()), "derivation_sha256": sha256(Path(__file__)),
        "formula": "base_reward - lambda * cohort_reward_scale * ramp_abs_kw / community_ramp_scale_kw",
        "scales": scales, "variants": VARIANTS,
        "passive_rows": "Preserved unchanged; zero-action cohorts are not trained",
        "split": "actual simulator seed22=chronologically reserved training; actual23=diagnostic validation",
        "validation_use": "Frozen training scales applied; no scale fitted on validation",
        "files": [], "complete": False,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    for name in VARIANTS:
        (output_dir / name).mkdir()
    source_files = [Path(item["file"]) for item in raw_audit["files"]]
    for file in source_files:
        table = pq.read_table(file)
        obs = table["obs_dim"].to_numpy()
        act = table["action_dim"].to_numpy()
        base = table["base_reward"].to_numpy()
        ramp = table["ramp_abs_kw"].to_numpy()
        group_scale = np.asarray([
            scales["cohorts"][f"obs{o}_act{a}"]["scale"] if a > 0 else 0.0
            for o, a in zip(obs, act)
        ])
        index = table.schema.get_field_index("reward")
        per_file = {"source_file": str(file), "source_sha256": sha256(file),
                    "rows": table.num_rows, "outputs": {}}
        for name, weight in VARIANTS.items():
            reward = base - weight * group_scale * ramp / scales["community_ramp_scale_kw"]
            if not np.isfinite(reward).all():
                raise ValueError("Derived reward is non-finite")
            target = output_dir / name / file.name
            derived = table.set_column(index, table.schema.field(index),
                                       pa.array(reward, type=table.schema.field(index).type))
            pq.write_table(derived, target, compression="zstd", row_group_size=1920)
            per_file["outputs"][name] = {"file": str(target), "sha256": sha256(target),
                                          "reward_min": float(np.min(reward)),
                                          "reward_max": float(np.max(reward)),
                                          "penalty_mean_active": float(np.mean((base - reward)[act > 0]))}
        manifest["files"].append(per_file)
        manifest_path.write_text(json.dumps(manifest, indent=2))
    manifest["complete"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = derive(args.data_dir, args.output_dir)
    print(json.dumps({"complete": manifest["complete"], "source_files": len(manifest["files"]),
                      "cohorts": len(manifest["scales"]["cohorts"]), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
