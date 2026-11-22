"""Audit frozen collection artifacts before any reward or policy selection."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from scripts.indin_core30.data import sha256, validate_windows


def audit(root: Path) -> dict:
    manifests = sorted(root.glob("*_manifest.json"))
    if not manifests:
        raise ValueError("No collection manifests found")
    columns = ["seed", "episode", "split", "timestep", "source_step", "agent_idx",
               "obs_dim", "action_dim", "reward", "base_reward", "terminated", "truncated",
               "community_net_energy_kwh", "community_net_power_kw", "ramp_abs_kw", "ramp_up_kw",
               "duration_seconds", "building_net_energy_kwh", "reward_components_json"]
    report = {"files": [], "train_only_scales": {}, "checks_passed": False}
    train_ramps = []
    groups = defaultdict(lambda: {"reward": [], "cost": [], "hard_penalty": []})
    seen_train = set()
    seen_val = set()
    for manifest_file in manifests:
        manifest = json.loads(manifest_file.read_text())
        if not manifest["complete"]:
            raise ValueError(f"Incomplete collection: {manifest_file.name}")
        validate_windows(manifest["windows"])
        for item in manifest["episodes"]:
            window = item["window"]
            file = root / window["split"] / item["file"]
            if sha256(file) != item["sha256"]:
                raise ValueError(f"Parquet hash mismatch: {file}")
            frame = pq.read_table(file, columns=columns).to_pandas()
            if len(frame) != item["rows"]:
                raise ValueError("Row count differs from manifest")
            assert frame["episode"].eq(window["id"]).all()
            assert frame["split"].eq(window["split"]).all()
            expected_seed = window.get("simulator_seed", manifest["config"]["simulator"].get("random_seed"))
            assert frame["seed"].eq(expected_seed).all()
            assert frame["duration_seconds"].eq(900).all()
            np.testing.assert_allclose(frame["reward"], frame["base_reward"], rtol=0, atol=0)
            assert not frame.duplicated(["timestep", "agent_idx"]).any()
            counts = frame.groupby("timestep").size()
            assert counts.eq(30).all() and len(counts) == item["transitions"]
            per_step = frame.groupby("timestep", sort=True)
            for column in ("source_step", "community_net_energy_kwh", "community_net_power_kw",
                           "ramp_abs_kw", "ramp_up_kw", "terminated", "truncated"):
                assert per_step[column].nunique().eq(1).all()
            physical = per_step.first()
            assert physical["source_step"].tolist() == list(range(window["start_step"], window["start_step"] + item["transitions"]))
            np.testing.assert_allclose(per_step["building_net_energy_kwh"].sum(),
                                       physical["community_net_energy_kwh"], rtol=1e-12, atol=1e-10)
            np.testing.assert_allclose(physical["community_net_power_kw"],
                                       physical["community_net_energy_kwh"] * 4, rtol=1e-12, atol=1e-10)
            delta = np.r_[0, np.diff(physical["community_net_power_kw"])]
            np.testing.assert_allclose(physical["ramp_abs_kw"], np.abs(delta), rtol=0, atol=1e-10)
            np.testing.assert_allclose(physical["ramp_up_kw"], np.maximum(delta, 0), rtol=0, atol=1e-10)
            done = np.maximum(physical["terminated"], physical["truncated"])
            assert done.iloc[:-1].eq(0).all() and done.iloc[-1] == 1
            steps = set(physical["source_step"])
            if window["split"] == "train":
                seen_train.update(steps)
                train_ramps.extend(physical["ramp_abs_kw"].iloc[1:].tolist())
                for row in frame.itertuples(index=False):
                    components = json.loads(row.reward_components_json)
                    np.testing.assert_allclose(components["reward_total"], row.base_reward,
                                               rtol=1e-10, atol=1e-8)
                    key = f"obs{row.obs_dim}_act{row.action_dim}"
                    groups[key]["reward"].append(float(row.base_reward))
                    groups[key]["cost"].append(float(components["local_cost_weighted_reward"]) +
                                                float(components.get("community_settlement_reward", 0)))
                    groups[key]["hard_penalty"].append(float(components["hard_constraint_penalty"]))
            elif window["split"] == "validation":
                seen_val.update(steps)
            report["files"].append({"file": str(file), "split": window["split"],
                                    "seed": expected_seed, "rows": len(frame),
                                    "transitions": item["transitions"], "seconds": item["seconds"],
                                    "sha256": item["sha256"], "bytes": file.stat().st_size})
    assert seen_train.isdisjoint(seen_val)
    if train_ramps:
        values = np.asarray(train_ramps)
        report["train_only_scales"] = {
            "ramp_mean_kw": float(np.mean(values)), "ramp_median_kw": float(np.median(values)),
            "ramp_q90_kw": float(np.quantile(values, 0.9)),
            "ramp_q95_kw": float(np.quantile(values, 0.95)),
            "ramp_sample_count_excluding_episode_first_intervals": len(values),
            "cohorts": {key: {
                "rows": len(value["reward"]),
                "mean_abs_base_reward": float(np.mean(np.abs(value["reward"]))),
                "mean_abs_cost_component": float(np.mean(np.abs(value["cost"]))),
                "mean_hard_constraint_penalty": float(np.mean(value["hard_penalty"])),
            } for key, value in sorted(groups.items())},
        }
    report["checks_passed"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.data_dir)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"checks_passed": report["checks_passed"], "files": len(report["files"]),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
