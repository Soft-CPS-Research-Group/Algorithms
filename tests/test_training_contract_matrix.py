import json
from pathlib import Path

import yaml

from scripts import audit_training_contract_matrix as matrix


def test_training_contract_matrix_generates_profile_variants(monkeypatch, tmp_path):
    calls = []

    def fake_run_contract_audit(*, config_path, output_dir, job_id, skip_export):
        calls.append(
            {
                "config_path": config_path,
                "output_dir": output_dir,
                "job_id": job_id,
                "skip_export": skip_export,
            }
        )
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        profile = cfg["simulator"]["entity_encoding"]["profile"]
        encoded_dim = 10 if profile == "maddpg_v1" else 6
        return {
            "config_path": config_path,
            "dataset_name": cfg["simulator"]["dataset_name"],
            "algorithm_name": cfg["algorithm"]["name"],
            "interface": "entity",
            "topology_mode": "static",
            "seconds_per_time_step": 15,
            "num_agents": 1,
            "total_actions": 2,
            "ev_actions": 1,
            "bounds_issues": [],
            "topology": {
                "observation_dimensions": [encoded_dim],
                "action_dimensions": [2],
            },
            "entity_encoding": {
                "profile": profile,
                "serving_observation_names": "encoded",
            },
            "agents": [
                {
                    "agent_index": 0,
                    "building_name": "Building_1",
                    "raw_observation_count": 12,
                    "serving_observation_count": encoded_dim,
                    "encoded_observation_dimension": encoded_dim,
                    "action_count": 2,
                }
            ],
            "manifest_path": str(output_dir / "bundle" / "artifact_manifest.json"),
            "manifest_valid": True,
            "agent_artifact_count": 1,
            "agent_format": "onnx",
        }

    monkeypatch.setattr(matrix, "run_contract_audit", fake_run_contract_audit)

    payload = matrix.run_training_contract_matrix(
        config_paths=[Path("configs/templates/maddpg/maddpg_local.yaml")],
        profiles=["maddpg_v1", "maddpg_v2_compact"],
        output_dir=tmp_path,
        matrix_name="test_matrix",
        skip_export=False,
    )

    assert len(calls) == 2
    assert len(payload["variants"]) == 2
    assert {row["profile"] for row in payload["variants"]} == {"maddpg_v1", "maddpg_v2_compact"}
    assert (tmp_path / "matrix_summary.csv").is_file()
    assert (tmp_path / "agent_contract_summary.csv").is_file()
    assert (tmp_path / "kpi_contract.json").is_file()

    generated = sorted((tmp_path / "generated_configs").glob("*.yaml"))
    assert len(generated) == 2
    generated_profiles = {
        yaml.safe_load(path.read_text(encoding="utf-8"))["simulator"]["entity_encoding"]["profile"]
        for path in generated
    }
    assert generated_profiles == {"maddpg_v1", "maddpg_v2_compact"}

    matrix_summary = json.loads((tmp_path / "matrix_summary.json").read_text(encoding="utf-8"))
    assert matrix_summary["tracked_kpis"] == matrix.DEFAULT_KPIS
