import json
import pytest
from pathlib import Path

import yaml

from scripts.build_remote_job_report import build_report
from scripts.collect_remote_results import _read_jobs_file


def test_read_jobs_file_accepts_submission_manifest(tmp_path: Path):
    manifest = tmp_path / "submitted.json"
    manifest.write_text(
        json.dumps(
            {
                "submissions": [
                    {"response": {"job_id": "job-a"}},
                    {"response": {"job_id": "job-b"}},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert _read_jobs_file(manifest) == ["job-a", "job-b"]


def test_build_remote_job_report_extracts_runtime_and_config_flags(tmp_path: Path):
    results_dir = tmp_path / "remote_results"
    job_dir = results_dir / "jobs" / "job-a"
    job_dir.mkdir(parents=True)
    (job_dir / "job_info.json").write_text(
        json.dumps(
            {
                "run_duration_seconds": 876.0,
                "queue_wait_seconds": 12.0,
                "total_duration_seconds": 888.0,
                "deucalion_options": {
                    "partition": "normal-a100-80",
                    "time": "48:00:00",
                    "cpus_per_task": 8,
                    "mem_gb": 96,
                    "gpus": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "status.json").write_text(json.dumps({"status": "finished", "exit_code": 0}), encoding="utf-8")
    (job_dir / "result.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    (job_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "tracking": {"mlflow_enabled": False},
                "checkpointing": {"checkpoint_interval": None},
                "simulator": {
                    "dataset_name": "citylearn_challenge_2022_phase_all_plus_evs_data_2026_05_21",
                    "episodes": 1,
                    "simulation_start_time_step": 0,
                    "simulation_end_time_step": 8759,
                    "episode_time_steps": 8760,
                    "export": {
                        "export_kpis_on_episode_end": True,
                        "final_episode_only": True,
                        "include_business_as_usual": False,
                        "export_business_as_usual_timeseries": False,
                    },
                },
                "training": {"seed": 123, "steps_between_training_updates": 8, "target_update_interval": 8},
                "pipeline": [
                    {
                        "algorithm": "MADDPG",
                        "count": 1,
                        "hyperparameters": {"seed": 123},
                        "replay_buffer": {"batch_size": 512, "capacity": 200000},
                        "exploration": {"params": {"use_amp": True}},
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (job_dir / "logs_tail.txt").write_text(
        "Step Duration: 0.10\nStep Duration: 0.20\nDevice selected: cuda\n",
        encoding="utf-8",
    )

    rows = build_report(
        results_dir,
        [
            {
                "job_id": "job-a",
                "status": "finished",
                "target_host": "deucalion",
                "image_tag": "sha-test",
                "device_selected": "cuda",
            }
        ],
    )

    row = rows[0]

    assert row["configured_env_steps"] == 8760
    assert row["algorithm"] == "MADDPG"
    assert row["full_year_check"] == "ok"
    assert row["seconds_per_env_step"] == 0.1
    assert row["batch_size"] == 512
    assert row["use_amp"] == "true"
    assert row["checkpoint_effective"] == "false"
    assert row["export_include_business_as_usual"] == "false"
    assert row["export_kpis_final_episode_only"] == "true"
    assert row["export_timeseries_final_episode_only"] == "true"
    assert row["tail_step_duration_median_seconds"] == pytest.approx(0.15)
    assert row["risk_flags"] == ""
