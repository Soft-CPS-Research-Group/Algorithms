import csv
import json

from scripts.build_phase8_v102_remote_scorecard import build_scorecard


def _write_summary(path, rows):
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_phase8_scorecard_infers_wave_policy_and_retry(tmp_path):
    summary = tmp_path / "summary.csv"
    _write_summary(
        summary,
        [
            {
                "job_id": "retry",
                "job_name": "p8-v102-wave2-retry8h-2022-rbc-smart-deucalion-cpu-shac8fbe78",
                "status": "queued",
                "config_path": "configs/remote_20260524_wave2_retry8h_2022_rbc_smart_deucalion_cpu.yaml",
                "deucalion_time": "08:00:00",
            },
            {
                "job_id": "maddpg",
                "job_name": "p8-v102-wave3-2022-maddpg-v3-direct-seed456-deucalion-gpu-shac8fbe78",
                "status": "running",
                "config_path": "configs/remote_20260524_wave3_2022_maddpg_v3_direct_u16_b256_seed456_deucalion_gpu.yaml",
                "run_duration_seconds": "3600",
            },
        ],
    )

    scorecard = build_scorecard([summary])
    retry = next(row for row in scorecard if row["job_id"] == "retry")
    maddpg = next(row for row in scorecard if row["job_id"] == "maddpg")

    assert retry["wave"] == "wave2_retry8h"
    assert retry["policy"] == "RBCSmartPolicy"
    assert retry["attempt"] == "retry8h"
    assert retry["decision"] == "wait_queue"
    assert maddpg["wave"] == "wave3_maddpg_direct"
    assert maddpg["policy"] == "MADDPG_v3_direct"
    assert maddpg["recipe"] == "maddpg_v3_direct_u16_b256"
    assert maddpg["seed"] == "456"
    assert maddpg["runtime_hours"] == "1.000"


def test_phase8_scorecard_later_summary_overrides_older_status(tmp_path):
    old = tmp_path / "old.csv"
    new = tmp_path / "new.csv"
    common = {
        "job_id": "job-1",
        "job_name": "p8-v102-wave2-2022-random-server-shac8fbe78",
        "config_path": "configs/remote_20260524_wave2_2022_random_server.yaml",
    }
    _write_summary(old, [{**common, "status": "running"}])
    _write_summary(
        new,
        [
            {
                **common,
                "status": "finished",
                "community_cost_eur": "10.0",
                "ev_min_acceptable_feasible_rate": "0.1",
                "electrical_violation_kwh": "0.0",
            }
        ],
    )

    scorecard = build_scorecard([old, new])

    assert len(scorecard) == 1
    assert scorecard[0]["status"] == "finished"
    assert scorecard[0]["decision"] == "finished_sanity_only_ev_fail"
    assert scorecard[0]["community_cost_eur"] == "10.0000"


def test_phase8_scorecard_uses_submitted_jobs_metadata_when_summary_omits_names(tmp_path):
    summary = tmp_path / "summary.csv"
    submitted = tmp_path / "submitted_jobs.json"
    _write_summary(
        summary,
        [
            {
                "job_id": "gpu-1",
                "status": "running",
                "slurm_state": "RUNNING",
            }
        ],
    )
    submitted.write_text(
        json.dumps(
            [
                {
                    "job_id": "gpu-1",
                    "job_name": "p8-v102-wave3-2022-maddpg-v3-direct-seed123-deucalion-gpu-shac8fbe78",
                    "file_name": "remote_20260524_wave3_2022_maddpg_v3_direct_u16_b256_seed123_deucalion_gpu.yaml",
                    "target_host": "deucalion",
                    "image_tag": "sha-c8fbe78",
                }
            ]
        ),
        encoding="utf-8",
    )

    scorecard = build_scorecard([summary], [submitted])

    assert scorecard[0]["wave"] == "wave3_maddpg_direct"
    assert scorecard[0]["policy"] == "MADDPG_v3_direct"
    assert scorecard[0]["recipe"] == "maddpg_v3_direct_u16_b256"
    assert scorecard[0]["seed"] == "123"
    assert scorecard[0]["target_host"] == "deucalion"
    assert scorecard[0]["image_tag"] == "sha-c8fbe78"
