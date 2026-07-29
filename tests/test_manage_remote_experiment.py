from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from scripts import manage_remote_experiment as manager


REPO_ROOT = Path(__file__).resolve().parents[1]


def _prepare_payload(tmp_path: Path) -> Path:
    output = tmp_path / "payload.json"
    result = manager.main(
        [
            "prepare",
            "--config",
            str(REPO_ROOT / "configs/templates/baselines/random_local.yaml"),
            "--job-name",
            "skill-test-job",
            "--submitted-by",
            "Test User",
            "--image-tag",
            "sha-test123",
            "--target-host",
            "server",
            "--output",
            str(output),
        ]
    )
    assert result == 0
    return output


def test_prepare_validates_config_and_builds_inline_payload(tmp_path: Path):
    output = _prepare_payload(tmp_path)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["job_name"] == "skill-test-job"
    assert payload["target_host"] == "server"
    assert payload["image_tag"] == "sha-test123"
    assert payload["save_as"] == "random_local.yaml"
    assert payload["config"]["pipeline"][0]["algorithm"] == "RandomPolicy"


def test_prepare_rejects_save_as_with_directory_components(tmp_path: Path, capsys):
    output = tmp_path / "payload.json"
    result = manager.main(
        [
            "prepare",
            "--config",
            str(REPO_ROOT / "configs/templates/baselines/random_local.yaml"),
            "--job-name",
            "skill-test-job",
            "--submitted-by",
            "Test User",
            "--image-tag",
            "sha-test123",
            "--target-host",
            "server",
            "--save-as",
            "configs/random_local.yaml",
            "--output",
            str(output),
        ]
    )

    assert result == 2
    assert "filename only" in capsys.readouterr().err
    assert not output.exists()


def test_submit_requires_confirmation_before_network_call(tmp_path: Path, monkeypatch, capsys):
    payload = _prepare_payload(tmp_path)

    def fail_request(*args, **kwargs):
        raise AssertionError("network must not be called without confirmation")

    monkeypatch.setattr(manager, "_request_json", fail_request)
    result = manager.main(
        [
            "submit",
            "--server",
            "http://orchestrator.test",
            "--payload",
            str(payload),
            "--campaign-dir",
            str(tmp_path / "campaign"),
        ]
    )

    assert result == 2
    assert "--confirm-submit" in capsys.readouterr().err


def test_submit_persists_payload_response_and_collector_compatible_manifest(tmp_path: Path, monkeypatch):
    payload_path = _prepare_payload(tmp_path)
    campaign_dir = tmp_path / "campaign"

    def fake_request(server, path, **kwargs):
        assert server == "http://orchestrator.test"
        assert path == "/run-simulation"
        assert kwargs["method"] == "POST"
        return {
            "job_id": "job-123",
            "status": "queued",
            "host": "server",
            "image_tag": "sha-test123",
            "image": "calof/opeva_simulator:sha-test123",
        }

    monkeypatch.setattr(manager, "_request_json", fake_request)
    result = manager.main(
        [
            "submit",
            "--server",
            "http://orchestrator.test",
            "--payload",
            str(payload_path),
            "--campaign-dir",
            str(campaign_dir),
            "--confirm-submit",
        ]
    )

    assert result == 0
    submitted = json.loads((campaign_dir / "submitted_jobs.json").read_text(encoding="utf-8"))
    assert submitted[0]["job_id"] == "job-123"
    assert submitted[0]["payload"]["config"]["pipeline"][0]["algorithm"] == "RandomPolicy"

    from scripts.collect_remote_results import _read_jobs_file

    assert _read_jobs_file(campaign_dir / "submitted_jobs.json") == ["job-123"]


def test_preflight_strict_verifies_host_and_image_without_mutation(tmp_path: Path, monkeypatch):
    responses = {
        "/health": {"status": "ok"},
        "/hosts": {
            "available_hosts": ["server"],
            "hosts": {"server": {"online": True}},
        },
        "/queue": [],
        "/job-images/versions": [{"tag": "sha-test123"}],
    }

    def fake_request(server, path, **kwargs):
        assert kwargs.get("method", "GET") == "GET"
        return responses[path]

    monkeypatch.setattr(manager, "_request_json", fake_request)
    output = tmp_path / "preflight.json"
    result = manager.main(
        [
            "preflight",
            "--server",
            "http://orchestrator.test",
            "--target-host",
            "server",
            "--image-tag",
            "sha-test123",
            "--strict",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    snapshot = json.loads(output.read_text(encoding="utf-8"))
    assert snapshot["target_host_found"] is True
    assert snapshot["target_host_online"] is True
    assert snapshot["image_tag_found"] is True
    assert snapshot["errors"] == {}


def test_preflight_strict_rejects_stale_union_authentication(tmp_path: Path, monkeypatch):
    responses = {
        "/health": {"status": "ok"},
        "/hosts": {
            "available_hosts": ["union-inesctec"],
            "hosts": {
                "union-inesctec": {
                    "online": True,
                    "info": {
                        "union_auth": {
                            "status": "authenticated",
                            "updated_at": 1_000.0,
                        }
                    },
                }
            },
        },
        "/queue": [],
        "/job-images/versions": {
            "tags": [{"name": "sha-union123", "union_ready": True}]
        },
    }

    def fake_request(server, path, **kwargs):
        assert kwargs.get("method", "GET") == "GET"
        return responses[path]

    monkeypatch.setattr(manager, "_request_json", fake_request)
    monkeypatch.setattr(manager.time, "time", lambda: 100_000.0)
    output = tmp_path / "union-preflight.json"
    result = manager.main(
        [
            "preflight",
            "--server",
            "http://orchestrator.test",
            "--target-host",
            "union-inesctec",
            "--image-tag",
            "sha-union123",
            "--max-union-auth-age-seconds",
            "3600",
            "--strict",
            "--output",
            str(output),
        ]
    )

    assert result == 3
    snapshot = json.loads(output.read_text(encoding="utf-8"))
    assert snapshot["union_auth_status"] == "authenticated"
    assert snapshot["union_auth_age_seconds"] == 99_000.0
    assert snapshot["union_auth_fresh"] is False
    assert snapshot["union_image_ready"] is True


def test_watch_reads_submission_manifest_and_writes_status_history(tmp_path: Path, monkeypatch):
    manifest = tmp_path / "submitted_jobs.json"
    manifest.write_text(json.dumps([{"response": {"job_id": "job-123"}}]), encoding="utf-8")

    def fake_request(server, path, **kwargs):
        assert path == "/status/job-123"
        return {"job_id": "job-123", "status": "finished", "exit_code": 0}

    monkeypatch.setattr(manager, "_request_json", fake_request)
    output_dir = tmp_path / "monitoring"
    result = manager.main(
        [
            "watch",
            "--server",
            "http://orchestrator.test",
            "--jobs-file",
            str(manifest),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0
    latest = json.loads((output_dir / "latest_status.json").read_text(encoding="utf-8"))
    assert latest[0]["status"] == "finished"
    assert "job-123" in (output_dir / "status_history.jsonl").read_text(encoding="utf-8")


def test_archive_writes_campaign_document_and_single_ledger_row(tmp_path: Path):
    config = yaml.safe_load(
        (REPO_ROOT / "configs/templates/baselines/random_local.yaml").read_text(encoding="utf-8")
    )
    config["simulator"]["simulation_start_time_step"] = 0
    config["simulator"]["simulation_end_time_step"] = 95
    config["training"]["seed"] = 123
    manifest = tmp_path / "submitted_jobs.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "job_id": "job-123",
                    "status": "queued",
                    "payload": {
                        "config": config,
                        "job_name": "archive-test",
                        "submitted_by": "Test User",
                        "target_host": "server",
                        "image_tag": "sha-test123",
                        "save_as": "archive-test.yaml",
                    },
                    "response": {"job_id": "job-123", "status": "queued"},
                }
            ]
        ),
        encoding="utf-8",
    )
    summary = tmp_path / "summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "job_id",
                "job_name",
                "algorithm",
                "seed",
                "status",
                "community_cost_eur",
                "ev_min_acceptable_feasible_rate",
                "ev_within_tolerance_feasible_rate",
                "electrical_violation_kwh",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "job_id": "job-123",
                "job_name": "archive-test",
                "algorithm": "RandomPolicy",
                "seed": "123",
                "status": "finished",
                "community_cost_eur": "100.0",
                "ev_min_acceptable_feasible_rate": "1.0",
                "ev_within_tolerance_feasible_rate": "0.9",
                "electrical_violation_kwh": "0.0",
            }
        )
    scorecard = tmp_path / "scorecard.csv"
    scorecard.write_text("job_id,decision\njob-123,reference\n", encoding="utf-8")
    history_dir = tmp_path / "history"

    args = [
        "archive",
        "--campaign",
        "test-campaign",
        "--jobs-file",
        str(manifest),
        "--summary-csv",
        str(summary),
        "--scorecard-csv",
        str(scorecard),
        "--gates-profile",
        "phase6-default",
        "--baseline",
        "RBCSmart same-window",
        "--evidence-horizon",
        "smoke",
        "--source-commit",
        "abcdef123456",
        "--history-dir",
        str(history_dir),
        "--note",
        "Synthetic test evidence only.",
    ]
    assert manager.main(args) == 0

    documents = list(history_dir.glob("*_test-campaign.md"))
    assert len(documents) == 1
    document_text = documents[0].read_text(encoding="utf-8")
    assert "abcdef123456" in document_text
    assert "RandomPolicy" in document_text
    assert "reference" in document_text

    with (history_dir / "index.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["campaign"] == "test-campaign"
    assert rows[0]["job_count"] == "1"
    assert rows[0]["finished_count"] == "1"
    assert rows[0]["scorecard_decisions"] == "reference:1"

    assert manager.main(args) == 2


def test_archive_replace_refuses_document_outside_history_directory(tmp_path: Path):
    config = yaml.safe_load(
        (REPO_ROOT / "configs/templates/baselines/random_local.yaml").read_text(encoding="utf-8")
    )
    manifest = tmp_path / "submitted_jobs.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "job_id": "job-escape",
                    "payload": {
                        "config": config,
                        "job_name": "archive-escape-test",
                        "image_tag": "sha-test123",
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    summary = tmp_path / "summary.csv"
    summary.write_text("job_id,status\njob-escape,finished\n", encoding="utf-8")
    history_dir = tmp_path / "history"
    history_dir.mkdir()
    outside_document = tmp_path / "outside.md"
    index = history_dir / "index.csv"
    with index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manager.HISTORY_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "campaign": "escape-campaign",
                "history_document": str(outside_document),
            }
        )

    result = manager.main(
        [
            "archive",
            "--campaign",
            "escape-campaign",
            "--jobs-file",
            str(manifest),
            "--summary-csv",
            str(summary),
            "--gates-profile",
            "test-gates",
            "--baseline",
            "test-baseline",
            "--history-dir",
            str(history_dir),
            "--replace",
        ]
    )

    assert result == 2
    assert not outside_document.exists()
