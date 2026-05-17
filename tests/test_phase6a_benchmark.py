import json
from argparse import Namespace

import yaml

from scripts import run_phase6a_benchmark as phase6a


def test_phase6a_dry_run_generates_configs_and_summary(tmp_path):
    output_dir = tmp_path / "phase6a"
    args = Namespace(
        output_dir=str(output_dir),
        dataset=["15s"],
        agent=["random", "maddpg"],
        maddpg_variant=["noop_centered"],
        seed=[123],
        episodes=1,
        steps=16,
        steps_15s=None,
        steps_2022=None,
        start=0,
        full_window=False,
        no_kpi_export=True,
        metric_interval=8,
        batch_size=16,
        buffer_capacity=1000,
        actor_layers="32",
        critic_layers="64,32",
        random_exploration_steps=4,
        sigma=0.1,
        dry_run=True,
        fail_fast=False,
    )

    payload = phase6a.run_phase6a(args)

    assert payload["output_dir"] == str(output_dir)
    assert len(payload["rows"]) == 2
    assert {row["status"] for row in payload["rows"]} == {"planned"}
    assert (output_dir / "benchmark_summary.csv").is_file()
    assert (output_dir / "benchmark_summary.json").is_file()
    assert (output_dir / "README.md").is_file()

    random_config = output_dir / "generated_configs" / "phase6a_15s_random_random_seed123.yaml"
    maddpg_config = output_dir / "generated_configs" / "phase6a_15s_maddpg_noop_centered_seed123.yaml"
    assert random_config.is_file()
    assert maddpg_config.is_file()

    random_payload = yaml.safe_load(random_config.read_text(encoding="utf-8"))
    maddpg_payload = yaml.safe_load(maddpg_config.read_text(encoding="utf-8"))

    assert random_payload["simulator"]["export"]["mode"] == "none"
    assert random_payload["simulator"]["episode_time_steps"] == 16
    assert maddpg_payload["algorithm"]["name"] == "MADDPG"
    assert maddpg_payload["algorithm"]["replay_buffer"]["batch_size"] == 16
    assert maddpg_payload["algorithm"]["networks"]["actor"]["layers"] == [32]
    assert maddpg_payload["algorithm"]["networks"]["critic"]["layers"] == [64, 32]
    assert (
        maddpg_payload["algorithm"]["exploration"]["params"]["initial_exploration_strategy"]
        == "noop_centered"
    )
    assert maddpg_payload["algorithm"]["exploration"]["params"]["random_exploration_steps"] == 4

    summary = json.loads((output_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert summary["settings"]["dry_run"] is True
    assert summary["settings"]["kpi_export"] is False
