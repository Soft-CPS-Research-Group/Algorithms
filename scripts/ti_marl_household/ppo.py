"""Train the native V2 prefix PPO from the transferred EV with household reward."""

import argparse
import json
import subprocess
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import yaml

from scripts.ti_marl_household import experiment
from scripts.ti_marl_household.audit import metrics, timing
from scripts.ti_marl_household.fixture import SLOTS


def evaluate(i):
    episodes = experiment.episodes
    weights = episodes.agent.actor.coordination.detach().cpu().numpy().copy()
    return episodes.episode(i, weights)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--processes", type=int, default=3)
    p.add_argument("--episodes", type=int, default=48)
    p.add_argument("--job-command", default="algorithms-job")
    args = p.parse_args()
    root = args.directory.resolve()
    trial = experiment.Episodes(root, args.seed)
    cases = trial.cases
    train = [i for i, c in enumerate(cases) if c["split"] == "train"]
    rng = np.random.default_rng(1128)
    indices = rng.choice(train, args.episodes, replace=False)
    trial.episode(int(indices[0]), np.zeros((5, 8)))
    initial = root / f"ppo-initial-{args.seed}"
    trial.agent.save_checkpoint(str(initial), step=0)
    cfg = trial.cfg
    cfg["training"].update(episodes=args.episodes, seed=args.seed)
    cfg["simulator"].update(
        simulation_start_time_step=0,
        simulation_end_time_step=max(cases[i]["start"] for i in train) + SLOTS - 1,
        repeat_episode_scenario=False,
        episode_time_steps=[[cases[i]["start"], cases[i]["start"] + SLOTS - 1] for i in indices],
    )
    h = cfg["pipeline"][0]["hyperparameters"]
    h.update(
        learning_rate=3e-5,
        rollout_steps=SLOTS - 1,
        ppo_epochs=2,
        trace={"enabled": False},
        reward_penalties={
            "ev_excess_kwh": 0.05,
            "stationary_terminal_deficit_kwh": float(0.16 / np.sqrt(0.94)),
        },
    )
    cfg["checkpointing"] = {
        "resume_training": True,
        "checkpoint_local_path": str(initial / "latest_checkpoint.pth"),
        "restore_optimizers": False,
    }
    config = root / f"ppo-{args.seed}.yaml"
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    job = f"household-ppo-{args.seed}"
    subprocess.run(
        [
            args.job_command,
            "--config",
            str(config),
            "--job_id",
            job,
            "--base-dir",
            str(root / "public"),
        ],
        check=True,
    )
    checkpoint = root / "public/jobs" / job / "checkpoints/latest_checkpoint.pth"
    ids = [i for i, c in enumerate(cases) if c["split"] in {"dev", "test"}]
    with get_context("fork").Pool(
        args.processes, experiment.worker_init, (root, args.seed, checkpoint)
    ) as pool:
        rows = pool.map(evaluate, ids, chunksize=1)
    results = {}
    for split in ("dev", "test"):
        selected = [r for r in rows if r["split"] == split]
        results[split] = dict(
            overall=metrics(selected),
            price=timing(selected, "price"),
            solar=timing(selected, "solar"),
        )
    (root / f"ppo-score-{args.seed}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
