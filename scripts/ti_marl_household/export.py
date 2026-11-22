"""Export a frozen household policy and prepare its ordinary Algorithms evaluation job."""

import argparse
import json
from pathlib import Path

import torch
import yaml

from scripts.ti_marl_household.experiment import Episodes
from scripts.ti_marl_household.fixture import evaluation_job


def export(directory, output, seed, case_index, tolerance_soc=0.05, ppo=None, margin=0):
    run = Episodes(directory, seed)
    agent = run.agent
    step = 0  # Reward-only parameter search produces no PPO optimiser steps.
    if ppo:
        step = int(torch.load(ppo, map_location="cpu", weights_only=False)["step"])
        agent.load_checkpoint(str(ppo))
        weights = agent.actor.coordination.detach().cpu().tolist()
    else:
        weights = json.loads((directory / f"refined-{seed}.json").read_text())["weights"]
    agent.actor.tolerance_soc = tolerance_soc
    agent.actor.departure_margin_seconds = margin
    for hyper in (
        run.cfg["pipeline"][0]["hyperparameters"],
        agent.config["algorithm"]["hyperparameters"],
    ):
        hyper["departure_margin_seconds"] = margin
        hyper.setdefault("feasibility", {})["ev_service_tolerance_ratio"] = tolerance_soc
        hyper["reward_penalties"] = {
            "ev_excess_kwh": 0.05,
            "stationary_terminal_deficit_kwh": 0.16 / 0.94**0.5,
        }
    # Exercise the policy with the actual topology before serialising compiler compatibility.
    result = run.episode(case_index, weights)
    output.mkdir(exist_ok=True, parents=True)
    agent.save_checkpoint(str(output), step=step)
    cfg = evaluation_job(run.cfg, output / "latest_checkpoint.pth", result["start"])
    (directory / f"eval-{seed}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (directory / f"reference-{seed}.json").write_text(json.dumps(result))
    with torch.no_grad():
        agent.actor.coordination.zero_()
    agent.load_checkpoint(str(output))
    torch.testing.assert_close(
        agent.actor.coordination.detach().cpu(), torch.tensor(weights, dtype=torch.float32)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--case", type=int, default=90)
    p.add_argument("--service-tolerance", type=float, default=0.05)
    p.add_argument(
        "--ppo", type=Path, help="Export a native PPO checkpoint instead of fitted readouts"
    )
    p.add_argument("--departure-margin", type=int, default=0)
    args = p.parse_args()
    export(
        args.directory,
        args.output,
        args.seed,
        args.case,
        args.service_tolerance,
        args.ppo,
        args.departure_margin,
    )
