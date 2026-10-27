"""Resolve explicit training + evaluation counts while preserving legacy runs."""

from __future__ import annotations


def episode_plan(config):
    training = config.get("training") or {}
    evaluation = config.get("evaluation") or {}
    simulator = config.get("simulator") or {}
    profile = (config.get("job") or {}).get("profile", "train")
    if profile == "evaluate":
        return {"training": 0, "evaluation": 1, "source": "evaluate_profile"}
    if profile == "smoke":
        return {"training": 1, "evaluation": 1, "source": "smoke_profile"}
    if training.get("episodes") is not None:
        return {
            "training": int(training["episodes"]),
            "evaluation": int(evaluation.get("episodes", 1)),
            "source": "explicit",
        }
    total = int(simulator.get("episodes", 1))
    final = int(bool(simulator.get("deterministic_finish", False)))
    return {"training": total - final, "evaluation": final, "source": "legacy_total"}


def set_legacy_episode_budget(config, total, deterministic_finish=False):
    """Compatibility for research CLI options whose episodes include evaluation."""
    total = int(total)
    final = int(bool(deterministic_finish))
    if total < 1:
        raise ValueError("episodes must be positive")
    config.setdefault("training", {})["episodes"] = total - final
    config.setdefault("evaluation", {})["episodes"] = final
    config.setdefault("simulator", {}).update(episodes=total, deterministic_finish=bool(final))
