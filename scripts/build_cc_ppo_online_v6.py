#!/usr/bin/env python3
"""Build causal online CC-PPO V6 training configurations."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


DEFAULT_BASE_CONFIG = Path(
    "configs/experiments/cc_causal_price_control_v4/"
    "cc_ppo_base_price_fixed_1p00_seed789.yaml"
)
DEFAULT_OUTPUT_DIR = Path("configs/experiments/cc_ppo_online_v6")

NEUTRAL_REFERENCES = {
    "target_import": 5.742010629852302,
    "reference_cost": 1.483289225480985,
    "reference_peak": 11.635772715640455,
    "reference_ramping": 1.632929404854114,
    "reference_export": 4.878997665643692,
    "reference_price": 0.16398,
}

PROFILES: Mapping[str, Mapping[str, float]] = {
    "cost_first": {
        "w_peak": 0.05,
        "w_ramp": 0.05,
        "w_export": 0.01,
        "w_smoothness": 0.002,
    },
    "balanced": {
        "w_peak": 0.20,
        "w_ramp": 0.15,
        "w_export": 0.03,
        "w_smoothness": 0.005,
    },
    "peak_guarded": {
        "w_peak": 0.40,
        "w_ramp": 0.30,
        "w_export": 0.05,
        "w_smoothness": 0.010,
    },
}


def _manager_stage(profile: Mapping[str, float]) -> Dict[str, Any]:
    return {
        "algorithm": "CCLevel1",
        "count": 1,
        "frozen": False,
        "hyperparameters": {
            "num_steps": 336,
            "lr": 1.0e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "num_epochs": 4,
            "mini_batch_size": 96,
            "clip_coef": 0.2,
            "vf_coef": 0.5,
            "ent_coef": 0.002,
            "max_grad_norm": 0.5,
            "target_kl": 0.05,
            "hidden_dims": [128, 128],
            "c_dim": 17,
            "cc_action_interval": 4,
            "price_min": 0.5,
            "price_max": 1.3,
            "initial_log_std": -2.0,
            "reference_multiplier": 1.0,
            "policy_residual_scale": 1.0,
            "w_factor": 0.0,
            "w_smoothness": profile["w_smoothness"],
            "bc_pretrain_enabled": True,
            "bc_collect_steps": 8760,
            "bc_train_steps": 2000,
            "bc_lr": 1.0e-3,
            "bc_w_cost": 1.0,
            "bc_w_peak": profile["w_peak"],
            "bc_w_ramp": profile["w_ramp"],
            "bc_w_export": profile["w_export"],
            "bc_w_violation": 2.0,
            "bc_w_headroom": 0.3,
            "bc_reference_headroom": 2.0,
            "bc_target_import": NEUTRAL_REFERENCES["target_import"],
            "bc_reference_peak": NEUTRAL_REFERENCES["reference_peak"],
            "bc_reference_ramping": NEUTRAL_REFERENCES["reference_ramping"],
            "bc_reference_export": NEUTRAL_REFERENCES["reference_export"],
            "bc_reference_price": NEUTRAL_REFERENCES["reference_price"],
            "bc_dt_hours": 0.25,
            "bc_mult_scale": 1.0,
        },
    }


def build_config(
    base: Mapping[str, Any],
    *,
    profile_name: str,
    signal_price_charge_rate: float,
    episodes: int = 6,
    smoke_steps: int | None = None,
) -> Dict[str, Any]:
    """Create one V6 candidate while preserving the audited frozen leaf."""

    if profile_name not in PROFILES:
        raise ValueError(f"Unknown V6 profile: {profile_name}")
    if not 0.0 <= signal_price_charge_rate <= 1.0:
        raise ValueError("signal_price_charge_rate must lie within [0, 1]")
    if episodes < 2:
        raise ValueError("V6 requires at least one training and one evaluation episode")

    profile = PROFILES[profile_name]
    config = deepcopy(dict(base))
    config["metadata"] = {
        **dict(config.get("metadata") or {}),
        "experiment_name": "cc_ppo_online_v6",
        "run_name": f"CC-PPO V6 online {profile_name} seed 789",
        "description": (
            "Online Level-1 coordinator trained against exact community-market "
            "settlement while seventeen deterministic PPO leaves remain frozen, "
            "strict-local and community-blind."
        ),
    }
    tracking = config.setdefault("tracking", {})
    tracking["tags"] = {
        "protocol": "cc_ppo_online_v6",
        "controller": "trainable_cc_over_frozen_ppo",
        "objective_profile": profile_name,
        "settlement": "enabled",
        "settlement_reward": "exact_member_settlement",
        "leaf_frozen": "True",
        "leaf_community_blind": "True",
        "ppo_actor_price_conditioning": "False",
        "cc_price_scope": "strict_local_residual_base_only",
        "signal_price_charge_rate": str(signal_price_charge_rate),
        "cc_action_interval": "4",
        "price_range": "0.5_1.3",
        "evidence_horizon": "full_year_repeated_training_then_clean_eval",
        "promotion_eligible": "False",
    }
    tracking["log_frequency"] = 512
    tracking["progress_update_interval"] = 128

    simulator = config["simulator"]
    simulator["episodes"] = int(episodes)
    simulator["deterministic_finish"] = True
    simulator["reward_function"] = "CCRewardLevel1"
    simulator["reward_function_kwargs"] = {
        "cost_aggregation": "community_settled",
        "community_local_price_ratio": 0.8,
        "community_grid_export_price": 0.0,
        "w_cost": 1.0,
        "w_peak": profile["w_peak"],
        "w_ramp": profile["w_ramp"],
        "w_export": profile["w_export"],
        "w_violation": 2.0,
        "target_import": NEUTRAL_REFERENCES["target_import"],
        "reference_cost": NEUTRAL_REFERENCES["reference_cost"],
        "reference_peak": NEUTRAL_REFERENCES["reference_peak"],
        "reference_ramping": NEUTRAL_REFERENCES["reference_ramping"],
        "reference_export": NEUTRAL_REFERENCES["reference_export"],
        "reference_violation": 1.0,
    }
    simulator["community_market"]["enabled"] = True
    simulator["community_market"]["local_price_ratio_to_grid_import"] = 0.8
    simulator["community_market"]["intra_community_sell_ratio"] = 0.8
    simulator["community_market"]["grid_export_price"] = 0.0
    simulator["export"]["session_name"] = (
        f"cc-ppo-v6-online-{profile_name.replace('_', '-')}-seed789"
    )

    leaf = deepcopy(config["pipeline"][1])
    leaf["frozen"] = True
    leaf_params = leaf["exploration"]["params"]
    leaf_params["local_price_conditioning_enabled"] = False
    leaf_params["local_price_forecast_mode"] = "real_unmodified"
    leaf_params["residual_base_policy"] = "SignalAwareRBCSmartLocal"
    leaf_params["residual_base_price_conditioning_enabled"] = True
    leaf_params.setdefault("residual_base_policy_hyperparameters", {})[
        "signal_price_charge_rate"
    ] = float(signal_price_charge_rate)
    config["pipeline"] = [_manager_stage(profile), leaf]

    checkpointing = config["checkpointing"]
    checkpointing["checkpoint_interval"] = 35040
    checkpointing["restore_reward_normalizer"] = False

    if smoke_steps is not None:
        if smoke_steps < 8:
            raise ValueError("smoke_steps must be at least 8")
        simulator["simulation_start_time_step"] = 0
        simulator["simulation_end_time_step"] = int(smoke_steps) - 1
        simulator["episode_time_steps"] = int(smoke_steps)
        simulator["episodes"] = max(int(episodes), 3)
        simulator["export"]["session_name"] += f"-smoke-{smoke_steps}"
        manager = config["pipeline"][0]["hyperparameters"]
        manager["bc_collect_steps"] = max(1, int(smoke_steps) // 4)
        manager["bc_train_steps"] = 16
        manager["num_steps"] = max(8, min(48, int(smoke_steps) // 4))
        manager["mini_batch_size"] = min(16, manager["num_steps"])
        checkpointing["checkpoint_interval"] = None
        tracking["tags"]["evidence_horizon"] = f"real_smoke_{smoke_steps}_steps"

    return config


def build_configs(
    *,
    base_config: Path,
    output_dir: Path,
    signal_price_charge_rate: float = 0.45,
    episodes: int = 6,
    smoke_steps: int | None = None,
) -> list[Path]:
    base = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    rate_token = f"{signal_price_charge_rate:.2f}".replace(".", "p")
    for profile_name in PROFILES:
        config = build_config(
            base,
            profile_name=profile_name,
            signal_price_charge_rate=signal_price_charge_rate,
            episodes=episodes,
            smoke_steps=smoke_steps,
        )
        smoke_token = "" if smoke_steps is None else f"_smoke{smoke_steps}"
        output = output_dir / (
            f"cc_ppo_online_{profile_name}_charge_{rate_token}_seed789"
            f"{smoke_token}.yaml"
        )
        output.write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--signal-price-charge-rate", type=float, default=0.45)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--smoke-steps", type=int)
    args = parser.parse_args()

    for output in build_configs(
        base_config=args.base_config,
        output_dir=args.output_dir,
        signal_price_charge_rate=args.signal_price_charge_rate,
        episodes=args.episodes,
        smoke_steps=args.smoke_steps,
    ):
        print(output)


if __name__ == "__main__":
    main()
