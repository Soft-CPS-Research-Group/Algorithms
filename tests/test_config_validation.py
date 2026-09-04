import copy
from pathlib import Path

import pytest
import yaml

from utils.config_schema import validate_config


@pytest.fixture
def base_config():
    config_path = Path("configs/config.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_validate_config_success(base_config):
    # Should not raise
    validate_config(base_config)


def test_validate_config_accepts_metadata_community_name(base_config):
    config = copy.deepcopy(base_config)
    config["metadata"]["community_name"] = "porto_cluster_a"
    validate_config(config)


def test_validate_config_accepts_building_local_entity_profile(base_config):
    config = copy.deepcopy(base_config)
    config["simulator"]["entity_encoding"]["profile"] = "building_local_v1"
    validate_config(config)


def test_validate_config_accepts_strict_local_rbc_policy(base_config):
    config = copy.deepcopy(base_config)
    config["pipeline"] = [
        {
            "algorithm": "RBCSmartLocalPolicy",
            "count": 1,
            "hyperparameters": {},
        }
    ]
    validate_config(config)


def _ti_marl_stage():
    return {
        "algorithm": "TIMARL",
        "count": 1,
        "hyperparameters": {
            "contract_version": "ti_marl_v1",
            "typed_interfaces_dir": "local/generated_interfaces",
            "backbone": {"name": "mappo"},
            "actor": {
                "d_model": 128,
                "attention_heads": 4,
                "relation_layers": 2,
            },
            "critic": {"kind": "set"},
            "feasibility": {
                "kind": "analytic_projection",
                "deferrable_service_margin_seconds": 3600.0,
                "ev_service_jit_buffer_seconds": 1800.0,
                "ev_service_jit_minimum_average_fraction": 0.25,
                "protect_ev_service_target": True,
                "enforce_ev_discharge_reserve": True,
                "ev_v2g_reserve_margin_ratio": 0.02,
                "enforce_ev_economic_guard": True,
                "ev_v2g_avoided_import_value_ratio": 0.8,
                "ev_v2g_minimum_profit_margin_eur_per_kwh": 0.015,
                "ev_v2g_degradation_cost_eur_per_kwh": 0.0,
                "ev_v2g_require_local_demand": True,
            },
        },
    }


def test_validate_config_accepts_ti_marl_entity_dynamic(base_config):
    config = copy.deepcopy(base_config)
    config["simulator"]["interface"] = "entity"
    config["simulator"]["topology_mode"] = "dynamic"
    config["simulator"]["central_agent"] = False
    config["pipeline"] = [_ti_marl_stage()]
    parsed = validate_config(config)
    assert parsed.pipeline[0].algorithm == "TIMARL"
    assert (
        parsed.pipeline[0].hyperparameters.feasibility.deferrable_service_margin_seconds
        == 3600.0
    )
    feasibility = parsed.pipeline[0].hyperparameters.feasibility
    assert feasibility.ev_service_jit_buffer_seconds == pytest.approx(1800.0)
    assert feasibility.ev_service_jit_minimum_average_fraction == pytest.approx(
        0.25
    )
    assert feasibility.protect_ev_service_target
    assert feasibility.enforce_ev_discharge_reserve
    assert feasibility.ev_v2g_reserve_margin_ratio == pytest.approx(0.02)
    assert feasibility.enforce_ev_economic_guard
    assert feasibility.ev_v2g_avoided_import_value_ratio == pytest.approx(0.8)
    assert feasibility.ev_v2g_minimum_profit_margin_eur_per_kwh == pytest.approx(
        0.015
    )
    assert feasibility.ev_v2g_degradation_cost_eur_per_kwh == pytest.approx(0.0)
    assert feasibility.ev_v2g_require_local_demand


def test_validate_config_accepts_ti_marl_electrical_service_preflight(base_config):
    config = copy.deepcopy(base_config)
    config["simulator"]["interface"] = "entity"
    config["simulator"]["central_agent"] = False
    stage = _ti_marl_stage()
    stage["hyperparameters"]["require_declared_electrical_service"] = True
    config["pipeline"] = [stage]

    parsed = validate_config(config)

    assert parsed.pipeline[0].hyperparameters.require_declared_electrical_service


def test_validate_config_accepts_ti_marl_typed_behavior_cloning(base_config):
    config = copy.deepcopy(base_config)
    config["simulator"]["interface"] = "entity"
    config["simulator"]["central_agent"] = False
    stage = _ti_marl_stage()
    stage["hyperparameters"]["actor"]["group_context_kind"] = "action_conditioned"
    stage["hyperparameters"]["actor"][
        "deterministic_mode_strategy"
    ] = "expected_signed"
    stage["hyperparameters"]["actor"][
        "deterministic_mode_strategy_by_group_type"
    ] = {"ev_session": "argmax"}
    stage["hyperparameters"]["actor"][
        "deterministic_expected_signed_gain_by_group_type"
    ] = {"stationary_storage": 2.0}
    stage["hyperparameters"]["actor"][
        "deterministic_expected_signed_deadband_by_group_type"
    ] = {"stationary_storage": 0.05}
    stage["hyperparameters"]["actor"][
        "deterministic_non_idle_logit_margin_by_group_type"
    ] = {"ev_session": 0.25}
    stage["hyperparameters"]["advantage_normalization"] = "per_agent"
    stage["hyperparameters"]["policy_credit_assignment"] = "typed_group"
    stage["hyperparameters"]["ppo_policy_group_types"] = ["ev_session"]
    stage["hyperparameters"]["actor_update_scope"] = "selected_group_heads"
    stage["hyperparameters"]["actor_update_group_types"] = ["ev_session"]
    stage["hyperparameters"]["policy_anchor_coeff"] = 0.05
    stage["hyperparameters"]["policy_anchor_coeff_by_group_type"] = {
        "stationary_storage": 0.0,
        "ev_session": 0.1,
        "deferrable": 0.2,
    }
    stage["hyperparameters"]["policy_anchor_reset_on_resume"] = True
    stage["hyperparameters"][
        "exclude_intervened_actions_from_policy_loss"
    ] = True
    stage["hyperparameters"]["intervention_distillation_coeff"] = 0.1
    stage["hyperparameters"]["discount_timebase_seconds"] = 3600.0
    stage["hyperparameters"]["ev_planning"] = {
        "auxiliary_coeff": 0.25,
        "balance_targets": True,
        "fraction_coeff": 0.2,
        "replay_capacity_per_reason": 64,
        "replay_samples_per_reason": 8,
        "charge_fraction": 0.95,
        "discharge_fraction": 0.40,
        "service_tolerance_ratio": 0.05,
        "v2g_service_margin_ratio": 0.06,
        "urgency_duty_ratio": 0.85,
        "minimum_price_spread": 0.001,
        "minimum_v2g_price_spread": 0.02,
        "minimum_v2g_departure_hours": 1.5,
        "v2g_avoided_import_value_ratio": 0.8,
        "v2g_minimum_profit_margin_eur_per_kwh": 0.03,
        "v2g_degradation_cost_eur_per_kwh": 0.01,
        "opportunity_value_kind": "community_marginal_import",
    }
    stage["hyperparameters"]["storage_planning"] = {
        "auxiliary_coeff": 0.15,
        "balance_targets": True,
        "fraction_coeff": 0.3,
        "replay_capacity_per_reason": 32,
        "replay_samples_per_reason": 6,
        "charge_fraction": 0.6,
        "discharge_fraction": 0.5,
        "minimum_soc_ratio": 0.25,
        "maximum_soc_ratio": 0.85,
        "minimum_price_spread": 0.02,
        "pv_surplus_threshold_kw": 0.5,
        "import_threshold_kw": 0.75,
        "price_regime_kind": "relative_forecast",
        "forecast_mean_margin_fraction": 0.15,
        "forecast_edge_margin_fraction": 0.08,
        "forecast_spread_floor_ratio": 0.04,
        "scale_price_fraction_by_opportunity": True,
        "minimum_price_fraction_scale": 0.45,
    }
    stage["hyperparameters"]["entropy_coeff_by_group_type"] = {
        "stationary_storage": 0.05,
        "ev_session": 0.005,
    }
    stage["hyperparameters"]["behavior_cloning"] = {
        "enabled": True,
        "demonstration_episodes": 1,
        "max_samples": 672,
        "pretraining_epochs": 2,
        "batch_size": 32,
        "learning_rate": 1.0e-4,
        "balance_action_modes": True,
        "mode_balance_exponent": 0.5,
        "max_mode_weight": 3.0,
        "balanced_loss_kind": "hierarchical_mode_mean",
        "calibration_epochs": 2,
        "calibration_learning_rate": 5.0e-5,
        "teacher": {
            "policy": "RBCSmartPolicy",
            "hyperparameters": {"allow_v2g": True},
        },
    }
    config["pipeline"] = [stage]

    parsed = validate_config(config)

    behavior_cloning = parsed.pipeline[0].hyperparameters.behavior_cloning
    assert parsed.pipeline[0].hyperparameters.policy_anchor_coeff == 0.05
    assert parsed.pipeline[0].hyperparameters.ppo_policy_group_types == [
        "ev_session"
    ]
    assert (
        parsed.pipeline[0].hyperparameters.actor_update_scope
        == "selected_group_heads"
    )
    assert parsed.pipeline[0].hyperparameters.actor_update_group_types == [
        "ev_session"
    ]
    assert (
        parsed.pipeline[0].hyperparameters.policy_anchor_coeff_by_group_type
        == {
            "stationary_storage": 0.0,
            "ev_session": 0.1,
            "deferrable": 0.2,
        }
    )
    assert parsed.pipeline[0].hyperparameters.policy_anchor_reset_on_resume
    assert (
        parsed.pipeline[0]
        .hyperparameters.exclude_intervened_actions_from_policy_loss
    )
    assert (
        parsed.pipeline[0].hyperparameters.intervention_distillation_coeff
        == 0.1
    )
    assert parsed.pipeline[0].hyperparameters.discount_timebase_seconds == 3600.0
    assert parsed.pipeline[0].hyperparameters.ev_planning.auxiliary_coeff == 0.25
    assert parsed.pipeline[0].hyperparameters.ev_planning.balance_targets
    assert parsed.pipeline[0].hyperparameters.ev_planning.fraction_coeff == 0.2
    assert (
        parsed.pipeline[0].hyperparameters.ev_planning.replay_capacity_per_reason
        == 64
    )
    assert (
        parsed.pipeline[0].hyperparameters.ev_planning.replay_samples_per_reason
        == 8
    )
    assert (
        parsed.pipeline[0].hyperparameters.ev_planning.minimum_price_spread
        == 0.001
    )
    assert (
        parsed.pipeline[0].hyperparameters.ev_planning.discharge_fraction
        == 0.40
    )
    assert (
        parsed.pipeline[0].hyperparameters.ev_planning.v2g_service_margin_ratio
        == 0.06
    )
    assert (
        parsed.pipeline[0].hyperparameters.ev_planning.minimum_v2g_price_spread
        == 0.02
    )
    assert (
        parsed.pipeline[0]
        .hyperparameters.ev_planning.minimum_v2g_departure_hours
        == 1.5
    )
    assert (
        parsed.pipeline[0]
        .hyperparameters.ev_planning.v2g_avoided_import_value_ratio
        == 0.8
    )
    assert (
        parsed.pipeline[0]
        .hyperparameters.ev_planning.v2g_minimum_profit_margin_eur_per_kwh
        == 0.03
    )
    assert (
        parsed.pipeline[0]
        .hyperparameters.ev_planning.v2g_degradation_cost_eur_per_kwh
        == 0.01
    )
    assert (
        parsed.pipeline[0]
        .hyperparameters.ev_planning.opportunity_value_kind
        == "community_marginal_import"
    )
    storage_planning = parsed.pipeline[0].hyperparameters.storage_planning
    assert storage_planning.auxiliary_coeff == 0.15
    assert storage_planning.balance_targets
    assert storage_planning.fraction_coeff == 0.3
    assert storage_planning.replay_capacity_per_reason == 32
    assert storage_planning.replay_samples_per_reason == 6
    assert storage_planning.charge_fraction == 0.6
    assert storage_planning.discharge_fraction == 0.5
    assert storage_planning.minimum_soc_ratio == 0.25
    assert storage_planning.maximum_soc_ratio == 0.85
    assert storage_planning.minimum_price_spread == 0.02
    assert storage_planning.pv_surplus_threshold_kw == 0.5
    assert storage_planning.import_threshold_kw == 0.75
    assert storage_planning.price_regime_kind == "relative_forecast"
    assert storage_planning.forecast_mean_margin_fraction == 0.15
    assert storage_planning.forecast_edge_margin_fraction == 0.08
    assert storage_planning.forecast_spread_floor_ratio == 0.04
    assert storage_planning.scale_price_fraction_by_opportunity
    assert storage_planning.minimum_price_fraction_scale == 0.45
    assert (
        parsed.pipeline[0].hyperparameters.actor.deterministic_mode_strategy
        == "expected_signed"
    )
    assert parsed.pipeline[0].hyperparameters.actor.deterministic_mode_strategy_by_group_type == {
        "ev_session": "argmax"
    }
    assert parsed.pipeline[0].hyperparameters.actor.deterministic_expected_signed_gain_by_group_type == {
        "stationary_storage": 2.0
    }
    assert parsed.pipeline[0].hyperparameters.actor.deterministic_expected_signed_deadband_by_group_type == {
        "stationary_storage": 0.05
    }
    assert parsed.pipeline[0].hyperparameters.actor.deterministic_non_idle_logit_margin_by_group_type == {
        "ev_session": 0.25
    }
    assert behavior_cloning is not None
    assert behavior_cloning.teacher.policy == "RBCSmartPolicy"
    assert behavior_cloning.max_samples == 672
    assert behavior_cloning.balance_action_modes
    assert behavior_cloning.mode_balance_exponent == 0.5
    assert behavior_cloning.max_mode_weight == 3.0
    assert behavior_cloning.balanced_loss_kind == "hierarchical_mode_mean"
    assert behavior_cloning.calibration_epochs == 2
    assert behavior_cloning.calibration_learning_rate == 5.0e-5
    assert parsed.pipeline[0].hyperparameters.advantage_normalization == "per_agent"
    assert (
        parsed.pipeline[0].hyperparameters.policy_credit_assignment
        == "typed_group"
    )
    assert parsed.pipeline[0].hyperparameters.entropy_coeff_by_group_type == {
        "stationary_storage": 0.05,
        "ev_session": 0.005,
    }
    assert (
        parsed.pipeline[0].hyperparameters.actor.group_context_kind
        == "action_conditioned"
    )


def test_validate_config_accepts_ti_ppo_with_local_critic(base_config):
    config = copy.deepcopy(base_config)
    config["simulator"]["interface"] = "entity"
    config["simulator"]["central_agent"] = False
    stage = _ti_marl_stage()
    stage["hyperparameters"]["backbone"] = {"name": "ppo"}
    stage["hyperparameters"]["critic"] = {"kind": "local"}
    config["pipeline"] = [stage]

    parsed = validate_config(config)

    assert parsed.pipeline[0].hyperparameters.backbone.name == "ppo"
    assert parsed.pipeline[0].hyperparameters.critic.kind == "local"


@pytest.mark.parametrize(
    ("backbone", "critic"),
    [("ppo", "set"), ("mappo", "local")],
)
def test_validate_config_rejects_mismatched_ti_marl_critic(
    base_config,
    backbone,
    critic,
):
    config = copy.deepcopy(base_config)
    config["simulator"]["interface"] = "entity"
    config["simulator"]["central_agent"] = False
    stage = _ti_marl_stage()
    stage["hyperparameters"]["backbone"] = {"name": backbone}
    stage["hyperparameters"]["critic"] = {"kind": critic}
    config["pipeline"] = [stage]

    with pytest.raises(ValueError, match="requires critic.kind"):
        validate_config(config)


def test_validate_config_rejects_negative_expected_signed_gain(base_config):
    config = copy.deepcopy(base_config)
    config["simulator"]["interface"] = "entity"
    config["simulator"]["central_agent"] = False
    stage = _ti_marl_stage()
    stage["hyperparameters"]["actor"][
        "deterministic_expected_signed_gain_by_group_type"
    ] = {"stationary_storage": -0.1}
    config["pipeline"] = [stage]

    with pytest.raises(ValueError, match="expected-signed gains must be non-negative"):
        validate_config(config)


def test_validate_config_rejects_invalid_expected_signed_deadband(base_config):
    config = copy.deepcopy(base_config)
    config["simulator"]["interface"] = "entity"
    config["simulator"]["central_agent"] = False
    stage = _ti_marl_stage()
    stage["hyperparameters"]["actor"][
        "deterministic_expected_signed_deadband_by_group_type"
    ] = {"stationary_storage": 1.1}
    config["pipeline"] = [stage]

    with pytest.raises(ValueError, match="deadbands must be between zero and one"):
        validate_config(config)


def test_validate_config_rejects_intervention_mask_with_joint_credit(base_config):
    config = copy.deepcopy(base_config)
    config["simulator"]["interface"] = "entity"
    config["simulator"]["central_agent"] = False
    stage = _ti_marl_stage()
    stage["hyperparameters"][
        "exclude_intervened_actions_from_policy_loss"
    ] = True
    config["pipeline"] = [stage]

    with pytest.raises(
        ValueError,
        match="exclude_intervened_actions_from_policy_loss requires",
    ):
        validate_config(config)


def test_validate_config_rejects_selective_anchor_with_joint_credit(base_config):
    config = copy.deepcopy(base_config)
    config["simulator"]["interface"] = "entity"
    config["simulator"]["central_agent"] = False
    stage = _ti_marl_stage()
    stage["hyperparameters"]["policy_anchor_coeff_by_group_type"] = {
        "ev_session": 0.1
    }
    config["pipeline"] = [stage]

    with pytest.raises(
        ValueError,
        match="policy_anchor_coeff_by_group_type requires",
    ):
        validate_config(config)


def test_validate_config_rejects_selective_ppo_groups_with_joint_credit(
    base_config,
):
    config = copy.deepcopy(base_config)
    config["simulator"]["interface"] = "entity"
    config["simulator"]["central_agent"] = False
    stage = _ti_marl_stage()
    stage["hyperparameters"]["ppo_policy_group_types"] = ["ev_session"]
    config["pipeline"] = [stage]

    with pytest.raises(
        ValueError,
        match="ppo_policy_group_types requires",
    ):
        validate_config(config)


@pytest.mark.parametrize(
    "group_types",
    [[], [""], ["ev_session", "ev_session"]],
)
def test_validate_config_rejects_invalid_selective_ppo_groups(
    base_config,
    group_types,
):
    config = copy.deepcopy(base_config)
    config["simulator"]["interface"] = "entity"
    config["simulator"]["central_agent"] = False
    stage = _ti_marl_stage()
    stage["hyperparameters"]["policy_credit_assignment"] = "typed_group"
    stage["hyperparameters"]["ppo_policy_group_types"] = group_types
    config["pipeline"] = [stage]

    with pytest.raises(ValueError, match="ppo_policy_group_types"):
        validate_config(config)


@pytest.mark.parametrize(
    ("scope", "group_types", "credit", "error"),
    [
        (
            "selected_group_heads",
            None,
            "typed_group",
            "actor_update_group_types",
        ),
        (
            "selected_group_heads",
            ["ev_session"],
            "joint_agent",
            "selected group-head updates require",
        ),
        (
            "all",
            ["ev_session"],
            "typed_group",
            "actor_update_group_types requires",
        ),
    ],
)
def test_validate_config_rejects_invalid_actor_update_scope(
    base_config,
    scope,
    group_types,
    credit,
    error,
):
    config = copy.deepcopy(base_config)
    config["simulator"]["interface"] = "entity"
    config["simulator"]["central_agent"] = False
    stage = _ti_marl_stage()
    stage["hyperparameters"]["policy_credit_assignment"] = credit
    stage["hyperparameters"]["actor_update_scope"] = scope
    if group_types is not None:
        stage["hyperparameters"]["actor_update_group_types"] = group_types
    config["pipeline"] = [stage]

    with pytest.raises(ValueError, match=error):
        validate_config(config)


def test_validate_config_rejects_negative_selective_anchor(base_config):
    config = copy.deepcopy(base_config)
    config["simulator"]["interface"] = "entity"
    config["simulator"]["central_agent"] = False
    stage = _ti_marl_stage()
    stage["hyperparameters"]["policy_credit_assignment"] = "typed_group"
    stage["hyperparameters"]["policy_anchor_coeff_by_group_type"] = {
        "ev_session": -0.1
    }
    config["pipeline"] = [stage]

    with pytest.raises(
        ValueError,
        match="policy_anchor_coeff_by_group_type",
    ):
        validate_config(config)


def test_validate_config_rejects_intervention_distillation_without_mask(base_config):
    config = copy.deepcopy(base_config)
    config["simulator"]["interface"] = "entity"
    config["simulator"]["central_agent"] = False
    stage = _ti_marl_stage()
    stage["hyperparameters"]["policy_credit_assignment"] = "typed_group"
    stage["hyperparameters"]["intervention_distillation_coeff"] = 0.1
    config["pipeline"] = [stage]

    with pytest.raises(
        ValueError,
        match="intervention_distillation_coeff requires",
    ):
        validate_config(config)


def _ti_marl_protocol_config(base_config, *, phase: str):
    config = copy.deepcopy(base_config)
    config["simulator"].update(
        {
            "interface": "entity",
            "topology_mode": "static",
            "central_agent": False,
            "random_seed": 101,
            "episodes": 1,
            "deterministic_finish": phase != "train",
        }
    )
    config["simulator"]["export"].update(
        {"export_kpis_on_episode_end": True, "final_episode_only": True}
    )
    stage = _ti_marl_stage()
    stage["frozen"] = phase != "train"
    config["pipeline"] = [stage]
    config["checkpointing"].update(
        {
            "checkpoint_interval": None,
            "checkpoint_on_episode_end": phase == "train",
            "keep_episode_checkpoints": phase == "train",
            "resume_training": phase != "train",
            "checkpoint_local_path": (
                "runs/local/checkpoint.pth" if phase != "train" else None
            ),
        }
    )
    config["experiment_protocol"] = {
        "version": "ti_marl_experiment_protocol_v1",
        "protocol_id": "ti-marl-v1",
        "phase": phase,
        "role": "candidate",
        "data_split": phase,
        "window_id": "winter",
        "candidate_id": "candidate-1",
        "paired_reference_id": "smart-winter" if phase != "train" else None,
        "selection_rules_sha256": "a" * 64 if phase == "development" else None,
        "selection_record_sha256": "b" * 64 if phase == "confirmation" else None,
        "selected_checkpoint_sha256": "c" * 64 if phase == "confirmation" else None,
    }
    return config


@pytest.mark.parametrize("phase", ["train", "development", "confirmation"])
def test_validate_config_accepts_explicit_ti_marl_protocol_phases(base_config, phase):
    validate_config(_ti_marl_protocol_config(base_config, phase=phase))


def test_validate_config_requires_post_bc_ti_marl_learning_episode(base_config):
    config = _ti_marl_protocol_config(base_config, phase="train")
    config["pipeline"][0]["hyperparameters"]["behavior_cloning"] = {
        "enabled": True,
        "demonstration_episodes": 2,
        "pretraining_epochs": 1,
    }
    config["simulator"]["episodes"] = 2

    with pytest.raises(ValueError, match="at least one post-BC learning episode"):
        validate_config(config)

    config["simulator"]["episodes"] = 3
    validate_config(config)


def test_validate_config_accepts_train_with_final_deterministic_diagnostic(base_config):
    config = _ti_marl_protocol_config(base_config, phase="train")
    config["simulator"].update(
        {
            "episodes": 3,
            "deterministic_finish": True,
            "episode_time_steps": [[0, 23], [24, 47], [72, 95]],
        }
    )
    config["simulator"]["export"].update(
        {
            "mode": "end",
            "export_kpis_on_episode_end": True,
            "final_episode_only": True,
            "kpis_final_episode_only": True,
            "timeseries_final_episode_only": True,
        }
    )

    validate_config(config)


def test_validate_config_rejects_non_isolated_train_diagnostic_export(base_config):
    config = _ti_marl_protocol_config(base_config, phase="train")
    config["simulator"].update(
        {
            "episodes": 2,
            "deterministic_finish": True,
            "episode_time_steps": [[0, 23], [72, 95]],
        }
    )
    config["simulator"]["export"].update(
        {
            "export_kpis_on_episode_end": True,
            "final_episode_only": False,
        }
    )

    with pytest.raises(ValueError, match="only for the final diagnostic episode"):
        validate_config(config)


def test_validate_config_counts_deterministic_finish_outside_bc_learning(base_config):
    config = _ti_marl_protocol_config(base_config, phase="train")
    config["pipeline"][0]["hyperparameters"]["behavior_cloning"] = {
        "enabled": True,
        "demonstration_episodes": 2,
        "pretraining_epochs": 1,
    }
    config["simulator"].update(
        {
            "episodes": 3,
            "deterministic_finish": True,
            "episode_time_steps": [[0, 23], [24, 47], [72, 95]],
        }
    )
    config["simulator"]["export"].update(
        {
            "mode": "end",
            "export_kpis_on_episode_end": True,
            "final_episode_only": True,
            "kpis_final_episode_only": True,
            "timeseries_final_episode_only": True,
        }
    )

    with pytest.raises(ValueError, match="at least one post-BC learning episode"):
        validate_config(config)

    config["simulator"].update(
        {
            "episodes": 4,
            "episode_time_steps": [[0, 23], [24, 47], [48, 71], [72, 95]],
        }
    )
    validate_config(config)


def test_validate_config_requires_explicit_simulator_seed_for_evaluation(base_config):
    config = _ti_marl_protocol_config(base_config, phase="development")
    config["simulator"]["random_seed"] = None
    with pytest.raises(ValueError, match="simulator.random_seed"):
        validate_config(config)


def test_validate_config_prevents_confirmation_without_selection_record(base_config):
    config = _ti_marl_protocol_config(base_config, phase="confirmation")
    config["experiment_protocol"]["selection_record_sha256"] = None
    with pytest.raises(ValueError, match="selection_record_sha256"):
        validate_config(config)


def test_validate_config_requires_selected_checkpoint_hash_for_confirmation(base_config):
    config = _ti_marl_protocol_config(base_config, phase="confirmation")
    config["experiment_protocol"]["selected_checkpoint_sha256"] = None
    with pytest.raises(ValueError, match="selected_checkpoint_sha256"):
        validate_config(config)


def test_validate_config_rejects_retired_ti_marl_interface_sources(base_config):
    config = copy.deepcopy(base_config)
    config["simulator"]["interface"] = "entity"
    config["simulator"]["central_agent"] = False
    stage = _ti_marl_stage()
    stage["hyperparameters"]["typed_interface_path"] = "retired-global.yaml"
    config["pipeline"] = [stage]
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        validate_config(config)


@pytest.mark.parametrize(
    ("interface", "central_agent", "message"),
    [
        ("flat", False, "interface='entity'"),
        ("entity", True, "central_agent=false"),
    ],
)
def test_validate_config_rejects_invalid_ti_marl_environment(
    base_config,
    interface,
    central_agent,
    message,
):
    config = copy.deepcopy(base_config)
    config["simulator"]["interface"] = interface
    config["simulator"]["topology_mode"] = "static"
    config["simulator"]["central_agent"] = central_agent
    config["pipeline"] = [_ti_marl_stage()]
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_validate_config_rejects_non_leaf_transformer_ppo() -> None:
    config_path = Path("configs/templates/dynamic/transformer_ppo_entity_dynamic.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["pipeline"].append(
        {"algorithm": "RuleBasedPolicy", "count": 1, "hyperparameters": {}}
    )

    with pytest.raises(ValueError, match="must be the final pipeline stage"):
        validate_config(config)


def test_validate_config_rejects_legacy_algorithm_key(base_config):
    config = copy.deepcopy(base_config)
    config.pop("pipeline", None)
    config["algorithm"] = {
        "name": "RuleBasedPolicy",
        "hyperparameters": {},
    }
    with pytest.raises(ValueError, match="deprecated top-level 'algorithm'"):
        validate_config(config)


def test_validate_config_missing_pipeline(base_config):
    config = copy.deepcopy(base_config)
    config["pipeline"] = None
    with pytest.raises(Exception):
        validate_config(config)


def test_validate_config_empty_pipeline(base_config):
    config = copy.deepcopy(base_config)
    config["pipeline"] = []
    with pytest.raises(Exception):
        validate_config(config)


def test_validate_config_invalid_network_layers(base_config):
    config = copy.deepcopy(base_config)
    config["pipeline"][0]["networks"]["actor"]["layers"] = []
    with pytest.raises(Exception):
        validate_config(config)


def test_validate_config_accepts_ev_stratified_replay_sampling(base_config):
    config = copy.deepcopy(base_config)
    replay = config["pipeline"][0]["replay_buffer"]
    replay["behavior_action_priority_scope"] = "ev"
    replay["behavior_action_stratified_sampling"] = True
    replay["behavior_action_positive_threshold"] = 0.1

    model = validate_config(config)
    parsed = model.pipeline[0].replay_buffer

    assert parsed.behavior_action_stratified_sampling is True
    assert parsed.behavior_action_positive_threshold == pytest.approx(0.1)


def test_validate_config_rejects_ev_stratified_sampling_without_ev_scope(base_config):
    config = copy.deepcopy(base_config)
    replay = config["pipeline"][0]["replay_buffer"]
    replay["behavior_action_priority_scope"] = "all"
    replay["behavior_action_stratified_sampling"] = True

    with pytest.raises(ValueError, match="requires behavior_action_priority_scope='ev'"):
        validate_config(config)


def test_validate_config_accepts_late_fusion_critic_layers(base_config):
    config = copy.deepcopy(base_config)
    config["pipeline"][0]["networks"]["critic"] = {
        "class": "LateFusionCritic",
        "layers": [64, 32],
        "state_layers": [64],
        "action_layers": [32],
        "joint_layers": [64, 32],
        "lr": 1.0e-3,
    }
    validate_config(config)


@pytest.mark.parametrize(
    "config_path",
    [
        Path("configs/templates/hiro_local.yaml"),
    ],
)
def test_validate_config_accepts_hierarchical_templates(config_path):
    with config_path.open("r", encoding="utf-8") as handle:
        validate_config(yaml.safe_load(handle))


def test_fixed_price_schedule_requires_ordered_entries_from_step_zero():
    from utils.config_schema import FixedPriceSignalHyperparameters

    valid = FixedPriceSignalHyperparameters(
        schedule=[
            {"start_step": 0, "multiplier": 1.05},
            {"start_step": 96, "multiplier": 1.025},
        ]
    )
    assert valid.schedule is not None
    assert valid.schedule[1].start_step == 96

    with pytest.raises(ValueError, match="start at step 0"):
        FixedPriceSignalHyperparameters(
            schedule=[{"start_step": 96, "multiplier": 1.025}]
        )

    with pytest.raises(ValueError, match="strictly increasing"):
        FixedPriceSignalHyperparameters(
            schedule=[
                {"start_step": 0, "multiplier": 1.05},
                {"start_step": 0, "multiplier": 1.025},
            ]
        )

    vector = FixedPriceSignalHyperparameters(
        vector_schedule=[
            {"start_step": 0, "multipliers": [1.0, 1.0]},
            {"start_step": 96, "multipliers": [0.7, 1.3]},
        ]
    )
    assert vector.vector_schedule is not None
    assert vector.vector_schedule[1].multipliers == [0.7, 1.3]

    with pytest.raises(ValueError, match="equal widths"):
        FixedPriceSignalHyperparameters(
            vector_schedule=[
                {"start_step": 0, "multipliers": [1.0, 1.0]},
                {"start_step": 96, "multipliers": [0.7]},
            ]
        )

    with pytest.raises(ValueError, match="must not be empty"):
        FixedPriceSignalHyperparameters(vector_schedule=[])

    with pytest.raises(ValueError, match="mutually exclusive"):
        FixedPriceSignalHyperparameters(
            multipliers=[1.0, 1.0],
            vector_schedule=[
                {"start_step": 0, "multipliers": [1.0, 1.0]},
            ],
        )


def test_to_dict_removes_none_network_optional_layers_from_pipeline(base_config):
    config = copy.deepcopy(base_config)
    config["pipeline"][0]["networks"]["critic"] = {
        "class": "LateFusionCritic",
        "layers": [64, 32],
        "state_layers": None,
        "action_layers": None,
        "joint_layers": None,
        "lr": 1.0e-3,
    }
    config["pipeline"][0]["networks"]["actor"]["head_layers"] = None

    resolved = validate_config(config).to_dict()

    critic = resolved["pipeline"][0]["networks"]["critic"]
    actor = resolved["pipeline"][0]["networks"]["actor"]
    assert "state_layers" not in critic
    assert "action_layers" not in critic
    assert "joint_layers" not in critic
    assert "head_layers" not in actor


def test_validate_config_accepts_deucalion_execution(base_config):
    config = copy.deepcopy(base_config)
    config["execution"] = {
        "deucalion": {
            "command_mode": "run",
            "gpus": 0,
            "datasets": ["datasets/citylearn_charging_constraints_demo"],
            "required_paths": ["/projects/F202508843CPCAA0/tiagocalof/images/simulator.sif"],
        }
    }
    validate_config(config)


def test_validate_config_rejects_invalid_deucalion_dataset(base_config):
    config = copy.deepcopy(base_config)
    config["execution"] = {
        "deucalion": {
            "datasets": ["/absolute/path/not/allowed"],
        }
    }
    with pytest.raises(Exception):
        validate_config(config)


def test_validate_config_accepts_bundle_section(base_config):
    config = copy.deepcopy(base_config)
    config["bundle"] = {
        "bundle_version": "2026-03-10-v1",
        "description": "Validation test",
        "alias_mapping_path": "aliases.json",
        "require_observations_envelope": True,
        "artifact_config": {"input_site_key": "site_a"},
        "per_agent_artifact_config": {
            "0": {"input_site_key": "boavista"},
            "1": {"input_site_key": "sao_mamede"},
        },
    }
    validate_config(config)


def test_validate_config_accepts_selected_pipeline_stage_checkpoint(base_config):
    config = copy.deepcopy(base_config)
    config["pipeline"].insert(
        0,
        {
            "algorithm": "CCLevel1",
            "count": 1,
            "hyperparameters": {},
        },
    )
    config["checkpointing"].update(
        {
            "resume_training": True,
            "checkpoint_local_path": None,
            "checkpoint_run_id": None,
            "stage_checkpoint_local_paths": {1: "runs/jobs/local-ppo/checkpoints"},
        }
    )

    parsed = validate_config(config)

    assert parsed.checkpointing.stage_checkpoint_local_paths == {
        1: "runs/jobs/local-ppo/checkpoints"
    }


def test_validate_config_accepts_fixed_neutral_price_signal_manager(base_config):
    config = copy.deepcopy(base_config)
    config["pipeline"].insert(
        0,
        {
            "algorithm": "FixedPriceSignal",
            "count": 1,
            "frozen": True,
            "hyperparameters": {"multiplier": 1.0},
        },
    )

    parsed = validate_config(config)

    assert parsed.pipeline[0].algorithm == "FixedPriceSignal"
    assert parsed.pipeline[0].hyperparameters.multiplier == 1.0


def test_validate_config_accepts_fixed_per_member_price_signal_manager(base_config):
    config = copy.deepcopy(base_config)
    config["pipeline"].insert(
        0,
        {
            "algorithm": "FixedPriceSignal",
            "count": 1,
            "frozen": True,
            "hyperparameters": {"multipliers": [0.9, 1.05]},
        },
    )

    parsed = validate_config(config)

    assert parsed.pipeline[0].hyperparameters.multipliers == [0.9, 1.05]


@pytest.mark.parametrize("multipliers", [[], [0.9, 0.0], [-1.0, 0.9]])
def test_validate_config_rejects_invalid_fixed_price_signal_vector(base_config, multipliers):
    config = copy.deepcopy(base_config)
    config["pipeline"].insert(
        0,
        {
            "algorithm": "FixedPriceSignal",
            "count": 1,
            "frozen": True,
            "hyperparameters": {"multipliers": multipliers},
        },
    )

    with pytest.raises(Exception, match="multipliers"):
        validate_config(config)


def test_validate_config_rejects_invalid_cc_price_range(base_config):
    config = copy.deepcopy(base_config)
    config["pipeline"].insert(
        0,
        {
            "algorithm": "CCLevel1",
            "count": 1,
            "hyperparameters": {"price_min": 1.0, "price_max": 1.0},
        },
    )

    with pytest.raises(Exception, match="price_max must be greater"):
        validate_config(config)


def test_validate_config_rejects_out_of_range_pipeline_stage_checkpoint(base_config):
    config = copy.deepcopy(base_config)
    config["pipeline"].insert(
        0,
        {
            "algorithm": "CCLevel1",
            "count": 1,
            "hyperparameters": {},
        },
    )
    config["checkpointing"].update(
        {
            "resume_training": True,
            "checkpoint_local_path": None,
            "checkpoint_run_id": None,
            "stage_checkpoint_local_paths": {2: "runs/jobs/missing/checkpoints"},
        }
    )

    with pytest.raises(Exception, match="outside pipeline range"):
        validate_config(config)


def test_validate_config_rejects_invalid_per_agent_artifact_config(base_config):
    config = copy.deepcopy(base_config)
    config["bundle"]["per_agent_artifact_config"] = {"0": ["invalid"]}
    with pytest.raises(Exception):
        validate_config(config)


def test_validate_config_accepts_simulator_export_and_time_controls(base_config):
    config = copy.deepcopy(base_config)
    config["tracking"]["progress_updates_enabled"] = False
    config["tracking"]["progress_update_interval"] = 3
    config["tracking"]["system_metrics_enabled"] = True
    config["tracking"]["system_metrics_interval"] = 12
    config["checkpointing"]["require_update_step"] = False
    config["checkpointing"]["require_initial_exploration_done"] = False
    config["simulator"]["simulation_start_time_step"] = 0
    config["simulator"]["simulation_end_time_step"] = 95
    config["simulator"]["episodes"] = 2
    config["simulator"]["episode_time_steps"] = 24
    config["simulator"]["terminal_observation_padding"] = True
    config["simulator"]["export"] = {
        "mode": "end",
        "export_kpis_on_episode_end": True,
        "final_episode_only": True,
        "include_business_as_usual": False,
        "export_business_as_usual_timeseries": False,
        "kpi_round_decimals": 4,
        "session_name": "session-a",
    }
    validated = validate_config(config)
    assert validated.simulator.terminal_observation_padding is True


def test_validate_config_accepts_runtime_safety_guards(base_config):
    config = copy.deepcopy(base_config)
    config["tracking"]["progress_phase_updates_enabled"] = True
    config["tracking"]["progress_phase_start_step"] = 4500
    config["tracking"]["progress_phase_end_step"] = 5700
    config["tracking"]["max_step_seconds"] = 15.0
    config["tracking"]["stall_watchdog_enabled"] = True
    config["tracking"]["stall_watchdog_timeout_seconds"] = 900.0
    config["tracking"]["stall_watchdog_exit_on_timeout"] = True
    config["tracking"]["stall_watchdog_repeat"] = False
    config["tracking"]["stall_watchdog_traceback_file"] = "logs/stall_watchdog.log"
    config["tracking"]["stall_watchdog_context_interval_steps"] = 64
    config["tracking"]["resource_guard_enabled"] = True
    config["tracking"]["max_process_rss_mb"] = 12000.0
    config["tracking"]["min_available_ram_mb"] = 1024.0
    validate_config(config)


def test_validate_config_accepts_max_update_seconds(base_config):
    config = copy.deepcopy(base_config)
    config["tracking"]["max_update_seconds"] = 2400.0

    resolved = validate_config(config).to_dict()

    assert resolved["tracking"]["max_update_seconds"] == 2400.0


@pytest.mark.parametrize("value", [0, -1.0])
def test_validate_config_rejects_non_positive_max_update_seconds(base_config, value):
    config = copy.deepcopy(base_config)
    config["tracking"]["max_update_seconds"] = value

    with pytest.raises(Exception):
        validate_config(config)


@pytest.fixture
def transformer_ppo_template_config():
    config_path = Path("configs/templates/dynamic/transformer_ppo_entity_dynamic.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_validate_config_accepts_transformer_ppo_require_cuda(transformer_ppo_template_config):
    transformer_ppo_template_config["pipeline"][0]["hyperparameters"]["require_cuda"] = True

    resolved = validate_config(transformer_ppo_template_config).to_dict()

    assert resolved["pipeline"][0]["hyperparameters"]["require_cuda"] is True


def test_validate_config_rejects_enabled_transformer_ppo_bc_without_demonstrations(
    transformer_ppo_template_config,
):
    transformer_ppo_template_config["pipeline"][0]["behavior_cloning"] = {
        "enabled": True,
        "demonstration_episodes": 0,
    }

    with pytest.raises(ValueError, match="demonstration_episodes.*at least 1"):
        validate_config(transformer_ppo_template_config)


def test_validate_config_accepts_disabled_transformer_ppo_bc_without_demonstrations(
    transformer_ppo_template_config,
):
    transformer_ppo_template_config["pipeline"][0]["behavior_cloning"] = {
        "enabled": False,
        "demonstration_episodes": 0,
    }

    validate_config(transformer_ppo_template_config)


def test_validate_config_accepts_enabled_transformer_ppo_bc_with_demonstrations(
    transformer_ppo_template_config,
):
    transformer_ppo_template_config["pipeline"][0]["behavior_cloning"] = {
        "enabled": True,
        "demonstration_episodes": 1,
    }

    validate_config(transformer_ppo_template_config)


def test_validate_config_rejects_invalid_runtime_safety_guards(base_config):
    config = copy.deepcopy(base_config)
    config["tracking"]["progress_phase_start_step"] = 5700
    config["tracking"]["progress_phase_end_step"] = 4500
    with pytest.raises(Exception):
        validate_config(config)

    config = copy.deepcopy(base_config)
    config["tracking"]["max_step_seconds"] = 0
    with pytest.raises(Exception):
        validate_config(config)

    config = copy.deepcopy(base_config)
    config["tracking"]["stall_watchdog_timeout_seconds"] = 0
    with pytest.raises(Exception):
        validate_config(config)

    config = copy.deepcopy(base_config)
    config["tracking"]["stall_watchdog_context_interval_steps"] = 0
    with pytest.raises(Exception):
        validate_config(config)


def test_validate_config_accepts_wrapper_reward_overrides(base_config):
    config = copy.deepcopy(base_config)
    config["simulator"]["wrapper_reward"] = {
        "enabled": True,
        "profile": "cost_limits_v1",
        "clip_enabled": True,
        "clip_min": -5.0,
        "clip_max": 5.0,
        "squash": "tanh",
    }
    validate_config(config)


def test_validate_config_rejects_wrapper_reward_invalid_clip_range(base_config):
    config = copy.deepcopy(base_config)
    config["simulator"]["wrapper_reward"]["clip_min"] = 1.0
    config["simulator"]["wrapper_reward"]["clip_max"] = -1.0
    with pytest.raises(Exception):
        validate_config(config)


def test_validate_config_rejects_invalid_simulator_export_mode(base_config):
    config = copy.deepcopy(base_config)
    config["simulator"]["export"] = {
        "mode": "invalid-mode",
        "export_kpis_on_episode_end": False,
    }
    with pytest.raises(Exception):
        validate_config(config)


def test_validate_config_rejects_invalid_simulation_window(base_config):
    config = copy.deepcopy(base_config)
    config["simulator"]["simulation_start_time_step"] = 50
    config["simulator"]["simulation_end_time_step"] = 10
    with pytest.raises(Exception):
        validate_config(config)

    config = copy.deepcopy(base_config)
    config["simulator"]["episodes"] = 0
    with pytest.raises(Exception):
        validate_config(config)


def test_validate_config_rejects_dynamic_topology_without_entity_interface(base_config):
    config = copy.deepcopy(base_config)
    config["simulator"]["interface"] = "flat"
    config["simulator"]["topology_mode"] = "dynamic"
    with pytest.raises(Exception):
        validate_config(config)


def test_validate_config_rejects_maddpg_with_entity_dynamic(base_config):
    config = copy.deepcopy(base_config)
    config["simulator"]["interface"] = "entity"
    config["simulator"]["topology_mode"] = "dynamic"
    with pytest.raises(Exception):
        validate_config(config)


def test_validate_config_accepts_rule_based_with_entity_dynamic(base_config):
    config = copy.deepcopy(base_config)
    config["simulator"]["interface"] = "entity"
    config["simulator"]["topology_mode"] = "dynamic"
    config["simulator"]["dataset_name"] = "citylearn_three_phase_dynamic_topology_demo_v1"
    config["simulator"]["dataset_path"] = "./datasets/citylearn_three_phase_dynamic_topology_demo_v1/schema.json"
    config["pipeline"] = [
        {
            "algorithm": "RuleBasedPolicy",
            "count": 1,
            "hyperparameters": {
                "pv_charge_threshold": 0.0,
                "flexibility_hours": 3.0,
                "emergency_hours": 1.0,
                "pv_preferred_charge_rate": 0.6,
                "flex_trickle_charge": 0.0,
                "min_charge_rate": 0.0,
                "emergency_charge_rate": 1.0,
                "energy_epsilon": 1e-3,
                "default_capacity_kwh": 60.0,
                "non_flexible_chargers": [],
            },
            "networks": None,
            "replay_buffer": None,
            "exploration": None,
        }
    ]
    validate_config(config)


def test_validate_config_accepts_signal_aware_rbc_stage(base_config):
    config = copy.deepcopy(base_config)
    config["pipeline"] = [
        {
            "algorithm": "SignalAwareRBC",
            "count": 17,
            "hyperparameters": {},
            "networks": None,
            "replay_buffer": None,
            "exploration": None,
        }
    ]

    validate_config(config)


def test_validate_config_rejects_invalid_mlflow_artifacts_profile(base_config):
    config = copy.deepcopy(base_config)
    config["tracking"]["mlflow_artifacts_profile"] = "all"
    with pytest.raises(Exception):
        validate_config(config)


def test_validate_config_rejects_invalid_tracking_intervals(base_config):
    config = copy.deepcopy(base_config)
    config["tracking"]["progress_update_interval"] = 0
    with pytest.raises(Exception):
        validate_config(config)

    config = copy.deepcopy(base_config)
    config["tracking"]["system_metrics_interval"] = 0
    with pytest.raises(Exception):
        validate_config(config)


def test_validate_all_templates():
    # Experimental templates intentionally use placeholder algorithms (e.g.
    # SingleAgentRL) that are not yet runtime-backed.  Exclude them here.
    template_paths = sorted(
        p
        for p in Path("configs/templates").rglob("*.yaml")
        if "experimental" not in p.parts
    )
    assert template_paths, "No template files found under configs/templates"

    for template_path in template_paths:
        with template_path.open("r", encoding="utf-8") as handle:
            template_config = yaml.safe_load(handle)
        validate_config(template_config)


def test_rbc_community_templates_use_community_settlement_objective():
    template_paths = [
        Path("configs/templates/baselines/rbc_community_local.yaml"),
        Path("configs/templates/baselines/rbc_community_2022_all_plus_evs_local.yaml"),
    ]

    for template_path in template_paths:
        with template_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        kwargs = config["simulator"]["reward_function_kwargs"]
        hyper = config["pipeline"][0]["hyperparameters"]

        assert kwargs["local_cost_weight"] == pytest.approx(0.0)
        assert kwargs["community_settlement_cost_weight"] == pytest.approx(1.0)
        assert kwargs["community_local_price_ratio"] == pytest.approx(0.8)
        assert kwargs["community_grid_export_price"] == pytest.approx(0.0)
        assert hyper["community_local_price_ratio"] == pytest.approx(0.8)
        assert hyper["community_grid_export_price"] == pytest.approx(0.0)


def test_rbc_smart_templates_keep_house_level_cost_objective():
    template_paths = [
        Path("configs/templates/baselines/rbc_smart_local.yaml"),
        Path("configs/templates/baselines/rbc_smart_2022_all_plus_evs_local.yaml"),
    ]

    for template_path in template_paths:
        with template_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        kwargs = config["simulator"]["reward_function_kwargs"]

        assert kwargs.get("local_cost_weight", 1.0) == pytest.approx(1.0)
        assert kwargs["community_settlement_cost_weight"] == pytest.approx(0.0)
