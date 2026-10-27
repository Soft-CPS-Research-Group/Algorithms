"""Explicit legacy actor/critic parameter extensions, validated without model imports.

No default values are injected here. The numerical agent owns defaults and
cross-parameter semantics. This boundary rejects misspellings and wrong types.
"""

from typing import Any

from pydantic import ConfigDict, TypeAdapter
from typing_extensions import TypedDict


class ActorParameters(TypedDict, total=False):
    __pydantic_config__ = ConfigDict(extra="forbid")
    alpha_lr: float
    actor_action_l2_penalty: float
    actor_action_saturation_penalty: float
    actor_action_saturation_threshold: float
    actor_auxiliary_hidden_layers: list[int] | None
    actor_auxiliary_loss_weight: float
    actor_behavior_cloning_clip_target_to_residual_authority: bool
    actor_behavior_cloning_decay_start_step: int
    actor_behavior_cloning_decay_steps: int
    actor_behavior_cloning_extra_update_end_step: int
    actor_behavior_cloning_extra_update_start_step: int
    actor_behavior_cloning_extra_updates: int
    actor_behavior_cloning_min_weight: float
    actor_behavior_cloning_replay_capacity: int
    actor_behavior_cloning_source: str
    actor_behavior_cloning_teacher_action_scope: str
    actor_behavior_cloning_teacher_hyperparameters: dict[str, Any] | None
    actor_behavior_cloning_teacher_policy: str | None
    actor_behavior_cloning_weight: float
    actor_community_context_enabled: bool
    actor_community_context_features: list[str] | None
    actor_community_observation_mask_tokens: list[str] | None
    actor_community_observation_masking: bool
    actor_deferrable_behavior_cloning_multiplier: float
    actor_deferrable_behavior_cloning_positive_target_power: float
    actor_deferrable_behavior_cloning_positive_target_weight: float
    actor_ev_behavior_cloning_multiplier: float
    actor_ev_behavior_cloning_positive_target_power: float
    actor_ev_behavior_cloning_positive_target_weight: float
    actor_ev_behavior_cloning_zero_target_threshold: float
    actor_ev_behavior_cloning_zero_target_weight: float
    actor_ev_v2g_action_l2_penalty: float
    actor_ev_v2g_action_mass_penalty: float
    actor_frame_stack_steps: int
    actor_offline_bc_pretrain_min_replay: int
    actor_offline_bc_pretrain_steps: int
    actor_offline_bc_pretrain_sync_targets: bool
    actor_offline_bc_pretrain_weight: float
    actor_other_behavior_cloning_multiplier: float
    actor_policy_loss_normalization: bool
    actor_policy_loss_normalization_epsilon: float
    actor_policy_loss_normalization_max_scale: float
    actor_policy_loss_warmup_start_step: int
    actor_policy_loss_warmup_steps: int
    actor_policy_loss_warmup_weight: float
    actor_policy_loss_weight: float
    actor_residual_behavior_cloning_neutral_target_threshold: float
    actor_residual_behavior_cloning_neutral_target_weight: float
    actor_residual_delta_l2_penalty: float
    actor_storage_action_l2_penalty: float
    actor_storage_behavior_cloning_multiplier: float
    actor_storage_smoothness_deadband: float
    actor_storage_smoothness_l2_penalty: float
    actor_td3_bc_loss_normalization: bool
    actor_update_interval: int
    agent_update_order: str
    automatic_entropy_tuning: bool
    bias: float
    clip_ratio: float
    critic_action_input_mode: str
    critic_huber_beta: float
    critic_loss: str
    critic_target_clip_abs: float
    critic_team_reward_mix: float
    critic_update_mode: str
    decay: float
    deferrable_on_probability: float
    deferrable_trigger_threshold: float
    end_initial_exploration_time_step: int
    entropy_alpha: float
    entropy_coef: float
    ev_negative_exploration_noise_multiplier: float
    frozen_checkpoint_path: str | None
    gae_lambda: float
    gamma: float
    initial_exploration_strategy: str | None
    initial_exploration_training_start_step: int
    initial_log_std: float
    local_action_safety_allow_discretionary_deferrable_start: bool
    local_action_safety_enabled: bool
    local_action_safety_ev_minimum_mode: str
    local_action_safety_fail_on_infeasible: bool
    local_action_safety_headroom_reserve_kw: float
    local_action_safety_protect_deferrable_must_start: bool
    local_action_safety_protect_ev_minimum: bool
    local_action_safety_protect_ev_service_target: bool
    local_action_safety_runtime_only_export: bool
    local_action_safety_service_teacher_enabled: bool
    local_action_safety_service_teacher_eval_enabled: bool
    local_price_conditioning_enabled: bool
    local_price_conditioning_trainable: bool
    local_price_forecast_mode: str
    max_grad_norm: float
    max_log_std: float
    min_log_std: float
    min_sigma: float
    minibatch_size: int
    n_step_gamma: float
    n_step_priority_aggregation: str
    n_step_returns: int
    noise_clip: float | None
    noop_actor_initialization: bool
    noop_actor_initialization_epsilon: float
    noop_noise_scale: float
    policy_delay: int
    policy_noise: float
    ppo_epochs: int
    random_exploration_steps: int
    require_cuda: bool
    residual_action_deadband: float
    residual_action_final_scale: float
    residual_action_growth_steps: int
    residual_action_scale: float
    residual_action_start_step: int
    residual_base_policy: str | None
    residual_base_policy_deterministic: bool
    residual_base_policy_hyperparameters: dict[str, Any] | None
    residual_base_price_conditioning_enabled: bool
    residual_building_deadbands: dict[str, float]
    residual_building_gain_multipliers: dict[str, float]
    residual_deferrable_action_scale_multiplier: float
    residual_ev_action_scale_multiplier: float
    residual_policy_enabled: bool
    residual_policy_runtime_only_export: bool
    residual_storage_action_scale_multiplier: float
    residual_zero_initialization: bool
    reward_normalization: bool
    reward_normalization_clip: float
    reward_normalization_epsilon: float
    rollout_length: int
    sigma: float
    signal_aware_actor_obs: bool
    signal_aware_actor_obs_gain: float
    signal_aware_base: bool
    storage_exploration_noise_multiplier: float
    target_entropy: float | None
    target_kl: float | None
    target_policy_noise: float
    target_policy_noise_clip: float
    target_policy_smoothing: bool
    tau: float
    train_during_initial_exploration: bool
    use_amp: bool
    value_loss_coef: float
    warm_start_policy: str | None
    warm_start_policy_deterministic: bool
    warm_start_policy_hyperparameters: dict[str, Any] | None
    warm_start_policy_noise_scale: float
    warm_start_policy_phaseout_mode: str
    warm_start_policy_phaseout_steps: int


PARAMETER_KEYS = {
    "MADDPG": frozenset(
        [
            "actor_action_l2_penalty",
            "actor_action_saturation_penalty",
            "actor_action_saturation_threshold",
            "actor_auxiliary_hidden_layers",
            "actor_auxiliary_loss_weight",
            "actor_behavior_cloning_clip_target_to_residual_authority",
            "actor_behavior_cloning_decay_start_step",
            "actor_behavior_cloning_decay_steps",
            "actor_behavior_cloning_extra_update_end_step",
            "actor_behavior_cloning_extra_update_start_step",
            "actor_behavior_cloning_extra_updates",
            "actor_behavior_cloning_min_weight",
            "actor_behavior_cloning_source",
            "actor_behavior_cloning_teacher_action_scope",
            "actor_behavior_cloning_teacher_hyperparameters",
            "actor_behavior_cloning_teacher_policy",
            "actor_behavior_cloning_weight",
            "actor_community_context_enabled",
            "actor_community_context_features",
            "actor_community_observation_mask_tokens",
            "actor_community_observation_masking",
            "actor_deferrable_behavior_cloning_multiplier",
            "actor_deferrable_behavior_cloning_positive_target_power",
            "actor_deferrable_behavior_cloning_positive_target_weight",
            "actor_ev_behavior_cloning_multiplier",
            "actor_ev_behavior_cloning_positive_target_power",
            "actor_ev_behavior_cloning_positive_target_weight",
            "actor_ev_behavior_cloning_zero_target_threshold",
            "actor_ev_behavior_cloning_zero_target_weight",
            "actor_ev_v2g_action_l2_penalty",
            "actor_ev_v2g_action_mass_penalty",
            "actor_frame_stack_steps",
            "actor_offline_bc_pretrain_min_replay",
            "actor_offline_bc_pretrain_steps",
            "actor_offline_bc_pretrain_sync_targets",
            "actor_offline_bc_pretrain_weight",
            "actor_policy_loss_normalization",
            "actor_policy_loss_normalization_epsilon",
            "actor_policy_loss_normalization_max_scale",
            "actor_policy_loss_warmup_start_step",
            "actor_policy_loss_warmup_steps",
            "actor_policy_loss_warmup_weight",
            "actor_policy_loss_weight",
            "actor_residual_delta_l2_penalty",
            "actor_storage_action_l2_penalty",
            "actor_storage_behavior_cloning_multiplier",
            "actor_storage_smoothness_deadband",
            "actor_storage_smoothness_l2_penalty",
            "actor_td3_bc_loss_normalization",
            "actor_update_interval",
            "bias",
            "critic_action_input_mode",
            "critic_huber_beta",
            "critic_loss",
            "critic_target_clip_abs",
            "critic_team_reward_mix",
            "critic_update_mode",
            "decay",
            "deferrable_on_probability",
            "deferrable_trigger_threshold",
            "end_initial_exploration_time_step",
            "ev_negative_exploration_noise_multiplier",
            "frozen_checkpoint_path",
            "gamma",
            "initial_exploration_strategy",
            "initial_exploration_training_start_step",
            "min_sigma",
            "n_step_gamma",
            "n_step_priority_aggregation",
            "n_step_returns",
            "noise_clip",
            "noop_actor_initialization",
            "noop_actor_initialization_epsilon",
            "noop_noise_scale",
            "random_exploration_steps",
            "require_cuda",
            "residual_action_final_scale",
            "residual_action_growth_steps",
            "residual_action_scale",
            "residual_action_start_step",
            "residual_building_gain_multipliers",
            "residual_deferrable_action_scale_multiplier",
            "residual_ev_action_scale_multiplier",
            "residual_policy_enabled",
            "residual_policy_runtime_only_export",
            "residual_storage_action_scale_multiplier",
            "reward_normalization",
            "reward_normalization_clip",
            "reward_normalization_epsilon",
            "sigma",
            "signal_aware_actor_obs",
            "signal_aware_actor_obs_gain",
            "signal_aware_base",
            "storage_exploration_noise_multiplier",
            "target_policy_noise",
            "target_policy_noise_clip",
            "target_policy_smoothing",
            "tau",
            "train_during_initial_exploration",
            "use_amp",
            "warm_start_policy",
            "warm_start_policy_deterministic",
            "warm_start_policy_hyperparameters",
            "warm_start_policy_noise_scale",
            "warm_start_policy_phaseout_mode",
            "warm_start_policy_phaseout_steps",
        ]
    ),
    "MATD3": frozenset(
        [
            "actor_action_l2_penalty",
            "actor_action_saturation_penalty",
            "actor_action_saturation_threshold",
            "actor_auxiliary_hidden_layers",
            "actor_auxiliary_loss_weight",
            "actor_behavior_cloning_clip_target_to_residual_authority",
            "actor_behavior_cloning_decay_start_step",
            "actor_behavior_cloning_decay_steps",
            "actor_behavior_cloning_extra_update_end_step",
            "actor_behavior_cloning_extra_update_start_step",
            "actor_behavior_cloning_extra_updates",
            "actor_behavior_cloning_min_weight",
            "actor_behavior_cloning_source",
            "actor_behavior_cloning_teacher_action_scope",
            "actor_behavior_cloning_teacher_hyperparameters",
            "actor_behavior_cloning_teacher_policy",
            "actor_behavior_cloning_weight",
            "actor_community_context_enabled",
            "actor_community_context_features",
            "actor_community_observation_mask_tokens",
            "actor_community_observation_masking",
            "actor_deferrable_behavior_cloning_multiplier",
            "actor_deferrable_behavior_cloning_positive_target_power",
            "actor_deferrable_behavior_cloning_positive_target_weight",
            "actor_ev_behavior_cloning_multiplier",
            "actor_ev_behavior_cloning_positive_target_power",
            "actor_ev_behavior_cloning_positive_target_weight",
            "actor_ev_behavior_cloning_zero_target_threshold",
            "actor_ev_behavior_cloning_zero_target_weight",
            "actor_ev_v2g_action_l2_penalty",
            "actor_ev_v2g_action_mass_penalty",
            "actor_frame_stack_steps",
            "actor_offline_bc_pretrain_min_replay",
            "actor_offline_bc_pretrain_steps",
            "actor_offline_bc_pretrain_sync_targets",
            "actor_offline_bc_pretrain_weight",
            "actor_policy_loss_normalization",
            "actor_policy_loss_normalization_epsilon",
            "actor_policy_loss_normalization_max_scale",
            "actor_policy_loss_warmup_start_step",
            "actor_policy_loss_warmup_steps",
            "actor_policy_loss_warmup_weight",
            "actor_policy_loss_weight",
            "actor_residual_delta_l2_penalty",
            "actor_storage_action_l2_penalty",
            "actor_storage_behavior_cloning_multiplier",
            "actor_storage_smoothness_deadband",
            "actor_storage_smoothness_l2_penalty",
            "actor_td3_bc_loss_normalization",
            "actor_update_interval",
            "bias",
            "critic_action_input_mode",
            "critic_huber_beta",
            "critic_loss",
            "critic_target_clip_abs",
            "critic_team_reward_mix",
            "critic_update_mode",
            "decay",
            "deferrable_on_probability",
            "deferrable_trigger_threshold",
            "end_initial_exploration_time_step",
            "ev_negative_exploration_noise_multiplier",
            "frozen_checkpoint_path",
            "gamma",
            "initial_exploration_strategy",
            "initial_exploration_training_start_step",
            "local_action_safety_allow_discretionary_deferrable_start",
            "local_action_safety_enabled",
            "local_action_safety_ev_minimum_mode",
            "local_action_safety_fail_on_infeasible",
            "local_action_safety_headroom_reserve_kw",
            "local_action_safety_protect_deferrable_must_start",
            "local_action_safety_protect_ev_minimum",
            "local_action_safety_protect_ev_service_target",
            "local_action_safety_runtime_only_export",
            "local_price_conditioning_enabled",
            "local_price_forecast_mode",
            "min_sigma",
            "n_step_gamma",
            "n_step_priority_aggregation",
            "n_step_returns",
            "noise_clip",
            "noop_actor_initialization",
            "noop_actor_initialization_epsilon",
            "noop_noise_scale",
            "policy_delay",
            "policy_noise",
            "random_exploration_steps",
            "require_cuda",
            "residual_action_final_scale",
            "residual_action_growth_steps",
            "residual_action_scale",
            "residual_action_start_step",
            "residual_building_gain_multipliers",
            "residual_deferrable_action_scale_multiplier",
            "residual_ev_action_scale_multiplier",
            "residual_policy_enabled",
            "residual_policy_runtime_only_export",
            "residual_storage_action_scale_multiplier",
            "reward_normalization",
            "reward_normalization_clip",
            "reward_normalization_epsilon",
            "sigma",
            "signal_aware_actor_obs",
            "signal_aware_actor_obs_gain",
            "signal_aware_base",
            "storage_exploration_noise_multiplier",
            "target_policy_noise",
            "target_policy_noise_clip",
            "target_policy_smoothing",
            "tau",
            "train_during_initial_exploration",
            "use_amp",
            "warm_start_policy",
            "warm_start_policy_deterministic",
            "warm_start_policy_hyperparameters",
            "warm_start_policy_noise_scale",
            "warm_start_policy_phaseout_mode",
            "warm_start_policy_phaseout_steps",
        ]
    ),
    "MASAC": frozenset(
        [
            "actor_action_l2_penalty",
            "actor_action_saturation_penalty",
            "actor_action_saturation_threshold",
            "actor_auxiliary_hidden_layers",
            "actor_auxiliary_loss_weight",
            "actor_behavior_cloning_clip_target_to_residual_authority",
            "actor_behavior_cloning_decay_start_step",
            "actor_behavior_cloning_decay_steps",
            "actor_behavior_cloning_extra_update_end_step",
            "actor_behavior_cloning_extra_update_start_step",
            "actor_behavior_cloning_extra_updates",
            "actor_behavior_cloning_min_weight",
            "actor_behavior_cloning_source",
            "actor_behavior_cloning_teacher_action_scope",
            "actor_behavior_cloning_teacher_hyperparameters",
            "actor_behavior_cloning_teacher_policy",
            "actor_behavior_cloning_weight",
            "actor_community_context_enabled",
            "actor_community_context_features",
            "actor_community_observation_mask_tokens",
            "actor_community_observation_masking",
            "actor_deferrable_behavior_cloning_multiplier",
            "actor_deferrable_behavior_cloning_positive_target_power",
            "actor_deferrable_behavior_cloning_positive_target_weight",
            "actor_ev_behavior_cloning_multiplier",
            "actor_ev_behavior_cloning_positive_target_power",
            "actor_ev_behavior_cloning_positive_target_weight",
            "actor_ev_behavior_cloning_zero_target_threshold",
            "actor_ev_behavior_cloning_zero_target_weight",
            "actor_ev_v2g_action_l2_penalty",
            "actor_ev_v2g_action_mass_penalty",
            "actor_frame_stack_steps",
            "actor_offline_bc_pretrain_min_replay",
            "actor_offline_bc_pretrain_steps",
            "actor_offline_bc_pretrain_sync_targets",
            "actor_offline_bc_pretrain_weight",
            "actor_policy_loss_normalization",
            "actor_policy_loss_normalization_epsilon",
            "actor_policy_loss_normalization_max_scale",
            "actor_policy_loss_warmup_start_step",
            "actor_policy_loss_warmup_steps",
            "actor_policy_loss_warmup_weight",
            "actor_policy_loss_weight",
            "actor_residual_delta_l2_penalty",
            "actor_storage_action_l2_penalty",
            "actor_storage_behavior_cloning_multiplier",
            "actor_storage_smoothness_deadband",
            "actor_storage_smoothness_l2_penalty",
            "actor_td3_bc_loss_normalization",
            "actor_update_interval",
            "automatic_entropy_tuning",
            "bias",
            "critic_action_input_mode",
            "critic_huber_beta",
            "critic_loss",
            "critic_target_clip_abs",
            "critic_team_reward_mix",
            "critic_update_mode",
            "decay",
            "deferrable_on_probability",
            "deferrable_trigger_threshold",
            "end_initial_exploration_time_step",
            "entropy_alpha",
            "ev_negative_exploration_noise_multiplier",
            "frozen_checkpoint_path",
            "gamma",
            "initial_exploration_strategy",
            "initial_exploration_training_start_step",
            "initial_log_std",
            "max_log_std",
            "min_log_std",
            "min_sigma",
            "n_step_gamma",
            "n_step_priority_aggregation",
            "n_step_returns",
            "noise_clip",
            "noop_actor_initialization",
            "noop_actor_initialization_epsilon",
            "noop_noise_scale",
            "random_exploration_steps",
            "require_cuda",
            "residual_action_final_scale",
            "residual_action_growth_steps",
            "residual_action_scale",
            "residual_action_start_step",
            "residual_building_gain_multipliers",
            "residual_deferrable_action_scale_multiplier",
            "residual_ev_action_scale_multiplier",
            "residual_policy_enabled",
            "residual_policy_runtime_only_export",
            "residual_storage_action_scale_multiplier",
            "reward_normalization",
            "reward_normalization_clip",
            "reward_normalization_epsilon",
            "sigma",
            "signal_aware_actor_obs",
            "signal_aware_actor_obs_gain",
            "signal_aware_base",
            "storage_exploration_noise_multiplier",
            "target_entropy",
            "target_policy_noise",
            "target_policy_noise_clip",
            "target_policy_smoothing",
            "tau",
            "train_during_initial_exploration",
            "use_amp",
            "warm_start_policy",
            "warm_start_policy_deterministic",
            "warm_start_policy_hyperparameters",
            "warm_start_policy_noise_scale",
            "warm_start_policy_phaseout_mode",
            "warm_start_policy_phaseout_steps",
        ]
    ),
    "PPO": frozenset(
        [
            "actor_action_l2_penalty",
            "actor_action_saturation_penalty",
            "actor_action_saturation_threshold",
            "actor_behavior_cloning_decay_start_step",
            "actor_behavior_cloning_decay_steps",
            "actor_behavior_cloning_extra_update_end_step",
            "actor_behavior_cloning_extra_update_start_step",
            "actor_behavior_cloning_extra_updates",
            "actor_behavior_cloning_min_weight",
            "actor_behavior_cloning_replay_capacity",
            "actor_behavior_cloning_weight",
            "actor_deferrable_behavior_cloning_multiplier",
            "actor_deferrable_behavior_cloning_positive_target_power",
            "actor_deferrable_behavior_cloning_positive_target_weight",
            "actor_ev_behavior_cloning_multiplier",
            "actor_ev_behavior_cloning_positive_target_power",
            "actor_ev_behavior_cloning_positive_target_weight",
            "actor_ev_behavior_cloning_zero_target_threshold",
            "actor_ev_behavior_cloning_zero_target_weight",
            "actor_ev_v2g_action_l2_penalty",
            "actor_other_behavior_cloning_multiplier",
            "actor_policy_loss_weight",
            "actor_residual_behavior_cloning_neutral_target_threshold",
            "actor_residual_behavior_cloning_neutral_target_weight",
            "actor_storage_action_l2_penalty",
            "actor_storage_behavior_cloning_multiplier",
            "agent_update_order",
            "clip_ratio",
            "end_initial_exploration_time_step",
            "entropy_coef",
            "gae_lambda",
            "gamma",
            "initial_exploration_strategy",
            "initial_exploration_training_start_step",
            "initial_log_std",
            "local_action_safety_allow_discretionary_deferrable_start",
            "local_action_safety_enabled",
            "local_action_safety_ev_minimum_mode",
            "local_action_safety_fail_on_infeasible",
            "local_action_safety_headroom_reserve_kw",
            "local_action_safety_protect_ev_minimum",
            "local_action_safety_protect_ev_service_target",
            "local_action_safety_runtime_only_export",
            "local_action_safety_service_teacher_enabled",
            "local_action_safety_service_teacher_eval_enabled",
            "local_price_conditioning_enabled",
            "local_price_conditioning_trainable",
            "local_price_forecast_mode",
            "max_grad_norm",
            "max_log_std",
            "min_log_std",
            "minibatch_size",
            "ppo_epochs",
            "random_exploration_steps",
            "require_cuda",
            "residual_action_deadband",
            "residual_action_scale",
            "residual_base_policy",
            "residual_base_policy_deterministic",
            "residual_base_policy_hyperparameters",
            "residual_base_price_conditioning_enabled",
            "residual_building_deadbands",
            "residual_building_gain_multipliers",
            "residual_deferrable_action_scale_multiplier",
            "residual_ev_action_scale_multiplier",
            "residual_policy_enabled",
            "residual_storage_action_scale_multiplier",
            "residual_zero_initialization",
            "rollout_length",
            "target_kl",
            "train_during_initial_exploration",
            "value_loss_coef",
            "warm_start_policy",
            "warm_start_policy_deterministic",
            "warm_start_policy_hyperparameters",
            "warm_start_policy_noise_scale",
            "warm_start_policy_phaseout_mode",
            "warm_start_policy_phaseout_steps",
        ]
    ),
    "TransformerMADDPG": frozenset(
        ["bias", "end_initial_exploration_time_step", "gamma", "sigma", "tau"]
    ),
}

PARAMETER_KEYS["MASAC"] |= {"alpha_lr"}
PARAMETER_KEYS["TD3"] = PARAMETER_KEYS["MADDPG"] | {
    "local_action_safety_enabled",
    "local_action_safety_fail_on_infeasible",
    "local_action_safety_protect_ev_minimum",
    "local_action_safety_ev_minimum_mode",
    "local_action_safety_protect_ev_service_target",
    "local_action_safety_headroom_reserve_kw",
    "local_action_safety_allow_discretionary_deferrable_start",
    "local_action_safety_runtime_only_export",
    "local_action_safety_service_teacher_enabled",
    "local_action_safety_service_teacher_eval_enabled",
    "local_price_conditioning_enabled",
    "local_price_forecast_mode",
}
PARAMETER_KEYS["MATD3"] -= {"policy_delay", "policy_noise"}


def validate_actor_parameters(name: str, values: dict[str, Any]) -> dict[str, Any]:
    family = {"IPPO": "PPO", "MAPPO": "PPO", "HAPPO": "PPO"}.get(name, name)
    allowed = PARAMETER_KEYS[family]
    unknown = values.keys() - allowed
    if unknown:
        raise ValueError(f"{name} exploration.params contains unsupported keys: {sorted(unknown)}")
    checked = dict(TypeAdapter(ActorParameters).validate_python(values))
    from algorithms.configuration.agents import RuleBasedHyperparameters

    for key, value in list(checked.items()):
        if key.endswith("_hyperparameters") and value is not None:
            checked[key] = RuleBasedHyperparameters.model_validate(value).model_dump(
                exclude_unset=True
            )
    return checked
