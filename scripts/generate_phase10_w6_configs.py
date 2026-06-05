"""Generate Phase 10 W6 guided-training configuration matrices.

W6 is intentionally narrow: train MADDPG with RBCSmart-guided behavior cloning
on fixed windows first, then promote only the best recipes to remote smoke and
full-year runs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config_schema import validate_config


MADDPG_2022_TEMPLATE = REPO_ROOT / "configs/templates/maddpg/maddpg_2022_all_plus_evs_local.yaml"
RBC_SMART_2022_TEMPLATE = REPO_ROOT / "configs/templates/baselines/rbc_smart_2022_all_plus_evs_local.yaml"
RBC_COMMUNITY_2022_TEMPLATE = REPO_ROOT / "configs/templates/baselines/rbc_community_2022_all_plus_evs_local.yaml"

DATASET_NAME = "citylearn_challenge_2022_phase_all_plus_evs"
DATASET_PATH = "./datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
REMOTE_DATASET_PATH = "datasets/citylearn_challenge_2022_phase_all_plus_evs"
REWARD_FUNCTION = "CostServiceCommunityFeasiblePrecisionRewardV46"
DEFAULT_SEEDS = (123, 456)
DEFAULT_LOCAL_RECIPES = (
    "w6_ev_only_bc_primary",
    "w6_balanced_bc_storage_light",
    "w6_fast_decay_less_teacher",
    "w6_clone_diagnostic",
)
DEFAULT_PROMOTION_RECIPES = (
    "w6_flex_margin_teacher_storage_tight",
    "w6_clone_cost_ev_v2g_masswall_gentle",
)
LOCAL_WINDOWS = (
    ("win0_0000_2048", 0, 2048),
    ("win1_2048_4096", 2048, 2048),
    ("win2_4096_6144", 4096, 2048),
    ("win3_6144_8192", 6144, 2048),
)


@dataclass(frozen=True)
class Recipe:
    name: str
    bc_weight: float
    bc_min_weight: float
    ev_bc_multiplier: float
    storage_bc_multiplier: float
    zero_ev_target_weight: float
    storage_l2: float
    ev_v2g_l2: float
    teacher_phaseout_steps: int
    ev_v2g_mass: float = 0.0
    actor_policy_loss_weight: float = 1.0
    actor_policy_loss_warmup_weight: float = 0.03
    extra_bc_updates: int = 1
    extra_bc_steps: int = 2048
    reward_function: str = REWARD_FUNCTION
    reward_kwargs: tuple[tuple[str, Any], ...] = ()
    reward_normalization_clip: float = 10.0
    teacher_policy: str = "RBCSmartPolicy"
    residual_policy_enabled: bool = False
    residual_action_scale: float = 0.0
    residual_action_final_scale: float = 0.0
    residual_action_start_step: int = 0
    residual_action_growth_steps: int = 0
    residual_storage_action_scale_multiplier: float = 1.0
    residual_ev_action_scale_multiplier: float = 1.0
    residual_deferrable_action_scale_multiplier: float = 1.0
    replay_observation_event_priority_mode: str = "ev_departure_service"
    note: str = ""


RECIPES: dict[str, Recipe] = {
    "w6_ev_only_bc_primary": Recipe(
        name="w6_ev_only_bc_primary",
        bc_weight=0.060,
        bc_min_weight=0.006,
        ev_bc_multiplier=4.0,
        storage_bc_multiplier=0.0,
        zero_ev_target_weight=5.0,
        storage_l2=0.004,
        ev_v2g_l2=0.050,
        teacher_phaseout_steps=4096,
        note="EV-only RBCSmart behavior cloning with storage controlled by RL regularization.",
    ),
    "w6_balanced_bc_storage_light": Recipe(
        name="w6_balanced_bc_storage_light",
        bc_weight=0.040,
        bc_min_weight=0.004,
        ev_bc_multiplier=4.0,
        storage_bc_multiplier=0.15,
        zero_ev_target_weight=5.0,
        storage_l2=0.008,
        ev_v2g_l2=0.050,
        teacher_phaseout_steps=4096,
        note="Same EV pressure as primary with light storage cloning and stronger storage L2.",
    ),
    "w6_fast_decay_less_teacher": Recipe(
        name="w6_fast_decay_less_teacher",
        bc_weight=0.030,
        bc_min_weight=0.0,
        ev_bc_multiplier=4.0,
        storage_bc_multiplier=0.0,
        zero_ev_target_weight=5.0,
        storage_l2=0.004,
        ev_v2g_l2=0.050,
        teacher_phaseout_steps=2048,
        note="Fast teacher phaseout to check whether RL improves after warm-start.",
    ),
    "w6_clone_diagnostic": Recipe(
        name="w6_clone_diagnostic",
        bc_weight=0.500,
        bc_min_weight=0.500,
        ev_bc_multiplier=16.0,
        storage_bc_multiplier=0.25,
        zero_ev_target_weight=5.0,
        storage_l2=0.004,
        ev_v2g_l2=0.050,
        teacher_phaseout_steps=4096,
        actor_policy_loss_weight=0.0,
        actor_policy_loss_warmup_weight=0.0,
        extra_bc_updates=4,
        extra_bc_steps=4096,
        note="Diagnostic clone of RBCSmart; not a promotion candidate by itself.",
    ),
    "w6_clone_cost_nudge": Recipe(
        name="w6_clone_cost_nudge",
        bc_weight=0.350,
        bc_min_weight=0.200,
        ev_bc_multiplier=16.0,
        storage_bc_multiplier=0.25,
        zero_ev_target_weight=5.0,
        storage_l2=0.006,
        ev_v2g_l2=0.075,
        teacher_phaseout_steps=4096,
        actor_policy_loss_weight=0.050,
        actor_policy_loss_warmup_weight=0.010,
        extra_bc_updates=4,
        extra_bc_steps=4096,
        note="Clone-like policy with a small critic cost gradient to try to close the RBCSmart cost gap.",
    ),
    "w6_clone_tight_v2g_storage": Recipe(
        name="w6_clone_tight_v2g_storage",
        bc_weight=0.500,
        bc_min_weight=0.500,
        ev_bc_multiplier=16.0,
        storage_bc_multiplier=0.25,
        zero_ev_target_weight=5.0,
        storage_l2=0.010,
        ev_v2g_l2=0.150,
        teacher_phaseout_steps=4096,
        actor_policy_loss_weight=0.0,
        actor_policy_loss_warmup_weight=0.0,
        extra_bc_updates=4,
        extra_bc_steps=4096,
        note="Clone diagnostic with stronger storage and EV V2G regularization.",
    ),
    "w6_clone_cost_gentle_regularized": Recipe(
        name="w6_clone_cost_gentle_regularized",
        bc_weight=0.450,
        bc_min_weight=0.300,
        ev_bc_multiplier=16.0,
        storage_bc_multiplier=0.25,
        zero_ev_target_weight=5.0,
        storage_l2=0.010,
        ev_v2g_l2=0.150,
        teacher_phaseout_steps=4096,
        actor_policy_loss_weight=0.020,
        actor_policy_loss_warmup_weight=0.005,
        extra_bc_updates=4,
        extra_bc_steps=4096,
        note="Clone-like policy with a smaller critic gradient and tighter V2G/storage regularization.",
    ),
    "w6_clone_cost_nudge_v2g_tight": Recipe(
        name="w6_clone_cost_nudge_v2g_tight",
        bc_weight=0.350,
        bc_min_weight=0.200,
        ev_bc_multiplier=16.0,
        storage_bc_multiplier=0.25,
        zero_ev_target_weight=5.0,
        storage_l2=0.012,
        ev_v2g_l2=0.250,
        teacher_phaseout_steps=4096,
        actor_policy_loss_weight=0.050,
        actor_policy_loss_warmup_weight=0.010,
        extra_bc_updates=4,
        extra_bc_steps=4096,
        reward_kwargs=(
            ("ev_v2g_service_penalty", 1200.0),
            ("battery_throughput_penalty", 0.030),
        ),
        note="Cost nudge with stronger actor and reward penalties against EV V2G/storage churn.",
    ),
    "w6_clone_cost_v47_precision": Recipe(
        name="w6_clone_cost_v47_precision",
        bc_weight=0.420,
        bc_min_weight=0.260,
        ev_bc_multiplier=16.0,
        storage_bc_multiplier=0.25,
        zero_ev_target_weight=5.0,
        storage_l2=0.010,
        ev_v2g_l2=0.180,
        teacher_phaseout_steps=4096,
        actor_policy_loss_weight=0.030,
        actor_policy_loss_warmup_weight=0.006,
        extra_bc_updates=4,
        extra_bc_steps=4096,
        reward_function="CostServiceCommunityFeasiblePrecisionRewardV47",
        reward_kwargs=(
            ("ev_v2g_service_penalty", 1000.0),
            ("battery_throughput_penalty", 0.020),
        ),
        note="V47 precision profile with moderate cost gradient and V2G/storage discipline.",
    ),
    "w6_clone_cost_v50_deadline": Recipe(
        name="w6_clone_cost_v50_deadline",
        bc_weight=0.420,
        bc_min_weight=0.260,
        ev_bc_multiplier=16.0,
        storage_bc_multiplier=0.25,
        zero_ev_target_weight=5.0,
        storage_l2=0.010,
        ev_v2g_l2=0.180,
        teacher_phaseout_steps=4096,
        actor_policy_loss_weight=0.030,
        actor_policy_loss_warmup_weight=0.006,
        extra_bc_updates=4,
        extra_bc_steps=4096,
        reward_function="CostServiceCommunityDeadlineValueRewardV50",
        reward_kwargs=(
            ("ev_v2g_service_penalty", 1100.0),
            ("battery_throughput_penalty", 0.020),
        ),
        note="Deadline-value reward test to see if stronger service shaping can preserve EV while reducing cost.",
    ),
    "w6_clone_cost_v52_peak": Recipe(
        name="w6_clone_cost_v52_peak",
        bc_weight=0.420,
        bc_min_weight=0.260,
        ev_bc_multiplier=16.0,
        storage_bc_multiplier=0.25,
        zero_ev_target_weight=5.0,
        storage_l2=0.010,
        ev_v2g_l2=0.180,
        teacher_phaseout_steps=4096,
        actor_policy_loss_weight=0.030,
        actor_policy_loss_warmup_weight=0.006,
        extra_bc_updates=4,
        extra_bc_steps=4096,
        reward_function="CostServiceCommunityPeakDeadlineRewardV52",
        reward_kwargs=(
            ("ev_v2g_service_penalty", 1100.0),
            ("battery_throughput_penalty", 0.018),
        ),
        note="Peak/deadline reward test for community peak pressure without hard action guards.",
    ),
    "w6_clone_cost_ev_v2g_softwall": Recipe(
        name="w6_clone_cost_ev_v2g_softwall",
        bc_weight=0.350,
        bc_min_weight=0.200,
        ev_bc_multiplier=16.0,
        storage_bc_multiplier=0.25,
        zero_ev_target_weight=5.0,
        storage_l2=0.012,
        ev_v2g_l2=8.0,
        teacher_phaseout_steps=4096,
        actor_policy_loss_weight=0.050,
        actor_policy_loss_warmup_weight=0.010,
        extra_bc_updates=4,
        extra_bc_steps=4096,
        reward_kwargs=(
            ("ev_v2g_service_penalty", 1200.0),
            ("battery_throughput_penalty", 0.030),
        ),
        note="Cost nudge with a soft EV-discharge wall in the actor loss, not a hard action guard.",
    ),
    "w6_clone_cost_ev_v2g_softwall_gentle": Recipe(
        name="w6_clone_cost_ev_v2g_softwall_gentle",
        bc_weight=0.450,
        bc_min_weight=0.300,
        ev_bc_multiplier=16.0,
        storage_bc_multiplier=0.25,
        zero_ev_target_weight=5.0,
        storage_l2=0.012,
        ev_v2g_l2=8.0,
        teacher_phaseout_steps=4096,
        actor_policy_loss_weight=0.020,
        actor_policy_loss_warmup_weight=0.005,
        extra_bc_updates=4,
        extra_bc_steps=4096,
        reward_kwargs=(
            ("ev_v2g_service_penalty", 1200.0),
            ("battery_throughput_penalty", 0.030),
        ),
        note="Gentler cost nudge with soft EV-discharge wall and stronger clone anchoring.",
    ),
    "w6_clone_cost_ev_v2g_softwall_storage": Recipe(
        name="w6_clone_cost_ev_v2g_softwall_storage",
        bc_weight=0.380,
        bc_min_weight=0.240,
        ev_bc_multiplier=16.0,
        storage_bc_multiplier=0.35,
        zero_ev_target_weight=5.0,
        storage_l2=0.020,
        ev_v2g_l2=8.0,
        teacher_phaseout_steps=4096,
        actor_policy_loss_weight=0.040,
        actor_policy_loss_warmup_weight=0.008,
        extra_bc_updates=4,
        extra_bc_steps=4096,
        reward_kwargs=(
            ("ev_v2g_service_penalty", 1200.0),
            ("battery_throughput_penalty", 0.040),
        ),
        note="Soft EV-discharge wall plus stronger storage anchoring to reduce battery churn.",
    ),
    "w6_clone_cost_v2g_highclip": Recipe(
        name="w6_clone_cost_v2g_highclip",
        bc_weight=0.350,
        bc_min_weight=0.200,
        ev_bc_multiplier=16.0,
        storage_bc_multiplier=0.25,
        zero_ev_target_weight=5.0,
        storage_l2=0.012,
        ev_v2g_l2=0.250,
        teacher_phaseout_steps=4096,
        actor_policy_loss_weight=0.050,
        actor_policy_loss_warmup_weight=0.010,
        extra_bc_updates=4,
        extra_bc_steps=4096,
        reward_kwargs=(
            ("ev_v2g_service_penalty", 1600.0),
            ("battery_throughput_penalty", 0.040),
        ),
        reward_normalization_clip=25.0,
        note="Cost nudge with less reward clipping so V2G/storage penalties reach the critic.",
    ),
    "w6_clone_cost_softwall_highclip": Recipe(
        name="w6_clone_cost_softwall_highclip",
        bc_weight=0.380,
        bc_min_weight=0.240,
        ev_bc_multiplier=16.0,
        storage_bc_multiplier=0.25,
        zero_ev_target_weight=5.0,
        storage_l2=0.012,
        ev_v2g_l2=8.0,
        teacher_phaseout_steps=4096,
        actor_policy_loss_weight=0.040,
        actor_policy_loss_warmup_weight=0.008,
        extra_bc_updates=4,
        extra_bc_steps=4096,
        reward_kwargs=(
            ("ev_v2g_service_penalty", 1600.0),
            ("battery_throughput_penalty", 0.040),
        ),
        reward_normalization_clip=25.0,
        note="Soft EV-discharge wall with higher reward clip to test critic sensitivity to penalties.",
    ),
    "w6_clone_cost_ev_v2g_masswall": Recipe(
        name="w6_clone_cost_ev_v2g_masswall",
        bc_weight=0.380,
        bc_min_weight=0.240,
        ev_bc_multiplier=16.0,
        storage_bc_multiplier=0.25,
        zero_ev_target_weight=5.0,
        storage_l2=0.012,
        ev_v2g_l2=4.0,
        ev_v2g_mass=8.0,
        teacher_phaseout_steps=4096,
        actor_policy_loss_weight=0.040,
        actor_policy_loss_warmup_weight=0.008,
        extra_bc_updates=4,
        extra_bc_steps=4096,
        reward_kwargs=(
            ("ev_v2g_service_penalty", 1400.0),
            ("battery_throughput_penalty", 0.035),
        ),
        reward_normalization_clip=25.0,
        note="Soft V2G mass penalty to reduce frequent EV micro-discharge without a hard guard.",
    ),
    "w6_clone_cost_ev_v2g_masswall_gentle": Recipe(
        name="w6_clone_cost_ev_v2g_masswall_gentle",
        bc_weight=0.480,
        bc_min_weight=0.340,
        ev_bc_multiplier=18.0,
        storage_bc_multiplier=0.25,
        zero_ev_target_weight=5.0,
        storage_l2=0.012,
        ev_v2g_l2=4.0,
        ev_v2g_mass=8.0,
        teacher_phaseout_steps=4096,
        actor_policy_loss_weight=0.020,
        actor_policy_loss_warmup_weight=0.005,
        extra_bc_updates=4,
        extra_bc_steps=4096,
        reward_kwargs=(
            ("ev_v2g_service_penalty", 1400.0),
            ("battery_throughput_penalty", 0.035),
        ),
        reward_normalization_clip=25.0,
        note="Mass-wall variant with stronger EV teacher anchoring to recover the EV gate.",
    ),
    "w6_clone_cost_ev_v2g_energywall": Recipe(
        name="w6_clone_cost_ev_v2g_energywall",
        bc_weight=0.450,
        bc_min_weight=0.320,
        ev_bc_multiplier=18.0,
        storage_bc_multiplier=0.25,
        zero_ev_target_weight=5.0,
        storage_l2=0.012,
        ev_v2g_l2=2.0,
        ev_v2g_mass=4.0,
        teacher_phaseout_steps=4096,
        actor_policy_loss_weight=0.030,
        actor_policy_loss_warmup_weight=0.006,
        extra_bc_updates=4,
        extra_bc_steps=4096,
        reward_kwargs=(
            ("ev_v2g_discharge_penalty", 1.0),
            ("ev_v2g_service_penalty", 900.0),
            ("battery_throughput_penalty", 0.025),
        ),
        reward_normalization_clip=25.0,
        note="Reward-level EV V2G energy penalty so the critic stops valuing V2G discharge as free cost reduction.",
    ),
    "w6_clone_cost_ev_v2g_energywall_battery_tight": Recipe(
        name="w6_clone_cost_ev_v2g_energywall_battery_tight",
        bc_weight=0.460,
        bc_min_weight=0.340,
        ev_bc_multiplier=18.0,
        storage_bc_multiplier=0.30,
        zero_ev_target_weight=5.0,
        storage_l2=0.020,
        ev_v2g_l2=3.0,
        ev_v2g_mass=6.0,
        teacher_phaseout_steps=4096,
        actor_policy_loss_weight=0.025,
        actor_policy_loss_warmup_weight=0.006,
        extra_bc_updates=4,
        extra_bc_steps=4096,
        reward_kwargs=(
            ("ev_v2g_discharge_penalty", 1.0),
            ("ev_v2g_service_penalty", 900.0),
            ("battery_throughput_penalty", 0.050),
        ),
        reward_normalization_clip=25.0,
        note="Energywall variant with stronger storage/throughput discipline after battery rose above RBCSmart.",
    ),
    "w6_flex_v2g_safe_value": Recipe(
        name="w6_flex_v2g_safe_value",
        bc_weight=0.420,
        bc_min_weight=0.260,
        ev_bc_multiplier=14.0,
        storage_bc_multiplier=0.20,
        zero_ev_target_weight=5.0,
        storage_l2=0.005,
        ev_v2g_l2=0.40,
        ev_v2g_mass=0.80,
        teacher_phaseout_steps=4096,
        actor_policy_loss_weight=0.035,
        actor_policy_loss_warmup_weight=0.008,
        extra_bc_updates=4,
        extra_bc_steps=4096,
        reward_function="CostServiceCommunityPeakDeadlineRewardV52",
        reward_kwargs=(
            ("community_settlement_cost_weight", 1.25),
            ("community_peak_import_penalty", 0.0020),
            ("community_export_penalty", 0.00035),
            ("ev_v2g_service_penalty", 1400.0),
            ("battery_throughput_penalty", 0.004),
        ),
        reward_normalization_clip=25.0,
        note="Flex recipe: allow useful V2G/storage, punish only service-risk V2G hard, value community peaks.",
    ),
    "w6_flex_storage_peak_value": Recipe(
        name="w6_flex_storage_peak_value",
        bc_weight=0.400,
        bc_min_weight=0.240,
        ev_bc_multiplier=14.0,
        storage_bc_multiplier=0.12,
        zero_ev_target_weight=5.0,
        storage_l2=0.002,
        ev_v2g_l2=1.00,
        ev_v2g_mass=2.00,
        teacher_phaseout_steps=4096,
        actor_policy_loss_weight=0.045,
        actor_policy_loss_warmup_weight=0.010,
        extra_bc_updates=4,
        extra_bc_steps=4096,
        reward_function="CostServiceCommunityPeakDeadlineRewardV52",
        reward_kwargs=(
            ("community_settlement_cost_weight", 1.30),
            ("community_peak_import_penalty", 0.0025),
            ("community_export_penalty", 0.00035),
            ("ev_v2g_service_penalty", 1100.0),
            ("battery_throughput_penalty", 0.0015),
        ),
        reward_normalization_clip=25.0,
        note="Flex recipe: prioritize stationary storage/peak value with low throughput cost and moderate EV V2G guard.",
    ),
    "w6_flex_v2g_margin_teacher": Recipe(
        name="w6_flex_v2g_margin_teacher",
        bc_weight=0.460,
        bc_min_weight=0.300,
        ev_bc_multiplier=18.0,
        storage_bc_multiplier=0.18,
        zero_ev_target_weight=5.0,
        storage_l2=0.006,
        ev_v2g_l2=0.80,
        ev_v2g_mass=1.50,
        teacher_phaseout_steps=4096,
        actor_policy_loss_weight=0.030,
        actor_policy_loss_warmup_weight=0.006,
        extra_bc_updates=4,
        extra_bc_steps=4096,
        reward_kwargs=(
            ("community_settlement_cost_weight", 1.18),
            ("community_peak_import_penalty", 0.0016),
            ("ev_v2g_service_penalty", 1600.0),
            ("battery_throughput_penalty", 0.006),
        ),
        reward_normalization_clip=25.0,
        note="Flex recipe: keep strong EV teacher margin while reopening safe V2G and battery value.",
    ),
    "w6_flex_margin_teacher_storage_tight": Recipe(
        name="w6_flex_margin_teacher_storage_tight",
        bc_weight=0.460,
        bc_min_weight=0.320,
        ev_bc_multiplier=18.0,
        storage_bc_multiplier=0.22,
        zero_ev_target_weight=5.0,
        storage_l2=0.014,
        ev_v2g_l2=0.80,
        ev_v2g_mass=1.50,
        teacher_phaseout_steps=4096,
        actor_policy_loss_weight=0.030,
        actor_policy_loss_warmup_weight=0.006,
        extra_bc_updates=4,
        extra_bc_steps=4096,
        reward_kwargs=(
            ("community_settlement_cost_weight", 1.16),
            ("community_peak_import_penalty", 0.0014),
            ("ev_v2g_service_penalty", 1600.0),
            ("battery_throughput_penalty", 0.014),
        ),
        reward_normalization_clip=25.0,
        note="Flex follow-up: same EV margin as margin_teacher, but tighter storage throughput discipline.",
    ),
    "w6_flex_ev_gate_repair_strong_bc": Recipe(
        name="w6_flex_ev_gate_repair_strong_bc",
        bc_weight=0.680,
        bc_min_weight=0.520,
        ev_bc_multiplier=28.0,
        storage_bc_multiplier=0.22,
        zero_ev_target_weight=5.0,
        storage_l2=0.014,
        ev_v2g_l2=1.00,
        ev_v2g_mass=2.00,
        teacher_phaseout_steps=8192,
        actor_policy_loss_weight=0.010,
        actor_policy_loss_warmup_weight=0.002,
        extra_bc_updates=4,
        extra_bc_steps=4096,
        reward_kwargs=(
            ("community_settlement_cost_weight", 1.16),
            ("community_peak_import_penalty", 0.0014),
            ("ev_v2g_service_penalty", 1800.0),
            ("battery_throughput_penalty", 0.018),
        ),
        reward_normalization_clip=25.0,
        note="Flex EV-gate repair: stronger RBCSmart EV anchoring after full-year flex missed feasible EV service.",
    ),
    "w6_flex_ev_gate_repair_mid_bc": Recipe(
        name="w6_flex_ev_gate_repair_mid_bc",
        bc_weight=0.560,
        bc_min_weight=0.360,
        ev_bc_multiplier=22.0,
        storage_bc_multiplier=0.20,
        zero_ev_target_weight=5.0,
        storage_l2=0.012,
        ev_v2g_l2=0.80,
        ev_v2g_mass=1.60,
        teacher_phaseout_steps=6144,
        actor_policy_loss_weight=0.018,
        actor_policy_loss_warmup_weight=0.004,
        extra_bc_updates=4,
        extra_bc_steps=4096,
        reward_kwargs=(
            ("community_settlement_cost_weight", 1.18),
            ("community_peak_import_penalty", 0.0015),
            ("ev_v2g_service_penalty", 1700.0),
            ("battery_throughput_penalty", 0.014),
        ),
        reward_normalization_clip=25.0,
        note="Flex EV-gate repair: reduce teacher lock-in while retaining stronger EV service anchoring.",
    ),
    "w6_flex_ev_gate_repair_cost_push": Recipe(
        name="w6_flex_ev_gate_repair_cost_push",
        bc_weight=0.520,
        bc_min_weight=0.300,
        ev_bc_multiplier=20.0,
        storage_bc_multiplier=0.18,
        zero_ev_target_weight=5.0,
        storage_l2=0.010,
        ev_v2g_l2=0.70,
        ev_v2g_mass=1.40,
        teacher_phaseout_steps=4096,
        actor_policy_loss_weight=0.026,
        actor_policy_loss_warmup_weight=0.006,
        extra_bc_updates=3,
        extra_bc_steps=4096,
        reward_kwargs=(
            ("community_settlement_cost_weight", 1.24),
            ("community_peak_import_penalty", 0.0017),
            ("ev_v2g_service_penalty", 1650.0),
            ("battery_throughput_penalty", 0.010),
        ),
        reward_normalization_clip=25.0,
        note="Flex EV-gate repair: push cost/peak objective harder after mid-BC service repair.",
    ),
    "w6_flex_ev_gate_repair_policy_open": Recipe(
        name="w6_flex_ev_gate_repair_policy_open",
        bc_weight=0.460,
        bc_min_weight=0.240,
        ev_bc_multiplier=18.0,
        storage_bc_multiplier=0.16,
        zero_ev_target_weight=5.0,
        storage_l2=0.008,
        ev_v2g_l2=0.55,
        ev_v2g_mass=1.00,
        teacher_phaseout_steps=3072,
        actor_policy_loss_weight=0.035,
        actor_policy_loss_warmup_weight=0.008,
        extra_bc_updates=2,
        extra_bc_steps=3072,
        reward_kwargs=(
            ("community_settlement_cost_weight", 1.28),
            ("community_peak_import_penalty", 0.0018),
            ("ev_v2g_service_penalty", 1600.0),
            ("battery_throughput_penalty", 0.008),
        ),
        reward_normalization_clip=25.0,
        note="Flex EV-gate repair: diagnostic for whether more policy freedom recovers cost without losing EV service.",
    ),
    "w6_flex_v2g_open_value": Recipe(
        name="w6_flex_v2g_open_value",
        bc_weight=0.450,
        bc_min_weight=0.300,
        ev_bc_multiplier=18.0,
        storage_bc_multiplier=0.14,
        zero_ev_target_weight=5.0,
        storage_l2=0.004,
        ev_v2g_l2=0.25,
        ev_v2g_mass=0.45,
        teacher_phaseout_steps=4096,
        actor_policy_loss_weight=0.035,
        actor_policy_loss_warmup_weight=0.008,
        extra_bc_updates=4,
        extra_bc_steps=4096,
        reward_function="CostServiceCommunityPeakDeadlineRewardV52",
        reward_kwargs=(
            ("community_settlement_cost_weight", 1.25),
            ("community_peak_import_penalty", 0.0022),
            ("community_export_penalty", 0.00035),
            ("ev_v2g_service_penalty", 2000.0),
            ("battery_throughput_penalty", 0.005),
        ),
        reward_normalization_clip=25.0,
        note="Flex follow-up: reopen EV V2G value while keeping strong EV margin and service-risk penalty.",
    ),
    "w6_residual_comm_constraint": Recipe(
        name="w6_residual_comm_constraint",
        bc_weight=0.120,
        bc_min_weight=0.020,
        ev_bc_multiplier=8.0,
        storage_bc_multiplier=0.40,
        zero_ev_target_weight=3.0,
        storage_l2=0.002,
        ev_v2g_l2=0.080,
        ev_v2g_mass=0.120,
        teacher_phaseout_steps=4096,
        actor_policy_loss_weight=0.080,
        actor_policy_loss_warmup_weight=0.020,
        extra_bc_updates=3,
        extra_bc_steps=4096,
        reward_function="CostServiceCommunityResidualConstraintRewardV53",
        reward_normalization_clip=25.0,
        teacher_policy="RBCCommunityPolicy",
        residual_policy_enabled=True,
        residual_action_scale=0.08,
        residual_action_final_scale=0.28,
        residual_action_growth_steps=4096,
        residual_storage_action_scale_multiplier=0.65,
        residual_ev_action_scale_multiplier=0.50,
        residual_deferrable_action_scale_multiplier=0.80,
        replay_observation_event_priority_mode="combined",
        note="Residual over RBCCommunity: conservative deltas, strict EV gate, community cost/peak/export value.",
    ),
    "w6_residual_comm_cost_push": Recipe(
        name="w6_residual_comm_cost_push",
        bc_weight=0.070,
        bc_min_weight=0.006,
        ev_bc_multiplier=6.0,
        storage_bc_multiplier=0.25,
        zero_ev_target_weight=2.5,
        storage_l2=0.0015,
        ev_v2g_l2=0.060,
        ev_v2g_mass=0.080,
        teacher_phaseout_steps=3072,
        actor_policy_loss_weight=0.120,
        actor_policy_loss_warmup_weight=0.025,
        extra_bc_updates=2,
        extra_bc_steps=3072,
        reward_function="CostServiceCommunityResidualConstraintRewardV53",
        reward_kwargs=(
            ("community_settlement_cost_weight", 1.32),
            ("community_peak_import_penalty", 0.0024),
            ("community_export_penalty", 0.00075),
            ("ev_departure_deficit_penalty", 1350.0),
            ("ev_v2g_service_penalty", 1250.0),
            ("battery_throughput_penalty", 0.0020),
        ),
        reward_normalization_clip=30.0,
        teacher_policy="RBCCommunityPolicy",
        residual_policy_enabled=True,
        residual_action_scale=0.10,
        residual_action_final_scale=0.36,
        residual_action_growth_steps=4096,
        residual_storage_action_scale_multiplier=0.80,
        residual_ev_action_scale_multiplier=0.60,
        residual_deferrable_action_scale_multiplier=1.00,
        replay_observation_event_priority_mode="combined",
        note="Residual over RBCCommunity: lower BC and stronger cost/peak/export pressure.",
    ),
    "w6_residual_smart_ev_safe": Recipe(
        name="w6_residual_smart_ev_safe",
        bc_weight=0.140,
        bc_min_weight=0.030,
        ev_bc_multiplier=10.0,
        storage_bc_multiplier=0.20,
        zero_ev_target_weight=3.5,
        storage_l2=0.0025,
        ev_v2g_l2=0.100,
        ev_v2g_mass=0.160,
        teacher_phaseout_steps=4096,
        actor_policy_loss_weight=0.070,
        actor_policy_loss_warmup_weight=0.015,
        extra_bc_updates=3,
        extra_bc_steps=4096,
        reward_function="CostServiceCommunityResidualConstraintRewardV53",
        reward_kwargs=(
            ("ev_departure_deficit_penalty", 1450.0),
            ("ev_departure_missed_penalty", 4200.0),
            ("ev_v2g_service_penalty", 1450.0),
        ),
        reward_normalization_clip=25.0,
        teacher_policy="RBCSmartPolicy",
        residual_policy_enabled=True,
        residual_action_scale=0.06,
        residual_action_final_scale=0.24,
        residual_action_growth_steps=4096,
        residual_storage_action_scale_multiplier=0.65,
        residual_ev_action_scale_multiplier=0.45,
        residual_deferrable_action_scale_multiplier=0.80,
        replay_observation_event_priority_mode="combined",
        note="Residual over RBCSmart: service-safe comparator for checking whether community teacher hurts EV gates.",
    ),
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=False)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("._-")
    return slug or "item"


def _validated(payload: Mapping[str, Any]) -> dict[str, Any]:
    return validate_config(copy.deepcopy(dict(payload))).to_dict()


def _rbc_smart_hyperparameters() -> dict[str, Any]:
    payload = _validated(_load_yaml(RBC_SMART_2022_TEMPLATE))
    return dict(payload.get("algorithm", {}).get("hyperparameters", {}) or {})


def _rbc_community_hyperparameters() -> dict[str, Any]:
    payload = _validated(_load_yaml(RBC_COMMUNITY_2022_TEMPLATE))
    return dict(payload.get("algorithm", {}).get("hyperparameters", {}) or {})


def _teacher_hyperparameters(policy_name: str) -> dict[str, Any]:
    if policy_name == "RBCCommunityPolicy":
        return _rbc_community_hyperparameters()
    if policy_name == "RBCSmartPolicy":
        return _rbc_smart_hyperparameters()
    return {}


def _normalise_requested(values: Sequence[str] | None, default: Sequence[str]) -> tuple[str, ...]:
    if not values:
        return tuple(default)
    return tuple(_slug(value) for value in values)


def _stage_names(stage: str) -> tuple[str, ...]:
    if stage == "all":
        return ("w6-smoke-local", "w6a-local", "w6b-remote-smoke", "w6c-full-year")
    return (stage,)


def _stage_windows(stage: str) -> tuple[tuple[str, int, int], ...]:
    if stage == "w6-smoke-local":
        return (("local_smoke_0000_0256", 0, 256),)
    if stage == "w6a-local":
        return LOCAL_WINDOWS
    if stage == "w6b-remote-smoke":
        return (("remote_smoke_0000_4096", 0, 4096),)
    if stage == "w6c-full-year":
        return (("full_year_0000_8760", 0, 8760),)
    raise ValueError(f"Unsupported W6 stage: {stage}")


def _stage_recipes(stage: str, requested: Sequence[str] | None) -> tuple[str, ...]:
    if requested:
        selected = _normalise_requested(requested, ())
    elif stage in {"w6-smoke-local", "w6a-local"}:
        selected = DEFAULT_LOCAL_RECIPES
    else:
        selected = DEFAULT_PROMOTION_RECIPES

    unknown = [recipe for recipe in selected if recipe not in RECIPES]
    if unknown:
        raise ValueError(f"Unknown W6 recipe(s): {', '.join(unknown)}")
    return selected


def _stage_algorithms(stage: str, requested: Sequence[str] | None) -> tuple[str, ...]:
    if requested:
        selected = tuple(str(value).upper() for value in requested)
    elif stage == "w6c-full-year":
        selected = ("MADDPG", "MATD3")
    else:
        selected = ("MADDPG",)

    unsupported = [name for name in selected if name not in {"MADDPG", "MATD3"}]
    if unsupported:
        raise ValueError(f"W6 only supports MADDPG/MATD3 here, got: {', '.join(unsupported)}")
    if stage in {"w6b-remote-smoke"} and any(
        name != "MADDPG" for name in selected
    ):
        raise ValueError(f"{stage} is intentionally MADDPG-only.")
    return selected


def _stage_runtime(stage: str) -> dict[str, Any]:
    if stage == "w6-smoke-local":
        return {
            "episodes": 1,
            "deterministic_finish": True,
            "random_exploration_steps": 64,
            "require_cuda": False,
            "updates_every": 4,
            "target_update_interval": 2,
            "deucalion": None,
            "watchdog": False,
            "max_rss_mb": None,
            "tracking_log_frequency": 64,
            "progress_interval": 32,
        }
    if stage == "w6a-local":
        return {
            "episodes": 8,
            "deterministic_finish": True,
            "random_exploration_steps": 512,
            "require_cuda": False,
            "updates_every": 4,
            "target_update_interval": 2,
            "deucalion": None,
            "watchdog": False,
            "max_rss_mb": None,
            "tracking_log_frequency": 128,
            "progress_interval": 64,
        }
    if stage == "w6b-remote-smoke":
        return {
            "episodes": 1,
            "deterministic_finish": True,
            "random_exploration_steps": 512,
            "require_cuda": True,
            "updates_every": 4,
            "target_update_interval": 2,
            "deucalion": {
                "partition": "a100",
                "time": "04:00:00",
                "cpus_per_task": 4,
                "mem_gb": 64,
                "gpus": 1,
                "datasets": [REMOTE_DATASET_PATH],
                "command_mode": "run",
            },
            "watchdog": False,
            "max_rss_mb": 56000.0,
            "tracking_log_frequency": 512,
            "progress_interval": 64,
        }
    if stage == "w6c-full-year":
        return {
            "episodes": 2,
            "deterministic_finish": True,
            "random_exploration_steps": 1024,
            "require_cuda": True,
            "updates_every": 4,
            "target_update_interval": 2,
            "deucalion": {
                "partition": "a100",
                "time": "12:00:00",
                "cpus_per_task": 4,
                "mem_gb": 96,
                "gpus": 1,
                "datasets": [REMOTE_DATASET_PATH],
                "command_mode": "run",
            },
            "watchdog": True,
            "max_rss_mb": 88000.0,
            "tracking_log_frequency": 512,
            "progress_interval": 128,
        }
    raise ValueError(f"Unsupported W6 stage: {stage}")


def _apply_tracking(config: dict[str, Any], *, stage: str, runtime: Mapping[str, Any]) -> None:
    tracking = config.setdefault("tracking", {})
    tracking.update(
        {
            "mlflow_enabled": False,
            "log_level": "INFO",
            "log_frequency": int(runtime["tracking_log_frequency"]),
            "mlflow_step_sample_interval": int(runtime["tracking_log_frequency"]),
            "mlflow_artifacts_profile": "minimal",
            "progress_updates_enabled": True,
            "progress_update_interval": int(runtime["progress_interval"]),
            "system_metrics_enabled": False,
            "system_metrics_interval": 32,
            "action_diagnostics_enabled": True,
            "action_diagnostics_detail": "summary",
            "training_diagnostics_enabled": True,
            "training_diagnostics_detail": "summary",
            "reward_diagnostics_enabled": True,
            "reward_diagnostics_detail": "summary",
            "runtime_profiling_enabled": stage not in {"w6-smoke-local", "w6a-local"},
            "runtime_profiling_interval": 512,
            "runtime_profiling_detail": "summary",
            "progress_phase_updates_enabled": stage not in {"w6-smoke-local", "w6a-local"},
            "progress_phase_start_step": 0 if stage not in {"w6-smoke-local", "w6a-local"} else None,
            "progress_phase_end_step": None,
            "max_step_seconds": 180.0 if stage not in {"w6-smoke-local", "w6a-local"} else None,
            "stall_watchdog_enabled": bool(runtime["watchdog"]),
            "stall_watchdog_timeout_seconds": 900.0 if runtime["watchdog"] else None,
            "stall_watchdog_exit_on_timeout": True,
            "stall_watchdog_repeat": False,
            "stall_watchdog_traceback_file": None,
            "stall_watchdog_context_interval_steps": 64,
            "resource_guard_enabled": runtime["max_rss_mb"] is not None,
            "max_process_rss_mb": runtime["max_rss_mb"],
            "min_available_ram_mb": 2048.0 if runtime["max_rss_mb"] is not None else None,
        }
    )


def _apply_simulator(
    config: dict[str, Any],
    *,
    session_name: str,
    start: int,
    steps: int,
    episodes: int,
    deterministic_finish: bool,
    reward_function: str = REWARD_FUNCTION,
    reward_kwargs: Mapping[str, Any] | None = None,
) -> None:
    simulator = config.setdefault("simulator", {})
    simulator["dataset_name"] = DATASET_NAME
    simulator["dataset_path"] = DATASET_PATH
    simulator["central_agent"] = False
    simulator["interface"] = "entity"
    simulator["topology_mode"] = "static"
    entity_encoding = simulator.setdefault("entity_encoding", {})
    entity_encoding.update(
        {
            "enabled": True,
            "normalization": "minmax_space",
            "profile": "maddpg_v3_operational",
            "clip": True,
        }
    )
    simulator["reward_function"] = str(reward_function)
    simulator["reward_function_kwargs"] = dict(reward_kwargs or {})
    simulator["episodes"] = int(episodes)
    simulator["deterministic_finish"] = bool(deterministic_finish)
    simulator["simulation_start_time_step"] = int(start)
    simulator["simulation_end_time_step"] = int(start + steps - 1)
    simulator["episode_time_steps"] = int(steps)
    export_cfg = simulator.setdefault("export", {})
    export_cfg.update(
        {
            "mode": "end",
            "export_kpis_on_episode_end": True,
            "final_episode_only": True,
            "kpis_final_episode_only": True,
            "timeseries_final_episode_only": True,
            "include_business_as_usual": True,
            "export_business_as_usual_timeseries": True,
            "kpi_round_decimals": None,
            "session_name": session_name,
        }
    )
    wrapper_reward = simulator.setdefault("wrapper_reward", {})
    wrapper_reward.update(
        {
            "enabled": False,
            "profile": "cost_limits_v1",
            "clip_enabled": True,
            "clip_min": -10.0,
            "clip_max": 10.0,
            "squash": "none",
        }
    )


def _apply_checkpointing(config: dict[str, Any]) -> None:
    checkpointing = config.setdefault("checkpointing", {})
    checkpointing.update(
        {
            "resume_training": False,
            "checkpoint_run_id": None,
            "checkpoint_artifact": "latest_checkpoint.pth",
            "use_best_checkpoint_artifact": False,
            "reset_replay_buffer": False,
            "freeze_pretrained_layers": False,
            "fine_tune": False,
            "checkpoint_interval": None,
            "require_update_step": True,
            "require_initial_exploration_done": True,
        }
    )


def _apply_execution(config: dict[str, Any], runtime: Mapping[str, Any]) -> None:
    deucalion = runtime.get("deucalion")
    config["execution"] = {"deucalion": copy.deepcopy(deucalion)} if deucalion else None


def _apply_common_metadata(config: dict[str, Any], *, run_id: str, run_name: str, description: str) -> None:
    metadata = config.setdefault("metadata", {})
    metadata.update(
        {
            "experiment_name": run_id,
            "run_name": run_name,
            "community_name": "citylearn_2022",
            "description": description,
            "bundle_version": "phase10-w6",
            "alias_mapping_path": None,
        }
    )
    bundle = config.setdefault("bundle", {})
    bundle.update(
        {
            "bundle_version": "phase10-w6",
            "description": description,
            "alias_mapping_path": None,
            "require_observations_envelope": False,
            "artifact_config": {},
            "per_agent_artifact_config": {},
        }
    )


def _build_baseline_config(
    *,
    policy_name: str,
    template_path: Path,
    stage: str,
    window_name: str,
    start: int,
    steps: int,
    seed: int,
    runtime: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    config = _validated(_load_yaml(template_path))
    run_id = _slug(f"phase10_w6a_{policy_name}_{window_name}_seed{seed}")
    _apply_common_metadata(
        config,
        run_id=run_id,
        run_name=f"Phase 10 W6A {policy_name} {window_name} seed {seed}",
        description="W6 same-window rule-based baseline for guided training gates.",
    )
    _apply_tracking(config, stage=stage, runtime=runtime)
    config.setdefault("tracking", {}).setdefault("tags", {}).update(
        {
            "stage": stage,
            "recipe": policy_name,
            "window": window_name,
            "seed": int(seed),
        }
    )
    _apply_checkpointing(config)
    _apply_simulator(
        config,
        session_name=run_id.replace("_", "-"),
        start=start,
        steps=steps,
        episodes=1,
        deterministic_finish=True,
    )
    training = config.setdefault("training", {})
    training["seed"] = int(seed)
    training["steps_between_training_updates"] = 1
    training["target_update_interval"] = 0
    _apply_execution(config, runtime)
    return run_id, _validated(config)


def _build_rl_config(
    *,
    stage: str,
    algorithm_name: str,
    recipe: Recipe,
    window_name: str,
    start: int,
    steps: int,
    seed: int,
    runtime: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    config = _validated(_load_yaml(MADDPG_2022_TEMPLATE))
    run_id = _slug(f"phase10_{stage}_{recipe.name}_{algorithm_name}_{window_name}_seed{seed}")
    total_steps = max(int(runtime["episodes"]) * int(steps), 1)
    random_steps = min(int(runtime["random_exploration_steps"]), max(total_steps - 1, 1))
    decay_steps = max(total_steps - random_steps, 1)
    extra_bc_end = min(total_steps, random_steps + int(recipe.extra_bc_steps))

    _apply_common_metadata(
        config,
        run_id=run_id,
        run_name=f"Phase 10 W6 {stage} {algorithm_name} {recipe.name} {window_name} seed {seed}",
        description=f"W6 guided training recipe: {recipe.note}",
    )
    _apply_tracking(config, stage=stage, runtime=runtime)
    config.setdefault("tracking", {}).setdefault("tags", {}).update(
        {
            "stage": stage,
            "algorithm": str(algorithm_name),
            "recipe": recipe.name,
            "window": window_name,
            "seed": int(seed),
            "reward_function": recipe.reward_function,
            "teacher_policy": recipe.teacher_policy,
            "residual_policy": bool(recipe.residual_policy_enabled),
        }
    )
    _apply_checkpointing(config)
    _apply_simulator(
        config,
        session_name=run_id.replace("_", "-"),
        start=start,
        steps=steps,
        episodes=int(runtime["episodes"]),
        deterministic_finish=bool(runtime["deterministic_finish"]),
        reward_function=recipe.reward_function,
        reward_kwargs=dict(recipe.reward_kwargs),
    )

    training = config.setdefault("training", {})
    training["seed"] = int(seed)
    training["steps_between_training_updates"] = int(runtime["updates_every"])
    training["target_update_interval"] = int(runtime["target_update_interval"])

    algorithm = config.setdefault("algorithm", {})
    algorithm["name"] = str(algorithm_name)
    algorithm["hyperparameters"] = {
        "gamma": 0.995,
        "require_cuda": bool(runtime["require_cuda"]),
    }
    algorithm.setdefault("networks", {}).setdefault("critic", {}).update(
        {
            "class": "LateFusionCritic",
            "layers": [1024, 512, 256],
            "state_layers": [1024, 512],
            "action_layers": [256],
            "joint_layers": [512, 256],
        }
    )
    algorithm["replay_buffer"] = {
        "class": "RewardWeightedMultiAgentReplayBuffer",
        "capacity": 200000,
        "batch_size": 256,
        "priority_fraction": 0.35,
        "priority_alpha": 0.60,
        "priority_epsilon": 0.001,
        "priority_mode": "negative_reward",
        "priority_max": 50.0,
        "behavior_action_priority_weight": 1.0,
        "behavior_action_priority_mode": "positive",
        "behavior_action_priority_scope": "ev",
        "observation_event_priority_weight": 8.0,
        "observation_event_priority_mode": str(recipe.replay_observation_event_priority_mode),
    }

    exploration = algorithm.setdefault("exploration", {}).setdefault("params", {})
    exploration.update(
        {
            "bias": 0.0,
            "sigma": 0.08,
            "decay": 0.9995,
            "min_sigma": 0.01,
            "noise_clip": 0.15,
            "storage_exploration_noise_multiplier": 0.25,
            "ev_negative_exploration_noise_multiplier": 0.10,
            "gamma": 0.99,
            "tau": 0.001,
            "use_amp": True,
            "end_initial_exploration_time_step": random_steps,
            "random_exploration_steps": random_steps,
            "initial_exploration_strategy": "policy",
            "warm_start_policy": str(recipe.teacher_policy),
            "warm_start_policy_hyperparameters": _teacher_hyperparameters(recipe.teacher_policy),
            "warm_start_policy_deterministic": True,
            "warm_start_policy_noise_scale": 0.0,
            "warm_start_policy_phaseout_steps": int(recipe.teacher_phaseout_steps),
            "warm_start_policy_phaseout_mode": "blend",
            "train_during_initial_exploration": False,
            "initial_exploration_training_start_step": 0,
            "noop_noise_scale": 0.0,
            "deferrable_on_probability": 0.2,
            "deferrable_trigger_threshold": 0.5,
            "noop_actor_initialization": True,
            "noop_actor_initialization_epsilon": 0.05,
            "critic_update_mode": "joint_mean",
            "actor_update_interval": 2 if algorithm_name == "MATD3" else 1,
            "target_policy_smoothing": algorithm_name == "MATD3",
            "target_policy_noise": 0.05,
            "target_policy_noise_clip": 0.10,
            "critic_loss": "huber",
            "critic_huber_beta": 1.0,
            "critic_target_clip_abs": 25.0,
            "actor_policy_loss_weight": float(recipe.actor_policy_loss_weight),
            "actor_policy_loss_warmup_weight": float(recipe.actor_policy_loss_warmup_weight),
            "actor_policy_loss_warmup_start_step": random_steps,
            "actor_policy_loss_warmup_steps": int(recipe.teacher_phaseout_steps),
            "actor_action_l2_penalty": 0.0,
            "actor_action_saturation_penalty": 0.01,
            "actor_storage_action_l2_penalty": float(recipe.storage_l2),
            "actor_ev_v2g_action_l2_penalty": float(recipe.ev_v2g_l2),
            "actor_ev_v2g_action_mass_penalty": float(recipe.ev_v2g_mass),
            "actor_action_saturation_threshold": 0.85,
            "actor_behavior_cloning_weight": float(recipe.bc_weight),
            "actor_behavior_cloning_min_weight": float(recipe.bc_min_weight),
            "actor_behavior_cloning_decay_start_step": random_steps,
            "actor_behavior_cloning_decay_steps": decay_steps,
            "actor_behavior_cloning_extra_updates": int(recipe.extra_bc_updates),
            "actor_behavior_cloning_extra_update_start_step": random_steps,
            "actor_behavior_cloning_extra_update_end_step": extra_bc_end,
            "actor_behavior_cloning_source": "warm_start_policy",
            "actor_ev_behavior_cloning_multiplier": float(recipe.ev_bc_multiplier),
            "actor_storage_behavior_cloning_multiplier": float(recipe.storage_bc_multiplier),
            "actor_ev_behavior_cloning_positive_target_weight": 1.0,
            "actor_ev_behavior_cloning_positive_target_power": 1.0,
            "actor_ev_behavior_cloning_zero_target_weight": float(recipe.zero_ev_target_weight),
            "actor_ev_behavior_cloning_zero_target_threshold": 0.05,
            "reward_normalization": True,
            "reward_normalization_clip": float(recipe.reward_normalization_clip),
            "reward_normalization_epsilon": 1.0e-8,
            "residual_policy_enabled": bool(recipe.residual_policy_enabled),
            "residual_action_scale": float(recipe.residual_action_scale),
            "residual_action_final_scale": float(recipe.residual_action_final_scale),
            "residual_action_start_step": int(recipe.residual_action_start_step),
            "residual_action_growth_steps": int(recipe.residual_action_growth_steps),
            "residual_storage_action_scale_multiplier": float(
                recipe.residual_storage_action_scale_multiplier
            ),
            "residual_ev_action_scale_multiplier": float(recipe.residual_ev_action_scale_multiplier),
            "residual_deferrable_action_scale_multiplier": float(
                recipe.residual_deferrable_action_scale_multiplier
            ),
        }
    )

    _apply_execution(config, runtime)
    return run_id, _validated(config)


def _matrix_row(
    *,
    stage: str,
    run_id: str,
    path: Path,
    algorithm: str,
    recipe: str,
    window_name: str,
    start: int,
    steps: int,
    seed: int,
    runtime: Mapping[str, Any],
    reward_function: str = REWARD_FUNCTION,
) -> dict[str, Any]:
    deucalion = runtime.get("deucalion") or {}
    return {
        "stage": stage,
        "job_id": run_id,
        "config_path": str(path),
        "algorithm": algorithm,
        "recipe": recipe,
        "seed": seed,
        "window": window_name,
        "start_step": start,
        "end_step": start + steps - 1,
        "episode_steps": steps,
        "episodes": runtime["episodes"] if recipe not in {"RBCSmartPolicy", "RBCCommunityPolicy"} else 1,
        "deterministic_finish": runtime["deterministic_finish"],
        "reward_function": reward_function,
        "teacher_policy": "RBCSmartPolicy" if recipe.startswith("w6_") else "",
        "deucalion_partition": deucalion.get("partition", ""),
        "deucalion_time": deucalion.get("time", ""),
        "deucalion_cpus_per_task": deucalion.get("cpus_per_task", ""),
        "deucalion_mem_gb": deucalion.get("mem_gb", ""),
        "deucalion_gpus": deucalion.get("gpus", ""),
        "watchdog_enabled": runtime["watchdog"],
        "promotion_note": (
            "Promote only if EV gate and window cost beat RBCSmart."
            if recipe.startswith("w6_")
            else "Same-window baseline."
        ),
    }


def _readme_text(rows: Sequence[Mapping[str, Any]]) -> str:
    by_stage: dict[str, int] = {}
    for row in rows:
        by_stage[str(row["stage"])] = by_stage.get(str(row["stage"]), 0) + 1

    counts = "\n".join(f"- {stage}: {count} configs" for stage, count in sorted(by_stage.items()))
    return f"""# Phase 10 W6 Guided Training Configs

Official target: RBCSmart.

Generated matrix:

{counts}

Promotion gates:

- ev_min_acceptable_feasible_rate >= 0.99
- electrical_violation_kwh == 0
- ev_within_tolerance_rate >= 0.40 initially
- full-year cost <= 17884.3
- battery throughput preferably <= 49000 kWh and low V2G
- community metrics no worse than RBCSmart by more than roughly 3%, unless cost clearly improves

Local run example:

```bash
python3 run_experiment.py --config <config_path> --job_id <job_id> --base-dir runs/phase10_w6
```

Use `run_matrix.csv` as the source of truth for job ids, windows and remote
resource requests. BAU export remains disabled intentionally, so
`cost_ratio_to_bau` is not a W6 gate.
"""


def generate_w6_configs(
    *,
    output_dir: Path,
    stage: str,
    recipes: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    algorithms: Sequence[str] | None = None,
    include_baselines: bool = True,
    validate_only: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_seeds = tuple(int(seed) for seed in (seeds or DEFAULT_SEEDS))

    for stage_name in _stage_names(stage):
        runtime = _stage_runtime(stage_name)
        windows = _stage_windows(stage_name)
        stage_dir = output_dir / stage_name

        if include_baselines and stage_name == "w6a-local":
            for policy_name, template in (
                ("RBCSmartPolicy", RBC_SMART_2022_TEMPLATE),
                ("RBCCommunityPolicy", RBC_COMMUNITY_2022_TEMPLATE),
            ):
                for window_name, start, steps in windows:
                    run_id, config = _build_baseline_config(
                        policy_name=policy_name,
                        template_path=template,
                        stage=stage_name,
                        window_name=window_name,
                        start=start,
                        steps=steps,
                        seed=selected_seeds[0],
                        runtime=runtime,
                    )
                    path = stage_dir / f"{run_id}.yaml"
                    if not validate_only:
                        _write_yaml(path, config)
                    rows.append(
                        _matrix_row(
                            stage=stage_name,
                            run_id=run_id,
                            path=path,
                            algorithm=policy_name,
                            recipe=policy_name,
                            window_name=window_name,
                            start=start,
                            steps=steps,
                            seed=selected_seeds[0],
                            runtime=runtime,
                            reward_function=REWARD_FUNCTION,
                        )
                    )

        for algorithm_name in _stage_algorithms(stage_name, algorithms):
            for recipe_name in _stage_recipes(stage_name, recipes):
                recipe = RECIPES[recipe_name]
                for seed in selected_seeds:
                    for window_name, start, steps in windows:
                        run_id, config = _build_rl_config(
                            stage=stage_name,
                            algorithm_name=algorithm_name,
                            recipe=recipe,
                            window_name=window_name,
                            start=start,
                            steps=steps,
                            seed=int(seed),
                            runtime=runtime,
                        )
                        path = stage_dir / f"{run_id}.yaml"
                        if not validate_only:
                            _write_yaml(path, config)
                        rows.append(
                            _matrix_row(
                                stage=stage_name,
                                run_id=run_id,
                                path=path,
                                algorithm=algorithm_name,
                                recipe=recipe_name,
                                window_name=window_name,
                                start=start,
                                steps=steps,
                                seed=int(seed),
                                runtime=runtime,
                                reward_function=recipe.reward_function,
                            )
                        )

    if not validate_only:
        fields = [
            "stage",
            "job_id",
            "config_path",
            "algorithm",
            "recipe",
            "seed",
            "window",
            "start_step",
            "end_step",
            "episode_steps",
            "episodes",
            "deterministic_finish",
            "reward_function",
            "teacher_policy",
            "deucalion_partition",
            "deucalion_time",
            "deucalion_cpus_per_task",
            "deucalion_mem_gb",
            "deucalion_gpus",
            "watchdog_enabled",
            "promotion_note",
        ]
        _write_csv(output_dir / "run_matrix.csv", rows, fields)
        _write_json(output_dir / "run_matrix.json", rows)
        (output_dir / "README.md").write_text(_readme_text(rows), encoding="utf-8")

    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Phase 10 W6 guided-training configs.")
    parser.add_argument(
        "--stage",
        choices=["w6-smoke-local", "w6a-local", "w6b-remote-smoke", "w6c-full-year", "all"],
        default="w6a-local",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/generated_configs/phase10_w6"),
        help="Directory where configs and matrix files are written.",
    )
    parser.add_argument("--recipe", action="append", dest="recipes", help="Recipe name; can be repeated.")
    parser.add_argument("--seed", action="append", type=int, dest="seeds", help="Seed; can be repeated.")
    parser.add_argument("--algorithm", action="append", dest="algorithms", help="MADDPG or MATD3.")
    parser.add_argument("--no-baselines", action="store_true", help="Skip W6A RBCSmart/RBCCommunity configs.")
    parser.add_argument("--validate-only", action="store_true", help="Validate generated configs without writing files.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = generate_w6_configs(
        output_dir=args.output_dir,
        stage=args.stage,
        recipes=args.recipes,
        seeds=args.seeds,
        algorithms=args.algorithms,
        include_baselines=not args.no_baselines,
        validate_only=args.validate_only,
    )
    print(f"Generated {len(rows)} W6 config entries for stage={args.stage} under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
