"""Strict single-agent TD3 built from the validated MATD3 implementation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from algorithms.agents.matd3_agent import MATD3
from algorithms.utils.citylearn_local_action_safety import (
    CityLearnLocalSafetyAdapter,
    CityLearnSafetyConfig,
    preserve_teacher_service_with_storage_fallback,
    replace_service_actions_with_teacher,
)
from algorithms.utils.price_multiplier_adapter import (
    ForecastMode,
    PriceMultiplierObservationAdapter,
    normalize_price_multiplier_context,
    price_feature_bounds_from_metadata,
    price_observation_names_from_metadata,
)


class TD3(MATD3):
    """Twin Delayed DDPG controller for exactly one local environment slot.

    MATD3 reduces exactly to TD3 when the joint state/action contains one
    agent.  Keeping this thin, explicit class lets the single-agent baseline
    reuse the mature replay, exploration, checkpoint and export paths without
    pretending that one object owns the complete community topology.
    """

    single_agent_only = True

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        if int(self.num_agents) != 1:
            raise ValueError(
                "TD3 controls exactly one environment slot. For a distributed "
                "multi-building run configure one TD3 stage with count equal to "
                "the number of buildings."
            )
        exploration_cfg = ((config["algorithm"].get("exploration") or {}).get("params") or {})
        self.local_action_safety_enabled = bool(
            exploration_cfg.get("local_action_safety_enabled", False)
        )
        self.local_action_safety_fail_on_infeasible = bool(
            exploration_cfg.get("local_action_safety_fail_on_infeasible", False)
        )
        self.local_action_safety_headroom_reserve_kw = max(
            0.0,
            float(exploration_cfg.get("local_action_safety_headroom_reserve_kw", 0.0) or 0.0),
        )
        self.local_action_safety_allow_discretionary_deferrable_start = bool(
            exploration_cfg.get(
                "local_action_safety_allow_discretionary_deferrable_start",
                False,
            )
        )
        self.local_action_safety_runtime_only_export = bool(
            exploration_cfg.get("local_action_safety_runtime_only_export", False)
        )
        self.local_action_safety_protect_ev_minimum = bool(
            exploration_cfg.get("local_action_safety_protect_ev_minimum", True)
        )
        self.local_action_safety_ev_minimum_mode = str(
            exploration_cfg.get("local_action_safety_ev_minimum_mode", "average")
            or "average"
        )
        self.local_action_safety_protect_ev_service_target = bool(
            exploration_cfg.get(
                "local_action_safety_protect_ev_service_target",
                False,
            )
        )
        self.local_action_safety_service_teacher_enabled = bool(
            exploration_cfg.get("local_action_safety_service_teacher_enabled", False)
        )
        self.local_action_safety_service_teacher_eval_enabled = bool(
            exploration_cfg.get(
                "local_action_safety_service_teacher_eval_enabled",
                self.local_action_safety_service_teacher_enabled,
            )
        )
        self.local_price_conditioning_enabled = bool(
            exploration_cfg.get("local_price_conditioning_enabled", False)
        )
        self.local_price_forecast_mode = ForecastMode(
            str(
                exploration_cfg.get(
                    "local_price_forecast_mode",
                    ForecastMode.REAL_UNMODIFIED.value,
                )
            )
        )
        self._local_action_safety_adapter: Optional[CityLearnLocalSafetyAdapter] = None
        self._local_price_adapter: Optional[PriceMultiplierObservationAdapter] = None
        self._last_local_price_diagnostics = None
        self._last_local_price_context_non_neutral = False
        self._last_local_action_projection = None
        self._service_teacher_runtime_call_count = 0
        self._last_service_teacher_applied = False
        self._last_raw_behavior_teacher_actions: Optional[List[List[float]]] = None
        self._last_projected_behavior_teacher_actions: Optional[List[List[float]]] = None
        self._last_raw_next_behavior_teacher_actions: Optional[List[List[float]]] = None
        self._last_projected_next_behavior_teacher_actions: Optional[
            List[List[float]]
        ] = None
        self._behavior_teacher_projection_diagnostics = (
            self._empty_behavior_teacher_projection_diagnostics()
        )
        self._next_behavior_teacher_projection_diagnostics = (
            self._empty_behavior_teacher_projection_diagnostics()
        )
        if self.local_action_safety_enabled:
            self.requires_raw_observation_context = True

    def attach_environment(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().attach_environment(
            observation_names=observation_names,
            action_names=action_names,
            action_space=action_space,
            observation_space=observation_space,
            metadata=metadata,
        )
        self._local_price_adapter = None
        if self.local_price_conditioning_enabled:
            if len(observation_names) != 1:
                raise ValueError("TD3 local price conditioning requires one environment slot.")
            feature_low, feature_high = price_feature_bounds_from_metadata(
                metadata=metadata,
                agent_index=0,
            )
            actor_observation_names = price_observation_names_from_metadata(
                metadata=metadata,
                agent_index=0,
                fallback_observation_names=observation_names[0],
            )
            self._local_price_adapter = PriceMultiplierObservationAdapter(
                observation_names=actor_observation_names,
                feature_low=feature_low,
                feature_high=feature_high,
                forecast_mode=self.local_price_forecast_mode,
            )
        if not self.local_action_safety_enabled:
            return
        if len(observation_names) != 1 or len(action_names) != 1:
            raise ValueError("TD3 local action safety requires exactly one environment slot.")
        self._local_action_safety_adapter = CityLearnLocalSafetyAdapter(
            observation_names=observation_names[0],
            action_names=action_names[0],
            action_low=self._action_low_for_agent(0),
            action_high=self._action_high_for_agent(0),
            metadata=metadata,
            config=CityLearnSafetyConfig(
                fail_on_infeasible=self.local_action_safety_fail_on_infeasible,
                protect_ev_minimum=self.local_action_safety_protect_ev_minimum,
                ev_minimum_mode=self.local_action_safety_ev_minimum_mode,
                protect_ev_service_target=(
                    self.local_action_safety_protect_ev_service_target
                ),
                headroom_reserve_kw=self.local_action_safety_headroom_reserve_kw,
                allow_discretionary_deferrable_start=(
                    self.local_action_safety_allow_discretionary_deferrable_start
                ),
            ),
        )

    def predict(
        self,
        observations,
        deterministic: bool = False,
        *,
        context: Any = None,
    ) -> List[List[float]]:
        self._last_local_price_diagnostics = None
        self._last_local_price_context_non_neutral = False
        if self.local_price_conditioning_enabled:
            if self._local_price_adapter is None:
                raise RuntimeError(
                    "TD3 local price conditioning is enabled but the environment is not attached."
                )
            parsed_context = normalize_price_multiplier_context(context)
            if parsed_context is not None:
                if len(observations) != 1:
                    raise ValueError("TD3 local price conditioning requires one observation.")
                conditioned, diagnostics = self._local_price_adapter.transform(
                    observations[0],
                    parsed_context,
                )
                observations = [conditioned]
                self._last_local_price_diagnostics = diagnostics
                self._last_local_price_context_non_neutral = not diagnostics.neutral_noop
        teacher_schedule_step = self._service_teacher_runtime_call_count
        self._service_teacher_runtime_call_count += 1
        service_teacher_active = bool(
            self.local_action_safety_service_teacher_enabled
            and (
                not deterministic
                or self.local_action_safety_service_teacher_eval_enabled
            )
        )
        self._last_service_teacher_applied = False
        actions = super().predict(observations, deterministic=deterministic, context=context)
        if service_teacher_active:
            if self._warm_start_policy is None:
                raise RuntimeError(
                    "TD3 service-teacher safety requires a configured warm_start_policy."
                )
            predict_at_step = getattr(self._warm_start_policy, "predict_at_step", None)
            if callable(predict_at_step):
                teacher_actions = predict_at_step(
                    self._latest_raw_observations,
                    schedule_step=teacher_schedule_step,
                    deterministic=True,
                )
                self._last_warm_start_policy_actions = teacher_actions
            else:
                teacher_actions = (
                    self._last_warm_start_policy_actions
                    if self._last_warm_start_policy_actions is not None
                    else self._predict_warm_start_policy(
                        apply_noise=False,
                        deterministic=True,
                    )
                )
            actions = replace_service_actions_with_teacher(
                action_names=self.action_names,
                proposed_actions=actions,
                teacher_actions=teacher_actions,
            )
            self._last_service_teacher_applied = True
        if not self.local_action_safety_enabled:
            return actions
        if self._local_action_safety_adapter is None:
            raise RuntimeError("TD3 local action safety is enabled but the environment is not attached.")
        if not self._latest_raw_observations or len(self._latest_raw_observations) != 1:
            raise RuntimeError(
                "TD3 local action safety requires one raw observation context before predict."
            )
        result = self._local_action_safety_adapter.project(
            self._latest_raw_observations[0],
            actions[0],
        )
        self._last_local_action_projection = result
        executed = list(result.executed_actions)
        if service_teacher_active:
            executed = preserve_teacher_service_with_storage_fallback(
                action_names=self.action_names[0],
                teacher_merged_actions=actions[0],
                projected_actions=executed,
            )
        return [executed]

    def update(self, *args, **kwargs) -> None:
        if self._last_local_price_context_non_neutral:
            raise RuntimeError(
                "TD3 received a non-neutral local price context during learning. "
                "Price-conditioned leaves are currently inference-only and must be frozen "
                "under the community coordinator."
            )
        return super().update(*args, **kwargs)

    @staticmethod
    def _empty_behavior_teacher_projection_diagnostics() -> Dict[str, float]:
        return {
            "target_available": 0.0,
            "projection_applied": 0.0,
            "action_count": 0.0,
            "disagreement_count": 0.0,
            "disagreement_ratio": 0.0,
            "mae": 0.0,
            "max_abs": 0.0,
        }

    @staticmethod
    def _copy_action_groups(actions: List[Any]) -> List[List[float]]:
        return [
            np.asarray(action, dtype=np.float64).reshape(-1).tolist()
            for action in actions
        ]

    def _set_behavior_teacher_projection_diagnostics(
        self,
        *,
        raw_actions: Optional[List[List[float]]],
        projected_actions: Optional[List[List[float]]],
        projection_applied: bool,
        next_observation: bool,
    ) -> None:
        diagnostics = self._empty_behavior_teacher_projection_diagnostics()
        if raw_actions is not None and projected_actions is not None:
            raw_parts = [
                np.asarray(action, dtype=np.float64).reshape(-1)
                for action in raw_actions
            ]
            projected_parts = [
                np.asarray(action, dtype=np.float64).reshape(-1)
                for action in projected_actions
            ]
            raw_flat = (
                np.concatenate(raw_parts)
                if raw_parts
                else np.empty(0, dtype=np.float64)
            )
            projected_flat = (
                np.concatenate(projected_parts)
                if projected_parts
                else np.empty(0, dtype=np.float64)
            )
            if raw_flat.shape != projected_flat.shape:
                raise ValueError(
                    "TD3 behavior-teacher safety projection returned an action shape "
                    "different from the raw teacher target."
                )
            absolute_difference = np.abs(projected_flat - raw_flat)
            action_count = int(absolute_difference.size)
            disagreement_count = int(
                np.count_nonzero(absolute_difference > 1.0e-9)
            )
            diagnostics.update(
                {
                    "target_available": 1.0,
                    "projection_applied": float(projection_applied),
                    "action_count": float(action_count),
                    "disagreement_count": float(disagreement_count),
                    "disagreement_ratio": (
                        float(disagreement_count / action_count)
                        if action_count > 0
                        else 0.0
                    ),
                    "mae": (
                        float(np.mean(absolute_difference))
                        if action_count > 0
                        else 0.0
                    ),
                    "max_abs": (
                        float(np.max(absolute_difference))
                        if action_count > 0
                        else 0.0
                    ),
                }
            )
        if next_observation:
            self._next_behavior_teacher_projection_diagnostics = diagnostics
        else:
            self._behavior_teacher_projection_diagnostics = diagnostics

    def _project_behavior_teacher_actions(
        self,
        raw_actions: List[Any],
        *,
        raw_observations: Optional[List[np.ndarray]],
        next_observation: bool,
    ) -> List[List[float]]:
        raw_copy = self._copy_action_groups(raw_actions)
        projected = [list(action) for action in raw_copy]
        projection_applied = False
        if self.local_action_safety_enabled:
            if self._local_action_safety_adapter is None:
                raise RuntimeError(
                    "TD3 behavior-teacher safety projection requires an attached "
                    "local action safety adapter."
                )
            if raw_observations is None or len(raw_observations) != 1:
                raise RuntimeError(
                    "TD3 behavior-teacher safety projection requires one raw "
                    "observation context."
                )
            if len(raw_copy) != 1:
                raise RuntimeError(
                    "TD3 behavior-teacher safety projection requires one action group."
                )
            result = self._local_action_safety_adapter.project(
                raw_observations[0],
                raw_copy[0],
            )
            projected = [list(result.executed_actions)]
            projection_applied = True

        if next_observation:
            self._last_raw_next_behavior_teacher_actions = raw_copy
            self._last_projected_next_behavior_teacher_actions = projected
        else:
            self._last_raw_behavior_teacher_actions = raw_copy
            self._last_projected_behavior_teacher_actions = projected
        self._set_behavior_teacher_projection_diagnostics(
            raw_actions=raw_copy,
            projected_actions=projected,
            projection_applied=projection_applied,
            next_observation=next_observation,
        )
        return projected

    def _transition_behavior_actions(self, actions: List[Any]) -> List[Any]:
        raw_actions = super()._transition_behavior_actions(actions)
        teacher_target_available = bool(
            self.actor_behavior_cloning_source == "warm_start_policy"
            and self._warm_start_policy is not None
            and self._latest_raw_observations is not None
        )
        if not teacher_target_available:
            self._last_raw_behavior_teacher_actions = None
            self._last_projected_behavior_teacher_actions = None
            self._set_behavior_teacher_projection_diagnostics(
                raw_actions=None,
                projected_actions=None,
                projection_applied=False,
                next_observation=False,
            )
            return raw_actions
        return self._project_behavior_teacher_actions(
            raw_actions,
            raw_observations=self._latest_raw_observations,
            next_observation=False,
        )

    def _transition_next_behavior_actions(
        self,
        fallback_actions: List[Any],
    ) -> List[Any]:
        raw_actions = super()._transition_next_behavior_actions(fallback_actions)
        teacher_target_available = bool(
            self._warm_start_policy is not None
            and self._latest_raw_next_observations is not None
            and (
                self.residual_policy_enabled
                or self.actor_behavior_cloning_source == "warm_start_policy"
            )
        )
        if not teacher_target_available:
            self._last_raw_next_behavior_teacher_actions = None
            self._last_projected_next_behavior_teacher_actions = None
            self._set_behavior_teacher_projection_diagnostics(
                raw_actions=None,
                projected_actions=None,
                projection_applied=False,
                next_observation=True,
            )
            return raw_actions
        return self._project_behavior_teacher_actions(
            raw_actions,
            raw_observations=self._latest_raw_next_observations,
            next_observation=True,
        )

    @staticmethod
    def _rename_metric_prefixes(metrics: Dict[str, float]) -> Dict[str, float]:
        renamed: Dict[str, float] = {}
        for key, value in metrics.items():
            if key.startswith("MATD3/"):
                key = f"TD3/{key[len('MATD3/') :]}"
            elif key.startswith("MADDPG/"):
                key = f"TD3/{key[len('MADDPG/') :]}"
            renamed[key] = value
        return renamed

    def _record_training_metrics(self, metrics: Dict[str, float], step: int) -> None:
        super()._record_training_metrics(self._rename_metric_prefixes(metrics), step)

    def export_artifacts(
        self,
        output_dir: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if (
            self.local_action_safety_enabled
            and not self.local_action_safety_runtime_only_export
        ):
            raise RuntimeError(
                "TD3 local action safety is not embedded in the ONNX actor. "
                "Set local_action_safety_runtime_only_export=true only for "
                "non-deployable experiment evidence, or implement a composite bundle."
            )
        metadata = super().export_artifacts(output_dir, context)
        if self.local_action_safety_enabled:
            for artifact in metadata.get("artifacts", []):
                artifact.setdefault("config", {}).update(
                    {
                        "deployable": False,
                        "runtime_only_reason": "external_local_action_safety_projector",
                        "requires_runtime_local_action_safety": True,
                        "requires_runtime_service_teacher": bool(
                            self.local_action_safety_service_teacher_enabled
                            and self.local_action_safety_service_teacher_eval_enabled
                        ),
                    }
                )
        if self.local_price_conditioning_enabled:
            for artifact in metadata.get("artifacts", []):
                artifact.setdefault("config", {}).update(
                    {
                        "requires_runtime_local_price_adapter": True,
                        "local_price_forecast_mode": self.local_price_forecast_mode.value,
                        "local_price_context_scope": "effective_local_price_only",
                        "community_observations_used_by_leaf": False,
                    }
                )
        return metadata

    def get_diagnostic_metrics(self) -> Dict[str, float]:
        metrics = self._rename_metric_prefixes(super().get_diagnostic_metrics())
        metrics["TD3/enabled"] = 1.0
        metrics["TD3/single_agent"] = 1.0
        metrics["TD3/local_action_safety_enabled"] = float(
            self.local_action_safety_enabled
        )
        metrics["TD3/local_action_safety_service_teacher_enabled"] = float(
            self.local_action_safety_service_teacher_enabled
        )
        metrics["TD3/local_action_safety_service_teacher_eval_enabled"] = float(
            self.local_action_safety_service_teacher_eval_enabled
        )
        metrics["TD3/local_action_safety_service_teacher_applied"] = float(
            self._last_service_teacher_applied
        )
        metrics["TD3/local_price_conditioning_enabled"] = float(
            self.local_price_conditioning_enabled
        )
        diagnostics = self._last_local_price_diagnostics
        metrics["TD3/local_price_context_non_neutral"] = float(
            self._last_local_price_context_non_neutral
        )
        metrics["TD3/local_price_clipping_count"] = float(
            diagnostics.clipping_count if diagnostics is not None else 0
        )
        projection = self._last_local_action_projection
        metrics["TD3/local_action_safety_interventions"] = float(
            len(projection.interventions) if projection is not None else 0
        )
        metrics["TD3/local_action_safety_infeasible"] = float(
            len(projection.infeasible_reasons) if projection is not None else 0
        )
        for key, value in self._behavior_teacher_projection_diagnostics.items():
            metrics[f"TD3/behavior_teacher_raw_projected_{key}"] = float(value)
        for key, value in self._next_behavior_teacher_projection_diagnostics.items():
            metrics[f"TD3/next_behavior_teacher_raw_projected_{key}"] = float(value)
        return metrics
