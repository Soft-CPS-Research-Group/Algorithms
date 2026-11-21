"""Run the thesis policy and its optional two-timescale community coordinator."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

from algorithms.configuration.agents import TIMARLHierarchyConfig, TIMARLRewardPenalties
from algorithms.ti_marl.learning.community_signal import CommunitySignal, community_input
from algorithms.ti_marl.runtime.forecasts import (
    BLOCK_SECONDS,
    ForecastInputs,
    IssuedForecastArchive,
    OperatingSnapshot,
    reservation_envelopes,
)


class OperatingControl:
    """Add forecasts and the optional coordinator around the common typed lifecycle."""

    def __init__(self, config):
        super().__init__(config)
        hyper = config["algorithm"]["hyperparameters"]
        self.manager = None
        self.forecasts = None
        self.reservation_error = hyper.get("reservation_error_kw")
        archive_path = hyper.get("issued_forecasts_path")
        self.archive = IssuedForecastArchive(archive_path) if archive_path else None
        default_reward = "thesis_common_reward_v1" if self.thesis_method else "environment"
        self.reward_profile = hyper.get("operating_reward", default_reward)
        self.reference_sites = int(hyper.get("reference_sites", 30))
        self.reward_penalties = TIMARLRewardPenalties.model_validate(
            hyper.get("reward_penalties", {})
        )
        self.realised_reward = None
        if hyper.get("hierarchy") is not None:
            if self.method_variant != "v2":
                raise ValueError("Community coordination requires a V2 local policy")
            settings = TIMARLHierarchyConfig.model_validate(hyper["hierarchy"]).model_dump()
            width = hyper.get("actor", {}).get("d_model", 128)
            self.manager = CommunitySignal(
                settings,
                width,
                self.device,
                self.seed,
                hyper.get("gamma", 0.99),
                hyper.get("discount_timebase_seconds") or 3600.0,
            )
            self.requires_deterministic_transition_observation = True
        self.bootstrap_manager = 0.0
        self.calendar = None
        path = config.get("simulator", {}).get("dataset_path")
        if self.manager and path and Path(path).is_file() and Path(path).suffix == ".json":
            schema = json.loads(Path(path).read_text())
            if schema.get("start_date") and schema.get("timezone"):
                zone = ZoneInfo(schema["timezone"])
                first = datetime.fromisoformat(schema["start_date"])
                if first.tzinfo is None:
                    first = first.replace(tzinfo=zone)
                self.calendar = (first.timestamp(), zone)
        if self.reward_profile != "environment":
            from algorithms.ti_marl.runtime.reward_feedback import SoftCPSRewardFeedback

            self.interval_reward_feedback = SoftCPSRewardFeedback().transition_info
            self.requires_deterministic_transition_observation = True

    def _market_day(self, seconds):
        if self.calendar is None:
            return math.floor(seconds / 86400)
        origin, zone = self.calendar
        return datetime.fromtimestamp(origin + seconds, zone).date()

    def attach_environment(self, **kwargs):
        super().attach_environment(**kwargs)
        if not self.thesis_method:
            return
        seconds = float(kwargs["metadata"]["seconds_per_time_step"])
        if self.manager and (
            seconds > BLOCK_SECONDS or not math.isclose(BLOCK_SECONDS % seconds, 0)
        ):
            raise ValueError("Local decision steps must divide the 15-minute coordinator period")
        if self.forecasts is None or seconds != self.forecasts.step_seconds:
            self.forecasts = ForecastInputs(seconds)

    def on_episode_start(self, *, episode, training):
        super().on_episode_start(episode=episode, training=training)
        if self.forecasts:
            self.forecasts = ForecastInputs(self.forecasts.step_seconds)
        if self.manager:
            if self.manager.phase == "coordinator":
                self.actor.order_seed = self.critic.order_seed = 0
            self.manager.reset()
            self.bootstrap_manager = 0.0

    def _prepare_outlook(self, snapshot, payload, info):
        if isinstance(snapshot, OperatingSnapshot):
            return snapshot
        start = (info or {}).get("episode_start_time_step", 0)
        now = (start + snapshot.time_step) * self.forecasts.step_seconds
        records = self.archive.at(now) if self.archive else {}
        for site, record in payload.get("meta", {}).get("ti_marl_forecasts", {}).items():
            if record["received_at_seconds"] <= now and (
                site not in records
                or record["issued_at_seconds"] >= records[site]["issued_at_seconds"]
            ):
                records[site] = record
        # The exogenous demand is for t. SoftCPS tags the PV meter with the
        # endogenous time step, usually t-1. Exclude reset-time or untagged PV
        # from residuals: no completed physical interval is certified there.
        meta = payload.get("meta", {})
        decision_step, metered_step = meta.get("time_step"), meta.get("endogenous_time_step")
        pv_lag_steps = None
        if decision_step is not None and metered_step is not None and decision_step > 0:
            pv_lag_steps = decision_step - metered_step
        enriched = self.forecasts.snapshot(
            snapshot, records, now, error_kw=self.reservation_error, pv_lag_steps=pv_lag_steps
        )
        bounds = reservation_envelopes(
            enriched, self.compiler.interface_registry, self.reservation_error
        )
        return replace(enriched, reservation_envelopes=bounds)

    def _signal_snapshot(self, snapshot, training):
        if self.manager is None:
            return snapshot
        peak = self.manager.peak
        day = self._market_day(snapshot.clock_seconds)
        snapshot = replace(snapshot, measured_peak_kw=peak.peak if day == peak.day else 0.0)
        curve = self.manager.curve_for(snapshot, training)
        return replace(snapshot, incentive=curve)

    def set_entity_observation_context(self, *, observation_payload, info=None):
        super().set_entity_observation_context(observation_payload=observation_payload, info=info)
        if self.thesis_method:
            current = self._prepare_outlook(self._current_snapshot, observation_payload, info)
            self._current_snapshot = self._signal_snapshot(
                current, self._current_episode_is_training
            )

    def set_entity_transition_context(
        self, *, observation_payload, next_observation_payload, info=None
    ):
        super().set_entity_transition_context(
            observation_payload=observation_payload,
            next_observation_payload=next_observation_payload,
            info=info,
        )
        self.realised_reward = None
        if self.thesis_method:
            self._next_snapshot = self._prepare_outlook(
                self._next_snapshot, next_observation_payload, info
            )

    def predict(self, observations, deterministic=None, *, context=None):
        if self.manager and self.manager.phase == "coordinator":
            deterministic = True
        return super().predict(observations, deterministic, context=context)

    def _real_reward(self, current, rewards):
        if self.reward_profile == "environment":
            return super()._thesis_rewards(current, rewards)[0]
        from algorithms.ti_marl.learning.operating_reward import operating_reward

        if self.realised_reward is not None:
            return self.realised_reward
        info = dict(self._transition_info)
        if self._pending is not None:
            from algorithms.ti_marl.runtime.reservation_metrics import prelimit_exceedance_kwh
            from algorithms.ti_marl.runtime.reward_feedback import merge_reward_evidence

            # A supervised step has no policy proposal. Its independently
            # verified fallback is the command proposed to the hardware.
            # For a regular step (including V0), measure before projection.
            traces = self._pending["prefix_traces"]
            verified = {bundle.agent_id: bundle for bundle in self._pending["final_bundles"]}
            proposals = []
            for proposal in self._pending["raw_bundles"]:
                site = proposal.agent_id
                if traces[site].supervised:
                    if not self._pending["reservations"][site].get("verified"):
                        raise ValueError(f"Unverified supervised proposal at {site}")
                    proposal = verified[site]
                proposals.append(proposal)
            violations = [
                prelimit_exceedance_kwh(
                    current,
                    bundle,
                    step_seconds=self.forecasts.step_seconds,
                    declared=self.compiler.interface_registry.for_agent(
                        bundle.agent_id
                    ).constraints,
                )
                for bundle in proposals
            ]
            if all(value is not None for value in violations):
                info["thesis_reward_terms"] = merge_reward_evidence(
                    info.get("thesis_reward_terms", {}),
                    {"proposal_exceedance_kwh": math.fsum(violations)},
                )
        result = operating_reward(
            info,
            reference_sites=self.reference_sites,
            profile=self.reward_profile,
            penalties=self.reward_penalties,
        )
        self._latest_diagnostics["TI_MARL/reward_terms_unavailable"] = len(result.missing_terms)
        components = dict(info.get("reward_components", {}))
        components["thesis"] = {
            "profile": result.profile,
            "value": result.value,
            "terms": result.terms,
            "missing_terms": result.missing_terms,
        }
        self._transition_info["reward_components"] = components
        self.realised_reward = result.value
        return self.realised_reward

    def _thesis_rewards(self, current, rewards):
        real = self._real_reward(current, rewards)
        if self.manager is None or self.manager.phase != "local":
            return real, {}
        rows = self._transition_info["reward_components"]["per_agent"]
        if len(rows) != len(current.agent_ids):
            raise ValueError("Missing member settlement for the local training incentive")
        delta = current.incentive[0]
        adjustments = {}
        for site, row in zip(current.agent_ids, rows):
            measured = float(row["net_electricity_consumption"])
            if not math.isfinite(measured):
                raise ValueError(f"Non-finite member import for {site}")
            adjustments[site] = -delta * max(0.0, measured) / self.reference_sites
        return real, adjustments

    def _advance_manager(self, rewards, *, training, terminated, truncated):
        # Settlement belongs to the action just executed. Assign it before
        # selecting the next block's signal, including at midnight.
        current = self._pending["snapshot"]
        community = self._transition_info["reward_components"]["community"]
        kwh = float(community["community_grid_import_after_settlement"])
        seconds = self.forecasts.step_seconds
        rise = self.manager.peak.observe(
            current.clock_seconds,
            seconds,
            kwh,
            float(community["community_grid_export_after_settlement"]),
            day=self._market_day(current.clock_seconds),
        )
        if training and self.manager.phase == "coordinator":
            realised = (
                self._real_reward(current, rewards)
                - self.manager.peak_weight * rise / self.reference_sites
            )
            self.manager.observe_reward(realised, seconds)
        future = self._next_snapshot
        sample_next = (
            training and not terminated and (not truncated or self.manager.phase == "local")
        )
        self._next_snapshot = self._signal_snapshot(future, sample_next)
        if truncated and not terminated and self.manager.phase == "coordinator":
            with torch.no_grad():
                features = torch.as_tensor(community_input(self._next_snapshot), device=self.device)
                self.bootstrap_manager = float(self.manager.network(features)[1])
        self._latest_diagnostics.update(
            {
                "TI_MARL/community_peak_kw": self.manager.peak.peak,
                "TI_MARL/active_adjustment_eur_kwh": current.incentive[0],
                "TI_MARL/external_import_kwh": kwh,
            }
        )

    def update(
        self, observations, actions, rewards, next_observations, terminated, truncated, **kwargs
    ):
        if self.manager is not None:
            self._advance_manager(
                rewards, training=True, terminated=terminated, truncated=truncated
            )
            if self.manager.phase == "coordinator":
                # In stage two the local actors and critics stay frozen.
                return super().observe_transition(
                    observations, actions, rewards, next_observations, terminated, truncated
                )
        return super().update(
            observations, actions, rewards, next_observations, terminated, truncated, **kwargs
        )

    def observe_transition(
        self, observations, actions, rewards, next_observations, terminated, truncated
    ):
        if self.thesis_method and self.reward_profile != "environment":
            self._real_reward(self._pending["snapshot"], rewards)
        if self.manager is not None:
            self._advance_manager(
                rewards, training=False, terminated=terminated, truncated=truncated
            )
        return super().observe_transition(
            observations, actions, rewards, next_observations, terminated, truncated
        )

    def on_episode_end(self, *, episode, training):
        super().on_episode_end(
            episode=episode,
            training=training and not (self.manager and self.manager.phase == "coordinator"),
        )
        if self.manager and training and self.manager.phase == "coordinator":
            self._latest_training_metrics.update(self.manager.finish(self.bootstrap_manager))

    def load_checkpoint(self, checkpoint_path):
        frozen = self.manager is not None and self.manager.phase == "coordinator"
        restore = self.restore_rollout
        if frozen:
            self.restore_rollout = False
        try:
            super().load_checkpoint(checkpoint_path)
        finally:
            self.restore_rollout = restore

    def _extra_policy_state(self):
        if self.manager is None:
            return {}
        return {
            "manager": self.manager.network.state_dict(),
            "manager_optimizer": self.manager.optimizer.state_dict(),
            "manager_rng": self.manager.rng.bit_generator.state,
            "curve_knots": self.manager.knots,
            "max_adjustment_eur_kwh": self.manager.bound,
            "signal": self.manager.kind,
            "training_phase": self.manager.phase,
        }

    def _extra_model_state(self):
        if self.manager is None:
            return {}
        return {**self._extra_deployment_state(), "manager": self.manager.network.state_dict()}

    def _extra_deployment_state(self):
        if self.manager is None:
            return {}
        return {
            "signal": self.manager.kind,
            "curve_knots": self.manager.knots,
            "max_adjustment_eur_kwh": self.manager.bound,
            "manager_actor": {
                key: value
                for key, value in self.manager.network.state_dict().items()
                if not key.startswith("value.")
            },
        }

    def _restore_extra_policy_state(self, state):
        if self.manager is None:
            return super()._restore_extra_policy_state(state)
        if self.manager.phase == "coordinator":
            self.learner.rollout.clear()
        if not state:
            # A V2 checkpoint can initialise local agents for a new hierarchy.
            return
        if (
            state["curve_knots"] != self.manager.knots
            or state["max_adjustment_eur_kwh"] != self.manager.bound
        ):
            raise ValueError("Incompatible community incentive curve in checkpoint")
        self.manager.network.load_state_dict(state["manager"])
        if self.restore_optimizers and self.manager.phase == state["training_phase"]:
            self.manager.optimizer.load_state_dict(state["manager_optimizer"])
        self.manager.rng.bit_generator.state = state["manager_rng"]
