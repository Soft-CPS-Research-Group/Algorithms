"""Typed SMART demonstrations used only to warm-start the TI-MARL actor."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import math
from random import Random
from typing import Any, Mapping, Sequence, Tuple

import torch
from torch import nn

from algorithms.ti_marl.contracts.models import InterfaceSnapshot, LocalActionBundle
from algorithms.ti_marl.policy.networks import TypedActor


@dataclass(frozen=True)
class TypedDemonstration:
    snapshot: InterfaceSnapshot
    bundles: Tuple[LocalActionBundle, ...]


class TypedBehaviorCloningWarmStart:
    """Reservoir-store typed demonstrations and pretrain the shared actor.

    Demonstration transitions are intentionally absent: this component never
    writes to the PPO rollout and never trains the centralized critic.
    """

    def __init__(
        self,
        *,
        demonstration_episodes: int,
        max_samples: int,
        pretraining_epochs: int,
        batch_size: int,
        learning_rate: float,
        balance_action_modes: bool,
        mode_balance_exponent: float,
        max_mode_weight: float,
        seed: int,
        calibration_epochs: int = 0,
        calibration_learning_rate: float | None = None,
    ) -> None:
        self.demonstration_episodes = int(demonstration_episodes)
        self.max_samples = int(max_samples)
        self.pretraining_epochs = int(pretraining_epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.balance_action_modes = bool(balance_action_modes)
        self.mode_balance_exponent = float(mode_balance_exponent)
        self.max_mode_weight = float(max_mode_weight)
        self.calibration_epochs = int(calibration_epochs)
        self.calibration_learning_rate = (
            self.learning_rate
            if calibration_learning_rate is None
            else float(calibration_learning_rate)
        )
        self._rng = Random(int(seed))
        self._demonstrations: list[TypedDemonstration] = []
        self.seen_samples = 0
        self.training_samples = 0
        self.mode_counts: Counter[tuple[str, str]] = Counter()
        self.mode_weights: dict[tuple[str, str], float] = {}
        self.pretraining_complete = False
        self.latest_loss = 0.0
        self.latest_batches = 0
        self.latest_balanced_loss = 0.0
        self.latest_balanced_batches = 0
        self.latest_calibration_loss = 0.0
        self.latest_calibration_batches = 0
        self.mode_diagnostic_metrics: dict[str, float] = {}

    def record(
        self,
        snapshot: InterfaceSnapshot,
        bundles: Sequence[LocalActionBundle],
    ) -> None:
        if self.pretraining_complete:
            raise RuntimeError("Cannot append demonstrations after TI-MARL BC pretraining")
        demonstration = TypedDemonstration(snapshot, tuple(bundles))
        demonstration_counts = self._demonstration_mode_counts(demonstration)
        self.seen_samples += 1
        if len(self._demonstrations) < self.max_samples:
            self._demonstrations.append(demonstration)
            self.mode_counts.update(demonstration_counts)
            return
        replacement = self._rng.randrange(self.seen_samples)
        if replacement < self.max_samples:
            evicted_counts = self._demonstration_mode_counts(
                self._demonstrations[replacement]
            )
            self.mode_counts.subtract(evicted_counts)
            self.mode_counts = Counter(
                {
                    key: count
                    for key, count in self.mode_counts.items()
                    if count > 0
                }
            )
            self._demonstrations[replacement] = demonstration
            self.mode_counts.update(demonstration_counts)

    @staticmethod
    def _demonstration_mode_counts(
        demonstration: TypedDemonstration,
    ) -> Counter[tuple[str, str]]:
        group_types = {
            (group.owner_agent_id, group.group_id): group.group_type
            for group in demonstration.snapshot.action_groups
        }
        counts: Counter[tuple[str, str]] = Counter()
        for bundle in demonstration.bundles:
            for decision in bundle.decisions:
                group_type = group_types.get((bundle.agent_id, decision.group_id))
                if group_type is None:
                    raise ValueError(
                        "Typed demonstration references an unknown action group: "
                        f"{bundle.agent_id}/{decision.group_id}"
                    )
                counts[(group_type, decision.mode)] += 1
        return counts

    def pretrain(self, actor: TypedActor, *, max_grad_norm: float) -> Mapping[str, float]:
        if self.pretraining_complete:
            return self.metrics()
        if not self._demonstrations:
            raise RuntimeError("TI-MARL behavior cloning has zero typed demonstrations")
        optimizer = torch.optim.Adam(actor.parameters(), lr=self.learning_rate)
        self.mode_weights = self._build_mode_weights()
        actor.train()
        balanced_losses, balanced_batches = self._train_epochs(
            actor,
            optimizer,
            epochs=self.pretraining_epochs,
            mode_weights=self.mode_weights,
            max_grad_norm=max_grad_norm,
        )
        calibration_losses: list[float] = []
        calibration_batches = 0
        if self.calibration_epochs > 0:
            for group in optimizer.param_groups:
                group["lr"] = self.calibration_learning_rate
            calibration_losses, calibration_batches = self._train_epochs(
                actor,
                optimizer,
                epochs=self.calibration_epochs,
                mode_weights={key: 1.0 for key in self.mode_counts},
                max_grad_norm=max_grad_norm,
            )
        losses = balanced_losses + calibration_losses
        batches = balanced_batches + calibration_batches
        if batches == 0:
            raise RuntimeError("TI-MARL behavior cloning produced zero trainable batches")
        self.latest_loss = float(sum(losses) / len(losses))
        self.latest_batches = int(batches)
        self.latest_balanced_loss = float(
            sum(balanced_losses) / len(balanced_losses)
        )
        self.latest_balanced_batches = int(balanced_batches)
        self.latest_calibration_loss = (
            0.0
            if not calibration_losses
            else float(sum(calibration_losses) / len(calibration_losses))
        )
        self.latest_calibration_batches = int(calibration_batches)
        self.training_samples = len(self._demonstrations)
        self.mode_diagnostic_metrics = self._evaluate_mode_diagnostics(actor)
        self.pretraining_complete = True
        # The demonstrations have already been distilled into the actor.  PPO
        # never revisits them, so retaining full typed snapshots would inflate
        # every subsequent checkpoint without changing resume semantics.
        self._demonstrations.clear()
        return self.metrics()

    def _train_epochs(
        self,
        actor: TypedActor,
        optimizer: torch.optim.Optimizer,
        *,
        epochs: int,
        mode_weights: Mapping[tuple[str, str], float],
        max_grad_norm: float,
    ) -> tuple[list[float], int]:
        losses: list[float] = []
        batches = 0
        for _epoch in range(int(epochs)):
            indices = list(range(len(self._demonstrations)))
            self._rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                demos = [
                    self._demonstrations[index]
                    for index in indices[start : start + self.batch_size]
                ]
                decisions = []
                for demo in demos:
                    decisions.append(
                        {
                            bundle.agent_id: {
                                decision.group_id: decision
                                for decision in bundle.decisions
                            }
                            for bundle in demo.bundles
                        }
                    )
                evaluation = actor.evaluate_actions_many(
                    tuple(
                        (demo.snapshot, action_map)
                        for demo, action_map in zip(demos, decisions)
                    )
                )
                sample_losses = []
                for demo, group_log_probs in zip(
                    demos, evaluation.log_prob_by_group_step
                ):
                    groups = {
                        (group.owner_agent_id, group.group_id): group
                        for group in demo.snapshot.action_groups
                    }
                    for bundle in demo.bundles:
                        for decision in bundle.decisions:
                            group = groups[(bundle.agent_id, decision.group_id)]
                            log_prob = group_log_probs[bundle.agent_id][
                                decision.group_id
                            ]
                            weight = mode_weights.get(
                                (group.group_type, decision.mode), 1.0
                            )
                            sample_losses.append(-log_prob * float(weight))
                if not sample_losses:
                    continue
                loss = torch.stack(sample_losses).mean()
                if not bool(torch.isfinite(loss).all()):
                    raise FloatingPointError("Non-finite TI-MARL behavior-cloning loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = nn.utils.clip_grad_norm_(actor.parameters(), float(max_grad_norm))
                if not bool(torch.isfinite(torch.as_tensor(gradient))):
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        "Non-finite TI-MARL behavior-cloning gradient"
                    )
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                batches += 1
        return losses, batches

    def in_demonstration_phase(self, *, episode: int, training: bool) -> bool:
        return bool(
            training
            and not self.pretraining_complete
            and int(episode) < self.demonstration_episodes
        )

    def metrics(self) -> Mapping[str, float]:
        metrics = {
            "bc_demonstration_samples": float(
                self.training_samples
                if self.pretraining_complete
                else len(self._demonstrations)
            ),
            "bc_seen_samples": float(self.seen_samples),
            "bc_pretraining_complete": float(self.pretraining_complete),
            "bc_pretraining_loss": float(self.latest_loss),
            "bc_pretraining_batches": float(self.latest_batches),
            "bc_balanced_loss": float(self.latest_balanced_loss),
            "bc_balanced_batches": float(self.latest_balanced_batches),
            "bc_calibration_loss": float(self.latest_calibration_loss),
            "bc_calibration_batches": float(self.latest_calibration_batches),
        }
        for (group_type, mode), count in sorted(self.mode_counts.items()):
            key = f"bc_mode_count_{group_type.lower()}_{mode.lower()}"
            metrics[key] = float(count)
            if (group_type, mode) in self.mode_weights:
                metrics[key.replace("count", "weight")] = float(
                    self.mode_weights[(group_type, mode)]
                )
        metrics.update(self.mode_diagnostic_metrics)
        return metrics

    def _evaluate_mode_diagnostics(self, actor: TypedActor) -> dict[str, float]:
        totals: Counter[tuple[str, str]] = Counter()
        correct: Counter[tuple[str, str]] = Counter()
        predicted: Counter[tuple[str, str]] = Counter()
        target_probability: Counter[tuple[str, str]] = Counter()
        actor.eval()
        with torch.no_grad():
            for start in range(0, len(self._demonstrations), self.batch_size):
                demos = self._demonstrations[start : start + self.batch_size]
                decisions = [
                    {
                        bundle.agent_id: {
                            decision.group_id: decision
                            for decision in bundle.decisions
                        }
                        for bundle in demo.bundles
                    }
                    for demo in demos
                ]
                evaluation = actor.evaluate_actions_many(
                    tuple(
                        (demo.snapshot, action_map)
                        for demo, action_map in zip(demos, decisions)
                    )
                )
                for step_index, demo in enumerate(demos):
                    groups = {
                        (group.owner_agent_id, group.group_id): group
                        for group in demo.snapshot.action_groups
                    }
                    for bundle in demo.bundles:
                        for decision in bundle.decisions:
                            group = groups[(bundle.agent_id, decision.group_id)]
                            key = (group.group_type, decision.mode)
                            predicted_index = int(
                                evaluation.predicted_mode_by_group_step[step_index][
                                    bundle.agent_id
                                ][decision.group_id].item()
                            )
                            predicted_mode = actor.group_modes[group.group_type][
                                predicted_index
                            ]
                            totals[key] += 1
                            correct[key] += int(predicted_mode == decision.mode)
                            predicted[(group.group_type, predicted_mode)] += 1
                            target_probability[key] += float(
                                evaluation.mode_log_prob_by_group_step[step_index][
                                    bundle.agent_id
                                ][decision.group_id].exp().cpu()
                            )
        actor.train()
        metrics: dict[str, float] = {}
        for key, count in sorted(totals.items()):
            group_type, mode = key
            prefix = f"bc_mode_{group_type.lower()}_{mode.lower()}"
            metrics[f"{prefix}_recall"] = float(correct[key] / count)
            metrics[f"{prefix}_target_probability"] = float(
                target_probability[key] / count
            )
            metrics[f"{prefix}_predicted_count"] = float(predicted[key])
        return metrics

    def _build_mode_weights(self) -> dict[tuple[str, str], float]:
        if not self.balance_action_modes:
            return {key: 1.0 for key in self.mode_counts}
        result: dict[tuple[str, str], float] = {}
        group_types = sorted({key[0] for key in self.mode_counts})
        for group_type in group_types:
            counts = {
                mode: count
                for (candidate_type, mode), count in self.mode_counts.items()
                if candidate_type == group_type and count > 0
            }
            if not counts:
                continue
            raw = {
                mode: float(count) ** (-self.mode_balance_exponent)
                for mode, count in counts.items()
            }
            normalizer = sum(counts.values()) / sum(
                counts[mode] * raw[mode] for mode in counts
            )
            for mode, value in raw.items():
                result[(group_type, mode)] = min(
                    value * normalizer,
                    self.max_mode_weight,
                )
        if not all(math.isfinite(value) and value > 0.0 for value in result.values()):
            raise FloatingPointError("Invalid TI-MARL behavior-cloning mode weights")
        return result

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "format": "ti_marl_behavior_cloning_v1",
            "demonstrations": tuple(self._demonstrations),
            "seen_samples": self.seen_samples,
            "training_samples": self.training_samples,
            "mode_counts": dict(self.mode_counts),
            "mode_weights": dict(self.mode_weights),
            "rng_state": self._rng.getstate(),
            "pretraining_complete": self.pretraining_complete,
            "latest_loss": self.latest_loss,
            "latest_batches": self.latest_batches,
            "latest_balanced_loss": self.latest_balanced_loss,
            "latest_balanced_batches": self.latest_balanced_batches,
            "latest_calibration_loss": self.latest_calibration_loss,
            "latest_calibration_batches": self.latest_calibration_batches,
            "mode_diagnostic_metrics": dict(self.mode_diagnostic_metrics),
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if payload.get("format") != "ti_marl_behavior_cloning_v1":
            raise ValueError("Unsupported TI-MARL behavior-cloning checkpoint format")
        demonstrations = list(payload.get("demonstrations", ()))
        if len(demonstrations) > self.max_samples:
            raise ValueError("TI-MARL BC checkpoint exceeds configured sample capacity")
        self._demonstrations = demonstrations
        self.seen_samples = int(payload.get("seen_samples", len(demonstrations)))
        self.training_samples = int(
            payload.get(
                "training_samples",
                self.seen_samples if payload.get("pretraining_complete", False) else 0,
            )
        )
        self.mode_counts = Counter(payload.get("mode_counts", {}))
        self.mode_weights = {
            tuple(key): float(value)
            for key, value in dict(payload.get("mode_weights", {})).items()
        }
        self._rng.setstate(payload["rng_state"])
        self.pretraining_complete = bool(payload.get("pretraining_complete", False))
        self.latest_loss = float(payload.get("latest_loss", 0.0))
        self.latest_batches = int(payload.get("latest_batches", 0))
        self.latest_balanced_loss = float(
            payload.get("latest_balanced_loss", self.latest_loss)
        )
        self.latest_balanced_batches = int(
            payload.get("latest_balanced_batches", self.latest_batches)
        )
        self.latest_calibration_loss = float(
            payload.get("latest_calibration_loss", 0.0)
        )
        self.latest_calibration_batches = int(
            payload.get("latest_calibration_batches", 0)
        )
        self.mode_diagnostic_metrics = {
            str(key): float(value)
            for key, value in dict(
                payload.get("mode_diagnostic_metrics", {})
            ).items()
        }
        if self.pretraining_complete:
            self._demonstrations.clear()
