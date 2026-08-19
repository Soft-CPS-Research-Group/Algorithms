"""Typed SMART demonstrations used only to warm-start the TI-MARL actor."""

from __future__ import annotations

from dataclasses import dataclass
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
        seed: int,
    ) -> None:
        self.demonstration_episodes = int(demonstration_episodes)
        self.max_samples = int(max_samples)
        self.pretraining_epochs = int(pretraining_epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self._rng = Random(int(seed))
        self._demonstrations: list[TypedDemonstration] = []
        self.seen_samples = 0
        self.training_samples = 0
        self.pretraining_complete = False
        self.latest_loss = 0.0
        self.latest_batches = 0

    def record(
        self,
        snapshot: InterfaceSnapshot,
        bundles: Sequence[LocalActionBundle],
    ) -> None:
        if self.pretraining_complete:
            raise RuntimeError("Cannot append demonstrations after TI-MARL BC pretraining")
        demonstration = TypedDemonstration(snapshot, tuple(bundles))
        self.seen_samples += 1
        if len(self._demonstrations) < self.max_samples:
            self._demonstrations.append(demonstration)
            return
        replacement = self._rng.randrange(self.seen_samples)
        if replacement < self.max_samples:
            self._demonstrations[replacement] = demonstration

    def pretrain(self, actor: TypedActor, *, max_grad_norm: float) -> Mapping[str, float]:
        if self.pretraining_complete:
            return self.metrics()
        if not self._demonstrations:
            raise RuntimeError("TI-MARL behavior cloning has zero typed demonstrations")
        optimizer = torch.optim.Adam(actor.parameters(), lr=self.learning_rate)
        actor.train()
        losses: list[float] = []
        batches = 0
        for _epoch in range(self.pretraining_epochs):
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
                for demo, log_probs in zip(demos, evaluation.log_prob_by_step):
                    for agent_id, log_prob in log_probs.items():
                        group_count = max(len(demo.snapshot.groups_for(agent_id)), 1)
                        sample_losses.append(-log_prob / float(group_count))
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
        if batches == 0:
            raise RuntimeError("TI-MARL behavior cloning produced zero trainable batches")
        self.latest_loss = float(sum(losses) / len(losses))
        self.latest_batches = int(batches)
        self.training_samples = len(self._demonstrations)
        self.pretraining_complete = True
        # The demonstrations have already been distilled into the actor.  PPO
        # never revisits them, so retaining full typed snapshots would inflate
        # every subsequent checkpoint without changing resume semantics.
        self._demonstrations.clear()
        return self.metrics()

    def in_demonstration_phase(self, *, episode: int, training: bool) -> bool:
        return bool(
            training
            and not self.pretraining_complete
            and int(episode) < self.demonstration_episodes
        )

    def metrics(self) -> Mapping[str, float]:
        return {
            "bc_demonstration_samples": float(
                self.training_samples
                if self.pretraining_complete
                else len(self._demonstrations)
            ),
            "bc_seen_samples": float(self.seen_samples),
            "bc_pretraining_complete": float(self.pretraining_complete),
            "bc_pretraining_loss": float(self.latest_loss),
            "bc_pretraining_batches": float(self.latest_batches),
        }

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "format": "ti_marl_behavior_cloning_v1",
            "demonstrations": tuple(self._demonstrations),
            "seen_samples": self.seen_samples,
            "training_samples": self.training_samples,
            "rng_state": self._rng.getstate(),
            "pretraining_complete": self.pretraining_complete,
            "latest_loss": self.latest_loss,
            "latest_batches": self.latest_batches,
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
        self._rng.setstate(payload["rng_state"])
        self.pretraining_complete = bool(payload.get("pretraining_complete", False))
        self.latest_loss = float(payload.get("latest_loss", 0.0))
        self.latest_batches = int(payload.get("latest_batches", 0))
        if self.pretraining_complete:
            self._demonstrations.clear()
