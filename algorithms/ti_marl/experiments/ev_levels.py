"""Discrete power ablation of the isolated V2 EV actor, using identical trainable parameters."""

from dataclasses import dataclass, replace

import torch
from torch.distributions import Categorical

from .ev_pretraining import LongEVActor


@dataclass(frozen=True)
class LevelDistribution:
    fractions: tuple[float, ...]
    probabilities: Categorical


class LevelEVActor(LongEVActor):
    """The mode head chooses IDLE/CHARGE; conditionally choose legal 2 or 4 kW."""

    @staticmethod
    def levels(choice):
        fractions = tuple(kw / choice.scale_kw for kw in (2.0, 4.0))
        allowed = tuple(q for q in fractions if choice.lower - 1e-8 <= q <= choice.upper + 1e-8)
        if not allowed:
            raise ValueError(
                f"No 2/4 kW setting in the EV's physical interval {choice.lower, choice.upper}"
            )
        return allowed

    def _distribution(self, context, state, choice):
        dist = super()._distribution(context, state, choice)
        if choice.stage != "power":
            return dist
        band, inside = dist
        levels = self.levels(choice)
        if len(levels) == 1:
            # This choice is forced. Keep a zero-gradient link for replay.
            logits = band.logits[:1] * 0
        else:
            # Reuse all heads. Allocate the interior component to the two
            # discrete levels according to the same learned Beta mean.
            low, middle, high = band.logits.double().unbind()
            a, b = inside.concentration1.log(), inside.concentration0.log()
            total = torch.logaddexp(a, b)
            logits = torch.stack(
                (
                    torch.logaddexp(low, middle + b - total),
                    torch.logaddexp(high, middle + a - total),
                )
            )
        return LevelDistribution(levels, Categorical(logits=logits))

    @staticmethod
    def _score(dist, choice, ref):
        if isinstance(dist, LevelDistribution):
            index = dist.fractions.index(choice.fraction)
            return dist.probabilities.logits[index], dist.probabilities.entropy()
        return LongEVActor._score(dist, choice, ref)

    def _choose(self, choice, group_context, state, deterministic):
        if choice.stage != "power":
            return super()._choose(choice, group_context, state, deterministic)
        if not choice.sampled:
            (level,) = self.levels(choice)
            return replace(choice, fraction=level)
        dist = self._distribution(group_context, state, choice)
        index = int(
            dist.probabilities.logits.argmax() if deterministic else dist.probabilities.sample()
        )
        fraction = dist.fractions[index]
        return replace(
            choice,
            fraction=fraction,
            unit_sample=min(
                1.0, max(0.0, (fraction - choice.lower) / (choice.upper - choice.lower))
            ),
            old_log_prob=float(dist.probabilities.logits[index].detach()),
        )
