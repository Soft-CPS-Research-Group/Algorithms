"""Thesis variants. Domains and permutations are data, independent of policy weights."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from math import cos, log, log1p, sin, sqrt
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.distributions import Beta, Categorical
from torch.nn import functional as F

from algorithms.ti_marl.contracts.models import ActionDecision, InterfaceSnapshot, LocalActionBundle
from algorithms.ti_marl.policy.forecast_encoder import ForecastEncoder
from algorithms.ti_marl.policy.networks import TypedActor, TypedSnapshotEncoder
from algorithms.ti_marl.runtime.reservation import SiteReservation, group_order_for
from algorithms.ti_marl.runtime.reservation_solver import ReservationUnavailable


def has_power(mode: str) -> bool:
    return mode.startswith(("CHARGE_", "DISCHARGE_"))


@dataclass(frozen=True)
class Choice:
    """One prefix stage. Power bounds are fractions of the available port rating."""

    group_id: str
    group_type: str
    stage: str
    mode: int
    allowed: tuple[int, ...] = ()
    lower: float = 0.0
    upper: float = 0.0
    scale_kw: float = 0.0
    fraction: float = 0.0
    unit_sample: float = 0.0
    old_log_prob: float = 0.0
    margins_kw: tuple[float | None, float | None] = (None, None)
    # Fractions protected and completion times, learned only after the witness is fixed.
    witness: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def domain_features(self, mode_count):
        mask = [0.0] * mode_count
        for index in self.allowed:
            mask[index] = 1.0
        margins = []
        for value in self.margins_kw:
            signed = 0.0 if value is None else math.copysign(log1p(abs(value)), value)
            margins.extend((float(value is not None), signed))
        return [
            float(self.stage == "power"),
            self.lower,
            self.upper,
            log1p(self.scale_kw),
            *margins,
            *mask,
        ]

    @property
    def sampled(self) -> bool:
        if self.stage == "mode":
            return len(self.allowed) > 1
        return self.stage == "power" and self.upper > self.lower


@dataclass(frozen=True)
class SiteTrace:
    order: tuple[str, ...]
    choices: tuple[Choice, ...]
    supervised: bool = False


@dataclass
class JointEvaluation:
    bundles: tuple[LocalActionBundle, ...]
    executed: tuple[LocalActionBundle, ...]
    traces: dict[str, SiteTrace]
    diagnostics: dict[str, dict]


class PrefixMemory(nn.Module):
    """Attention over the ordered choices already made in one physical interval."""

    def __init__(self, width: int, mode_count: int):
        super().__init__()
        self.mode_count = mode_count
        self.token = nn.Linear(2 * mode_count + 15, width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.order_encoder = nn.GRU(width, width, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def start(self, ordered_groups: list[Tensor], context: Tensor) -> tuple[Tensor, ...]:
        if not ordered_groups:
            return (context,)
        _, hidden = self.order_encoder(
            torch.stack(ordered_groups).unsqueeze(0), context[None, None]
        )
        return (hidden[0, 0],)

    def read(self, prefix: tuple[Tensor, ...], query: Tensor) -> Tensor:
        tokens = torch.stack(prefix)
        scores = self.key(tokens) @ self.query(query) / sqrt(query.numel())
        return self.norm(query + scores.softmax(0) @ self.value(tokens))

    def extend(
        self, prefix: tuple[Tensor, ...], group: Tensor, choice: Choice
    ) -> tuple[Tensor, ...]:
        if choice.stage == "inapplicable":
            return prefix
        selected = [0.0] * self.mode_count
        selected[choice.mode] = 1.0
        position = len(prefix)
        features = [
            *selected,
            *choice.domain_features(self.mode_count),
            choice.fraction,
            log1p(position),
            sin(position),
            cos(position),
            *choice.witness,
        ]
        token = group + self.token(group.new_tensor(features))
        return (*prefix, token)


class ConstraintMessages(nn.Module):
    """Pass equipment messages along the declared shared-limit memberships."""

    def __init__(self, width: int, layers: int):
        super().__init__()
        self.limit_encoder = nn.Linear(8, width)
        self.updates = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.LayerNorm(width))
            for _ in range(layers)
        )

    def forward(self, contexts: Mapping[str, Tensor], limits):
        if not contexts or not limits:
            return contexts
        keys = tuple(contexts)
        by_key = {key: i for i, key in enumerate(keys)}
        incidence, fields = [], []
        for limit in limits:
            if not limit.active:
                continue
            members = limit.member_group_ids or keys
            weights = dict(limit.member_group_coefficients) or dict.fromkeys(members, 1.0)
            edges = [0.0] * len(keys)
            for key in members:
                if key in by_key:
                    edges[by_key[key]] = weights.get(key, 0.0)
            if not any(edges):
                continue
            incidence.append(edges)
            kind = limit.constraint_type
            bounds = []
            for v in (limit.lower_bound, limit.upper_bound):
                valid = v is not None and math.isfinite(v)
                bounds.extend((float(valid), math.copysign(log1p(abs(v)), v) if valid else 0.0))
            fields.append(
                (
                    float("export" in kind),
                    float("phase" in kind),
                    float("group" in kind),
                    log1p(sum(x != 0 for x in edges)),
                    *bounds,
                )
            )
        if not incidence:
            return contexts
        tokens = torch.stack([contexts[key] for key in keys])
        adjacency = tokens.new_tensor(incidence)
        limits = self.limit_encoder(tokens.new_tensor(fields))
        to_limits = adjacency / adjacency.abs().sum(1, keepdim=True).clamp_min(1e-9)
        to_equipment = adjacency.T / adjacency.abs().sum(0).clamp_min(1e-9)[:, None]
        linked = adjacency.abs().sum(0)[:, None] > 0
        for update in self.updates:
            messages = to_equipment @ (limits + to_limits @ tokens)
            tokens = tokens + update(torch.cat((tokens, messages), dim=-1)) * linked
        return dict(zip(keys, tokens.unbind()))


class JointActor(TypedActor):
    """Sample all modes first, fix a witness, then sample each required power."""

    def __init__(self, *args, variant: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.variant = variant
        self.forecasts = ForecastEncoder(self.d_model)
        self.connections = ConstraintMessages(self.d_model, kwargs["relation_layers"])
        self.independent = variant == "v0"
        self.step_seconds = 900.0
        self.tolerance_soc = 0.05
        self.departure_margin_seconds = 0.0
        self.reserve_programmes_in_certified_window = False
        self.order_seed = 0
        self.memory = PrefixMemory(self.d_model, max(map(len, self.group_modes.values())))
        self.domain_context = nn.Linear(8 + self.memory.mode_count, self.d_model)
        self.selected_mode = nn.Embedding(self.memory.mode_count, self.d_model)
        self.condition = nn.LayerNorm(self.d_model)

    def _contexts(self, snapshot):
        requests = [(snapshot, site) for site in snapshot.agent_ids]
        if not requests:
            return {}
        device = next(self.parameters()).device
        local, parts = self._encode_actor_requests(requests, device)
        local = local + self.forecasts(snapshot, snapshot.agent_ids)
        groups = [snapshot.groups_for(site) for site in snapshot.agent_ids]
        flat = self._group_contexts_many(requests, groups, local, device, preencoded_parts=parts)
        offset, result = 0, {}
        for i, (site, site_groups) in enumerate(zip(snapshot.agent_ids, groups)):
            group_context = {g.group_id: flat[offset + j] for j, g in enumerate(site_groups)}
            limits = [limit for limit in snapshot.constraints if limit.owner_agent_id == site]
            result[site] = local[i], self.connections(group_context, limits)
            offset += len(site_groups)
        return result

    def _distribution(self, group_context: Tensor, state: tuple[Tensor, ...], c: Choice):
        context = group_context if self.independent else self.memory.read(state, group_context)
        context = self.condition(
            context
            + self.domain_context(context.new_tensor(c.domain_features(self.memory.mode_count)))
        )
        if c.stage == "mode":
            logits = self.mode_heads[c.group_type](context)
            return Categorical(logits=logits[list(c.allowed)])
        context = self.condition(context + self.selected_mode.weight[c.mode])
        a, b = (F.softplus(self.beta_heads[c.group_type](context)) + 1.0).unbind()
        return Beta(a, b)

    @staticmethod
    def _score(dist, c: Choice, ref: Tensor):
        if c.stage == "mode":
            idx = torch.tensor(c.allowed.index(c.mode), device=ref.device)
            return dist.log_prob(idx), dist.entropy()
        log_width_kw = log((c.upper - c.lower) * c.scale_kw)
        return (
            dist.log_prob(ref.new_tensor(c.unit_sample)) - log_width_kw,
            dist.entropy() + log_width_kw,
        )

    def _choose(self, c, group_context, state, deterministic):
        if not c.sampled:
            return c
        dist = self._distribution(group_context, state, c)
        if c.stage == "mode":
            idx = dist.probs.argmax() if deterministic else dist.sample()
            c = replace(c, mode=c.allowed[int(idx)])
        else:
            sample = float(dist.mean if deterministic else dist.sample())
            c = replace(c, unit_sample=sample, fraction=c.lower + (c.upper - c.lower) * sample)
        score, _ = self._score(dist, c, group_context)
        return replace(c, old_log_prob=float(score.detach()))

    def _decode_site(self, snapshot, site, domain, site_latent, contexts, deterministic):
        order = tuple(domain.group_order)
        groups = {g.group_id: g for g in snapshot.groups_for(site)}
        state = self.memory.start([contexts[key] for key in order], site_latent)
        modes, fractions, log_probs, choices = {}, {}, {}, []
        for key in order:
            group = groups[key]
            vocabulary = self.group_modes[group.group_type]
            allowed = (
                domain.equipment_modes(key) if self.independent else domain.admissible_modes(key)
            )
            indices = tuple(i for i, name in enumerate(vocabulary) if name in allowed)
            if not indices:
                raise ValueError(f"No registered admissible mode for {key}: {allowed}")
            margins = (None, None) if self.independent else domain.remaining_margin_kw(key)
            c = Choice(
                key, group.group_type, "mode", indices[0], allowed=indices, margins_kw=margins
            )
            c = self._choose(c, contexts[key], state, deterministic)
            modes[key] = c.mode
            log_probs[key] = c.old_log_prob
            if not self.independent:
                domain.select_mode(key, vocabulary[c.mode])
                if key == order[-1]:
                    verified = domain.diagnostics
                    levels = verified["protected_fractions"].values()
                    completion = verified["earliest_completion_seconds"].values()
                    c = replace(
                        c,
                        witness=(
                            1.0,
                            min(levels, default=1.0),
                            max(completion, default=0.0) / 86400.0,
                        ),
                    )
                state = self.memory.extend(state, contexts[key], c)
            choices.append(c)
        for key in order:
            group = groups[key]
            vocabulary = self.group_modes[group.group_type]
            idx = modes[key]
            mode = vocabulary[idx]
            value = 0.0 if mode == "IDLE" else 1.0
            if has_power(mode):
                lo, hi = (
                    domain.equipment_fraction_bounds(key, mode)
                    if self.independent
                    else domain.fraction_bounds(key)
                )
                scale = domain.power_scale_kw(key, mode)
                if scale <= 0 and hi > lo:
                    raise ValueError(f"Non-positive physical power range for {key}/{mode}")
                c = Choice(
                    key,
                    group.group_type,
                    "power",
                    idx,
                    lower=lo,
                    upper=hi,
                    scale_kw=scale,
                    fraction=lo,
                    margins_kw=(None, None)
                    if self.independent
                    else domain.remaining_margin_kw(key),
                )
                c = self._choose(c, contexts[key], state, deterministic)
                value = c.fraction
                log_probs[key] += c.old_log_prob
                choices.append(c)
                if not self.independent:
                    state = self.memory.extend(state, contexts[key], c)
            elif any(has_power(name) for name in vocabulary):
                choices.append(Choice(key, group.group_type, "inapplicable", idx))
            if not self.independent:
                domain.select_fraction(key, value)
            fractions[key] = value
        bundle = LocalActionBundle(
            site,
            tuple(
                ActionDecision(
                    key,
                    self.group_modes[groups[key].group_type][modes[key]],
                    fractions[key],
                    modes[key],
                    log_probs[key],
                )
                for key in order
            ),
        )
        final = domain.reconcile(bundle) if self.independent else bundle
        return bundle, final, SiteTrace(order, tuple(choices)), domain.diagnostics

    def forward(self, snapshot: InterfaceSnapshot, *, deterministic: bool = False):
        raw, final, traces, diagnostics = [], [], {}, {}
        for site, (latent, contexts) in self._contexts(snapshot).items():
            # If even the initial reservation is unavailable, there is no verified
            # command to hand to the supervisor. Propagate the failure before dispatch.
            domain = SiteReservation.from_snapshot(
                snapshot,
                site,
                step_seconds=self.step_seconds,
                order_seed=self.order_seed,
                tolerance_soc=self.tolerance_soc,
                departure_margin_seconds=self.departure_margin_seconds,
                reserve_programmes_in_certified_window=self.reserve_programmes_in_certified_window,
            )
            try:
                proposal, verified, traces[site], diagnostics[site] = self._decode_site(
                    snapshot, site, domain, latent, contexts, deterministic
                )
            except ReservationUnavailable as error:
                verified = domain.verified_fallback(site)
                proposal = LocalActionBundle(site, ())
                traces[site] = SiteTrace(tuple(domain.group_order), (), supervised=True)
                diagnostics[site] = {"supervised": True, "verified": True, "reason": str(error)}
            raw.append(proposal)
            final.append(verified)
        return JointEvaluation(tuple(raw), tuple(final), traces, diagnostics)

    def replay(self, snapshot: InterfaceSnapshot, traces: Mapping[str, SiteTrace]):
        """Evaluate only sampled choices using the recorded order, mask and interval."""
        result = {}
        for site, (latent, contexts) in self._contexts(snapshot).items():
            trace = traces[site]
            state = self.memory.start([contexts[g] for g in trace.order], latent)
            scores = []
            for k, choice in enumerate(trace.choices):
                group = contexts[choice.group_id]
                if choice.sampled:
                    dist = self._distribution(group, state, choice)
                    score, entropy = self._score(dist, choice, group)
                    scores.append((k, score, entropy))
                if not self.independent:
                    state = self.memory.extend(state, group, choice)
            result[site] = scores
        return result


class PrefixCritic(nn.Module):
    """One value head for every prefix, with an independent central encoder."""

    def __init__(
        self, type_registry, *, d_model: int, relation_layers: int, centralised: bool = True
    ):
        super().__init__()
        self.encoder = TypedSnapshotEncoder(type_registry, d_model, relation_layers)
        self.forecasts = ForecastEncoder(d_model)
        self.connections = ConstraintMessages(d_model, relation_layers)
        types = type_registry["action_group_types"]
        self.group_index = {name: i for i, name in enumerate(sorted(types))}
        self.group_type = nn.Embedding(len(types), d_model)
        self.memory = PrefixMemory(d_model, max(len(data["modes"]) for data in types.values()))
        self.centralised = centralised
        self.site_score = nn.Linear(d_model, 1) if centralised else None
        self.site_query = nn.Linear(d_model * 2 + 3 if centralised else d_model, d_model)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )
        self.order_seed = 0

    def prefix_values(self, snapshot, traces=None):
        sites = snapshot.agent_ids
        if not sites:
            return {}
        device = next(self.parameters()).device
        requests = [(snapshot, site) for site in sites]
        local, parts, encoded = self.encoder.forward_many_with_parts(requests, device)
        local = local + self.forecasts(snapshot, sites)
        if self.centralised:
            community = torch.sum(self.site_score(local).softmax(dim=0) * local, dim=0)
            counts = local.new_tensor(
                [
                    log1p(len(sites)),
                    log1p(len(snapshot.action_groups)),
                    log1p(
                        sum(
                            max(
                                g.max_charge_power_kw,
                                g.max_discharge_power_kw,
                                g.activation_power_kw,
                            )
                            for g in snapshot.action_groups
                        )
                    ),
                ]
            ).expand(len(sites), -1)
            queries = self.site_query(
                torch.cat((local, community.expand_as(local), counts), dim=-1)
            )
        else:
            queries = self.site_query(local)
        values, offset = {}, 0
        for i, site in enumerate(sites):
            groups = snapshot.groups_for(site)
            context = {}
            for g in groups:
                indices = [
                    offset + j
                    for j, p in enumerate(parts[i])
                    if TypedActor._group_part_relation(g, p) == 0
                ]
                token = self.group_type.weight[self.group_index[g.group_type]]
                if indices:
                    token = token + encoded[indices].mean(0)
                context[g.group_id] = token
            offset += len(parts[i])
            limits = [limit for limit in snapshot.constraints if limit.owner_agent_id == site]
            context = self.connections(context, limits)
            trace = (
                traces[site]
                if traces is not None
                else SiteTrace(group_order_for(snapshot, site, self.order_seed), ())
            )
            state = self.memory.start([context[key] for key in trace.order], queries[i])
            states = [self.memory.read(state, queries[i])]
            for choice in trace.choices:
                state = self.memory.extend(state, context[choice.group_id], choice)
                states.append(
                    states[-1]
                    if choice.stage == "inapplicable"
                    else self.memory.read(state, queries[i])
                )
            values[site] = self.value_head(torch.stack(states)).squeeze(-1)
        return values

    def forward(self, snapshot):
        return {key: value[0] for key, value in self.prefix_values(snapshot).items()}
