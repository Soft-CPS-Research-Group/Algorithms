"""Community operating reward in physical units (chapter 7, section 7.3).

Only settled, post-action facts belong here. Logs of proposed actions can
supply their checked exceedance, but proposals and forecasts cannot establish
delivered energy. Settlement and service facts for the same physical interval
must be passed once; the reference population is fixed for an experiment.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

from algorithms.configuration.agents import TIMARLRewardPenalties

THESIS_PROFILE = "thesis_common_reward_v1"
SMOKE_PROFILE = "settlement_smoke"
TERM_WEIGHTS = {
    "energy_cost_eur": 1.0,
    "dr_cost_eur": 1.0,
    **TIMARLRewardPenalties().model_dump(),
}
SMOKE_TERMS = ("energy_cost_eur", "stationary_throughput_kwh")


@dataclass(frozen=True)
class OperatingReward:
    value: float
    profile: str
    terms: dict[str, float]
    missing_terms: tuple[str, ...]


def _reading(value, name: str, *, signed: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"Expected a numeric {name}")
    value = float(value)
    if not math.isfinite(value) or (not signed and value < 0):
        raise ValueError(f"Invalid {name}: {value}")
    return value


def _settled_terms(info: Mapping) -> dict[str, float]:
    """Read only fields exposed by the real Algorithms / SoftCPS transition."""
    components = info.get("reward_components") or {}
    if not isinstance(components, Mapping):
        raise ValueError("Invalid reward_components")
    community = components.get("community") or {}
    terms = {}
    if "community_settlement_cost_total" in community:
        terms["energy_cost_eur"] = _reading(
            community["community_settlement_cost_total"], "settled cost", signed=True
        )
    rows = components.get("per_agent")
    if (
        isinstance(rows, (tuple, list))
        and rows
        and all(
            isinstance(row, Mapping) and row.get("battery_throughput") is not None for row in rows
        )
    ):
        terms["stationary_throughput_kwh"] = math.fsum(
            _reading(row["battery_throughput"], "stationary battery throughput") for row in rows
        )
    return terms


def proposal_exceedance_kwh(
    site_kw: float, phase_kw: Sequence[float], group_kw: Sequence[float], interval_seconds: float
) -> float:
    """One site's pre-limiter excess; all inputs are nonnegative excess magnitudes."""
    seconds = _reading(interval_seconds, "duration")
    if seconds <= 0:
        raise ValueError("The physical interval must have positive duration")
    excess_kw = max(
        _reading(site_kw, "site excess"),
        math.fsum(_reading(n, "phase excess") for n in phase_kw),
        max((_reading(n, "group excess") for n in group_kw), default=0.0),
    )
    return excess_kw * seconds / 3600.0


def operating_reward(
    info: Mapping,
    *,
    reference_sites: int,
    profile: str = THESIS_PROFILE,
    penalties: TIMARLRewardPenalties | None = None,
) -> OperatingReward:
    """Evaluate the full thesis objective or an explicitly partial, fixed smoke objective.

    An authorised post-action ledger may supply `info["thesis_reward_terms"]`
    with the physical terms named in TERM_WEIGHTS. Settlement supplied twice
    must agree. A missing ledger entry is unknown, including for short episodes
    without any completed service. An explicit ledger zero denotes verified
    absence. The smoke profile always uses the same two terms.
    """
    if profile not in (THESIS_PROFILE, SMOKE_PROFILE):
        raise ValueError(f"Unknown operating reward profile: {profile}")
    if (
        isinstance(reference_sites, bool)
        or not isinstance(reference_sites, Integral)
        or reference_sites < 1
    ):
        raise ValueError("reference_sites must be a positive fixed population")
    terms = _settled_terms(info)
    ledger = info.get("thesis_reward_terms", {})
    if not isinstance(ledger, Mapping):
        raise ValueError("Expected a mapping of post-action thesis reward terms")
    for name, raw in ledger.items():
        if name not in TERM_WEIGHTS:
            raise ValueError(f"Unknown thesis reward term: {name}")
        if raw is None:
            continue
        value = _reading(raw, name, signed=name in ("energy_cost_eur", "dr_cost_eur"))
        if name in terms and not math.isclose(value, terms[name], rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(f"Physical ledger disagrees with the simulator for {name}")
        terms[name] = value
    missing = tuple(name for name in TERM_WEIGHTS if name not in terms)
    required = TERM_WEIGHTS if profile == THESIS_PROFILE else SMOKE_TERMS
    unavailable = [name for name in required if name in missing]
    if unavailable:
        raise ValueError(f"{profile} lacks physical reward terms: {', '.join(unavailable)}")
    weights = TERM_WEIGHTS if penalties is None else {**TERM_WEIGHTS, **penalties.model_dump()}
    reward = -math.fsum(weights[name] * terms[name] for name in required) / reference_sites
    return OperatingReward(value=reward, profile=profile, terms=terms, missing_terms=missing)
