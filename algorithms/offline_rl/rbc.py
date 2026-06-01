"""RBC — bug-fix wrapper around ``RuleBasedPolicy``.

The v1 ``RuleBasedPolicy`` (algorithms/agents/rbc_agent.py) looks up EV-related
observations using bare names like ``electric_vehicle_charger_state`` and
``electric_vehicle_soc``. The CityLearn env, however, exposes those fields with
**charger-namespaced** names:

    electric_vehicle_charger_state       → electric_vehicle_charger_<id>_connected_state
    electric_vehicle_soc                 → connected_electric_vehicle_at_charger_<id>_soc
    electric_vehicle_required_soc_departure
                                         → connected_electric_vehicle_at_charger_<id>_required_soc_departure
    electric_vehicle_departure_time      → connected_electric_vehicle_at_charger_<id>_departure_time
    electric_vehicle_is_flexible         → (not exposed; remains at its default = 1.0)

Result in v1: the bare names never resolve, every value falls back to its
default (charger_state→0), and ``_compute_ev_action`` short-circuits to 0 at
the very first guard. The RBC has been emitting **constant zeros**.

This wrapper keeps the v1 algorithm intact and only overrides the
observation-name resolution: when asked for one of the v1 bare names, it tries
the v1 name first, then a per-charger namespaced fallback derived from the
agent's known chargers.

No v1 file is modified.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from algorithms.agents.rbc_agent import ChargerInfo, RuleBasedPolicy


# Mapping from v1 bare names to a list of "namespaced" patterns. Each pattern
# uses ``{charger_id}`` as a placeholder.
_NAMESPACED_FALLBACKS: Dict[str, List[str]] = {
    "electric_vehicle_charger_state": [
        "electric_vehicle_charger_{charger_id}_connected_state",
    ],
    "electric_vehicle_soc": [
        "connected_electric_vehicle_at_charger_{charger_id}_soc",
    ],
    "electric_vehicle_required_soc_departure": [
        "connected_electric_vehicle_at_charger_{charger_id}_required_soc_departure",
    ],
    "electric_vehicle_departure_time": [
        "connected_electric_vehicle_at_charger_{charger_id}_departure_time",
    ],
    "electric_vehicle_is_flexible": [
        # Env doesn't expose this; v1's default of 1.0 is fine.
    ],
}


class OfflineRBC(RuleBasedPolicy):
    """RuleBasedPolicy with charger-namespaced obs lookup.

    Behaviour and hyperparameters are identical to v1. The only change is
    ``_compute_ev_action`` now actually receives non-default values for
    ``charger_state``, ``soc``, ``required_soc_departure`` and
    ``departure_time``, so its EV charging logic runs as designed.
    """

    # ------------------------------------------------------------------
    # Internals: per-step charger context
    # ------------------------------------------------------------------

    def _compute_ev_action(  # type: ignore[override]
        self,
        agent_idx: int,
        obs: np.ndarray,
        obs_map: Dict[str, int],
        charger_info: Optional[ChargerInfo],
        bounds,
        action_name: str = "",
    ) -> float:
        # Build a *patched* obs_map that includes the v1 bare names mapped to
        # whichever namespaced index actually exists for this agent's chargers.
        patched = dict(obs_map)
        candidate_charger_ids: List[str] = []
        if charger_info is not None and charger_info.charger_id:
            candidate_charger_ids.append(charger_info.charger_id)
        # Also try every charger known to this agent (covers the case where
        # the action position doesn't have a registered charger but the obs
        # do).
        for info in self._ev_action_mapping.get(agent_idx, []) or []:
            if info is not None and info.charger_id and info.charger_id not in candidate_charger_ids:
                candidate_charger_ids.append(info.charger_id)

        for v1_name, patterns in _NAMESPACED_FALLBACKS.items():
            if v1_name in patched:
                continue  # nothing to do — env actually uses the bare name
            resolved_idx: Optional[int] = None
            for pattern in patterns:
                for cid in candidate_charger_ids:
                    real_name = pattern.format(charger_id=cid)
                    if real_name in obs_map:
                        resolved_idx = obs_map[real_name]
                        break
                if resolved_idx is not None:
                    break
            if resolved_idx is not None:
                patched[v1_name] = resolved_idx

        return super()._compute_ev_action(
            agent_idx,
            obs,
            patched,
            charger_info,
            bounds,
            action_name,
        )
