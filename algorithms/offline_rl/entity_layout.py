"""Bind member-specific entity IDs to typed local slots for shared policies.

Only identifiers are canonicalised. Feature names, asset types and multiplicity
are preserved, so equal vector dimensions alone never establish compatibility.
The same mapping must be applied at collection and inference. Physical vectors
are not reordered here; the returned names describe their existing order.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence


ENTITY_NAME_CONVENTION = "typed_local_slots_v1"


def canonicalize_entity_names(
    observation_names: Sequence[str], action_names: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Return position-preserving names with stable slots within each asset type.

    Assets are sorted by their full entity ID within a member. This defines a
    deterministic local binding, not a claim that device characteristics are
    identical. Those characteristics remain observations. Actions are bound by
    their actual device IDs to the same slots as their observation features.
    """
    if len(set(observation_names)) != len(observation_names):
        raise ValueError("Duplicate observation names cannot define a shared layout")
    if len(set(action_names)) != len(action_names):
        raise ValueError("Duplicate action names cannot define a shared layout")
    entities: dict[str, set[str]] = defaultdict(set)
    for name in observation_names:
        if "::" in name:
            parts = name.split("::", 2)
            if len(parts) != 3 or not all(parts):
                raise ValueError(f"Malformed scoped entity feature: {name}")
            kind, entity_id, _ = parts
            entities[kind].add(entity_id)
    slots = {kind: {entity_id: f"slot_{index:03d}"
                    for index, entity_id in enumerate(sorted(ids))}
             for kind, ids in entities.items()}
    observations = []
    for name in observation_names:
        if "::" in name:
            kind, entity_id, feature = name.split("::", 2)
            observations.append(f"{kind}::{slots[kind][entity_id]}::{feature}")
        else:
            observations.append(name)
    actions = []
    for name in action_names:
        matched = False
        for prefix, kind in (("electric_vehicle_storage_", "charger"),
                             ("deferrable_appliance_", "deferrable_appliance")):
            if not name.startswith(prefix):
                continue
            device_id = name[len(prefix):]
            candidates = [entity_id for entity_id in entities.get(kind, ())
                          if entity_id.rsplit("/", 1)[-1] == device_id]
            if len(candidates) != 1:
                raise ValueError(f"Action {name!r} has no unique matching {kind} observations")
            actions.append(prefix + slots[kind][candidates[0]])
            matched = True
            break
        if not matched:
            if name == "electrical_storage" and len(entities.get("storage", ())) == 1:
                actions.append(name)
            elif not entities:
                # Legacy unscoped layouts retain their original strict names.
                actions.append(name)
            else:
                raise ValueError(f"Unsupported entity action binding: {name!r}")
    if len(set(observations)) != len(observations) or len(set(actions)) != len(actions):
        raise ValueError("Canonical entity names collide")
    return observations, actions
