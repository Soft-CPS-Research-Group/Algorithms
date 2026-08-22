# CityLearn adapter to deployment-neutral TI-MARL contract map

This mapping is adapter-owned. None of these CityLearn names or versions
appears in a public `typed_agent_interface_v1` file.

## Existing entity contract

| CityLearn object | TI-MARL role |
|---|---|
| district table | `community` sensor channels and exogenous observations |
| building table | local meter/load/PV/headroom observations and logical agent |
| charger table | charger module features and EV-service evidence |
| EV table | EV/session entity state |
| storage table | stationary battery module/entity state |
| PV table | observation-only generation module in slice 1 |
| deferrable table | deferrable module/entity and start group |
| typed edge arrays | ownership, containment and connection relations |
| action specs/tables | executable route, ordering and static bounds |
| feature metadata | unit, source bundle and temporal semantics |
| topology version | snapshot/interface change boundary |

Current exogenous values are time `t`; endogenous operational/community values
are interpreted according to the payload's `temporal_semantics`, normally the
settled `t-1` state. Actors may only consume community fields on an explicit
allowlist that comparison baselines can also receive.

## Runtime-status extension

`runtime_status_v1` adds raw evidence in separate collections for asset
connections, asset availability, sensor channels, actuator channels,
communication links and active causal events. Default nominal states and field
vocabularies live in `entity_specs`; payloads may remain sparse.

Normal EV connection is represented by the charger-to-EV relation/mask and is
not a health failure. Asset unavailability does not imply sensor or actuator
channel loss unless an explicit event says so.

## Initial action mapping

| CityLearn action | Group | TI ports |
|---|---|---|
| electrical storage scalar | stationary storage mode | idle, charge, discharge |
| EV storage scalar per charger | EV charger mode | idle, charge, discharge |
| deferrable start scalar | deferrable operation | idle, start |

The public port parameter uses its declared physical unit. The Simulator
adapter/codec converts the final value to CityLearn's signed normalised scalar
using the current runtime bounds; resource effects and traces retain physical
kW separately.
