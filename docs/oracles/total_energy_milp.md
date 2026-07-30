# Total-energy MILP reference

## Purpose and comparator split

The repository now distinguishes four optimization references instead of
calling every perfect-foresight schedule "the MILP":

| Reference | Physical scope | Settlement | Intended comparator |
|---|---|---|---|
| Individual fixed-service | Stationary battery only; EV/deferrables frozen | Positive import per building | Earlier storage-assisted PPO/TD3 diagnosis |
| Community fixed-service | Stationary batteries only; EV/deferrables frozen | District net import | Earlier community battery diagnosis |
| Individual total-energy | Battery + EV/V2G + deferrables + building/phase service | Positive import per building | Building-local PPO/TD3 |
| Community total-energy | Same complete physical scope | District net import/market settlement | Future CC and MARL |

The individual total-energy problems contain no community observation or
coupling.  They answer the question "how good could this home be with perfect
foresight?".  The community problem uses the same physical models but permits
economic netting across buildings.  It answers a different question and must
not be used as the direct denominator for a community-blind local policy.

## Current linear formulation

`algorithms/oracles/total_energy_milp.py` jointly schedules:

- stationary battery charge, discharge, SOC, standby loss and direction;
- EV charge/V2G by contiguous connection session, including arrival SOC,
  minimum SOC, departure target, charger efficiency and minimum-power
  deadbands;
- one binary start for every must-run deferrable cycle and its complete fixed
  kWh/step profile;
- signed building import/export envelopes;
- signed L1/L2/L3 envelopes, including balanced base load and `all_phases`
  storage;
- either per-building positive imports (`individual`) or positive district
  net import (`community`).

Export currently has zero credit, matching the dataset's grid-export tariff.
The common-price community market cancels internal payments at district level,
so its total cost is the grid cost after netting.  Member bill allocation is a
post-processing question unless fairness or bill caps enter the objective.

## Lexicographic EV service tolerance

EV departure service is solved before energy cost.  The first solve minimizes
total departure shortfall.  Once that phase is certified optimal, the second
minimizes cost under one aggregate shortfall cap per building.  This is exact
at zero tolerance: service feasibility and all physical constraints factor by
building, while community settlement appears only in the economic objective,
so every global service optimum must attain each building's local minimum.
The per-building caps remove the otherwise large cross-building degeneracy.
The economic phase is not run when the service phase has only an incumbent;
such an incumbent remains explicitly non-optimal rather than being hidden
behind a later `optimal` status.

The default numerical slack is `0.001 kWh` (1 Wh), selected because tighter
`1e-7` to `1e-4 kWh` caps caused HiGHS to stall or report a false infeasibility
on the joint week-one problem even though every building subproblem was
feasible.  Buildings whose minimum shortfall is zero receive a hard zero cap;
the single 1 Wh allowance is distributed only among buildings with a positive
minimum.  Thus numerical slack cannot create missed service at an otherwise
fully serviceable home.  The CLI exposes
`--lexicographic-shortfall-tolerance-kwh`.

Every result records the service-phase status, global and per-building minima,
per-building caps, realized per-building shortfall and tolerance.  Consequently
the realized value, rather than only the first-stage minimum, is the auditable
service outcome.  This tolerance is numerical and must not be described as
unmet service hidden by the optimizer.

## Week-one corrected-SOC certificate

The first week-one extraction used `1 - depth_of_discharge` for EVs without an
explicit schema `initial_soc`.  CityLearn instead samples those initial SOCs
deterministically from MD5 of seed, EV type and EV identifier.  The shared
extractor now reproduces that algorithm exactly.  Pre-correction individual
and community certificates are superseded as dataset-aligned optima; their
CityLearn replays remain feasible diagnostic trajectories.

The corrected community instance has 17 buildings and 672 steps.  Its service
phase is optimal at 28.902728 kWh unavoidable shortfall, entirely in
`Building_15`.  The economic phase is optimal at EUR 347.7026176791, with dual
bound EUR 347.6787858945 and 0.0068547% MIP gap.  The realized shortfall is
28.903728 kWh: the exact minimum plus the configured 1 Wh numerical tolerance.

The zero-teacher CityLearn replay costs EUR 347.7026162444, only
-EUR 0.0000014346 relative to the model objective.  It passes 17/17 local hard
gates, improves all 17 buildings, and reduces matching RBCCommunity cost from
EUR 440.0609948344 by EUR 92.3583785900 (20.99%).  It is therefore accepted as:

- optimal for the supplied corrected linear model;
- a CityLearn-feasible replay schedule;
- a tight week-one conditional benchmark.

It is not called the global optimum of nonlinear CityLearn.  The `[0,672)`
window also right-truncates eight EV sessions, so this result is not seasonal
or annual promotion evidence.  Raw solve and replay-audit evidence lives under
`runs/analysis/total_energy_community_week1_corrected_soc_solve_20260731` and
`runs/analysis/total_energy_community_week1_corrected_soc_optimal_replay_audit_20260731`.

## Bounds and permitted claims

The continuous relaxation is a structural lower bound for the supplied
linear model.  The mixed-integer solution is feasible for that linear model
and emits semantic CityLearn actions.  It is not automatically feasible for
the nonlinear simulator because CityLearn also applies SOC-dependent power,
efficiency curves, degradation, float32 action conversion and requested-power
service checks.

The valid claim before replay is therefore:

```text
linear lower bound <= optimum of supplied linear model <= linear MILP cost
```

After a zero-intervention CityLearn replay, the replay cost is a valid
simulator-feasible upper bound.  A global "CityLearn optimum" claim additionally
requires an optimistic model proven to contain the nonlinear simulator's
feasible set and a sufficiently small lower-bound/replay interval.

## Building_15 electrical service

The total-energy model keeps `Building_15` physical rather than aggregating it:

- total import/export: ±12 kW;
- L1: ±7 kW, with `charger_15_1` on L1;
- L2: ±5 kW, with `charger_15_2` on L2;
- L3: ±4 kW;
- stationary battery: balanced across all phases;
- unassigned base load: balanced across all phases.

Initial replay candidates reserve 0.10 kW inside every applicable total/phase
limit.  This exceeds the approximately 0.0802 kW aggregate requested-power
excess observed across the three tiny violations in the earlier annual
fixed-service community replay.  The reserve may only be reduced after a
zero-violation requested/limited/applied trace.

## Annual execution strategy

An annual monolithic complete MILP is not the default.  At 15-minute resolution
it would have roughly 2.38 million continuous variables and 0.97 million binary
variables before piecewise-linear battery physics.  The execution plan is:

1. annual optimistic relaxation for a lower bound;
2. seven-day lookahead complete MILP with one-day commits;
3. exact carry of storage/EV SOC and already-started deferrable profiles;
4. strict semantic preflight;
5. CityLearn replay with no safety intervention accepted;
6. scorecard against a matching RBC on the same window and settlement;
7. archive model, schedule, replay, hashes, bounds and decision in experiment
   history.

The maximum observed EV session is about 67 hours and the maximum deferrable
window plus profile is below nine hours, so seven days contains the local
events while keeping each solve tractable.  Rolling horizon produces a strong
candidate, not by itself a global certificate.

## CC use of local PPO/TD3

The CC does not add community inputs to local actors.  The local checkpoint
keeps exactly the same observation dimension and sees only its effective local
price.  A price adapter decodes the normalized tariff, multiplies the raw
price, and re-encodes it.  Multiplier 1.0 must be bitwise neutral.  Official
costs always use the real simulator tariff; virtual prices are policy/training
signals only.

Existing eight-week checkpoints are mechanically compatible with this
adapter, but they are not yet validated as price-responsive or seasonal.  They
must be fine-tuned with price-domain randomization and evaluated on embargoed,
strictly held-out seasonal windows before use under a learned CC.
