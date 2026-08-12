# Autonomous CC-L2 over frozen PPO

This campaign trains a Level-2 coordinator without a Level-1 controller,
Level-1 observation, Level-1 action, or Level-1-derived incumbent. The only
reference is the exact neutral vector `[1.0] * 17`, which reproduces the frozen
local PPO contract.

## Protocol

The twelve annual episodes have fixed roles:

1. one exact PPO-neutral warm-up absorbs the simulator/leaf cold start;
2. one exact PPO-neutral episode collects per-decision, per-building rewards;
3. two stochastic CC-L2 training episodes;
4. one deterministic validation episode;
5. steps 3--4 repeat three times (six training and three validation years);
6. the final deterministic year restores the best validated policy.

Training uses the neutral episode only as an action-independent control
variate. Subtracting this baseline reduces seasonal reward variance without
providing a policy, schedule, or price to imitate. A validation policy is
promoted only if its full-episode reward exceeds the neutral objective. If no
candidate does, final evaluation falls back to the exact neutral coordinator.
After a rejected validation, training is rolled back to the selected policy
before the next exploration block, preventing a bad update sequence from
becoming the next starting point.
All protocol episodes replay the same CityLearn episode index, so EV drift and
the exogenous year are identical: validation differences are caused by the L2
policy rather than by a different stochastic EV realization.

The 17 PPO leaves remain frozen, deterministic and building-local. CC-L2 emits
one price per building. Its sparse centered parameterization has an exact
neutral deadband so weak/noisy actions do not perturb the PPO leaf.

## Candidates

| Config | Primary intent | Price range | Reward emphasis |
|---|---|---:|---|
| `cost_first_seed123` | strongest economic search | `[0.60, 1.00]` | settled cost |
| `balanced_seed456` | cost with physical quality | `[0.70, 1.00]` | cost, peak, ramping |
| `scorecard_seed789` | conservative physical policy | `[0.78, 1.00]` | stronger peak/ramping pressure |

`paired_neutral` is the episode-11 comparator with the same frozen PPO price
path, settlement, export contract, dataset and horizon.

## Promotion

Annual evidence is required. Apply the Phase-6 hard gates first: EV minimum
service at least `0.999`, EV tolerance at least `0.80`, and electrical
violations no more than `1e-6 kWh`. A candidate must then beat its paired
neutral PPO on settled cost; peak, ramping, emissions, solar, throughput and
per-building fairness remain explicit scorecard trade-offs.
