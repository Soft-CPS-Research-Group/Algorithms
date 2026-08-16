# TI-MARL experiment plan

The first vertical slice is a software/evidence gate and is not required to
beat PPO or SMART. Performance claims begin only after the architecture gates
pass.

Campaign order:

1. one-agent TI-RL with multiple local assets;
2. nominal multi-agent TI-MAPPO;
3. runtime module and population changes;
4. health/channel/communication stress;
5. known-type held-out compositions;
6. semantically unknown-type rejection;
7. scale and annual evaluation.

Development uses three seeds and paired short windows. Confirmatory runs use
five frozen seeds and the final annual protocol. Reward weights and authorised
community observations are frozen on a development split before confirmation.

Control metrics retain cost, settlement, import/export, peak, ramping, solar,
emissions, EV feasibility, V2G, throughput, fairness and network violations.
TI metrics add binding/invalid-port rates, raw/final infeasibility, compiler and
policy latency, health detection/recovery, fallback/intervention, changing
cardinality, parameter-count stability and trace completeness.

Mandatory structural comparisons include fixed masked output, health as a
feature without closure, independent heads, per-port clipping, fixed critic and
disabled compatibility checks.

