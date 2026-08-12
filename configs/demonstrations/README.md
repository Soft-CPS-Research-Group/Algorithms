# Training demonstrations

`community_fixed_service_battery_oracle_annual_v1.json.gz` is a compressed
semantic stationary-battery schedule for the full 15-minute dataset. It was
produced by the conservative community fixed-service MILP over the exact
paired SMART episode used by the MATD3 V4/V5 protocol and is used only as a
behavior-cloning teacher.

Scientific scope:

- perfect foresight, training-only and not deployable;
- optimizes stationary batteries while SMART EV and deferrable actions remain
  fixed;
- certificate applies to the conservative linear fixed-service formulation,
  not to the complete CityLearn problem;
- evaluated MATD3 actors do not read the schedule.

Provenance:

- compressed artifact SHA-256:
  `f40c201d545ea03226ddb97688f3cf694fdab5471f93dd1295cf4fdb4843b425`
- uncompressed schedule SHA-256:
  `681e318e0ac6c517f13df6cb2b5081855b62fe0ad780a96758584371c8b68f6e`
- horizon: 35,039 steps at 15 minutes;
- paired SMART job:
  `c531d3ef-98d9-4229-bdda-085329cb8b5e`, exported episode 2;
- paired SMART reconstructed/model cost: EUR 21,957.37;
- conditional conservative MILP cost: EUR 18,545.55;
- exact paired-service CityLearn replay: EUR 18,515.03 (-15.68%),
  119,910.51 kWh community import and lower settled cost for 17/17 buildings;
- exact replay ramping ratio: 1.24965 versus 2.40401 for paired SMART;
- this cost-only point regresses daily peak (1.13337 versus 1.07062), all-time
  peak (1.17509 versus 1.13440) and daily load-factor penalty (1.11403 versus
  1.10301), which is why it is not the scorecard teacher;
- exact replay solar self-consumption: 80.678%, EV minimum-acceptable
  feasibility: 99.818%, and electrical violations: 0.5051 kWh over 212 events.
  The latter is inside the agreed sub-kWh tolerance but fails the strict
  zero-violation gate.

The schedule is intentionally packaged in the repository so remote workers
receive the exact teacher artifact with the configuration and code image.

## Global scorecard teacher

`community_fixed_service_battery_global_scorecard_teacher_annual_v5.json.gz`
is the promoted multiobjective teaching schedule. It starts from a feasible
global cost/peak/ramp point and applies physical per-building coordinate
updates that minimize gross member import emissions while preserving the
global scorecard constraints after every accepted update. Relaxed subproblem
actions are projected through the actual battery capacity, power and
efficiency model before they may be accepted. This is a feasible coordinate
solution, not a global-optimality certificate.

Provenance:

- compressed artifact SHA-256:
  `4524206b39faf54d9484a84e4386f03d21c408f9b1b35987183624cc4ec88912`;
- uncompressed schedule SHA-256:
  `6c77cc1ddcc3cac65f5229615d4dc7d5a505130836a198393ab292029cf4e38a`;
- horizon: 35,039 steps at 15 minutes;
- paired source job: `c531d3ef-98d9-4229-bdda-085329cb8b5e`, exported
  episode 2, seed 789 and community settlement on;
- exact annual CityLearn replay cost: EUR 20,244.08, versus EUR 21,957.37 for
  paired SMART (-7.80%);
- exact replay community import: 124,067.54 kWh, versus 132,708.21 kWh
  (-6.51%);
- gross member emissions: 21,811.00 kgCO2, versus 22,407.36 kgCO2 for paired
  SMART (-2.66%);
- daily peak ratio: 0.99560 versus 1.07062; all-time peak ratio: 1.08398
  versus 1.13440;
- ramping ratio: 0.92556 versus 2.40401 for paired SMART;
- daily load-factor penalty ratio: 1.07827 versus 1.10301 for paired SMART;
- solar self-consumption: 77.366%, EV minimum-acceptable feasibility:
  99.855%, and V2G export: 42.35 kWh;
- settled member cost improves for all 17 of 17 buildings;
- electrical violations: 0.3337 kWh over 159 events. This is within the
  explicitly accepted sub-kWh research tolerance, but remains reported and
  fails the repository's zero-violation hard gate.

This replay demonstrates that a materially better joint cost/peak/ramping
operating point is physically attainable while retaining the neutral
`SignalAwareRBC` service controller (behaviorally identical to
`RBCSmartPolicy` at multiplier 1.0) and overriding only stationary-battery
actions. Because CityLearn recomputes that controller after the battery
trajectory changes, its realized service actions are not claimed to be
byte-for-byte identical to the source trace. This does not demonstrate that a
causal MATD3 actor has learned the point; the actor must still pass paired
seasonal and annual evaluation without access to the perfect-foresight
schedule.
