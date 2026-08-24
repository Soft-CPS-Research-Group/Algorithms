# TI-MARL experiment protocol v1

## Purpose

This protocol prevents checkpoint cherry-picking and comparisons made on
different simulator trajectories. It separates three roles:

1. `train`: stochastic learning; no result is a final claim;
2. `development`: deterministic paired replay used to choose hyperparameters
   and checkpoints;
3. `confirmation`: deterministic replay opened only after one checkpoint and
   all selection rules have been frozen.

Confirmation results are rejected as inputs to the checkpoint selector.

## Two independent seeds

`training.seed` initializes the learning algorithm. `simulator.random_seed`
initializes the Simulator and exogenous stochastic processes. A candidate and
its paired reference must share the latter, time window, dataset/schema,
building set, settlement, reward surface, topology mode and Simulator version.
They do not need to share the neural seed.

The runner records a content-addressed `paired_simulator_surface_v1`
fingerprint in `result.json` and `summary.json` whenever
`experiment_protocol` is configured.

## Frozen annual split

The 35,040-step annual 15-minute dataset is divided into four equal 8,760-step
blocks. Development and confirmation each use a different 14-day window in
every block:

| block start | development inclusive | confirmation inclusive |
|---:|---:|---:|
| 0 | 4,032–5,375 | 6,720–8,063 |
| 8,760 | 12,792–14,135 | 15,480–16,823 |
| 17,520 | 21,552–22,895 | 24,240–25,583 |
| 26,280 | 30,312–31,655 | 33,000–34,343 |

The remaining 24,288 steps are the tuning/training surface. Development uses
neural seeds `123`, `456`, `789`. Confirmatory training uses the previously
unseen seeds `1009`, `2027`, `3037`, `4051`, `5059`.

The four reserved temporal-confirmation windows use paired Simulator seeds
`2601`, `2602`, `2603`, and `2604`, respectively. These seeds are frozen before
opening the confirmation data, are shared by the candidate and reference in
each window, and are distinct from neural-training seeds.

The eventual deterministic full-year replay is a separate **annual benchmark**.
If its policy has trained on that same yearly trace, it is reported as a
transductive control benchmark, not as temporal generalization. Held-out
composition, topology and fault generalization are evaluated separately.

## Checkpoint production and evaluation

Protocol training must preserve every episode-end checkpoint. A development or
confirmation candidate:

- runs exactly one deterministic episode;
- is frozen and loads one explicit checkpoint;
- writes no new checkpoint;
- exports final-episode Simulator KPIs;
- records its paired reference ID and frozen rule/selection hash.

A training job may append one explicitly-windowed deterministic final episode
as an operational diagnostic. In that case the final episode performs no
learning, is reported to agent lifecycle hooks as `training=false`, writes no
episode checkpoint, and is the only episode allowed to export KPIs or
timeseries. This removes a redundant replay when screening the final training
state, but its KPIs are not a formal development or confirmation record. The
checkpoint selected for promotion is still replayed frozen under the paired
development/confirmation protocol above.

Entity-layout bootstrap must not consume a configured episode window. The
runner records the actual start/end, length, deterministic/training mode and
export flag of every executed episode in `result.json`; promotion checks use
this runtime evidence rather than trusting the requested schedule alone.

Safety-projected PPO is treated as a declared training ablation. When
`exclude_intervened_actions_from_policy_loss` is enabled, the run must report
the number of raw action groups masked from the actor objective and the number
that remained eligible. The transition is never discarded: critics, traces,
requested/applied diagnostics and the scorecard still use the real executed
outcome. This prevents reward produced by a substituted safety action from
being attributed to an unexecuted raw policy sample.

If `intervention_distillation_coeff` is positive, the run must additionally
report the auxiliary loss and its sample count. Distillation applies only to
masked typed groups and targets their final locally feasible decisions. It is
a safety-competence ablation, not an economic teacher and not a replacement
for the paired scorecard.

`scripts/ti_marl_experiment_protocol.py record` converts each deterministic KPI
export into `ti_marl_evaluation_record_v1`. The record hashes the checkpoint,
KPI file and simulator surface. Paths, records, configs and outputs remain in
the ignored local `runs/` tree.

After all reserved pairs finish, the `confirm` command in
`scripts/ti_marl_experiment_protocol.py` produces
`ti_marl_confirmation_report_v1`. It verifies the frozen selection-record and
checkpoint hashes, requires the exact same paired surfaces, and reports
aggregate and per-window deltas against the reference. It never selects or
replaces a checkpoint.

## Frozen selection rule

The current canonical rule is
`algorithms/ti_marl/experiments/selection_rules_v3.yaml`. The original v1 and
v2 files remain immutable so their content hashes continue to verify the
development evidence already collected with them.

Version 3 corrects the semantics of EV safety before its canonical development
campaign. Required departure SoC is a lower service bound. The symmetric
target-tolerance KPI also penalizes an EV that arrives above its requested
departure target, even when service is fully satisfied and discharging is not
authorized. Consequently, minimum acceptable feasible SoC remains the hard
safety gate, while the symmetric KPI becomes a no-regression guardrail against
the paired reference. This does not authorize V2G and does not permit a
candidate to reduce minimum service.

It is deliberately cost-first, but a checkpoint is promotable only when:

- total electrical violation energy across development is at most 0.5 kWh;
- EV minimum acceptable departure service is at least 99%;
- the symmetric EV target-tolerance rate does not fall below the reference;
- it improves paired-reference cost by at least 0.1%;
- daily peak and ramping do not degrade by more than 5%;
- solar self-consumption does not fall by more than two percentage points;
- deferrable service level does not fall below the paired reference.

All four development windows must be present for both checkpoint and paired
reference. Additive quantities are summed; shape ratios are averaged; EV hard
gates use the worst window. Ties are resolved by daily peak, ramping and solar,
in that order.

The selector emits `ti_marl_checkpoint_selection_v1`, including the exact
checkpoint SHA-256. Confirmation embeds `selected_checkpoint_sha256`, and the
runner verifies the resolved file before loading any weights. Five
confirmatory seeds are all reported; no best-seed filtering is allowed.
