# Proposed thesis results — supervisor discussion summary

> **Historical illustrative draft — retained 2026-09-25.** Performance values
> are hypothetical placeholders, not measured or provisional experimental
> results. The author has retired the previous `THESIS-EXP-V1` campaign plan.
> These tables preserve discussion context, not acceptance criteria or a plan
> to execute. See [current research status](../README.md) and the separate
> [observed TI-MARL evidence](../ti-marl/provisional-results-2026-09.md).

The [detailed version](proposed-thesis-results-for-supervisors.md) contains
the complete method catalogue, dataset audit, development ladder, mechanism
tests, flexibility programmes, integrated stress, and deployment panels. This
short version keeps only the decisions that a supervisor may want to discuss
first.

## 1. Condensed nominal comparison

Results represent annual-equivalent values scaled from four held-out seasonal
windows, with five independent training seeds for learned methods. Cross-family
rows are descriptive because information and action authority differ. The
registered inferential comparison is TI-MAPPO against matched MAPPO `REF*`; that
selected pair is then expanded to HEAD10 with five additional training seeds.

| Family | Representative controller | Cost (EUR) | Saving vs BAU | EV minimum service | Daily peak ratio | Control-computation time |
|---|---|---:|---:|---:|---:|---:|
| Reference | Normal policy, BAU | 24,900 | 0.0% | 100.0% | 1.000 | 0.09 ms |
| Rule-based | RBC Smart Local | 23,232 | 6.7% | 100.0% | 0.931 | 0.18 ms |
| Rule-based | RBC Community | 22,858 | 8.2% | 100.0% | 0.912 | 0.31 ms |
| Optimisation | Rolling causal MILP, local | 21,065 | 15.4% | 100.0% | 0.844 | 8.4 min per daily optimisation |
| Optimisation | Rolling causal MILP, community | 20,294 | 18.5% | 100.0% | 0.810 | 46.7 min per daily optimisation |
| Single-agent | PPO, selected variant | 22,559 | 9.4% | 100.0% | 0.914 | 3.8 ms |
| Single-agent | TD3, selected variant | 22,310 | 10.4% | 100.0% | 0.902 | 4.4 ms |
| Entity-based | Agent Transformer MATD3 | 21,937 | 11.9% | 100.0% | 0.879 | 7.6 ms |
| Decentralised multi-agent | MATD3 | 21,688 | 12.9% | 100.0% | 0.865 | 6.4 ms |
| Coordinated multi-agent | **MAPPO, selected `REF*`** | **21,240** | **14.7%** | **100.0%** | **0.842** | **7.4 ms** |
| Hierarchical coordination | Community Coordinator Level 2 | 21,339 | 14.3% | 100.0% | 0.844 | 8.9 ms |
| Typed-interface | TI-PPO | 21,588 | 13.3% | 100.0% | 0.861 | 9.7 ms |
| Typed-interface | **TI-MAPPO, selected** | **21,367** | **14.2%** | **100.0%** | **0.846** | **11.4 ms** |
| Privileged reference | Perfect-information annual MILP, best incumbent | 19,322 | 22.4% | 100.0% | 0.775 | >72 h |

- The rolling local and community formulations would optimise seven days at a
  time and commit the next day. Their reported times therefore refer to one
  daily optimisation, not to the extraction of an individual 15-minute action.
- Learned coordination would improve on strong rule-based baselines.
- TI-MAPPO would not be the nominal winner; its value would have to appear in
  the structural and lifecycle evidence.
- The full table retains every principal PPO, TD3, Transformer, independent,
  coordinated, hierarchical, and typed method.

## 2. The intended TI-MARL argument

TI-MAPPO is the selected implementation of TI-MARL in these comparisons. Its
nominal result is deliberately tested for non-inferiority. Superiority is then
tested where typed interfaces are expected to matter.

### 2.1 Comparative evidence across the thesis requirements

| Dimension | Strongest relevant alternative | TI-MAPPO | What the result demonstrates |
|---|---:|---:|---|
| Nominal control | MAPPO is 0.60% cheaper | One-sided upper confidence limit is +1.42% | TI-MAPPO remains inside the frozen 2% non-inferiority margin while preserving every service gate. The structural capability is therefore not obtained through a material nominal loss. |
| Scalability | Padded MAPPO grows from 2.20 M to 6.52 M actor parameters and reaches 34.8 ms inference at 100 members | One 1.86 M-parameter actor; 19.6 ms inference; all gates pass from 10 to 100 members | TI-MAPPO processes a tenfold population range and one to six chargers per member without resizing or replacing the actor. The alternative uses almost 3.5 times more parameters at 100 members. |
| Reconfiguration | Fixed-layout MAPPO handles 8/14 events; padded MAPPO handles 13/14 and produces six invalid ports | 14/14 events; zero wrong bindings; zero invalid ports; valid control resumes in the next 15-minute step; no checkpoint swap | Every declared member, asset and EV-session change is incorporated while the surviving observations and actions retain their identity. The strongest alternative still fails one event and cannot provide a complete semantic trace. |
| Safety | Matched MAPPO leaves 2.1 kWh of requested violation energy after its common protection and requires Simulator correction | 0.0 kWh residual; maximum exceedance below 1×10⁻⁵ kW; no Simulator correction | TI-MAPPO produces a locally feasible bundle before physical application while retaining 100% feasible EV service and every mandatory deferrable cycle. |
| Health | Matched MAPPO incurs +10.4% fault cost, 98.9% adjusted EV service, a 4.00 h p95 recovery time and only 4/48 exact dependency closures | +2.8% fault cost, 100% service, 48/48 exact closures and valid stable operation restored within 0.75 h at p95 | Relative to paired clean runs, typed health closure confines each registered fault to the dependent interface. After a fault ends, 95% of the evaluated cases return to sustained valid control and service within 45 minutes, compared with four hours for the matched alternative. |
| Integrated robustness | MAPPO achieves 10.9% saving, 98.9% EV service, 99.6% mandatory service and 75% demand-response delivery; 1/5 seeds fails | 13.3% saving, 100% EV and mandatory service, 88% demand-response delivery and 0/5 failed seeds | Under simultaneous faults, topology changes, limits and demand-response requests, TI-MAPPO is the only learned method that retains every service gate and passes the integrated acceptance rule. |
| Deployability | A flat MAPPO bundle rebuilds numerically but detects only 2/7 semantic incompatibilities and requires manual recovery in 4/5 failed deployments | Same bundle on 5/5 targets; maximum action difference 2.4×10⁻⁶; 7/7 invalid cases rejected; 5/5 automatic rollbacks | TI-MAPPO preserves the policy-interface contract across PC, server, Deucalion, Union and edge execution. Invalid or failed deployments emit no learned action and return to the last validated bundle. |

The two recovery values refer to different events. For reconfiguration, the
next-step result means that the controller uses the new valid bindings in the
first 15-minute decision after a topology change. For health, the 0.75 h p95 is
measured from the end of a fault until valid control and the applicable service
conditions are restored and remain satisfied for four consecutive steps. It
does not represent the time required to repair the physical equipment.

### 2.2 Computational and storage footprint

The comparison separates model size from the work performed when the number of
members and assets changes. A fixed-layout controller encodes particular
positions in its input and output layers. Supporting a new layout can therefore
require a resized network and a separately stored policy. A padded controller
avoids several policies by allocating one network for the largest catalogue,
but carries those inactive positions at smaller scales. Set-based and
Transformer controllers reuse shared encoders across a variable number of
entities, so their neural parameter count can remain fixed. TI-MAPPO adds typed
encoders, action-group decoders and a variable-set critic to preserve the
meaning and validity of those entities. Its fixed parameter count is therefore
not a property of the Transformer alone.

A fixed parameter count does not imply constant computational cost. More active
entities produce more tokens, attention operations, compiled bindings and
actions. Runtime memory and latency must consequently be measured at each
scale. Training time is omitted from this proposed comparison. All matched
learned methods will receive the same interaction and tuning budgets, while
their measured wall-clock time, failed trials and hardware consumption will be
reported after the campaign. No training-time advantage is assumed here. Trace
storage is shown separately because complete typed evidence is intentionally
richer than a flat action log.

| Controller strategy | Models required for 10–100 members | Largest deployed actor | Deployed policy storage | Peak runtime memory at 100 members | End-to-end inference p95 at 100 members | Compressed annual trace | Trace completeness |
|---|---:|---:|---:|---:|---:|---:|---:|
| Size-specific fixed-layout MAPPO | 6 | 1.42 M parameters per actor | 38.4 MB | 1.46 GB | 18.2 ms | 0.58 GB | 47% |
| Padded-and-masked MAPPO | 1 | 6.52 M parameters | 28.7 MB | 1.94 GB | 34.8 ms | 0.92 GB | 64% |
| Agent Transformer MATD3 | 1 | 2.40 M parameters | 11.2 MB | 1.32 GB | 26.4 ms | 1.02 GB | 71% |
| **TI-MAPPO** | **1** | **1.86 M parameters** | **9.8 MB** | **1.08 GB** | **34.4 ms, including 14.8 ms compilation** | **1.48 GB** | **100%** |

- TI-MAPPO stores one deployable policy instead of a portfolio indexed by
  population or layout. Its end-to-end latency includes interface compilation
  and remains 0.0038% of the 900-second control interval.
- The typed trace requires more storage because it retains identities, health,
  bounds, requested actions, projected actions and applied commands. That cost
  buys complete reconstruction rather than a smaller but incomplete log.

### 2.3 Overall acceptance across the complete claim

The scale resource gate requires one deployable actor, at most 5 M parameters,
at most 1.5 GB peak runtime memory and p95 inference below 50 ms at 100 members.
The deployment gate requires parity on 5/5 targets, safe rejection of 7/7
invalid cases, successful rollback in 5/5 failures and no unsafe learned action.

| Method | Nominal | Scale | Reconfiguration | Safety | Health | Integrated robustness | Deployment | Overall |
|---|---|---|---|---|---|---|---|---|
| Rolling community MILP | Best cost | Fail compute gate | Fail dynamic-interface gate | Pass with fallback | Fail | Pass with fallback | Fail real-time gate | **Fail** |
| Fixed-layout MAPPO `REF*` | Pass | Fail model-continuity gate | Fail | Fail | Fail | Fail | Fail semantic-contract gate | **Fail** |
| Padded-and-masked MAPPO | Pass | Fail resource gate | Fail | Fail | Fail | Fail | Fail admission gate | **Fail** |
| Agent Transformer MATD3 | Fail 2% nominal gate | Pass | Fail | Fail | Fail | Fail | Fail lifecycle gate | **Fail** |
| **TI-MAPPO** | **Pass** | **Pass** | **Pass** | **Pass** | **Pass** | **Pass** | **Pass** | **Pass** |

TI-MARL is therefore not described as the cheapest controller under perfect or
stable information. It is the only evaluated approach that passes the complete
real-world applicability claim. The conclusion remains bounded by the tested
semantic types, fault families, physical constraints, computing targets and
population range.

## 3. Proposed split between thesis body and annexes

| Destination | Tables or evidence | Reason |
|---|---|---|
| Main body | Dataset ladder and compact audit | Defines the experimental surfaces and their boundaries |
| Main body | Two-part nominal comparison with all principal algorithms | Preserves the historical controller progression |
| Main body | Non-inferiority and matched structural baselines | Carries the central TI-MARL claim |
| Main body | Safety, health, and scale | Tests the three main structural mechanisms and their costs |
| Main body | Integrated stress and deployment lifecycle | Connects frozen control to execution under difficult conditions |
| Main body | Demonstrator evidence levels and four-dimension synthesis | States exactly what was observed and what remains outside scope |
| Annex | Hyperparameter search, rewards, replay, warm start, behaviour cloning, and detailed ablations | Explains how final variants were obtained without interrupting the main argument |
| Annex | Per-seed estimates, learning curves, full scorecards, event traces, MILP diagnostics, and asset inventories | Supplies uncertainty, provenance, and reproducibility detail |
| Machine-readable supplement | Run registry and long-format metrics | Retains commits, hashes, seeds, hardware, failures, and every measured value |

*This is a proposed reporting structure, not an experimental result.*

The point of the split is to keep every principal method visible while avoiding
a thesis chapter organised around thousands of individual runs. The execution
ledger remains necessary, but it serves the tables rather than becoming the
story itself.
