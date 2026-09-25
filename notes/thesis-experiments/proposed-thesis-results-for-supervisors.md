# Proposed thesis result tables for supervisor discussion

> **Historical illustrative draft — retained 2026-09-25.** Performance values
> are hypothetical placeholders, not measured or provisional experimental
> results. The author has retired the previous `THESIS-EXP-V1` campaign plan.
> These tables preserve discussion context, not acceptance criteria or a plan
> to execute. See [current research status](../README.md) and the separate
> [observed TI-MARL evidence](../ti-marl/provisional-results-2026-09.md).

The original draft hypothetically assumed that all blocked implementations
would be completed and that its proposed protocol would be followed. Those
assumptions and the illustrative conclusions below are not observed outcomes.

For a short meeting handout, use the
[supervisor summary](proposed-thesis-results-supervisor-summary.md). The
remainder of this file is the detailed table prototype.

## Executive view

This first table contains the shortest version of the proposed result narrative.
It is an index to the detailed tables, not a substitute for them.

| Question | Headline result | Why it would matter | Detailed evidence |
|---|---|---|---|
| Does typed control retain nominal quality? | TI-MAPPO is 0.60% more expensive than matched MAPPO, with a one-sided 95% upper limit of 1.42%, below the frozen 2% margin | The structural contribution would not be purchased through a material nominal loss | Sections 2 and 4 |
| What does typing add? | 14/14 known-type changes handled, no wrong binding, explicit rejection of an unknown type, and a complete semantic action trace | It separates typed semantics from padding or generic set processing | Section 5 |
| Where does safety come from? | Joint projection reduces represented local exceedance to below 1×10⁻⁵ kW before Simulator enforcement | It distinguishes a feasible command from an unsafe request later corrected by the plant layer | Section 6 |
| Does health handling reduce degradation? | Dependency closure identifies the registered affected interface elements in 48/48 fault cases and shortens recovery | Robustness would be measured against paired clean twins rather than claimed from nominal operation | Section 7 |
| Can the artefact move from training to execution? | One bundle is reconstructed with parity on heterogeneous inference targets; invalid bundles are rejected or rolled back | Deployment evidence would include contracts and failure handling, not latency alone | Sections 12 and 13 |

## Reading conventions

- `Selected` identifies a method hypothetically frozen using development data.
- `REF*` identifies the best eligible non-typed learned comparator with matched
  information, action authority, reward, and protection.
- `Pass` and `Fail` refer only to a predeclared gate.
- `N/A` means that the mechanism or metric does not apply on that surface.
- `NE` means that the quantity would apply but is not evaluated in the table.
- `Blocked` means that evidence cannot yet be produced because an implementation
  or data dependency remains open.
- A measured zero is written as `0`; it is never used in place of `N/A` or `NE`.
- A perfect-information MILP is an oracle only when the relevant optimum is
  certified. The annual table below reports a time-limited feasible incumbent
  and its remaining optimality gap. It is a privileged-information reference,
  not a deployable comparator or a certified oracle optimum.

### Abbreviation key

| Abbreviation | Meaning |
|---|---|
| BAU | Business-as-usual operational reference |
| RBC | Rule-based controller |
| MILP | Mixed-integer linear programming |
| PPO | Proximal policy optimisation |
| TD3 | Twin delayed deep deterministic policy gradient |
| IPPO | Independent proximal policy optimisation |
| MADDPG | Multi-agent deep deterministic policy gradient |
| MATD3 | Multi-agent twin delayed deep deterministic policy gradient |
| MASAC | Multi-agent soft actor-critic |
| MAPPO | Multi-agent proximal policy optimisation |
| HAPPO | Heterogeneous-agent proximal policy optimisation |
| TI-MARL | Typed-Interface Multi-Agent Reinforcement Learning |
| V2G | Vehicle-to-grid charging |
| DR | Demand response |

## 1. Annual datasets and their experimental role

The population and asset counts below reproduce the current dataset contract.
All datasets cover the 2023 civil year at 15-minute resolution, corresponding
to 35,040 control transitions.

| Experimental surface | Active members | PV | Batteries | Chargers | Main added mechanisms | Intended use | Audit outcome |
|---|---:|---:|---:|---:|---|---|---|
| `MICRO-4-Q` | 4 | 2 | 1 | 2 | Minimal heterogeneous assets | Smokes, debugging, interface probes | Pass within declared scope |
| `CORE-15-STRIPPED` | 15 | 7 | 3 | 7 | Static clean population | Development, ablations, initial tuning | Pass within declared scope |
| `CORE-30-NOMINAL` | 30 | 15 | 6 | 15 | Static population, no electrical-service limits | Common nominal algorithm comparison | Pass within declared scope |
| `CORE-30-SAFETY` | 30 | 15 | 6 | 15 | Installation-total and per-phase limits | Local feasibility and protection | Pass within declared scope |
| `CORE-30-HEALTH` | 30 | 15 | 6 | 15 | Telemetry, forecast, storage, and charger faults | Paired health and robustness tests | Pass within declared scope |
| `CORE-30-DYNAMIC` | 26 → 30 → 28 | 15 in full catalogue | 6 in full catalogue | 15 in full catalogue | 14 member and asset topology events | Known-type reconfiguration | Pass within declared scope |
| `CORE-30-COMBINED` | 26 → 30 → 28 | 15 in full catalogue | 6 in full catalogue | 15 in full catalogue | Safety, health, dynamics, and four DR requests | Integrated Core-30 stress | Pass within declared scope |
| `PREMIUM-100-CLEAN` | 70–100 | 55 | 22 | 80 | Scale, heterogeneity, dynamics, limits, and DR | Large clean integrated confirmation | Pass within declared scope |
| `PREMIUM-100-ALLIN` | 70–100 | 55 | 22 | 80 | Clean mechanisms plus eight faults and outages | Closed final stress test | Pass within declared scope |

- This progression would let the thesis introduce complexity one mechanism at a
  time before opening the integrated stress cases.
- The main comparison would remain on `CORE-30-NOMINAL`; safety, health, V2G,
  and demand response would be tested only where those mechanisms are present.
- The populations would establish coverage of mechanisms and profiles. They
  would not constitute a demographic sample of Portuguese energy communities.
- The represented electrical boundary would stop at installation-total and
  per-phase active-power limits. Transformer, feeder, voltage, reactive-power,
  and protection claims would remain outside scope.

### 1.1 Compact Simulator and dataset audit

| Dataset family | Required transitions | Missing required values | Maximum energy-balance residual | Declared infeasible EV sessions | Infeasible mandatory deferrables | Clean-twin mismatches | Identity violations | Repeated fingerprints |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Micro-4 | 35,040/35,040 | 0 | 4.2×10⁻⁹ kWh | 0/554 | 0/214 | 0 | 0 | 4/4 identical |
| Core-15 | 35,040/35,040 | 0 | 5.8×10⁻⁹ kWh | 4/1,757 | 0/928 | 0 | 0 | 4/4 identical |
| Core-30 and overlays | 35,040/35,040 | 0 | 6.4×10⁻⁹ kWh | 9/3,774 | 0/1,892 | 0 | 0 | 4/4 identical |
| Premium-100 and overlays | 35,040/35,040 | 0 | 8.9×10⁻⁹ kWh | 101/41,173 | 0/7,360 | 0 | 0 | 4/4 identical |

- A small, explicitly labelled infeasible EV set would retain difficult cases
  without silently counting physically impossible departures as controller
  failures.
- Zero mandatory-deferrable infeasibility would confirm that every `must_run`
  cycle can be scheduled inside its declared window.
- Clean-twin equality would be checked outside the declared experimental
  dimension, not inferred from matching filenames.
- The detailed member, asset, phase, session, and limit inventory would still
  belong in the thesis annex.

## 2. Held-out nominal controller comparison on Core-30

The following two panels contain exactly the same controller rows. Each learned
row represents one variant selected on development data, rather than the best
confirmation seed. The estimates use five independent training seeds
and four seasonal confirmation windows per policy. Aggregation occurs first by
policy seed, followed by a hierarchical bootstrap over seeds and repeated
windows. The annual-equivalent costs and energy totals are scaled from held-out
per-member-day estimates so that the values remain easy to read. They are not
full-year replays. A replay on the same 2023 year used for learning would be
reported separately as a transductive benchmark.

The omnibus comparison is descriptive across families because information and
action authority differ. Inferential claims are reserved for registered matched
pairs, particularly selected TI-MAPPO versus `REF*`.

| Comparison class | Information available at decision time | Action authority | Valid interpretation |
|---|---|---|---|
| Grid and operational references | Realised exogenous or strict local | None or declared local assets | Economic and operational context |
| Local rule-based and single-agent methods | Strict local | Matched local assets | Within-class control comparison |
| Community rule-based methods | Community state | Declared community assets | Strong deterministic community reference |
| Causal optimisation | Current data and registered causal forecasts | Matched local or community assets | Control-versus-computation comparison |
| Decentralised and coordinated learning | Local actor; local or community information during training as declared | Matched assets | Descriptive family progression; inference only for preregistered matched pairs |
| Hierarchical coordination | Community signal plus fixed downstream controller | Indirect signal authority | Complete-system comparison, not a direct actor comparison |
| Typed-interface learning | Typed local actor; declared local or variable-set critic | All matched typed groups | Primary matched contrast and structural tests |

### 2.1 Energy and economic outcomes

| Family | Controller | Annual-equivalent cost (EUR) ↓ | 95% CI | Saving vs BAU ↑ | Import (MWh) ↓ | Export (MWh) | Net exchange (MWh) ↓ | Solar self-consumption ↑ | Battery throughput (MWh) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Economic reference | Grid-only counterfactual | 27,714 | N/A | −11.3% | N/A | N/A | N/A | N/A | N/A |
| Operational reference | Normal without battery control | 26,120 | N/A | −4.9% | 181.2 | 37.4 | 143.8 | 67.8% | 0.0 |
| Operational reference | Normal policy, BAU | 24,900 | N/A | 0.0% | 175.5 | 29.8 | 145.7 | 73.5% | 31.8 |
| Rule-based | RBC Basic | 24,203 | N/A | 2.8% | 174.5 | 28.6 | 145.9 | 74.2% | 35.4 |
| Rule-based | RBC Smart Local | 23,232 | N/A | 6.7% | 168.5 | 22.1 | 146.4 | 76.8% | 43.7 |
| Rule-based | RBC Community | 22,858 | N/A | 8.2% | 166.1 | 19.5 | 146.6 | 77.5% | 46.9 |
| Optimisation | Rolling-horizon causal MILP, local | 21,065 | N/A | 15.4% | 159.1 | 12.4 | 146.7 | 81.2% | 48.8 |
| Optimisation | Rolling-horizon causal MILP, community | 20,294 | N/A | 18.5% | 156.5 | 9.6 | 146.9 | 83.5% | 52.4 |
| Single-agent learning | PPO, selected variant | 22,559 | [22,240, 22,910] | 9.4% | 165.1 | 18.7 | 146.4 | 78.0% | 44.1 |
| Single-agent learning | TD3, selected variant | 22,310 | [21,980, 22,670] | 10.4% | 164.4 | 17.8 | 146.6 | 78.6% | 46.0 |
| Entity-based learning | Agent Transformer PPO, selected variant | 22,186 | [21,900, 22,500] | 10.9% | 163.4 | 16.9 | 146.5 | 79.0% | 45.8 |
| Entity-based learning | Agent Transformer MATD3, selected variant | 21,937 | [21,630, 22,260] | 11.9% | 161.6 | 14.8 | 146.8 | 79.8% | 49.2 |
| Decentralised multi-agent learning | IPPO, selected variant | 22,485 | [22,100, 22,870] | 9.7% | 164.7 | 18.2 | 146.5 | 78.2% | 44.7 |
| Decentralised multi-agent learning | MADDPG, selected variant | 22,111 | [21,790, 22,450] | 11.2% | 162.8 | 16.1 | 146.7 | 79.2% | 48.0 |
| Decentralised multi-agent learning | MATD3, selected variant | 21,688 | [21,370, 22,020] | 12.9% | 159.7 | 12.7 | 147.0 | 80.4% | 52.6 |
| Decentralised multi-agent learning | MASAC, selected variant | 21,887 | [21,540, 22,230] | 12.1% | 160.8 | 14.0 | 146.8 | 80.0% | 50.1 |
| Coordinated multi-agent learning | **MAPPO, selected `REF*`** | **21,240** | **[20,970, 21,520]** | **14.7%** | **158.3** | **11.4** | **146.9** | **81.4%** | **51.0** |
| Coordinated multi-agent learning | Faithful HAPPO, selected variant | 21,514 | [21,160, 21,870] | 13.6% | 159.5 | 12.5 | 147.0 | 80.7% | 51.8 |
| Hierarchical coordination | Community Coordinator Level 1, selected variant | 21,439 | [21,100, 21,790] | 13.9% | 159.2 | 12.1 | 147.1 | 81.1% | 53.4 |
| Hierarchical coordination | Community Coordinator Level 2, selected variant | 21,339 | [21,020, 21,670] | 14.3% | 158.9 | 11.8 | 147.1 | 81.3% | 54.1 |
| Typed-interface learning | TI-PPO, selected variant | 21,588 | [21,230, 21,950] | 13.3% | 159.6 | 12.9 | 146.7 | 80.8% | 48.7 |
| Typed-interface learning | **TI-MAPPO, selected TI method** | **21,367** | **[21,060, 21,690]** | **14.2%** | **158.4** | **11.6** | **146.8** | **81.3%** | **49.4** |

#### Separate privileged-information panel

| Role | Controller | Settled cost (EUR) | Saving vs BAU | Import (MWh) | Export (MWh) | Net exchange (MWh) | Solar self-consumption | Battery throughput (MWh) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Time-limited privileged reference | Perfect-information annual community MILP, best incumbent | 19,322 | 22.4% | 153.0 | 5.7 | 147.3 | 86.1% | 56.8 |

The annual formulation uses the realised future trajectory, but reaches the
72-hour time limit with an 8.7% incumbent gap. Its feasible schedule is not a
certificate of the annual optimum and is not a deployable controller.

- This pattern would show a clear progression from BAU and rule-based control to
  learned coordination, without forcing every later method to win every metric.
- The causal community MILP would remain the strongest operational economic
  comparator, at a computational cost shown in the next panel.
- TI-MAPPO would be close to the best matched learned controller, but would not
  be presented as the nominal cost winner.
- Export and battery throughput have no universal better direction. Their value
  depends on settlement, degradation, service, and the origin of the energy.
- Net exchange remains almost constant because the annual demand and generation
  are shared across controllers; its small increase follows storage losses as
  throughput rises. The economic gains come mainly from timing and settlement,
  not from making annual demand disappear.

### 2.2 Service, load shape, equity, and computation

The Core-30 denominator contains 3,765 jointly feasible EV departures
after nine infeasible sessions are labelled and excluded, 1,892 mandatory
deferrable cycles, and 30 participants. The final table would retain the exact
numerator and denominator in the machine-readable results and annex even when a
rounded percentage is printed here.

| Controller | EV minimum feasible service ↑ | EV within tolerance ↑ | Mandatory deferrables ↑ | Daily peak ratio ↓ | All-time peak ratio ↓ | Load-factor penalty ↓ | Ramping ratio ↓ | Members benefited | Training time | Control-computation time ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Grid-only counterfactual | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Normal without battery control | 100.0% | 96.8% | 99.6% | 1.045 | 1.064 | 1.071 | 1.083 | 4/30 | N/A | 0.08 ms |
| Normal policy, BAU | 100.0% | 97.2% | 100.0% | 1.000 | 1.000 | 1.000 | 1.000 | 0/30 | N/A | 0.09 ms |
| RBC Basic | 100.0% | 97.8% | 100.0% | 0.982 | 0.994 | 0.981 | 0.961 | 20/30 | N/A | 0.12 ms |
| RBC Smart Local | 100.0% | 98.8% | 100.0% | 0.931 | 0.948 | 0.923 | 0.904 | 23/30 | N/A | 0.18 ms |
| RBC Community | 100.0% | 99.0% | 100.0% | 0.912 | 0.925 | 0.904 | 0.879 | 25/30 | N/A | 0.31 ms |
| Rolling-horizon causal MILP, local | 100.0% | 99.7% | 100.0% | 0.844 | 0.862 | 0.835 | 0.812 | 28/30 | N/A | 8.4 min per daily optimisation |
| Rolling-horizon causal MILP, community | 100.0% | 99.9% | 100.0% | 0.810 | 0.823 | 0.798 | 0.744 | 29/30 | N/A | 46.7 min per daily optimisation |
| PPO, selected variant | 100.0% | 98.9% | 100.0% | 0.914 | 0.936 | 0.921 | 0.865 | 24/30 | 2.7 h | 3.8 ms |
| TD3, selected variant | 100.0% | 99.0% | 100.0% | 0.902 | 0.920 | 0.908 | 0.851 | 24/30 | 3.2 h | 4.4 ms |
| Agent Transformer PPO, selected variant | 100.0% | 99.1% | 100.0% | 0.896 | 0.910 | 0.895 | 0.842 | 25/30 | 4.4 h | 6.7 ms |
| Agent Transformer MATD3, selected variant | 100.0% | 99.3% | 100.0% | 0.879 | 0.895 | 0.876 | 0.824 | 25/30 | 5.1 h | 7.6 ms |
| IPPO, selected variant | 100.0% | 98.8% | 100.0% | 0.908 | 0.930 | 0.916 | 0.872 | 24/30 | 4.0 h | 5.2 ms |
| MADDPG, selected variant | 100.0% | 99.1% | 100.0% | 0.895 | 0.914 | 0.899 | 0.836 | 25/30 | 4.8 h | 5.9 ms |
| MATD3, selected variant | 100.0% | 99.4% | 100.0% | 0.865 | 0.882 | 0.861 | 0.797 | 26/30 | 5.4 h | 6.4 ms |
| MASAC, selected variant | 100.0% | 99.2% | 100.0% | 0.873 | 0.892 | 0.874 | 0.809 | 25/30 | 5.8 h | 6.8 ms |
| **MAPPO, selected `REF*`** | **100.0%** | **99.7%** | **100.0%** | **0.842** | **0.856** | **0.847** | **0.768** | **28/30** | **5.6 h** | **7.4 ms** |
| Faithful HAPPO, selected variant | 100.0% | 99.5% | 100.0% | 0.856 | 0.870 | 0.854 | 0.786 | 27/30 | 6.1 h | 7.8 ms |
| Community Coordinator Level 1, selected variant | 100.0% | 99.6% | 100.0% | 0.848 | 0.862 | 0.851 | 0.774 | 27/30 | 5.9 h | 8.3 ms |
| Community Coordinator Level 2, selected variant | 100.0% | 99.7% | 100.0% | 0.844 | 0.858 | 0.849 | 0.770 | 28/30 | 6.1 h | 8.9 ms |
| TI-PPO, selected variant | 100.0% | 99.5% | 100.0% | 0.861 | 0.876 | 0.868 | 0.791 | 27/30 | 5.8 h | 9.7 ms |
| **TI-MAPPO, selected TI method** | **100.0%** | **99.7%** | **100.0%** | **0.846** | **0.859** | **0.850** | **0.772** | **28/30** | **6.7 h** | **11.4 ms** |

#### Separate privileged-information panel

| Controller | EV minimum feasible service | EV within tolerance | Mandatory deferrables | Daily peak ratio | All-time peak ratio | Load-factor penalty | Ramping ratio | Members benefited | Decision time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Perfect-information annual community MILP, best incumbent | 100.0% | 100.0% | 100.0% | 0.775 | 0.791 | 0.762 | 0.691 | 30/30 | >72 h; time limit reached |

Faithful HAPPO is shown only after its current implementation blocker has been
resolved.

- Cost alone would not determine the preferred controller. All compulsory
  services, load-shape effects, distribution of benefit, and computation would
  remain visible.
- A rolling MILP optimisation would take minutes locally and tens of minutes
  at community level, despite its stronger energy result. A scheduled action
  can subsequently be read quickly, but producing the schedule remains much
  more expensive than neural inference.
- The extra latency of the typed controller would be visible rather than hidden.
  At 11.4 ms it would still use a negligible part of a 900-second control step.
- Electrical violations and V2G are absent because neither mechanism exists in
  `CORE-30-NOMINAL`. Reporting zero would be scientifically incorrect.

### 2.3 Optimisation conditions

| Optimisation reference | Horizon | Future information | Model build p95 | Solve p95 | Total p95 | Final optimality gap | Timeout rate | Frozen fallback |
|---|---|---|---:|---:|---:|---:|---:|---|
| Rolling causal MILP, local | Seven-day look-ahead; one-day commit | Registered causal forecasts only | 1.2 min | 7.2 min | 8.4 min | 0.8% | 2.3% | RBC Smart Local |
| Rolling causal MILP, community | Seven-day look-ahead; one-day commit | Registered causal forecasts only | 4.1 min | 42.6 min | 46.7 min | 3.7% | 18.9% | RBC Community |
| Perfect-information annual community MILP | Full year; 35,040 steps | Realised annual trajectory | 1.6 h | >70.4 h | >72 h | 8.7% incumbent gap | 100.0% | N/A |

- The rolling methods would be causal operational comparators, not oracles.
- Model construction, solve time, timeout, incumbent quality, and fallback would
  be reported alongside control quality.
- The rolling times refer to one daily re-optimisation. Actions are still
  scheduled at 15-minute resolution during the committed day.
- The annual monolithic formulation reaches the 72-hour time limit. Its row
  reports the best feasible incumbent and solver bound, not a certified annual
  optimum. It therefore remains a privileged-information reference rather
  than an exact oracle result.

### 2.4 Compact controller-development ladder

This panel illustrates one cumulative development branch. It would explain why
the final variants exist without moving every reward weight, replay parameter,
and rejected checkpoint into the thesis body. All rows belong to
`CORE-15-STRIPPED` development with DEV3; none is a confirmation result.

| Development stage | Main mechanism introduced | Saving vs BAU | EV minimum service | Daily peak ratio | Steps to all screening gates | Training time | Failed DEV3 seeds |
|---|---|---:|---:|---:|---:|---:|---:|
| A | Flat direct PPO, cost-only reward, uniform replay | 5.2% | 98.2% | 1.045 | Not reached | 1.4 h | 1/3 |
| B | Residual action over local RBC and local feasibility preprocessing | 7.6% | 99.5% | 0.982 | 97,152 | 1.5 h | 0/3 |
| C | Service-aware reward, hard service penalty, and dense urgency | 8.9% | 100.0% | 0.955 | 72,864 | 1.6 h | 0/3 |
| D | Scorecard-aligned cost, peak, ramping, solar, and normalisation terms | 9.8% | 100.0% | 0.905 | 48,576 | 1.8 h | 0/3 |
| E | Warm start, event-prioritised replay, and n-step returns | 10.0% | 100.0% | 0.901 | 24,288 | 1.9 h | 0/3 |
| F | Entity-based Transformer representation | 11.4% | 100.0% | 0.887 | 48,576 | 2.8 h | 0/3 |
| G | Coordinated training with a central value function | 14.2% | 100.0% | 0.835 | 48,576 | 3.5 h | 0/3 |
| H | Typed entities, versioned ports, variable-set critic, and dependency closure | 13.9% | 100.0% | 0.840 | 48,576 | 4.1 h | 0/3 |

- The table would connect reward, replay, input representation, coordination,
  and interface engineering to the historical controller progression.
- Stage H would not be described as an economic improvement over Stage G. It
  would add the semantics later tested under reconfiguration and faults.
- Only matched one-factor or registered grouped ablations could support causal
  claims about an individual mechanism. Those fuller tables would remain in the
  annex.

## 3. Cross-dataset finalist ladder

Values are cost savings relative to the corresponding BAU. A `Pass` concerns
only the gates available on that surface. The integrated surfaces are opened
after controller selection has been frozen.

| Frozen finalist | Core-15 Stripped | Core-30 Nominal | Core-30 Combined | Premium-100 Clean | Premium-100 AllIn |
|---|---:|---:|---:|---:|---:|
| RBC Community with its DR-aware extension where requests exist | 7.5% · Pass | 8.2% · Pass | 7.4% · Pass | 7.9% · Pass | 6.9% · Fail DR |
| Rolling causal community MILP | 17.1% · Pass | 18.5% · Pass | 16.9% · Pass | 18.6% · Pass | 15.3% · Pass¹ |
| **MAPPO, selected `REF*`** | **12.6% · Pass** | **14.7% · Pass** | **12.7% · Pass** | **14.6% · Pass** | **10.9% · Fail service and DR** |
| **TI-MAPPO, selected TI method** | **11.9% · Pass** | **14.2% · Pass** | **12.9% · Pass** | **14.0% · Pass** | **13.3% · Pass** |

¹ The MILP result includes a 1.3% solver-timeout rate resolved by the
frozen safe fallback.

- This pattern would keep the nominal result modest. TI-MAPPO would trail the
  matched learned reference slightly on the clean common surfaces.
- Its relative value would become clearer when known topology changes and faults
  occur together.
- The MILP would remain economically stronger, while retaining its very different
  information, optimisation, and runtime profile.
- `PREMIUM-100-ALLIN` would show an integrated envelope only. It could not isolate
  which fault or event caused an observed difference.

## 4. Primary non-inferiority claim

This is the claim-carrying comparison. The one-sided confidence limit, rather
than the point estimate alone, is compared with the frozen 2% margin. After the
CONF5 omnibus comparison, the selected pair receives the five additional
HEAD_EXTRA5 seeds, forming HEAD10. This table therefore uses ten independently
trained policies per method and four repeated held-out seasonal windows per
policy; the training seed is the experimental unit.

| Quantity | MAPPO, selected `REF*` | Selected TI-MAPPO | Registered interpretation |
|---|---:|---:|---|
| Mean held-out cost | EUR 1.9397/member-day | EUR 1.9513/member-day | Comparator is slightly cheaper |
| Annual-equivalent community cost | EUR 21,240 | EUR 21,367 | Display quantity, not a full-year replay |
| Mean paired cost difference | Reference | +0.60% | TI is numerically worse |
| One-sided 95% upper confidence limit | Reference | +1.42% | Below the +2.00% margin |
| Practical non-inferiority margin | Reference | +EUR 0.0388/member-day | Approximately EUR 425/community-year or EUR 14.2/member-year |
| Daily peak difference | Reference | +0.48% | Within the +5% guardrail |
| Ramping difference | Reference | +0.52% | Within the +5% guardrail |
| Solar self-consumption difference | Reference | −0.1 percentage points | Within the −2 pp guardrail |
| Jointly feasible EV departures above minimum | 100.0% | 100.0% | Pass |
| Mandatory deferrables completed | 100.0% | 100.0% | Pass |
| Cost-endpoint conclusion | Reference | Nominally non-inferior | Superiority is not claimed |
| Overall system acceptance | Pass | Pass | Cost result plus all predeclared service and safety gates |

The displayed point estimate is internally consistent because
`(21,367 − 21,240) / 21,240 = 0.598%`.

- If obtained under the frozen protocol, this result would support nominal
  non-inferiority, not nominal superiority.
- The structural claim would still require the reconfiguration, safety, health,
  scale, and deployment evidence shown next.
- A failed service or safety gate would block overall system acceptance. The
  statistical conclusion for the cost endpoint would still be reported
  separately rather than erased.

## 5. Structural response to changing interfaces

### 5.1 Matched nominal control and resource cost

The structural baselines use matched observations, action authority, reward,
protection, interaction budget, and confirmation windows. Actor, training-only
critic, and exported-bundle parameters are reported separately.

| Controller | Cost difference vs `REF*` | One-sided upper limit | EV minimum service | Daily peak ratio | Actor parameters | Critic parameters | Exported parameters | Peak memory | Inference p95 | Nominal control gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MAPPO `REF*` | Reference | Reference | 100.0% | 0.842 | 1.42 M | 1.58 M | 1.42 M | 0.62 GB | 7.4 ms | Pass |
| Padded-and-masked MAPPO | +0.30% | +1.30% | 100.0% | 0.845 | 2.74 M | 1.65 M | 2.74 M | 0.81 GB | 9.3 ms | Pass |
| Untyped set MAPPO | +0.50% | +1.50% | 100.0% | 0.847 | 1.78 M | 1.12 M | 1.78 M | 0.64 GB | 10.2 ms | Pass |
| Agent Transformer MATD3 | +3.28% | +4.40% | 100.0% | 0.879 | 2.40 M | 1.70 M | 2.40 M | 0.76 GB | 7.6 ms | Pass |
| **Selected TI-MAPPO** | **+0.60%** | **+1.42%** | **100.0%** | **0.846** | **1.86 M** | **1.14 M** | **1.86 M** | **0.68 GB** | **11.4 ms** | **Pass** |

- Padded and untyped set controllers would match the typed controller closely in
  nominal control. The thesis could not attribute a nominal cost advantage to
  semantic typing.
- The typed claim would therefore depend on the event and lifecycle evidence,
  while accepting modest compiler, memory, or inference overhead.
- A nominal control gate is not the 2% cost non-inferiority test. The latter is
  evaluated only for the frozen primary pair in Section 4.

### 5.2 Known-type reconfiguration and unknown-type rejection

The padded and untyped MAPPO rows are required matched final baselines; in the
current implementation plan they remain blockers rather than completed methods.
TI-PPO is retained separately to measure the additional value of centralised
training within the typed family.

The fourteen events are the known-type topology changes declared for the
Core-30 dynamic scenario. The unknown-type test is separate and examines safe
rejection, not zero-shot generalisation.

| Controller | Known-type events handled | Wrong bindings | Invalid ports | Recovery p95 | Neural parameter change | Checkpoint swap or retraining | Unknown semantic type | Trace completeness |
|---|---:|---:|---:|---:|---:|---|---|---:|
| MAPPO `REF*`, fixed-layout boundary | 8/14 | 0 | 27 | Not recovered within 24 h for failed events | +18.4% if resized | Required | Shape failure after compilation | 47% |
| Padded-and-masked MAPPO | 13/14 | 0 | 6 | 1.25 h | 0.0% | No | Accepted only as an untyped spare slot | 64% |
| Untyped set MAPPO | 11/14 | 8 | 12 | 2.00 h | 0.0% | No | Accepted as a generic entity | 58% |
| Agent Transformer MATD3 | 13/14 | 0 | 5 | 1.00 h | 0.0% | No | No versioned compatibility decision | 71% |
| TI-PPO | 14/14 | 0 | 0 | 0.50 h | 0.0% | No | Rejected before action generation | 100% |
| **TI-MAPPO, selected TI method** | **14/14** | **0** | **0** | **0.25 h** | **0.0%** | **No** | **Rejected before action generation** | **100%** |

- Padding and set processing would explain part of the ability to accept a
  varying number of entities. The thesis should acknowledge that result.
- The evidence specific to the typed interface would be correct binding,
  compatibility checking, dependency-aware recompilation, and a complete trace
  from semantic port to applied action.
- Rejecting an unknown type would be a safety property. It would not demonstrate
  that the policy understands unseen semantics.
- If the padded baseline matched TI-MARL in control, resources, and adaptation,
  the contribution would need to be narrowed to semantic validation,
  traceability, and integration.

Trace completeness requires the policy identifier, interface-schema version,
entity identifier, semantic port, requested value, projected value, applied
value, and health or fallback state for every issued action. Recovery is the
first point followed by four consecutive steps in which bindings are valid and
all applicable service gates are restored. Events that fail this criterion
inside the 24-hour observation window are treated as right-censored.

At the 15-minute resolution used here, a topology recovery of 0.25 h means that
the first control decision after the declared change uses the new valid
bindings. It does not denote the duration of retraining, since no retraining or
checkpoint replacement is performed. The following four valid steps are used
to confirm that this was a sustained recovery rather than a single valid
decision.

## 6. Local physical feasibility

The first three rows replay the same actor outputs. The Simulator remains the
last physical enforcement layer in every row, allowing requested, projected,
and applied actions to be distinguished.

| Protection mechanism | Maximum raw exceedance (kW) | Maximum after common protection (kW) | Raw requested violation energy (kWh) | Residual after common protection (kWh) | Applied violation energy (kWh) | Common intervention rate | Simulator limiter rate | EV minimum service | Mandatory deferrables |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No common projector | 7.84 | 7.84 | 32.6 | 32.6 | 0.000 | 0.0% | 6.8% | 100.0% | 100.0% |
| Independent per-port clipping | 7.84 | 3.26 | 32.6 | 12.4 | 0.000 | 9.7% | 3.1% | 100.0% | 100.0% |
| Joint local projector | 7.84 | <1×10⁻⁵ | 32.6 | 0.0 | 0.000 | 13.8% | 0.0% | 100.0% | 100.0% |
| Reward-only safety | 1.92 | 1.92 | 4.7 | 4.7 | 0.000 | 0.0% | 1.2% | 100.0% | 100.0% |
| Safe local RBC | 0.0 | 0.0 | 0.0 | 0.0 | 0.000 | 0.0% | 0.0% | 100.0% | 100.0% |

- Zero applied violation in the first row would not make the raw policy safe. It
  would show that the Simulator corrected 6.8% of its decisions.
- Independent clipping would reduce exceedance but could not enforce a joint
  installation or phase constraint.
- Reward shaping would reduce unsafe requests without guaranteeing feasibility.
- The joint projector would be the only learned-system row that reaches local
  feasibility before Simulator correction while preserving service.
- The claim would remain limited to installation-total and per-phase active-power
  constraints represented by the Simulator.

## 7. Health-aware control

The table aggregates 48 predeclared fault conditions, each paired with its
seasonal clean twin. Service is adjusted for physical availability, so an
unavailable charger is not treated as a policy failure when the requested
service is no longer jointly feasible. The scenario explicitly supplies the
registered health event. The latency below measures ingestion and propagation
through the control interface, not anomaly detection inferred from raw
telemetry. Each fault has a ground-truth set of affected semantic ports against
which closure precision and recall are calculated.

Health-event propagation and recovery measure different intervals. A median
propagation value of 0.25 h means that the declared health change reaches the
compiled interface by the following 15-minute decision. Recovery begins when
the fault ends. A p95 value of 0.75 h means that 95% of eligible fault episodes
return to sustained valid control and service within 45 minutes. It does not
mean that the physical fault itself is repaired in 45 minutes.

| Method | Fault-minus-clean cost, 95% CI | Availability-adjusted EV service | Exact closure cases | Closure precision / recall | Health-event propagation, median | Recovery p95 | Healthy-asset impact | Fallback rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **TI-MAPPO with dependency closure** | **+2.8% [2.1, 3.6]** | **100.0%** | **48/48** | **100.0% / 100.0%** | **0.25 h** | **0.75 h** | **4.2 kWh** | **0.8%** |
| TI-MAPPO, health feature only | +7.9% [6.6, 9.3] | 99.6% | 15/48 | 68.0% / 73.0% | 0.25 h | 2.25 h | 16.8 kWh | 1.8% |
| TI-MAPPO, mask only | +5.6% [4.5, 6.8] | 99.8% | 33/48 | 82.0% / 88.0% | 0.25 h | 1.50 h | 10.1 kWh | 1.2% |
| Matched non-typed MAPPO `REF*` | +10.4% [8.7, 12.2] | 98.9% | 4/48 | 26.0% / 35.0% | 0.75 h | 4.00 h | 25.7 kWh | 4.6% |
| Safe local RBC | +6.7% | 100.0% | N/A | N/A | N/A | 3.25 h | 30.4 kWh | 6.1% |

- A health feature alone would not reproduce dependency-aware action closure.
- The safe RBC would preserve service, but would rely more often on fallback and
  would not expose a dependency graph against which semantic closure could be
  scored.
- TI-MARL would still incur a cost under faults. A credible robustness result is
  smaller degradation and faster recovery, not immunity to disturbance.
- The conclusion would remain limited to the four registered fault families,
  their severities, and the tested dependency graph.

## 8. Evaluated scale envelope

### 8.1 One frozen TI-MAPPO checkpoint across population sizes

The checkpoint is trained on the 30-member Core-30 surface and frozen before
all six scale evaluations. Savings use the BAU of each proportional Premium
view, avoiding a comparison of raw costs across different population mixes.

| Active members | TI saving vs size-specific BAU | TI cost difference vs matched `REF*` | Actor parameters | Training critic parameters | Exported parameters | Peak memory | Compiler p95 | Inference p95 | Wrong bindings | Gates |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 13.8% | +0.7% | 1.86 M | 1.14 M | 1.86 M | 0.38 GB | 1.9 ms | 6.3 ms | 0 | Pass |
| 20 | 14.1% | +0.9% | 1.86 M | 1.14 M | 1.86 M | 0.46 GB | 3.2 ms | 8.7 ms | 0 | Pass |
| 40 | 13.9% | +1.1% | 1.86 M | 1.14 M | 1.86 M | 0.61 GB | 5.9 ms | 12.9 ms | 0 | Pass |
| 60 | 14.3% | +1.0% | 1.86 M | 1.14 M | 1.86 M | 0.75 GB | 8.4 ms | 15.2 ms | 0 | Pass |
| 80 | 13.7% | +1.4% | 1.86 M | 1.14 M | 1.86 M | 0.91 GB | 11.2 ms | 17.5 ms | 0 | Pass |
| 100 | 14.0% | +1.5% | 1.86 M | 1.14 M | 1.86 M | 1.08 GB | 14.8 ms | 19.6 ms | 0 | Pass |

### 8.2 Matched-size training comparison

Parameter triplets below are `actor / training-only critic / exported bundle`.

| Members | MAPPO `REF*` saving | Padded MAPPO saving | TI-MAPPO saving | TI cost difference vs `REF*` | TI one-sided upper limit | `REF*` parameter triplet | Padded MAPPO parameter triplet | TI parameter triplet |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 14.6% | 13.2% | 13.8% | +0.94% | +1.35% | 1.42 / 0.78 / 1.42 M | 2.20 / 1.20 / 2.20 M | 1.86 / 1.14 / 1.86 M |
| 60 | 15.1% | 13.6% | 14.2% | +1.06% | +1.66% | 1.42 / 4.28 / 1.42 M | 4.36 / 3.00 / 4.36 M | 1.86 / 1.14 / 1.86 M |
| 100 | 15.0% | 13.4% | 14.1% | +1.06% | +1.88% | 1.42 / 7.88 / 1.42 M | 6.52 / 4.80 / 6.52 M | 1.86 / 1.14 / 1.86 M |

### 8.3 Multiple chargers controlled by one member agent

| Chargers in one member | Normalised cost | EV minimum service | Wrong bindings | Invalid ports | Inference p95 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.000 | 100.0% | 0 | 0 | 5.0 ms |
| 2 | 1.002 | 100.0% | 0 | 0 | 5.6 ms |
| 4 | 0.999 | 100.0% | 0 | 0 | 6.8 ms |
| 6 | 1.004 | 100.0% | 0 | 0 | 8.4 ms |

The aggregate charging power and flexible energy are held constant in the
multiple-charger microbenchmark.

- A constant typed actor, critic, and exported-bundle parameter count would
  support architectural compatibility across the evaluated populations.
- Memory, compilation time, and inference latency would still grow because the
  controller processes more entities. The thesis should show that cost rather
  than hide it behind a constant parameter count.
- The matched-size panel would test control quality separately from running one
  frozen checkpoint at several cardinalities.
- The evidence would support an evaluated envelope from 10 to 100 members and
  from one to six chargers per agent. It would not establish unlimited scale.

### 8.4 Computational and storage footprint

The footprint comparison separates the number of neural parameters from the
storage required to retain every deployable model needed across the evaluated
population range. A fixed-layout actor assigns meaning to particular input and
output positions. Changing the supported layout can therefore require wider
layers or a separate network. The size-specific strategy below retains six
policies, one for each evaluated population. Padded MAPPO uses one network but
allocates its input and output space for the maximum catalogue, carrying unused
positions at smaller scales.

Set-based and Transformer strategies instead apply shared encoders to a
variable collection of active entities. This permits a fixed neural parameter
count, although memory and execution time still increase with the number of
tokens and attention operations. TI-MAPPO combines this property with shared
type-conditioned encoders and action-group decoders, a variable-set critic,
semantic bindings and compatibility checks. Its constant parameter count is
thus not explained by the Transformer alone, nor does it establish constant
computational cost.

Training time is deliberately absent from this proposed table. The matched
learned methods will receive the same interaction and tuning budgets, but
wall-clock duration may differ because of attention, interface compilation,
centralised critic updates and hardware utilisation. It will be reported only
after measurement, together with failed trials, CPU and GPU hours. The table
also separates policy storage from trace storage. Typed traces are larger
because they retain the semantic evidence needed to reconstruct a decision.

| Controller strategy | Models required for 10–100 members | Largest deployed actor | Deployed policy storage | Peak runtime memory at 100 members | End-to-end inference p95 at 100 members | Compressed annual trace | Trace completeness |
|---|---:|---:|---:|---:|---:|---:|---:|
| Size-specific fixed-layout MAPPO | 6 | 1.42 M parameters per actor | 38.4 MB | 1.46 GB | 18.2 ms | 0.58 GB | 47% |
| Padded-and-masked MAPPO | 1 | 6.52 M parameters | 28.7 MB | 1.94 GB | 34.8 ms | 0.92 GB | 64% |
| Agent Transformer MATD3 | 1 | 2.40 M parameters | 11.2 MB | 1.32 GB | 26.4 ms | 1.02 GB | 71% |
| **TI-MAPPO** | **1** | **1.86 M parameters** | **9.8 MB** | **1.08 GB** | **34.4 ms, including 14.8 ms compilation** | **1.48 GB** | **100%** |

- A fixed-layout portfolio retains one actor for every evaluated population
  layout. Padded MAPPO avoids that portfolio but carries its maximum catalogue
  in every deployment.
- TI-MAPPO retains one typed actor bundle and processes only the active
  interface. Its 34.4 ms end-to-end p95 uses 0.0038% of a 900-second control
  interval.
- The table makes no claim that TI-MAPPO trains faster. Its training cost must
  be measured under the common interaction budget and reported separately from
  the architectural advantage of retaining one deployable actor.
- Complete typed traces use more disk space than positional action logs. Their
  additional records include entity bindings, health, bounds, invalid ports,
  requested actions, projected actions and applied commands.

## 9. Isolated bidirectional vehicle charging

The table compares matched `V2G-OFF` and `V2G-ON` twins. Negative cost
differences mean that the ON twin is cheaper. The degradation-adjusted column
uses a degradation cost of EUR 0.05 per discharged kWh. The learned rows use
five paired training seeds and 1,559 eligible sessions; uncertainty is
aggregated first by policy.

| Controller | Discharged energy (kWh) | Sessions used | Used locally (kWh) | Residual export (kWh) | Settled cost difference | 95% CI | Difference after degradation | EV service | Reserve breaches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BAU invariance control | 0 | 0 | 0 | 0 | EUR 0 | N/A | EUR 0 | 100.0% | 0 |
| RBC Smart Local | 620 | 185 | 595 | 25 | −EUR 76 | N/A | −EUR 45 | 100.0% | 0 |
| RBC Community | 780 | 231 | 758 | 22 | −EUR 124 | N/A | −EUR 85 | 100.0% | 0 |
| Rolling causal community MILP | 1,010 | 298 | 998 | 12 | −EUR 196 | N/A | −EUR 145.5 | 100.0% | 0 |
| Selected non-typed MAPPO `REF*` | 920 | 276 | 894 | 26 | −EUR 163 | [−187, −141] | −EUR 117 | 100.0% | 0 |
| **Selected TI-MAPPO** | **900** | **270** | **884** | **16** | **−EUR 160** | **[−181, −139]** | **−EUR 115** | **100.0%** | **0** |

- This pattern would show economic value after preserving departure service and
  the physical reserve, rather than rewarding discharge by itself.
- Most discharged energy would offset local demand instead of becoming
  unremunerated residual export.
- The causal MILP would find the greatest value, while the selected TI and
  non-typed learners would remain close.
- At EUR 0.10/kWh of discharged energy, the degradation-adjusted
  benefits would fall to EUR 14, 46, 95, 71, and 70 for the five active methods.
  The V2G conclusion would therefore remain degradation-sensitive.

## 10. Isolated demand response

All methods receive the same aggregate request of 416.0 kWh across sixteen
confirmation events. BAU and RBC Smart are retained as non-responsive controls,
not as demand-response methods. Learned estimates use five training seeds, with
event direction and event identity retained as repeated measurements.

| Controller | Delivered energy | Delivered/requested ↑ | 95% CI | Tracking RMSE ↓ | Sustained compliance ↑ | Median time to 90% ↓ | 24 h rebound ↓ | Penalised shortfall / penalty ↓ | EV service | Decision p95 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BAU, non-responsive | 25.0 kWh | 6.0% | N/A | 15.2 kW | 0.0% | Not reached | 3.1 kWh | 372.4 kWh / EUR 372.4 | 100.0% | 0.08 ms |
| RBC Smart, non-responsive | 45.8 kWh | 11.0% | N/A | 14.1 kW | 3.1% | Not reached | 5.7 kWh | 350.1 kWh / EUR 350.1 | 100.0% | 0.18 ms |
| Demand-response-aware RBC | 358.2 kWh | 86.1% | N/A | 2.65 kW | 82.4% | 30 min | 46.2 kWh | 21.6 kWh / EUR 21.6 | 100.0% | 0.42 ms |
| Rolling causal community MILP | 407.7 kWh | 98.0% | N/A | 0.42 kW | 97.8% | 0 min | 18.4 kWh | 2.4 kWh / EUR 2.4 | 100.0% | 46.7 min per daily optimisation |
| Selected non-typed MAPPO `REF*` | 393.1 kWh | 94.5% | [92.3, 96.1] | 1.08 kW | 92.1% | 15 min | 27.6 kWh | 8.7 kWh / EUR 8.7 | 100.0% | 7.4 ms |
| **Selected TI-MAPPO** | **397.0 kWh** | **95.4%** | **[94.0, 96.6]** | **0.91 kW** | **94.0%** | **15 min** | **23.1 kWh** | **7.2 kWh / EUR 7.2** | **100.0%** | **11.4 ms** |

The shortfall penalty is calculated event by event after the registered
tolerance band. It is therefore not the raw aggregate energy shortfall
multiplied directly by one euro.

### 10.1 Direction and matched-twin checks

| Controller | UP delivered/requested | DOWN delivered/requested | Incremental ON-minus-OFF response | Maximum pre-activation trajectory difference |
|---|---:|---:|---:|---:|
| BAU, non-responsive | 4.0% | 8.0% | 0.0% | 0.000 kW |
| RBC Smart, non-responsive | 10.0% | 12.0% | 0.0% | 0.000 kW |
| Demand-response-aware RBC | 83.4% | 88.8% | 75.2% | 0.000 kW |
| Rolling causal community MILP | 97.0% | 99.0% | 91.6% | 0.000 kW |
| Selected non-typed MAPPO `REF*` | 92.5% | 96.5% | 86.3% | 0.000 kW |
| **Selected TI-MAPPO** | **94.2%** | **96.6%** | **88.7%** | **0.000 kW** |

- The non-responsive controls would show that incidental load movement is not a
  demand-response capability.
- The MILP would track best, but at much higher decision time.
- TI-MAPPO would improve tracking and rebound over its matched learned reference,
  without being made artificially better than the optimisation comparator.
- Separate UP and DOWN results would prevent aggregate performance from hiding a
  direction-specific failure, while zero pre-activation difference would verify
  the causal ON/OFF pairing.
- The conclusion would remain limited to the registered event directions,
  intensities, construction method, and experimental remuneration values.

## 11. Integrated stress envelope

Only frozen finalists enter the combined scenarios. Cost savings use the BAU of
the same surface. `PREMIUM-100-ALLIN` combines mechanisms and therefore cannot
be used to attribute an effect to one specific fault. In this table, the
integrated gate requires at least 85% aggregate DR delivery in addition to the
registered service and applied-safety conditions. These are full-year frozen
integration replays. Where the 2023 surface contributed to learning, they are
transductive and are not used as held-out estimates of generalisation.

### 11.1 Control and service

| Surface | Frozen system | Saving vs surface BAU | Availability-adjusted EV service | Mandatory deferrables | DR delivered/requested | Overall gate |
|---|---|---:|---:|---:|---:|---|
| Core-30 Combined | Demand-response-aware RBC Community | 7.4% | 100.0% | 100.0% | 86.0% | Pass |
| Core-30 Combined | Rolling causal community MILP | 16.9% | 100.0% | 100.0% | 97.0% | Pass with fallback |
| Core-30 Combined | MAPPO `REF*` | 12.7% | 100.0% | 100.0% | 92.0% | Pass |
| Core-30 Combined | **TI-MAPPO** | **12.9%** | **100.0%** | **100.0%** | **94.0%** | **Pass** |
| Premium-100 Clean | Demand-response-aware RBC Community | 7.9% | 100.0% | 100.0% | 86.0% | Pass |
| Premium-100 Clean | Rolling causal community MILP | 18.6% | 100.0% | 100.0% | 98.0% | Pass with fallback |
| Premium-100 Clean | MAPPO `REF*` | 14.6% | 100.0% | 100.0% | 93.0% | Pass |
| Premium-100 Clean | **TI-MAPPO** | **14.0%** | **100.0%** | **100.0%** | **95.0%** | **Pass** |
| Premium-100 AllIn | Demand-response-aware RBC Community | 6.9% | 100.0% | 100.0% | 78.0% | Fail DR |
| Premium-100 AllIn | Rolling causal community MILP | 15.3% | 100.0% | 100.0% | 91.0% | Pass with fallback |
| Premium-100 AllIn | MAPPO `REF*` | 10.9% | 98.9% | 99.6% | 75.0% | Fail service and DR |
| Premium-100 AllIn | **TI-MAPPO** | **13.3%** | **100.0%** | **100.0%** | **88.0%** | **Pass** |

### 11.2 Protection, recovery, and execution failures

| Surface | Frozen system | Raw requested violation energy | Residual after common protection | Simulator limiter rate | Applied violation energy | Recovery p95 | Catastrophic evaluation seeds | Solver timeouts |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Core-30 Combined | Demand-response-aware RBC Community | 1.2 kWh | 0.0 kWh | 0.0% | 0.000 kWh | 2.00 h | N/A | N/A |
| Core-30 Combined | Rolling causal community MILP | 0.3 kWh | 0.0 kWh | 0.0% | 0.000 kWh | 0.50 h | N/A | 1.1% |
| Core-30 Combined | MAPPO `REF*` | 9.8 kWh | 2.1 kWh | 0.8% | 0.000 kWh | 2.50 h | 0/5 | N/A |
| Core-30 Combined | **TI-MAPPO** | **8.4 kWh** | **0.0 kWh** | **0.0%** | **0.000 kWh** | **0.75 h** | **0/5** | **N/A** |
| Premium-100 Clean | Demand-response-aware RBC Community | 2.2 kWh | 0.0 kWh | 0.0% | 0.000 kWh | N/A | N/A | N/A |
| Premium-100 Clean | Rolling causal community MILP | 0.6 kWh | 0.0 kWh | 0.0% | 0.000 kWh | N/A | N/A | 0.8% |
| Premium-100 Clean | MAPPO `REF*` | 14.5 kWh | 3.8 kWh | 1.2% | 0.000 kWh | N/A | 0/5 | N/A |
| Premium-100 Clean | **TI-MAPPO** | **11.2 kWh** | **0.0 kWh** | **0.0%** | **0.000 kWh** | **N/A** | **0/5** | **N/A** |
| Premium-100 AllIn | Demand-response-aware RBC Community | 4.6 kWh | 0.0 kWh | 0.0% | 0.000 kWh | 3.25 h | N/A | N/A |
| Premium-100 AllIn | Rolling causal community MILP | 1.7 kWh | 0.0 kWh | 0.0% | 0.000 kWh | 1.50 h | N/A | 1.3% |
| Premium-100 AllIn | MAPPO `REF*` | 28.4 kWh | 7.2 kWh | 2.6% | 0.000 kWh | 4.00 h | 1/5 | N/A |
| Premium-100 AllIn | **TI-MAPPO** | **22.9 kWh** | **0.0 kWh** | **0.0%** | **0.000 kWh** | **1.00 h** | **0/5** | **N/A** |

- The clean Premium result would keep the matched reference slightly ahead in
  cost, avoiding a claim that TI-MARL wins by construction.
- Under coincident changes and faults, TI-MARL would preserve service and recover
  faster, while all methods would lose some economic performance.
- The MILP would remain economically strongest but would depend on timeouts and a
  frozen fallback.
- The AllIn table would confirm integration only. Fault-specific claims would
  continue to rely on the paired health and clean-twin experiments.
- Applied violations would be zero for every row because the Simulator remains
  the final limiter. Requested and post-protection values reveal which systems
  depend on that correction.

## 12. Training-to-inference lifecycle

Training portability and inference parity are separate experiments. The first
panel benchmarks matched full-budget training jobs. The second reconstructs one
identical frozen actor bundle on each eligible inference target.

### 12.1 Heterogeneous training benchmark

| Training host | Execution profile | Queue | Environment setup | Full-budget training | Export after training | Failed attempts |
|---|---|---:|---:|---:|---:|---:|
| Local PC | CPU | 0 min | 1.1 min | 46.8 h | 12.1 s | 0/5 |
| Server | GPU | 12.0 min | 2.4 min | 6.7 h | 9.4 s | 0/5 |
| Deucalion | GPU | 24.6 min | 8.2 min | 3.4 h | 15.8 s | 1/6 |
| Union | GPU | 7.8 min | 5.3 min | 4.1 h | 13.7 s | 0/5 |

### 12.2 Identical-bundle inference parity

The parity corpus contains 10,000 observations stratified across the
four held-out seasonal windows and the registered topology and health events.
All targets use float32 inference. `Pass` requires an absolute difference no
greater than 1×10⁻⁵ on every normalised action and an identical semantic port
binding for every observation.

| Inference target | Bundle hash verified | Bundle rebuilt | Cold load | Maximum action difference | Warm inference p95 | 900 s deadline use | Parity result |
|---|---|---|---:|---:|---:|---:|---|
| Local PC, CPU reference | Yes | Yes | 1.6 s | 0 | 22.6 ms | 0.0025% | Pass |
| Server, GPU | Yes | Yes | 2.8 s | 6.2×10⁻⁷ | 11.4 ms | 0.0013% | Pass |
| Deucalion, GPU | Yes | Yes | 5.6 s | 1.1×10⁻⁶ | 8.7 ms | 0.0010% | Pass |
| Union, GPU | Yes | Yes | 4.3 s | 8.6×10⁻⁷ | 9.6 ms | 0.0011% | Pass |
| Eligible edge CPU | Yes | Yes | 2.2 s | 2.4×10⁻⁶ | 31.8 ms | 0.0035% | Pass |

### 12.3 Negative lifecycle cases

| Test | Expected response | Unsafe action emitted | Outcome |
|---|---|---:|---|
| Incompatible schema version | Reject before inference | 0 | Pass |
| Corrupted typed actor bundle | Reject during hash validation | 0 | Pass |
| Missing checkpoint component | Reject incomplete artefact | 0 | Pass |
| Target without required capability | Reject scheduling request | 0 | Pass |
| Runtime timeout | Apply safe fallback within one control step | 0 | Pass |
| Candidate parity above tolerance | Block promotion and retain previous bundle | 0 | Pass |
| Failed deployment after promotion | Roll back to last validated bundle | 0 | Pass |

### 12.4 Lifecycle comparison with non-typed artefacts

| Controller package | Eligible targets reconstructed | Maximum action difference | Semantic incompatibilities rejected | Automatic rollback tests | Unsafe learned actions emitted | Lifecycle gate |
|---|---:|---:|---:|---:|---:|---|
| Fixed-layout MAPPO bundle | 5/5 | 1.8×10⁻⁶ | 2/7 | 1/5 | 3 | Fail |
| Padded-and-masked MAPPO bundle | 5/5 | 2.0×10⁻⁶ | 3/7 | 2/5 | 2 | Fail |
| Agent Transformer MATD3 bundle | 5/5 | 2.2×10⁻⁶ | 4/7 | 2/5 | 1 | Fail |
| **TI-MAPPO typed bundle** | **5/5** | **2.4×10⁻⁶** | **7/7** | **5/5** | **0** | **Pass** |

Numerical portability alone does not establish deployability. All four packages
can reproduce their numerical outputs on eligible hardware. Only the typed
bundle carries the interface, compatibility, health, fallback and rollback
contracts required to reject every registered invalid case before a learned
action reaches the execution path.

- This pattern would show that matched training time changes with hardware, then
  test the same exported actor independently for inference parity.
- Millisecond inference would be only one part of deployability. Reconstruction,
  compatibility rejection, fallback, and rollback would be equally important.
- Queue time would not be disguised as training time, particularly on shared
  infrastructure.
- The final evidence requires the generic inference service to reproduce this
  complete reconstruction, rejection and rollback campaign end to end.

## 13. Demonstrator evidence levels

This table will be completed from demonstrator logs, asset inventories,
protocols, and deliverables.

| Evidence level | OPEVA | DemFlex | Claim permitted after verification |
|---|---|---|---|
| Real operational data | Source material available; interval and coverage to verify | Inventory pending | The evaluation used data collected from a real installation |
| Software and system integration | Demonstrator integration described; exact boundary to catalogue | Inventory pending | Named components exchanged data under the documented configuration |
| Shadow or testbed execution | To verify from logs and test protocol | Inventory pending | Decisions were executed without controlling the live physical process |
| Closed-loop physical action | To verify; no value inserted here | Inventory pending | The controller applied actions to the represented physical assets |
| Duration, participants, and assets | To verify against the final inventory | Inventory pending | Conclusions are bounded by the observed installation and period |
| Fault, fallback, and recovery evidence | To verify from timestamped events | Inventory pending | A robustness or safety mechanism acted under a documented disturbance |

- Integration, shadow execution, and closed-loop physical control would be kept
  as distinct evidence levels.
- Simulation datasets would not be presented as extensions of either
  demonstrator.
- Any unverified cell would restrict the final real-world claim rather than being
  filled from memory or inferred from a project description.

## 14. Evidence supporting the four research dimensions

This final table would be a synthesis of earlier evidence. It would introduce
no new measurements.

| Research dimension | Mechanism examined | Evidence pattern | Conclusion that could be supported | Current evidence state | Scientific boundary |
|---|---|---|---|---|---|
| Scalability | Shared typed policy, variable entity sets, interface compilation | Constant 1.86 M exported parameters; applicable gates pass from 10 to 100 members and one to six chargers; latency rises from 6.3 to 19.6 ms | The same policy structure can control the evaluated heterogeneous populations without resizing the actor | Pending campaign | Evaluated envelope only; no claim beyond 100 members or for unknown semantic types |
| Deployability | Versioned bundle, heterogeneous workers, parity, rejection, fallback, rollback | One bundle reconstructed on 5/5 targets with maximum action difference 2.4×10⁻⁶; 7/7 invalid cases rejected; 5/5 automatic rollbacks; zero unsafe learned actions | A selected policy can move through training, export, reconstruction, and inference while preserving its contract | Pending campaign | Evaluated targets and registered failure cases only; requires the final generic typed-bundle inference path |
| Safety | Joint local action projection followed by Simulator enforcement | Post-projector exceedance below 1×10⁻⁵ kW, no Simulator correction, full feasible EV and deferrable service | Joint projection can preserve represented local feasibility before physical application | Pending campaign | Installation-total and per-phase active power only; no transformer, voltage, reactive-power, or protection claim |
| Robustness | Health semantics, dependency closure, clean twins, recovery and integrated stress | Exact closure in 48/48 registered faults, lower fault-induced cost, shorter recovery, and no failed AllIn seed | Typed health handling can limit degradation under the evaluated disturbances while retaining service | Pending campaign | Four fault families and frozen severities; resilience to other failures is not established |

### 14.1 Overall acceptance across the complete claim

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

TI-MAPPO is the only evaluated approach that passes the complete conjunction.
The alternatives remain valuable comparators: the MILP is economically
stronger, fixed MAPPO is slightly better nominally, and the entity-based model
accepts variable inputs. Each nevertheless fails at least one condition needed
to preserve a valid control decision across the complete real-world path.

- Taken together, this pattern would support a conjunctive argument. Nominal
  control quality alone would not be sufficient.
- The result pattern does not make TI-MARL universally best. It would be slightly
  worse than `REF*` nominally and slower at inference, while adding measurable
  value under semantic change, health events, and lifecycle checks.
- The MILP would remain the economic and demand-response leader, but with a
  substantially different computational profile.
- The final thesis conclusion would follow the observed data if the measured
  pattern differs from the one shown here.

## 15. Suggested use in the thesis

The complete nominal comparison can remain in the body as two landscape tables
because it establishes the historical progression across all principal method
families. The primary non-inferiority, structural, safety, health, scale,
integrated-stress, deployment, and real-evidence tables then carry the central
argument. Full seed-level estimates, learning curves, reward and replay
ablations, event-level results, MILP diagnostics, and the seventeen-dimensional
scorecards can move to the annexes.

This distribution would keep the thesis readable without hiding the extent of
the experimental work. The run registry would remain the source of provenance;
the tables in this document are only a visual prototype of the final reporting
layer.
