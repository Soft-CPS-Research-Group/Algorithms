# Community peak threshold: Core30 calibration protocol

Protocol fixed before submitting the November 3, 2026 campaign. This is a
configuration experiment, with no policy changes and no weakening of Basic.

## Comparison surface

- Algorithms image `honor-prepared-worker-attempts-without-hiding-baseline-results-a0eeb04`
  (source `a0eeb047fa1dc812862d627125af9112bc5d1045`); the baseline policy/runtime
  code is identical to merged main `197f80a429a1b75c6977471da209f5c1d38161a8`.
- Official Simulator 2.0.0; Core30 v3.3, Normal, all 30 members, entity/static.
- Frozen server dataset `rec_pt_2023_core_30_v2_sim200_a0adc12`, copied and
  hash-checked during the preceding baseline audit.
- Calibration uses `evaluation/preparation_2023/core_30_normal.json`.
  Validation uses `evaluation/core_30_normal.json`, with the same physical data.
- Four independent seven-day episodes per year: January 1–7, April 1–7,
  July 1–7 and October 1–7. Each has 672 completed 15-minute intervals and
  673 states; reset at the start of each window. This is seasonal partial-window
  evidence, not annual performance. January/July 2024 were inspected in the
  preceding audit; the new parameter selection nevertheless uses only 2023.
  Windows and daily-maximum groups are anchored at midnight **UTC**, using fixed
  dataset indices. During Portuguese summer time this is 01:00 local, rather
  than local midnight. The submitted indices are the authoritative boundaries.
- Seed 22 (deterministic evaluation stream 6763), zero training, one evaluation,
  same initial state protocol, reward, interface, final export and bundle checks.
- MLflow disabled. Native BAU counterfactual export disabled for every job in
  this campaign, avoiding an unnecessary second simulation. BAU ratios will be
  absent, not estimated. Own cost, service and energy exports remain enabled.
- Grid import uses the dataset price P; local trades settle at 0.8P on both
  sides; own PV has zero operational energy cost; external exports earn zero.
  The Simulator's simultaneous limited-pool sharing and default member weights
  remain unchanged. No negative tariffs are introduced.

## Calibration grid and decision rule

Run Basic and Smart Local unchanged, plus Community with
`community_import_threshold_kw` explicitly set to **15, 30, 45, 60 or 210 kW**,
in each 2023 window (28 jobs). The original automatic threshold for 30 members
is 7 × 30 = 210 kW; explicit 210 is the control.

Before cost/peak ranking, every candidate must pass in every calibration window:
EV minimum acceptable service on feasible departures ≥0.999; minimum-service
audit coverage 1 and unknown count 0; target-tolerance precision ≥0.80;
electrical violation energy ≤1e-6 kWh and zero events; zero missed appliance
cycles and full appliance audit coverage; battery SOC within physical [0,1]
(1e-6 numerical allowance); successful bundle parity checks. Non-finite or
missing required values require review and cannot win.

Relative to Community 210, eligible candidates must keep cost and weekly maximum
grid import within +1% **in every window**. Rank by the mean of the four
candidate/control weekly peak ratios, lower first. Ratios within 0.001 of the
best are tied; break that tie using mean daily-maximum import ratios, with the
same tolerance, then total cost within 0.1%, then the highest threshold.
Promotion also requires at least 0.1% improvement in either the mean weekly-peak
ratio or mean daily-peak ratio. Otherwise retain 210. If the original fails a
service gate, report the failure rather than claiming a calibrated reference.

Freeze the chosen threshold and complete 2023 scorecard, with timestamp and
hash, before preparing/submitting any 2024 job. Validate the frozen choice,
original Community 210, Basic and Smart Local on the four 2024 weeks (16 jobs,
or 12 if the original wins). Do not tune from 2024. Report per-window and pooled
cost, maximum import, mean daily maximum import, service and all gate failures.

## Interpretation constraints

The current community threshold enables additional battery discharge only when
the observed community import exceeds it **and the price is classified as
expensive**. It does not impose a hard import cap, shift all EV starts, or create
storage capacity. Core30 Normal has only three stationary batteries. A lower
threshold may therefore leave the largest (e.g. cheap-hour) peaks unchanged.
The report also inspects battery throughput and last exported SOC. These RBC
episodes do not impose an equal terminal SOC; energy costs exclude a monetary
battery-wear charge. These diagnostics do not change the frozen selection rule.

Basic remains an honest price/urgency baseline. A simpler policy would be a
separate behavioural specification and comparison, not a retrospective change
to improve the apparent ranking. BAU continues to represent immediate maximum
feasible EV charging toward 100%, earliest permitted appliance starts and either
no battery control or local self-consumption battery control. That is an explicit
reference behaviour, not a claim that every household behaves identically.

Raw evidence: `runs/community-peak-core30-20261103/`. Completed results and the final decision are recorded below.

## Frozen choice before 2024 validation

Selected **15 kW** at `2026-11-03T03:13:12.102161+00:00`, using only the 28 completed 2023 jobs. All passed the declared gates. The 15 and 30 kW candidates tied on weekly peaks; 15 kW won the daily-peak comparison under the predeclared tolerance. No 2024 job had been prepared or submitted.

| Threshold (kW) | 2023 cost, 28 sampled days (€) | Mean weekly peak ratio to 210 | Mean daily peak ratio to 210 | Eligible |
|---:|---:|---:|---:|---|
| 15 | 675.041405 | 0.983076918 | 0.991560679 | True |
| 30 | 686.489140 | 0.983076914 | 0.994208117 | True |
| 45 | 688.984310 | 1.000000000 | 1.000000000 | True |
| 60 | 688.984310 | 1.000000000 | 1.000000000 | True |
| 210 | 688.984310 | 1.000000000 | 1.000000000 | True |

2023 evidence SHA-256: `e2a8931a3f6453409c3636af83230f91c2ceede6035bd54071d2428213f01c41`. Selection file SHA-256: `e51605c5550c71f28ae8a4687e6fc3e22d73bea2829f2543395c582ecab3d40f`. Raw selection: `runs/community-peak-core30-20261103/selection.json`.

## Completed results and adoption decision

All **44 jobs** finished successfully: 28 calibration jobs and 16 validation jobs. Inputs, prices and official settled-cost KPIs matched exactly within each window. Every run passed the declared service/electrical/SOC gates and reproduced 32 bundle decisions with zero maximum error. No policy implementation was changed.

The selected **15 kW** threshold reduced 2024 energy cost, but its incremental peak benefit over the original Community did **not** meaningfully carry over to 2024. **Keep the existing Core30 reference (automatic 210 kW for 30 members); do not promote 15 kW as a validated peak-reduction setting.** The 15 kW recipe remains fully reproducible in the campaign payloads as an economical alternative with greater storage use. Basic remains unchanged.

### 2024: four sampled weeks, 28 days in total

| Controller | Energy cost (€) | Maximum 15-min import (kW) | Mean daily maximum (kW) | Feasible EV minimum service | Battery throughput (kWh) | Mean last exported storage energy (kWh) |
|---|---:|---:|---:|---:|---:|---:|
| Basic | 421.640563 | 58.700710 | 31.422166 | 100.00% | 510.071 | 20.340 |
| Smart Local | 420.035168 | 55.372711 | 30.571767 | 100.00% | 712.823 | 19.712 |
| Community 210 | 419.405015 | 55.372711 | 29.927672 | 100.00% | 646.568 | 20.773 |
| Community 15 | 410.510912 | 55.372711 | 29.911209 | 100.00% | 1018.530 | 16.659 |

Each controller covered **352 EV departures and 302 completed appliance cycles** in 2024, with zero missed cycles, zero electrical violation energy/events, complete audit coverage and no physically infeasible minimum-service departures. EV target-tolerance precision was **96.875%**; minimum-service compliance and target precision are separate indicators. Storage SOC stayed between approximately 19.75% and 90%; the strategic 20% reserve is not an exact post-step floor (the lowest validation sample accompanied a discharge), while the dataset physical minimum is 10%. This campaign did not tighten that strategic reserve.

Against Community 210, Community 15 reduced total cost by **2.121%**, left the largest peak **unchanged**, and reduced the pooled mean daily peak by only **0.055%**. The mean weekly-peak ratio was 0.999951980; the mean daily-peak ratio was 0.999260624. Both incremental peak effects are below the 0.1% tolerance used in the calibration ranking. Battery throughput increased **57.53%**, while the mean last exported storage energy was **4.114 kWh lower**. This is not a measured battery-degradation cost: wear was not monetized, and terminal SOC was not constrained to equality.

### 2024 seasonal detail

| UTC week | Controller | Cost (€) | Maximum import (kW) | Mean daily maximum (kW) |
|---|---|---:|---:|---:|
| 2024-01-01–07 | Basic | 187.310688 | 58.700710 | 36.989579 |
| 2024-01-01–07 | Smart Local | 186.576146 | 55.372711 | 35.896484 |
| 2024-01-01–07 | Community 210 | 186.533843 | 55.372711 | 35.870428 |
| 2024-01-01–07 | Community 15 | 184.783233 | 55.372711 | 36.126969 |
| 2024-04-01–07 | Basic | 13.222251 | 34.494264 | 30.339867 |
| 2024-04-01–07 | Smart Local | 13.237443 | 34.494264 | 30.986830 |
| 2024-04-01–07 | Community 210 | 13.234795 | 34.500891 | 30.987160 |
| 2024-04-01–07 | Community 15 | 11.856413 | 34.494264 | 30.835790 |
| 2024-07-01–07 | Basic | 68.686144 | 28.197354 | 21.379063 |
| 2024-07-01–07 | Smart Local | 68.537533 | 25.655154 | 20.858767 |
| 2024-07-01–07 | Community 210 | 68.094062 | 25.655154 | 20.122730 |
| 2024-07-01–07 | Community 15 | 66.204550 | 25.655154 | 20.122767 |
| 2024-10-01–07 | Basic | 152.421481 | 46.686434 | 36.980156 |
| 2024-10-01–07 | Smart Local | 151.684046 | 43.947282 | 34.544986 |
| 2024-10-01–07 | Community 210 | 151.542315 | 38.354294 | 32.730370 |
| 2024-10-01–07 | Community 15 | 147.666717 | 38.354294 | 32.559312 |

The 15 kW candidate made the January daily-peak average worse (36.126969 vs 35.870428 kW), improved April and October modestly, and was effectively equal in July. Its weekly maximum was effectively equal to the original in all four validation windows. The Community/Smart Local peak difference in October already existed with the original threshold; it must not be credited to the new calibration.

### 2023 calibration context

| Controller | Cost for 28 sampled days (€) | Maximum import (kW) | Mean daily maximum (kW) |
|---|---:|---:|---:|
| Basic | 703.254769 | 66.213479 | 32.842107 |
| Smart Local | 691.930669 | 62.744951 | 30.925951 |
| Community 15 | 675.041405 | 58.497598 | 30.208123 |
| Community 30 | 686.489140 | 58.497598 | 30.225822 |
| Community 45 | 688.984310 | 62.744951 | 30.479992 |
| Community 60 | 688.984310 | 62.744951 | 30.479992 |
| Community 210 | 688.984310 | 62.744951 | 30.479992 |

Each calibration controller covered 365 EV departures and 293 completed appliance cycles, with 100% feasible minimum service, zero missed cycles and zero electrical violations. Target-tolerance precision was 97.2603%. The selected 15 kW and 30 kW thresholds reduced the January maximum from 62.744951 to 58.497598 kW; the lower threshold won the predeclared daily-peak comparison across all four weeks. Higher thresholds 45/60/210 produced identical headline outcomes in these windows.

### Interpretation and next useful change

- The Community already provides lower mean daily peaks than Smart Local over the sampled 2024 weeks. Lowering its battery trigger mostly improved energy cost and increased battery use; it did not materially improve held-out peaks. The threshold is a trigger, not a guaranteed import cap.
- Core30 Normal has three batteries, 25.6 kWh total capacity and 12.68 kW nominal power. The default 65% peak-discharge rate requests at most 8.242 kW before further physical constraints. The threshold branch is price-gated, and dispatch is sequential rather than a joint optimizer.
- The Basic/Smart Local comparison is honest without weakening Basic. Smart Local was cheaper in all four 2023 weeks and in the 2024 aggregate, but not universally: April 2024 cost was 13.237443 vs 13.222251 EUR (a 0.015192 EUR difference). Its April daily peak average was also higher.
- July 2023 illustrates the cost/peak tradeoff: Smart Local cost less and improved average daily peaks, but its maximum exceeded Basic (26.775416 vs 25.973300 kW). At 2023-07-06 00:00 UTC, EV charging was 2.383924 kW higher and BESS charging 1.321915 kW lower than Basic; the net difference explains that interval. Charger export -1 sentinels were excluded from physical energy sums. These diagnostics do not replace the official cost KPI.
- If the next objective is stronger community peak control, test explicit coordination of flexible EV/appliance charging, including cheap-hour synchronization, with the same service protections. Do not weaken Basic to force a ranking. This work did not implement or claim that new policy.
- The two BAUs retain their preceding audited operational definitions: immediate maximum-feasible EV charging toward 100%, earliest permitted appliance start, and either no BESS control or local self-consumption BESS control. They are explicit reference behaviours, not empirical household-behaviour models. Their previous audit remains the evidence; this campaign compared the RBCs only.

### Evidence and reproducibility

Raw evidence: `runs/community-peak-core30-20261103/`, including both phase payloads, resolved configs, official KPI JSON/CSV, simulator manifests, energy/price/BESS series, bundle checks, exact input-matching audits, frozen selection and analysis scripts. Combined records: `remote-configs/submitted_jobs.json`, `remote-results/summary.csv`, `remote-results/scorecard.csv`, `remote-results/decision.json`. Standard Phase 6 reports are retained under each phase’s `remote-results/legacy-phase6/`; current decisions use explicit matching, not legacy filename inference.

Jobs remain in **Tiago Fonseca / Baselines Core30**. No remote jobs were canceled, deleted or archived. No Simulator changes were made. The broader iteration 5 and specialized TIMARL work remain deferred. These are independent seasonal windows, not continuous annual evaluation; January/July 2024 had appeared in the previous audit, and were not used for this threshold selection.
