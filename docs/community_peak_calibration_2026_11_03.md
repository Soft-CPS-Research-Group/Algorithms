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

Raw evidence: `runs/community-peak-core30-20261103/`. Results and final decision
will be appended after both phases finish.
