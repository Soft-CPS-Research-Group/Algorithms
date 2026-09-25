# Canonical annual REC datasets

Dataset context retained on 2026-09-25 after retirement of the previous
experimental campaign. This inventory records existing data and evidence;
it does not prescribe the next campaign's algorithms, splits, seeds or budget.

## Common contract

- Dataset contract: `2023-q15-v1.9`; generator: `1.9.0`; generation seed: `2023`.
- Calendar year 2023, 35,040 physical control intervals of 900 seconds.
- UTC physical timeline, with `Europe/Lisbon` civil-calendar attributes.
- Self-contained Parquet data, manifests, checksums and validation reports.
- Dataset families live under `Simulator/data/datasets/rec_2023_*` in the
  adjacent Simulator repository. Each family has a default `schema.json` and
  alternative schemas under `schemas/`.
- The inspected Algorithms runtime pins `softcpsrecsimulator==1.8.0`.
  Dataset, generator, package and source-commit versions are separate identities.

## Physical composition

| Family | Members | PV | BESS | Charging members | Physical chargers | Deferrables |
|---|---:|---:|---:|---:|---:|---:|
| `rec_2023_micro_4_q` | 4 | 2 | 1 | 2 | 2 | 1 |
| `rec_2023_core_15_stripped` | 15 | 7 | 3 | 6 | 7 | 4 |
| `rec_2023_core_30` | 30 | 15 | 6 | 12 | 15 | 8 |
| `rec_2023_premium_100` | 100 | 55 | 22 | 45 | 80 | 30 |

Core-15 retains a physical subset of the first 15 Core-30 members. Core-30
contains 24 residential members, four small services and two larger members.
Premium contains 70 residential members, 20 small services and ten larger
members. Member, charger, EV and session identities are distinct; a shared
charger may serve several EVs over the year.

## Available variants

| Variant | Conditions represented |
|---|---|
| Micro-4-Q | Small end-to-end inspection and smoke surface. |
| Core-15-Stripped | Reduced static composition for development and diagnosis. |
| Core-30-Nominal | Static population and complete asset mix, without injected faults. |
| Core-30-Safety | Installation-total and per-phase import/export limits. |
| Core-30-Health | Observation/forecast, communication, command and equipment disturbances. |
| Core-30-Dynamic | Member join/leave and independent asset removal/reinstallation. |
| Core-30-Combined | Safety, Health and Dynamic conditions plus demand-response requests. |
| Premium-100-Clean | Heterogeneous scale, multiple chargers, dynamic topology, limits, settlement and services, without injected health faults. |
| Premium-100-AllIn | Matching Premium physical base and topology with injected and coincident disturbances. |

Variants within a family share physical time series and asset identities.
Core-30 Dynamic/Combined starts with 26 active members, reaches 30 and ends
with 28. Premium starts with 70, reaches 100 and ends with 94. Removal and
reinstallation creates a new runtime asset instance at its declared initial
state; temporary unavailability of an existing instance is a different event.

Micro, Core-15 and Core-30 Nominal/Health/Dynamic do not instantiate the
electrical-service block. Electrical-limit safety is therefore not applicable
on those surfaces. A zero-valued historical export does not demonstrate a
limit guarantee there. Core-30 Safety/Combined and both Premium variants have
installation and phase limits; these are not a feeder power-flow model.

## Prices, services and interpretation

The suite uses Portuguese OMIE day-ahead prices for 2023, repeated from hourly
market periods into four physical quarter-hours. Local settlement uses 80% of
the import tariff, equal importer weights and zero remuneration for residual
grid export. Load/PV forecasts and price-publication boundaries have explicit
causal conventions in the dataset documentation.

Ordinary EV sessions are individually feasible at their declared charger.
That does not establish joint feasibility under concurrent load, local limits
or faults. Service denominators must account for active members and chargers:
Core-30 Dynamic/Combined exposes 3,690 departures and Premium exposes 28,108,
rather than all sessions in their full catalogues.

The populations are designed experimental scenarios, not a statistical sample
of Portuguese communities. Historical 17-member controller surfaces and real
demonstrator evidence remain separate from this suite.

## Existing annual baseline evidence

The local archive
`runs/annual_rec_baseline_audit_v1_9_asset_count_attested_full_summary/`
contains 18 paired BAU/RBCSmart runs, covering all nine variants. Its aggregate
validation, generated on 2026-08-21, reports 18/18 runs and integrity `pass`.

Examples from that archived audit:

| Variant | BAU cost EUR | RBCSmart cost EUR | RBC cost change |
|---|---:|---:|---:|
| Core-15 | 4078.732 | 3964.394 | -2.80% |
| Core-30-Nominal | 23497.824 | 23060.935 | -1.86% |
| Premium-Clean | 113639.380 | 112209.554 | -1.26% |
| Premium-AllIn | 113610.368 | 112170.261 | -1.27% |

The archived Phase-6 profile requires EV minimum service >=0.999, precision
>=0.80 and electrical-violation energy <=1e-6 kWh. Premium minimum-service
rates are approximately 0.9973 (Clean) and 0.9965 (AllIn), hence `fail/watch`.
Several Core variants also show increased ramping. Passing execution-integrity
checks is not equivalent to passing every controller-quality gate.

These are existing annual baseline observations, not a completed multi-seed
comparison of the learned algorithms. Raw reports remain in ignored `runs/`.

Source contracts: Simulator `docs/dataset_reference.md`, the four family
`dataset_manifest.json` files and their variant schemas. The thesis's simulator
chapter describes the same ladder. Performance summaries above are grounded in
the archived `campaign_summary.md` and `annual_analysis.md`, not in illustrative
thesis-result tables.
