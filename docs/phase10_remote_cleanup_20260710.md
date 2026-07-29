# Phase 10 Remote Cleanup - 2026-07-10
Purpose: free orchestrator/server space while preserving KPI evidence locally.
Local KPI archive committed in `docs/phase10_remote_kpi_archive_20260710.csv`. Raw collected remote artifacts are kept locally under `runs/remote_results/cleanup_archive_tiago_before_purge_20260710/` but are ignored by git.
## Retention Rules
- Keep PPO/TPPO runs.
- Keep one best hourly learned run.
- Keep final baseline references for the two active 15-minute datasets.
- Keep one best learned candidate for each active 15-minute dataset.
- Delete old variants, duplicate seeds, windows, smokes, failed runs, and stale configs once KPI/config snapshots are local.
## Kept Remote Jobs
| job | id | dataset | reason |
|---|---|---|---|
| `TPPO_53eps new` | `baaa9d13-03cf-45eb-ab24-27d3af174e97` | `citylearn_three_phase_dynamic_assets_only_demo_15s_parquet` | keep PPO/TPPO |
| `mctx4-123` | `2cfcf0b8-b82d-480e-8e1d-84dc7a47fd07` | `citylearn_challenge_2022_phase_all_plus_evs` | keep best hourly learned run |
| `15m-basic` | `27694ecd-25a9-4fcc-b8f9-09c2ff14afef` | `citylearn_three_phase_electrical_service_demo_15min_parquet` | keep final baseline reference |
| `15m-smart` | `66050c41-6454-4f6e-b49d-d5c2b65f1f0b` | `citylearn_three_phase_electrical_service_demo_15min_parquet` | keep final baseline reference |
| `15m-comm` | `86d498c1-f064-449d-b9b2-1ff6078051c7` | `citylearn_three_phase_electrical_service_demo_15min_parquet` | keep final baseline reference |
| `22m-basic-fy` | `376782a2-93c5-45f0-9b70-96e3ef93c17a` | `citylearn_challenge_2022_phase_all_plus_evs_15min_parquet` | keep final baseline reference |
| `22m-comm-fy` | `dba3132e-30d4-4e81-bbf0-8ce89a2cf968` | `citylearn_challenge_2022_phase_all_plus_evs_15min_parquet` | keep final baseline reference |
| `22m-smart-fy` | `503c119a-2e33-43c6-9ee1-0b3e493707e0` | `citylearn_challenge_2022_phase_all_plus_evs_15min_parquet` | keep final baseline reference |
| `22m-mctx4h-456-fy` | `eed07d0f-046a-49b5-ab87-160e25dc74f7` | `citylearn_challenge_2022_phase_all_plus_evs_15min_parquet` | keep best learned run for 2022 15min dataset |
| `15t-cost4-fy-srv-r2` | `f7dbed6b-b9d3-460b-8d8a-dfccbed944ce` | `citylearn_three_phase_electrical_service_demo_15min_parquet` | keep best current 15min three-phase candidate |

## Deleted Remote Jobs
Deleted `48` Tiago Fonseca jobs from the orchestrator. Full list is in `docs/phase10_remote_cleanup_plan_20260710.csv`.

## Dataset Decision
No datasets were deleted in this cleanup. The remaining remote jobs still reference the active datasets: dynamic 15s TPPO, dynamic 15min Gustavo runs, hourly `citylearn_challenge_2022_phase_all_plus_evs`, `citylearn_three_phase_electrical_service_demo_15min_parquet`, and `citylearn_challenge_2022_phase_all_plus_evs_15min_parquet`.

## Config Cleanup
Deleted configs associated with removed Tiago jobs, plus stale `15t-*` variants and orphan `mold-head-ef10014.yaml`. Kept configs for retained baselines, PPO/TPPO, best hourly, and current 15-minute candidate.
