# Experiment Campaign: baselines-core30-sim200

## Provenance

| Field | Value |
|---|---|
| Archived at | `2026-11-03T01:01:48.129308Z` |
| Source commit | `a0eeb047fa1dc812862d627125af9112bc5d1045` |
| Image tags | align-reference-baselines-with-simulator-2-0-and-core30-electrical-contr-2d203f5, complete-core30-oracle-plans-and-cap-bau-battery-demand-estimates-80fc555, honor-prepared-worker-attempts-without-hiding-baseline-results-a0eeb04, respect-basic-battery-ceiling-and-collect-current-runtime-kpis-987e72e |
| Datasets | rec_pt_2023_core_30_v2 |
| Windows | 35040..35712, 39072..39360, 52512..53184 |
| Algorithms | NormalNoBatteryPolicy, NormalPolicy, RBCBasicPolicy, RBCCommunityPolicy, RBCSmartLocalPolicy |
| Evidence horizon | `partial-window` |
| Gate profile | core30-reference-service-0.999-grid-1e-6-precision-0.8-review |
| Baseline | RBCSmartLocalPolicy matched by schema, window, Simulator 2.0.0 and seed 22/6763; BAU target is 100% |
| Submission manifest | `runs/baselines-core30-20261102/remote-configs/submitted_jobs.json` |
| Results summary | `runs/baselines-core30-20261102/remote-results/summary.csv` |
| Scorecard | `runs/baselines-core30-20261102/remote-results/reference-scorecard.csv` |

## Jobs and decisions

| Job | Job ID | Algorithm | Seed | Status | Cost EUR | EV min | EV tolerance | Grid kWh | Decision |
|---|---|---|---:|---|---:|---:|---:|---:|---|
| core30-dynamic2024-bau-no-battery | `0897730b-986c-4e12-9217-9d3f993a938a` | NormalNoBatteryPolicy | 22 | finished | 118.76649143991112 | 1.0 | 0.0 | 0.0 | superseded-dynamic-metadata |
| core30-dynamic2024-community | `f67f20cd-4284-4596-816d-ee0c2ed6d315` | RBCCommunityPolicy | 22 | finished | 72.24059646157637 | 1.0 | 1.0 | 0.0 | superseded-dynamic-metadata |
| core30-dynamic2024-smart-local | `bb219bff-169b-4113-82c5-2ffe268bc536` | RBCSmartLocalPolicy | 22 | finished | 72.30449997492036 | 1.0 | 1.0 | 0.0 | superseded-dynamic-metadata |
| core30-summer2024-bau-no-battery | `582d3350-f59a-424c-b758-ac8bfe1d191c` | NormalNoBatteryPolicy | 22 | finished | 133.73871692605093 | 1.0 | 0.012048192771084338 | 0.0 | accepted-reference |
| core30-summer2024-community | `58ba8485-c373-40a4-b7e9-653d854fefbc` | RBCCommunityPolicy | 22 | finished | 68.09406195362787 | 1.0 | 1.0 | 0.0 | accepted-reference |
| core30-summer2024-smart-local | `86f3465e-05b8-4e7c-90e1-71924f79323d` | RBCSmartLocalPolicy | 22 | finished | 68.53753316048609 | 1.0 | 1.0 | 0.0 | accepted-reference |
| core30-winter2024-bau-no-battery | `d731735d-6d73-405e-baf4-c29cd34044d5` | NormalNoBatteryPolicy | 22 | finished | 302.04202878005924 | 1.0 | 0.0 | 0.0 | accepted-reference |
| core30-winter2024-community | `9f6e3201-e33a-4e22-8ed6-3fbe1296e986` | RBCCommunityPolicy | 22 | finished | 186.53384295917917 | 1.0 | 0.8850574712643678 | 0.0 | accepted-reference |
| core30-winter2024-smart-local | `08cdeb2b-1868-4445-8231-6efd641fe836` | RBCSmartLocalPolicy | 22 | finished | 186.57614567801517 | 1.0 | 0.8850574712643678 | 0.0 | accepted-reference |
| core30-dynamic2024-bau-battery | `a7ec97d7-12a9-48dc-8821-386ff369afc7` | NormalPolicy | 22 | finished | 118.12189571453963 | 1.0 | 0.0 | 0.0 | superseded-dynamic-metadata |
| core30-summer2024-bau-battery | `215d9304-a8f1-4125-a66c-399320143844` | NormalPolicy | 22 | finished | 128.445730850492 | 1.0 | 0.012048192771084338 | 0.0 | accepted-reference |
| core30-winter2024-bau-battery | `a5dbfcd3-54b9-4b41-8653-2e682b8a1fba` | NormalPolicy | 22 | finished | 299.5290990616546 | 1.0 | 0.0 | 0.0 | accepted-reference |
| core30-summer2024-basic | `cf36dc86-dddc-419a-a508-26be6d331a57` | RBCBasicPolicy | 22 | finished | 68.68614372069044 | 1.0 | 1.0 | 0.0 | accepted-reference |
| core30-winter2024-basic | `41c3a5d7-3a6d-4c67-87f3-a7a8d5e4fc11` | RBCBasicPolicy | 22 | finished | 187.3106878544736 | 1.0 | 0.8850574712643678 | 0.0 | accepted-reference |
| core30-dynamic2024-bau-battery-final | `990a15c4-ce71-4c0e-9f9f-82331de26db1` | NormalPolicy | 22 | finished | 118.15198252877224 | 1.0 | 0.0 | 0.0 | accepted-reference |
| core30-dynamic2024-bau-no-battery-final | `34a53fa4-3473-4cc9-85c4-a33a07ea689d` | NormalNoBatteryPolicy | 22 | finished | 118.76649143991112 | 1.0 | 0.0 | 0.0 | accepted-reference |
| core30-dynamic2024-basic-final | `90ade406-f176-487f-a419-d54091582210` | RBCBasicPolicy | 22 | finished | 72.48895246382942 | 1.0 | 1.0 | 0.0 | accepted-reference |
| core30-dynamic2024-smart-local-final | `8521da97-ebd8-4517-bbd1-a8e827522564` | RBCSmartLocalPolicy | 22 | finished | 72.53923372853698 | 1.0 | 1.0 | 0.0 | accepted-reference |
| core30-dynamic2024-community-final | `5a2f69da-ba72-44ce-ae3b-d73e809c07cc` | RBCCommunityPolicy | 22 | finished | 72.47525394882572 | 1.0 | 1.0 | 0.0 | accepted-reference |

## Notes and limitations

- 15 accepted reference jobs plus 4 superseded dynamic diagnostics; all 19 finished successfully. [Full audit](../baselines_core30_audit_2026_11_02.md).
- Core30 v3.3, official PyPI Simulator 2.0.0. Normal winters/summers are seven days; Combined dynamic is three days. Cold reset per window; not annual evidence.
- Gates: feasible EV minimum >=0.999 with complete audit coverage and zero unknown departures, electrical violations <=1e-6 kWh and zero events, zero missed deferrable cycles, physical SOC review, RBC precision >=0.8. BAU precision against requested SOC is not applicable: its target is 100%.
- All 15 final bundles reproduced 32 decision samples with zero maximum error. Five final dynamic jobs used a0eeb04 and exposed outputs without manual path repair.
- Static jobs use the corrected version of each policy; later dynamic-storage metadata changes reproduced all stored static bundle references exactly. Tags and registry digests are in the raw campaign manifest.
- Official simulator KPIs supply costs. Peaks use exported net community energy divided by the 0.25-hour interval. There is no universal cost or peak dominance between RBCs.
- Legacy Phase 6 reports are retained separately; decisions use the explicit reference-scorecard.csv, not historical filename inference or a BAU target-precision gate.
