# Phase 6 Dataset Variant Remote-Pending Configs

Generated on 2026-05-20. These configs are local only and have not been uploaded to the orchestrator.

They cover the first ablation matrix for the new dataset variants:

- `no_v2g`: EV charger schemas have charge-only bounds; RBCSmart V2G is disabled in these configs.
- `multi_charger`: extra charger/action stress test with Building 15 repeated phases and one `all_phases` charger.

Scopes:

- `smoke`: short RBCSmart/MADDPG checks.
- `baseline`: full-window RBCSmart baselines.
- `full`: seed-123 MADDPG v48 training configs.

Upload later only after the current Deucalion CUDA smoke confirms `Device selected: cuda`.
