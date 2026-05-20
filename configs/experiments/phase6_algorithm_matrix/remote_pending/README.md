# Phase 6 Algorithm Matrix Remote-Pending Configs

Generated on 2026-05-20. These configs are local only and have not been uploaded to the orchestrator.

Purpose: compare the implemented learning candidates under the same reward, encoding and dataset variants while the current Deucalion/server jobs finish.

Algorithms: MATD3, MASAC, IPPO, MAPPO, HAPPO.

Datasets/variants: 15s original, 2022 original, 15s no_v2g, 2022 no_v2g, 15s multi_charger, 2022 multi_charger.

All configs use `CostServiceCommunityFeasiblePrecisionRewardV46`, `maddpg_v2_compact`, MLflow disabled, end-of-run KPI export enabled, and seed 123.

Do not submit these until a Docker/SIF image exists from a commit that includes the RL comparator agents and this matrix prep.
