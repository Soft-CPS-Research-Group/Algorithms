# Phase 6 2022 Full-Year Scorecard Matrix

Official clean full-year 2022 matrix for the current scorecard.

- Dataset: `citylearn_challenge_2022_phase_all_plus_evs_data_2026_05_21`
- Image tag: `sha-f20cc24`
- Window: `0..8759`, `episode_time_steps: 8760`
- Baselines: 1 full-year episode
- RL/MARL candidates: 6 full-year episodes, seeds 123/456/789
- MLflow: disabled
- Checkpoints: disabled
- BAU export: disabled
- KPI export: per episode; timeseries final episode only

See `docs/community_optimization_success_scorecard_pt.md` for the decision gates.

## Redistribution

After the first submission, secondary RL/MARL seeds were redistributed away from
the Deucalion GPU queue:

- seed `123` remains on Deucalion GPU for `MADDPG V48`,
  `MATD3ServiceStorageGuard`, and `MATD3StorageGuard`;
- seed `456` runs on the `server` CPU worker;
- seed `789` runs on Deucalion CPU (`normal-x86`);
- redistributed CPU configs have AMP disabled (`use_amp: false`).

See `redistribution_manifest.csv` for the replacement configs.
