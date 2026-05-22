# Phase 6 MADDPG Performance Profiles

Prepared configs for the next remote image. They are not submitted automatically.

Purpose: test whether MADDPG can keep learning with fewer network updates and larger replay batches, so GPU work is denser and simulator/encoding overhead is amortized.

Profiles:

- `update4_batch512`: conservative first step; should preserve learning behavior closest to current configs.
- `update8_batch512`: medium reduction in update frequency.
- `update16_batch1024`: aggressive profile; useful if cost per update dominates.

Datasets:

- `2022_full_year`: 6 x 8760 hourly steps, comparable to the fresh 2022 matrix.
- `15s_window`: 2 x 5760 steps, explicitly a performance window, not a KPI claim.

All configs keep MLflow off, checkpoints off, BAU exports off, AMP on, and runtime profiling on.
They export KPIs at every episode end (`kpis_final_episode_only: false`) but keep
timeseries/render output for the final episode only (`timeseries_final_episode_only: true`).
