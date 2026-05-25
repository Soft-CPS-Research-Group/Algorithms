# Experiments

This directory is reserved for active, hand-curated experiment configs.

Do not commit generated remote submission batches, stale scorecard manifests, or
one-off local profiling configs here. Generate those under `runs/remote_configs/`
or another ignored `runs/` subdirectory, then promote only reusable templates or
short documentation summaries.

Current reusable entry points live in:

- `configs/templates/baselines/`
- `configs/templates/maddpg/`
- `configs/templates/rl/`
- `configs/templates/dynamic/`
