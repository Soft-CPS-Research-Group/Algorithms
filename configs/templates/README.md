# Config Templates

Templates are grouped by purpose:

- `maddpg/`: current MADDPG training templates for the supported static datasets.
- `rl/`: RL/MARL comparators (`MATD3`, `MASAC`, `IPPO`, `MAPPO`, `HAPPO`) that
  use the same wrapper/export contract.
- `baselines/`: Random, Normal, NormalNoBattery, legacy RuleBased, RBCBasic,
  RBCSmart and RBCCommunity comparison baselines.
- `dynamic/`: entity dynamic-topology smoke/debug templates.

Use `simulator.entity_encoding.profile` to switch MADDPG observation profiles, e.g. `maddpg_v1`, `maddpg_v2_compact`, `maddpg_v3_operational`, or `maddpg_v3_realtime`, instead of creating duplicate templates for each profile.

Generated remote batches should live under `runs/remote_configs/`, not in
`configs/templates/` or `configs/experiments/`.
