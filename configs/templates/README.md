# Config Templates

Templates are grouped by purpose:

- `maddpg/`: current MADDPG training templates for the supported static datasets.
- `rl/`: experimental RL/MARL comparators (`MATD3`, `IPPO`, `MAPPO`) that use
  the same wrapper/export contract.
- `baselines/`: Random, Normal, NormalNoBattery, legacy RuleBased, RBCBasic and RBCSmart comparison baselines.
- `dynamic/`: entity dynamic-topology smoke/debug templates.
- `legacy/`: older demo/docker compatibility templates.
- `experimental/`: placeholders or non-primary experiments.

Use `simulator.entity_encoding.profile` to switch MADDPG observation profiles, e.g. `maddpg_v1` or `maddpg_v2_compact`, instead of creating duplicate templates for each profile.
