# Config Templates

Templates are grouped by purpose:

- `maddpg/`: current MADDPG training templates for the supported static datasets.
- `rl/`: RL/MARL comparators (`MATD3`, `MASAC`, `IPPO`, `MAPPO`, `HAPPO`) that
  use the same wrapper/export contract.
- `baselines/`: Random, Normal, NormalNoBattery, legacy RuleBased, RBCBasic,
  RBCSmart and RBCCommunity comparison baselines.
- `dynamic/`: entity dynamic-topology smoke/debug templates. This group includes
  Transformer PPO and Transformer MATD3 default, residual, and behavior-cloning
  variants.

Transformer MATD3 templates:

- `dynamic/transformer_matd3_entity_dynamic.yaml`: default training. Optional
  residual, safety, price, and behavior-cloning paths are disabled.
- `dynamic/transformer_matd3_entity_dynamic_residual.yaml`: residual control with
  `RBCSmartPolicy`. ONNX metadata declares the required runtime residual path.
- `dynamic/transformer_matd3_entity_dynamic_bc.yaml`: independent replay-based
  and demonstration-based behavior cloning.

15-minute static dataset templates are available for the Phase 2 work:

- `baselines/rbc_basic_15min_local.yaml`
- `baselines/rbc_smart_15min_local.yaml`
- `baselines/rbc_community_15min_local.yaml`

Use `simulator.entity_encoding.profile` to switch MADDPG observation profiles, e.g. `maddpg_v1`, `maddpg_v2_compact`, `maddpg_v3_operational`, or `maddpg_v3_realtime`, instead of creating duplicate templates for each profile.

Generated remote batches should live under `runs/remote_configs/`, not in
`configs/templates/`. Project-specific recipes, demonstrations and results are
local research material and are intentionally ignored by Git.
