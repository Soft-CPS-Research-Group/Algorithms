# TI-MARL implementation gap analysis

| Capability | Current repository state | Status | TI-MARL action |
|---|---|---|---|
| Entity IDs, tables and edges | CityLearn `entity_v1` and `EntityContractAdapter` | partial | Consume raw payload; do not flatten away ownership |
| Dynamic asset topology | Simulator topology events and wrapper rebuild | partial | Preserve structured transitions instead of flushing them |
| Dynamic population | Simulator supports member add/remove | partial | Stable-ID rollout and variable set critic |
| Runtime fault injection | Simulator robustness service | partial | Expose raw runtime facts without TI health decisions |
| Typed compiler | No canonical typed object graph | absent | New deterministic TIC |
| Health closure | Events are mostly diagnostics/features | absent | Versioned derivation, dependencies and causal effects |
| Grouped parameterised ports | Existing agents emit fixed scalar vectors | absent | Dynamic groups and categorical/continuous ports |
| Shared local actor | TPPO keeps per-building network stacks | absent | Shared actor per role |
| Variable central critic | MAPPO/MATD3 concatenate fixed vectors | absent | Set critic with per-agent values |
| Local action feasibility | Name-based analytic safety adapter | partial | Reuse mathematics behind typed constraints and bundles |
| Reward decomposition | `CostHardConstraintReward` exposes components | implemented | Reuse scalar reward and persist components |
| Cross-topology replay | Wrapper has special topology hook/flush | partial | Current/next immutable snapshots and stable-ID semantics |
| Checkpointing | Agent-specific checkpoints | implemented | Add TI contract hashes and population-independent restore |
| Traces | Metrics and projection diagnostics | partial | Buffered typed decision/transition trace |
| Dynamic deployment | Fixed-cardinality ONNX bundle | absent | Research-only Torch bundle first |
| Remote execution | Docker/Union/SIF and Job Orchestrator | implemented | Standard config and pinned image; no bespoke runner |

Existing RBC, TPPO, PPO, MAPPO, MATD3 and flat/entity adapters remain
backward-compatible and are regression baselines.

