# Agent catalogue

Declarations and capabilities live in `algorithms/catalogue.py`; inspect them with
`algorithms-job --list-agents` without loading model dependencies.
The authoritative name list is `algorithms.registry.supported_algorithms()`: **37**
registered agent/controller names after the BAU/RBC review. A registered name
means it can enter the common job builder; deployment readiness and model quality
are separate properties.

| Family | Registered names | State at consolidation |
|---|---|---|
| Actor–critic | `MADDPG`, `MATD3`, `MASAC`, `PPO`, `TD3`, `IPPO`, `MAPPO`, `HAPPO` | Existing main implementations retained. |
| Shared entity Transformers | `AgentTransformerPPO`, `AgentTransformerMATD3` | Current main implementations retained, including dynamic topology and BC paths. |
| Typed multi-agent | `TIMARL` | Existing main implementation retained; research export. |
| Hierarchical coordination | `BuildingAgent`, `CommunityCoordinator`, `CCLevel1`, `CCLevel2`, `CausalPriceSignal`, `FixedPriceSignal` | Existing implementations retained. |
| Baselines and signal policies | `RandomPolicy`, `NormalPolicy`, `NormalNoBatteryPolicy`, `RBCBasicPolicy`, `RBCCommunityPolicy`, `RBCSmartPolicy`, `RBCSmartLocalPolicy`, `SignalAwareRBC`, `SignalAwareRBCSmartLocal` | Existing implementations retained. |
| Oracle replays | `FixedServiceOracleReplayPolicy`, `TotalHomeOracleReplayPolicy`, `TotalOracleReplayPolicy` | Existing audit/reference implementations retained. |
| Historical Transformer | `TransformerMADDPG` | Recovered experimental joint actor. Static topology, uniform nonzero action dimensions. Joint ONNX includes learned tokenizer and actor; deterministic parity checked. Complete research checkpoint retained. |
| Offline policies | `BCAgent`, `IQLAgent`, `IQLEntityAgent`, `CQLEntityAgent` | Recovered trainers and policies. Common job preparation, simulation, KPI and model export. Static, matching dataset layout required. |
| Capacity coordinator | `CCLevel2Capacity` | Recovered distinct capacity-aware teacher/network. Does not replace current CCLevel2. Research checkpoint. |
| Recovered signal workers | `SignalAwareRBCCommunity`, `SignalDirectedRBC` | Community price multiplier / directional stationary-battery worker. Common hierarchy; requires manager context for serving. |

`BCAgent` and flat `IQLAgent` retain the historical Building_5 experiment: the
learned policy controls that building and OfflineRBC controls companions. They
expect its original 35-feature/two-action schema. Entity IQL/CQL dispatch one
Gaussian policy per trained group, with explicit named-input validation. They do
not claim compatibility with arbitrary datasets just because dimensions match.

New recipes live in `configs/templates/experimental/`. The three `*_smoke.yaml`
recipes execute short jobs on the bundled dataset. The offline evaluation recipes
require a trained `model_dir`; see [offline usage](offline_rl/README.md) to train
inside the same job instead.

`EVDataCollectionRBC` and `DistrictDataCollectionRBC` are preserved collection
utilities and are not advertised as ordinary model-training jobs. `SingleAgentRL`
is a schema placeholder and remains rejected. Old GNN/Transformer skeletons and
superseded prototypes are not counted as additional implemented agents.

Origins, validation scope and pending work: [first consolidation](consolidation/README.md).

## Export support after stage 4

All 37 names share finalization, KPI and bundle contracts. The [v2 export matrix](inference_bundle.md#delivery-formats) separates 14 neural names with verified ONNX, 13 rules/signal names with executable JSON, and 10 names requiring specialised research adapters. Counts describe the tested base configurations, not every optional residual/safety variant or model quality. Hierarchical leaves are consumed as part of their declared graph.

`bundle.require_deployable: true` rejects known specialised families during configuration validation and requires actual final-evaluation parity for the others. Ordinary exports retain research artifacts and explicit limitations. The [stage 4 delivery](runtime/export_delivery.md) records the acceptance evidence.

The [BAU/RBC behavior contract](baseline_controllers.md) defines the five principal baselines, retires `RuleBasedPolicy`, and retains Smart only for compatibility.
