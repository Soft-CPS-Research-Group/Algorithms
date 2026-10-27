# Agent catalogue

The authoritative list is `algorithms.registry.supported_algorithms()`: **38**
registered agent/controller names after the first consolidation. A registered name
means it can enter the common job builder; deployment readiness and model quality
are separate properties.

| Family | Registered names | State at consolidation |
|---|---|---|
| Actor–critic | `MADDPG`, `MATD3`, `MASAC`, `PPO`, `TD3`, `IPPO`, `MAPPO`, `HAPPO` | Existing main implementations retained. |
| Shared entity Transformers | `AgentTransformerPPO`, `AgentTransformerMATD3` | Current main implementations retained, including dynamic topology and BC paths. |
| Typed multi-agent | `TIMARL` | Existing main implementation retained; research export. |
| Hierarchical coordination | `BuildingAgent`, `CommunityCoordinator`, `CCLevel1`, `CCLevel2`, `CausalPriceSignal`, `FixedPriceSignal` | Existing implementations retained. |
| Baselines and signal policies | `RuleBasedPolicy`, `RandomPolicy`, `NormalPolicy`, `NormalNoBatteryPolicy`, `RBCBasicPolicy`, `RBCCommunityPolicy`, `RBCSmartPolicy`, `RBCSmartLocalPolicy`, `SignalAwareRBC`, `SignalAwareRBCSmartLocal` | Existing implementations retained. |
| Oracle replays | `FixedServiceOracleReplayPolicy`, `TotalHomeOracleReplayPolicy`, `TotalOracleReplayPolicy` | Existing audit/reference implementations retained. |
| Historical Transformer | `TransformerMADDPG` | Recovered experimental joint actor. Static topology, uniform nonzero action dimensions. Complete research checkpoint; no inference deployment claim. |
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
