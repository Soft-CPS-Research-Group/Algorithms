# Checkpoint state inventory

This inventory separates serialized contents from **supported exact continuation**.
A file containing optimizers is not sufficient evidence of complete simulator/job
resume. Every published generation also has the shared runtime sidecar and index.
`weights` in the common index means initializer support, even if the agent loader
also restores additional state. Only explicitly tested complete paths advertise
`episode`; none advertise mid-episode continuation.

| Family | Agent-owned state observed in serializers | Boundary capability and gaps |
|---|---|---|
| MADDPG | Actors/critics and targets, auxiliary heads, optimizers, replay, exploration sigma/step, reward running moments, Python/NumPy/Torch RNG | Static episode continuation tested in independent processes; common sidecar restores counters/streams/tracker |
| Entity Transformer PPO | Tokenizer/backbone/actor/critic per member, optimizers including BC optimizer, value normalizer, BC state, counters/layout/action bounds, RNG | Static episode continuation tested; nonempty rollout explicitly defers save instead of losing on-policy data |
| Entity Transformer MATD3 | Actors/targets, centralized twin critics/targets, optimizers, signature-bucket replay and private RNG, reward normalizers, exploration/training/BC counters, topology descriptors and RNG | Static full-state episode continuation tested; weights/reset/fine-tune paths cannot claim this capability |
| TI-MARL | Typed actor/critic, learner state (optimizers, rollout/normalizers and counters), contract/layout metadata, RNG | Static episode continuation tested; nullable restore options preserve complete-state defaults |
| IPPO/MAPPO/HAPPO | Actors/value networks, optimizers, rollout, BC replay, exploration step, RNG | Loader round-trip covered by family tests; common exact-episode capability withheld until dedicated independent-process continuation evidence |
| MATD3/TD3 variants | Actors/critics/targets and optimizers, replay, exploration and normalization state in full mode | Family round-trip tests; common capability weights only; no blanket claim for all reset/BC variants |
| MASAC | Actors, twin critics/targets, optimizers including entropy-temperature optimizer, log-alpha, replay, exploration, reward moments and RNG | Family round-trip tests; common capability weights only |
| Flat Transformer MADDPG | Actor/critic/targets, optimizers, tokenizer, Transformer configuration, replay where available | Does not serialize all exploration and RNG state itself; initializer capability only |
| CC-L1 / CommunityCoordinator | Policy and optimizer with family-specific dimensions/counters | Hierarchical round-trip tests cover routing; no complete operational/rollout continuation promise |
| CC-L2 directional | Policy/optimizer, reward moments, counters, selected validation-policy state/history and protocol state | Best-policy selection requires validation evidence. Common capability weights only; hierarchy continuation not certified |
| CC-L2 capacity | Policy/optimizer, reward moments, PPO/CC counters and BC references | Does not persist full interval/rollout/simulator state; weights only |
| BuildingAgent | No implemented training checkpoint (its methods raise NotImplementedError); catalogue corrected to save/load false | Explicit requests fail before execution; research model export remains available |
| RBC/BAU/random/reference/solver policies | Stateless/rule parameters or no training checkpoint, as declared in the catalogue | No learned optimizer/replay to resume; output requests respect declared capabilities |
| BCAgent/IQL/CQL entity/offline variants | Offline trainer's model/normalizer files and research export | Common checkpoint loading/saving not declared; offline optimization restart is not advertised as job resume |
| Pipeline/Ensemble | Manifest plus child payloads using stable stage/member routing | Full episode capability requires every leaf to be allowlisted and complete; frozen/partial-stage resumes are initializers |

The precise 38-name capabilities and implementation paths are in
[agent_catalogue.md](../agent_catalogue.md) and `algorithms/catalogue.py`; aliases
inherit their implementation family's limitations. Schedulers, warmup counters,
terminal/truncation behavior, temporal reset and topology transactions remain in
the family tests. Existing state round-trips are supplemented by end-to-end
four-family uninterrupted/repeated/resumed comparisons and tamper/partial-write
regressions. Supporting another exact-episode family requires extending both the
serializer audit and these independent-process tests before adding it to
`EPISODE_RESUME_AGENTS`.
