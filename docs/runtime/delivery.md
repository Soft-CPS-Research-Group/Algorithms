# Stage 2 delivery and acceptance

Scope: the agreed second stage, **common infrastructure with agents behind stable
boundaries**. Stage 1 branch consolidation remains complete. This is not an
additional algorithm-specific research iteration.

| Planned action | Delivered boundary |
| --- | --- |
| A09.01 | Job, simulator, shared loop, controller, resources, diagnostics, checkpoints and output modules |
| A09.02 | Common/agent configuration ownership; lazy catalogue and execution-unit builder |
| A09.03 | Typed contexts/transitions/schedules; common services; replay/rollout/offline learning owned by agents |
| A09.04 | Compatibility entry points delegate; research tools classified; simulator helper imports use their owner; offline settings shared |
| A09.05 | Installable wheel, CLI, optional extras, tested constraints and clean configuration/CPU installations |
| A09.06 | Ruff format/lint and mypy on new boundaries; errors propagate; cleanup cannot mask the job error |
| A09.07 | IDE files removed; contribution/ownership and license/provenance status documented |
| A01.01 | Configuration v1, migration, override precedence and null semantics |
| A01.02 | Strict common/agent schemas and validated actor/teacher/offline extensions |
| A01.03 | 38 capability declarations and lazy implementation loading |
| A01.04 | Capability combinations, validate/dry-run and agent-owned environment validators |
| A01.05 | Typed Pipeline/Ensemble routing preserves IDs, action roles, termination and duration |
| A01.06 | train/evaluate/smoke and none/checkpoint/bundle/both output intentions |
| A01.07 | 46 tracked YAML configurations migrated; explicit audit of previously ignored settings |
| A02.02 | Common environment factory and reset/step adapter |
| A10.02 | Parameterized contract suite plus representative complete jobs |
| A14.01 / A14.04 | Documentation index, runtime/agent/config/install guides, ownership, ADR and updated README/AGENTS |

Local acceptance on Python 3.10 / Simulator 1.8.3: **1,773 passed, 18 skipped**
in the default suite; **17 complete job tests passed**. Four skips require CUDA;
14 require external historical datasets/checkpoints. Ruff lint/format and mypy
on five typed boundary modules passed. A clean configuration-only wheel validated
both MADDPG and Transformer PPO configurations outside the checkout without
numerical dependencies. A separately installed CPU wheel completed a TransformerMADDPG
job and exported a valid bundle with MLflow absent. The exact CPU dependency
snapshot is checked in; `pip check` passed.

CI evidence is linked in the delivery PR. The real job suite
covers NormalPolicy (BAU), RBCBasicPolicy, MADDPG, TransformerMADDPG,
AgentTransformerPPO, AgentTransformerMATD3, TIMARL, two hierarchical compositions,
and offline IQL/CQL entity training. It checks actual results, final deterministic
execution, KPI files and model artifacts. Separate tests check explicit output
modes and evaluation from a previously saved model.

Remaining agreed stages are unchanged: stage 3 certifies behavior across the full
matrix (seeds, updates, topology, repeatability and meaningful run comparisons);
stage 4 settles export timing/defaults, MLflow cost and the final manifest;
stage 5 addresses model quality and inference readiness. This delivery does not
certify unavailable CUDA execution or turn experimental Torch artifacts into ONNX.
The unresolved license grant is an institutional ownership decision, documented
rather than guessed.
