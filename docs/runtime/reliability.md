# Reliable execution (stage 3)

The common runtime owns input validation, initialization, execution, checkpoint
publication and attempt completion. Agents retain their numerical methods. The
public entry point and 38 registered names remain available; this stage does not
claim scientific quality, production inference compatibility or GPU certification.

## Supported runtime

| Target | Python / PyTorch / CUDA / Simulator | Evidence and support decision |
|---|---|---|
| Linux CPU reference | 3.10 / 2.5.1 CPU / none / 1.8.3 | Clean installed package, actual training/reference jobs, independent-process repetition and episode resume |
| Development CPU on CUDA wheel | 3.10.12 / 2.5.1+cu121 with CUDA disabled / 1.8.3 | Full regression suite and tracking benchmark |
| Generic amd64 image | Dockerfile pins Torch 2.5.1, CUDA 12.1 wheel, Simulator 1.8.3 | Build support; GPU numerical equivalence remains a hardware validation task |
| Generic arm64 image | Same Python/Simulator contract; architecture-specific wheel | Build support; no ARM runtime certification in this stage |
| Union Blackwell | Dockerfile.union: Torch 2.7.1, CUDA 12.8, Simulator 1.8.3 | Separate image retained; no silent switch to CPU reference pins and no hardware certification claimed |
| Jetson Xavier L4T r35 | Legacy Python 3.8, NVIDIA Torch 2.0, CUDA 11.4, Simulator 1.8.0 | Previous image retained. Automatic replacement disabled: stage-3 job requires Python >=3.10. Migration and target-device acceptance are required before re-enabling publication |

Simulator source for the reference is
`0ca1696884cb372ff8f09f443eddf6f50dccd238`, package `softcpsrecsimulator==1.8.3`.
The package's Python range is >=3.10,<3.13; only 3.10 is certified here. Version
pins reproduce this evidence; they do not assert absence of dependency vulnerabilities.
The existing dependency-security action A10.06 remains open.

## Training, evaluation and randomness

```yaml
training:
  episodes: 20
  seed: 0
  determinism: strict
  steps_between_training_updates: 1
  target_update_interval: 2
evaluation:
  episodes: 1
  policy: final
```

This means **20 training episodes plus one evaluation**. Evaluation may be disabled
with `episodes: 0`. Smoke is one training plus one evaluation; evaluate is zero
plus one. Legacy `simulator.episodes=N` with `deterministic_finish=true` still means
N-1 training plus one evaluation. Migration preserves that effective budget:
[46-template inventory](episode_migration.json). Conflicting old/new counts fail.
Research helpers whose CLI count historically included evaluation use the explicit
compatibility helper in `algorithms.configuration.episodes`.

Python, NumPy and Torch are seeded before environment/model construction, including
seed zero. Stable stage/member identities derive separate initialization,
exploration, learning/replay and evaluation streams. Legacy global generators are
swapped around synchronous hooks; concurrent jobs/hooks in the same process are
unsupported. The simulator has its own recorded seed. Its derived default is
16-bit to accommodate the pinned Simulator's episode-seed multiplication; an
explicit seed whose random-episode budget exceeds NumPy's range fails early.
Strict mode requests deterministic Torch kernels and disables cuDNN benchmarking.
Unsupported deterministic kernels fail rather than silently weakening the setting.

The core suppresses learning during evaluation and frozen-stage callbacks receive
`training=false`. Torch modules, including networks in lists and per-building
state, run under eval/no-grad. Agents still retain operational observation/action
state needed for control. Rollout/replay, warmup, terminal/truncation and target
scheduling have family regressions; learning counters exclude evaluation steps.
BuildingAgent and CommunityCoordinator use deterministic actions without updating learned normalizers during evaluation; PPO/MASAC exploration counters remain fixed. BuildingAgent's unimplemented checkpoint capability is now declared unsupported.
The TPPO/TMATD3 constructors now apply the training seed; TMATD3's private replay
RNG is seeded; TI-MARL's nullable restore flags now preserve their enabled default.

The evaluated policy is selected before hashing. Default is final weights. The
existing CC-L2 validated-best option must have a selection record and reports its
source, episode and objective. It is never mislabeled as the final training
checkpoint. Results link the selection, parameter/buffer hash, available checkpoint
index entry, window, simulator seed and initial evaluation topology. Dynamic runs
also retain their topology history. The hash includes all stored Torch networks
(including critics), not only deployed actor weights.

Independent-process tests compare exact policy hashes and episode histories for
MADDPG, entity Transformer PPO, entity Transformer MATD3 and TI-MARL on the pinned
CPU runtime. They also compare uninterrupted training against an episode-boundary
resume. This is a supported-path regression, not a proof for every hyperparameter,
BC/residual/offline mode or device. Cross-device equality is not promised; tolerance
must be measured in the later hardware/scientific matrix.

## Data and actions

`inputs.json` records the resolved schema/overlays, declared window, ordered member
mapping and SHA-256/size of referenced selected data and auxiliary assets, including
tokenizer/typed-interface files and offline model/data directories. Initially excluded
members referenced by later topology events (including asset donors) are also
inventoried without changing their initial activation state. This is a
conservative inventory of declared inputs, not a system-call trace of individual
bytes read. Missing inputs fail; no substitute physical model is synthesized.

The loader checks duplicate JSON identities and columns, finite required physical
columns, infinities, basic calendar bounds, static horizon length and optional
terminal-observation padding. Valid nullable measurements remain identified in
the report. Dynamic member-window alignment is delegated to Simulator's existing
loader. An optional `simulator.data_contract_path` JSON can make column order,
required-finite fields, bounds, timestamp column, timezone and units explicit:

```json
{"files":{"building.csv":{"columns":["timestamp","non_shiftable_load"],"required_finite":["non_shiftable_load"],"bounds":{"non_shiftable_load":[0,100]},"timestamp_column":"timestamp","timezone":"UTC","units":{"non_shiftable_load":"kWh"}}}}
```

Timestamp grids must be unique, increasing and match the schema resolution.
Anonymous indexed files cannot prove absolute dates, timezone or physical units;
these remain explicitly undeclared or declared by the contract, never inferred.
Column values alone cannot verify a supplied unit label. Agent/schema layout
validators remain authoritative for supported observations/actions and topology.

Invalid action cardinality, dimensions and non-finite values fail before stepping.
`simulator.action_policy: clip` preserves bounded legacy behavior and logs proposed
and submitted actions when changed; `reject` fails on any bounds violation.
Static jobs and actual dynamic addition/removal of members/assets are tested.
RBC uses the member/asset demo; TPPO/TMATD3 use their supported assets-only fixture.
Unsupported feature/tokenizer combinations still fail; generic dynamic support is
not a promise that every physical device type has an implemented neural head.
Wrapper and composite rollback regressions cover failed reattachment.

## Checkpoints and retries

Each save publishes a new generation under `checkpoints/`, including repeated
requests at the same step. Serialization occurs in a hidden `.partial` directory;
payload hashes, `.complete` and then an atomic `index.json` publish it. Failure
leaves diagnostics without replacing `latest`. `best` remains null unless a real
selection mechanism supplies evidence; it is not guessed from filenames.
The index records format version, relative artifact, agent/stage identities,
step/episode/reason, hashes, boundary and resume capability. Old file layouts,
including TPPO's nested numbered files, remain readable. The index is authoritative;
legacy aliases are conveniences. All generations are retained even if the legacy
`keep_episode_checkpoints` switch is false; there is no automatic expiration.

`checkpointing.save_final` is independent of `checkpoint_interval`; disabling the
periodic cadence does not disable the final save. Explicit output requests still
require their artifacts. Free disk space is checked against `min_free_disk_mb`
(default 64); a serializer can still exhaust the disk, in which case no complete
index is published. Atomic publication is per file/generation on the local
filesystem, not a distributed transaction or an object-store guarantee.

```yaml
checkpointing:
  save_final: true
  checkpoint_interval: null
  resume_training: true
  resume_mode: episode
  checkpoint_local_path: /data/jobs/prior/checkpoints
  checkpoint_mode: full
```

`resume_mode: weights` remains the default initializer behavior. Complete episode
resume is explicitly gated to the four tested serializers, static topology, full
state, matching inputs/config/runtime versions and a remaining episode. It restores
agent payloads plus common counters, episode history, named RNG streams and the
version-pinned Simulator episode tracker. Partial reset/fine-tune/frozen options
are rejected for this mode. Periodic mid-episode and dynamic checkpoints advertise
weights-only capability: the environment state needed for exact continuation is
not fabricated. See [the family state inventory](checkpoint_inventory.md).

Repeated job IDs create isolated attempts and retain prior results, models,
checkpoints and logs. `latest_attempt.json` identifies the newest attempt. Invalid
configuration, cancellation/SIGTERM, serializer failure, export failure and disk
errors produce failed/cancelled evidence where the filesystem permits. All cleanup
hooks are attempted. Result/manifest records use atomic strict JSON writes; NaN
and Infinity are rejected. The common runner commits `completed` only after output
validation, local tracking finalization and resource cleanup. Remote curated result
snapshots may say `finalizing`; the local result/progress is the completion authority.

## Tracking and measured cost

MLflow defaults off and is an optional installation extra. Local JSONL, outputs and
checkpoints work without importing or installing it. One facade governs metrics
from agents, coordinators, rewards and the loop. `mlflow_step_sample_interval`
applies globally; episode/overall boundary metrics are retained. Artifact profiles
are `minimal` (none), `curated` (selected job metadata) and `full`; explicit model
and checkpoint upload flags can opt in. `mlflow_failure_policy: local` retains local
records after a remote failure; `abort` fails the attempt. Artifact retention has
no automatic expiry. Explicit checkpoint downloads remain possible independently
of whether logging is enabled.

The [recorded benchmark](tracking_benchmark.json) uses one unmeasured warmup and
three independent processes per mode: the same small MADDPG job, two training
and one evaluation episodes, 17 state points, CPU and a temporary file-based
MLflow store. No production service was installed or contacted.

| Mode | Median job wall / CPU seconds | Peak RSS MiB | Local job bytes | MLflow calls |
|---|---:|---:|---:|---:|
| Off | 1.272 / 1.245 | 647.7 | 867,606 | 0 |
| Local detailed metrics | 1.319 / 1.290 | 648.4 | 1,490,989 | 0 |
| MLflow minimal, sample 10 | 1.920 / 1.884 | 687.8 | 1,490,985 | 17 |
| MLflow full, every step + artifacts | 2.332 / 2.294 | 688.4 | 1,490,922 | 98 |

Peak RSS includes imports; timed wall/CPU exclude imports. File MLflow excludes
network latency and is not a production-size throughput estimate. Its separate
store used 2,871,893 bytes for all measured runs. Detailed profiling intentionally
increases local logs by about 0.62 MB even in this tiny job; keep it opt-in and use
normal sampling for routine runs. In a representative local sample, environment
steps took 0.090 s, encoding 0.009 s, prediction 0.036 s and learning 0.190 s;
checkpoint/output costs are separately recorded. These timers overlap and must
not be summed. Traces, detailed profiling and remote uploads remain opt-in; no
unmeasured universal cost limit is invented.

Decision: **keep the code integration optional, use local tracking normally**.
The measured overhead does not justify restoring the removed production service
(SOF-222). MLflow is useful only when an experiment explicitly needs its comparison
UI/remote store. Reproduce from the checkout with the tracking extra installed:

```bash
CUDA_VISIBLE_DEVICES='' python -m scripts.benchmark_runtime_tracking --output /tmp/tracking-comparison
```

Stage 4 still owns the final global KPI/export policy, complete delivery manifest
and inference handoff. Stage 5 owns exhaustive family/hardware acceptance and
scientific comparison. Their actions remain in the master plan.
