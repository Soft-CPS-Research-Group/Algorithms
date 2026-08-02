# Transformer PPO Recovery Campaign Design

## Purpose

Repair the correctness defects in `AgentTransformerPPO`, establish valid
teacher and policy baselines, and run a controlled remote campaign whose final
goal is to beat `RBCSmartPolicy` on district cost without worse safety or EV
service.

This campaign covers plain Transformer PPO (TPPO) and TPPO with behavioral
cloning (BC). It does not assume that BC is beneficial. BC advances only when
it improves on repaired plain TPPO.

## Current Evidence

The existing results are not valid evidence against PPO as an algorithm. The
current implementation and experiment protocol contain correctness and
comparison defects:

- Transformer dropout changes the policy between action collection and PPO
  probability evaluation.
- PPO reconstructs old log probabilities instead of retaining the exact
  collection-time values.
- BC phaseout executes teacher-blended actions and treats them as samples from
  the actor, which violates PPO's on-policy assumption.
- The critic clamps squared errors at 100, producing zero gradient for value
  residuals greater than 10.
- Truncation and termination share one terminal flag.
- Partial rollouts can be discarded without an update.
- The reported KPI data comes from stochastic online training trajectories,
  not frozen deterministic evaluation.
- Existing plain TPPO and BC runs use different scenarios and cannot serve as
  direct controls.

The TPPO+BC week KPI export also shows severe operational failure:

| KPI | Result |
|---|---:|
| District cost ratio to BAU | 1.1475 |
| Electrical violation events | 13,822 |
| Electrical violation energy | 8,457.02 kWh |
| EV minimum acceptable feasible service | 58.09% |
| EV feasible departure success | 51.50% |
| EV feasible departure within tolerance | 15.29% |
| Battery throughput ratio to BAU | 6.3094 |
| Solar self-consumption | 0.6331 |
| BAU solar self-consumption | 0.7435 |

Building 15 must always be reviewed separately because district aggregation
previously hid its dominant reward and safety failures.

## Success Criteria

The final policy must satisfy all of these requirements:

1. The three-seed mean district control cost is lower than the deterministic
   `RBCSmartPolicy` baseline on the same scenario.
2. No key safety violation is materially worse than the Smart baseline on any
   seed.
3. EV service is no worse than Smart on any seed.
4. Building 15 independently passes the safety and EV service gates.
5. Battery throughput remains within the approved limit relative to Smart.
6. Solar self-consumption has no material regression relative to Smart.
7. Training and evaluation pass all algorithm-validity checks.

Episode reward is a training diagnostic. It is not the final optimization
score.

## Remote Run Contract

### Source And Image Identity

Each experiment wave uses an immutable code and image baseline:

1. Production campaign changes land on `gj/tppo_bclonning`.
2. The branch is pushed before remote execution.
3. The container image is built from that exact commit.
4. The image tag or run metadata records the commit SHA.
5. All configuration-only variants in one wave use the same image.
6. Temporary `gj/*` branches and images are permitted only for explicit
   code-level ablations.
7. Any successful temporary change returns to `gj/tppo_bclonning`.

No result is comparable if its commit, image, resolved configuration, seed, or
horizon is unknown.

### Configuration Identity

Every remote run has a committed YAML configuration. Campaign configurations
will use a dedicated directory and descriptive names. The run name format is:

```text
tppo-recovery-w<wave>-<variant>-s<seed>-<short-sha>
```

The resolved configuration is authoritative when orchestration overrides a
template.

### Required Export

Export these artifacts for every run:

- complete runtime log;
- resolved configuration YAML;
- KPI JSON or CSV;
- job identifier;
- image tag and commit SHA when absent from the other artifacts.

Timeseries data is not required by default. Request a targeted subset only
when aggregate artifacts cannot explain a failure. Valid reasons include:

- Building 15 remains an outlier;
- safety violations increase without a matching reward component;
- teacher and student action mapping appears inconsistent;
- a topology transition causes an abrupt regression;
- battery throughput remains excessive without action-level evidence.

The requested subset should contain only affected buildings, actions, reward
components, and time windows.

## Common Correctness Foundation

These repairs apply to every TPPO configuration. They are not tuning
variables and must not be evaluated as optional ablations.

### Exact On-Policy Collection

- Use a deterministic Transformer representation for PPO probability
  evaluation. Set Transformer dropout to `0.0` for TPPO.
- During `predict()`, capture the exact sampled actor action, log probability,
  and critic value used for the environment decision.
- During `update()`, store those retained collection-time values instead of
  forwarding the network again to reconstruct them.
- Before the first optimizer step, recompute the policy probability in the
  same deterministic model mode and assert a probability ratio close to one.
- Log approximate KL divergence and clip fraction for every update.

The implementation must support the wrapper's `predict()` followed by
`update()` contract without silently pairing a transition with data from a
different action decision.

### Behavioral Cloning Separation

Teacher-generated or teacher-blended actions must never enter a PPO rollout as
actor samples.

The supported lifecycle is:

1. Collect Smart teacher demonstrations.
2. Run supervised actor pretraining without PPO updates on teacher actions.
3. Start PPO with actor-sampled environment actions only.
4. Optionally retain an auxiliary BC loss on stored teacher demonstrations.
5. Decay the auxiliary BC loss independently of environment action selection.

Remove teacher replacement and teacher blending from the PPO collection
phase. `RBCSmartPolicy` is the primary candidate teacher, subject to standalone
qualification.

### Critic Stability

- Replace clamped mean squared error with Huber loss.
- Ensure large residuals retain a finite, nonzero critic gradient.
- Add return or value-target normalization. Select one explicit method and
  persist its state in checkpoints.
- Make reward clipping explicit and symmetric if clipping remains enabled.
- Log raw and clipped reward distributions.
- Log value residual quantiles and explained variance.

### Rollout Boundaries

- Preserve separate `terminated` and `truncated` information.
- Do not bootstrap true terminal transitions.
- Bootstrap time-limit truncations from the next observation.
- Update on a valid partial final rollout instead of discarding it.
- Never clear a rollout when no optimizer update occurred.
- Reject configurations whose normal update cadence cannot produce the
  required minibatch.
- Mark dynamic-topology boundaries explicitly in GAE.
- Preserve unaffected network and optimizer state through topology changes.

### Checkpoint Completeness

Checkpoints used for campaign continuation must include:

- tokenizer, backbone, actor, and critic state;
- optimizer state;
- value-target normalization state;
- BC pretraining and decay progress;
- topology versions and action-layout identity;
- global learning and update counters.

An on-policy rollout does not need to survive a planned episode boundary. If
mid-episode resume is supported, its complete rollout state must also persist.

## Local Correctness Gates

No remote TPPO campaign starts until focused tests prove these invariants:

| Gate | Required Result |
|---|---|
| Unchanged-policy probability ratio | `1 +/- 1e-5` before optimization |
| Teacher action isolation | No teacher-generated action in a PPO buffer |
| Critic extreme residual | Finite, nonzero gradient |
| Truncation | Bootstraps from next-state value |
| True termination | Does not bootstrap |
| Partial rollout | Produces an update when large enough |
| Undersized rollout | Retained, not silently cleared |
| Topology mutation | Unaffected weights and optimizer state preserved |
| Deterministic inference | Repeated calls return equal actions |
| Extreme actions and rewards | Finite losses and gradients |
| Checkpoint round trip | Restores all campaign-relevant state |

Tests must include nonzero configured dropout to prove model-mode handling or
must reject nonzero dropout for TPPO configuration.

## Teacher Qualification

Run deterministic `RBCSmartPolicy` and `RBCCommunityPolicy` on the exact TPPO
scenario. Use the same dataset, topology events, reward, horizon, export
settings, and seed.

Smart qualifies as the BC teacher only when:

- it beats Community on the primary district cost objective;
- it has no materially worse safety violations;
- repeated Smart runs are reproducible;
- its action dimensions and ordering remain correct through topology changes;
- Building 15 has no unexplained severe failure.

If Smart fails qualification, repair or configure it before BC training. A
student is not expected to reliably outperform a teacher whose own behavior is
invalid on the target scenario.

## KPI Hierarchy

Use constraints first and cost second. A weighted score must not hide unsafe
behavior.

### Gate 1: Electrical Safety

Compare these district and per-building KPIs with Smart:

- electrical violation event count;
- electrical violation energy in kWh;
- phase peaks and imbalance as diagnostics.

The candidate cannot materially exceed Smart. If Smart has zero violations,
the campaign target is zero.

### Gate 2: EV Service

Compare these district and per-building KPIs with Smart:

- departure minimum acceptable feasible ratio;
- departure success feasible ratio;
- departure within-tolerance feasible ratio;
- shortfall beyond tolerance;
- SOC deficit and absolute error.

All primary EV service ratios must be no worse than Smart. Building 15 must
pass independently.

### Gate 3: Battery Health

Use battery throughput ratio to BAU and equivalent full cycles. The initial
provisional ceiling is `1.25 * Smart throughput`. Replace this with the exact
approved threshold after teacher qualification.

### Primary Objective: District Cost

Minimize `district_cost_total_control_eur`. A final TPPO candidate succeeds
when its three-seed mean beats deterministic Smart and no seed is catastrophic.

Community market savings is reported as supporting context, not as a
substitute for total control cost.

### Secondary Objective: Solar Self-Consumption

Maximize district solar self-consumption and report its ratio to BAU. Require
no material regression from Smart.

### Statistical Reporting

For every finalist report:

- each seed result;
- arithmetic mean;
- standard deviation;
- worst seed;
- safety and service gate result for each seed.

Safety and service gates apply per seed. Cost victory uses the three-seed mean.

## Compressed Remote Experiment Funnel

The campaign uses two required remote waves and one optional tuning wave.
Parallel capacity is six runs.

### Wave A: Qualification And Screening

Use one commit and image for all six slots:

| Slot | Variant | Purpose |
|---:|---|---|
| 1 | Deterministic `RBCSmartPolicy` | Exact baseline and teacher candidate |
| 2 | Deterministic `RBCCommunityPolicy` | Current-teacher comparison |
| 3 | Repaired plain TPPO reference | Plain learned control |
| 4 | Repaired plain TPPO, conservative exploration | Test action aggressiveness |
| 5 | Smart BC pretraining only | Isolate supervised initialization |
| 6 | Smart pretraining plus decaying auxiliary BC | Test continued imitation |

BAU does not need a separate slot when the KPI export includes BAU
counterfactual metrics.

Each TPPO job contains:

- one full-year training episode for screening;
- one frozen deterministic full-year evaluation episode;
- KPI export from the evaluation episode;
- training diagnostics kept separate from evaluation metrics.

The evaluation phase must disable optimizer updates, stochastic action
sampling, exploration, BC action replacement, and teacher action execution.

Wave A intentionally runs BC candidates concurrently with teacher
qualification to reduce wall-clock waves. If Smart fails qualification, reject
the BC results regardless of their apparent scores.

### Optional Wave B: Focused Tuning

Run this wave only if Wave A demonstrates valid, stable learning but no
candidate passes the KPI gates.

Choose six single-factor variants based on Wave A evidence. Candidate factors
are:

- actor initial standard deviation;
- entropy coefficient;
- learning rate;
- rollout length;
- BC pretraining dataset size or auxiliary weight;
- EV versus storage BC weighting;
- return normalization method, if Wave A critic evidence requires it.

One slot remains the Wave A control. Every other slot changes one factor.
Use the same seed for causal screening.

Do not run a broad hyperparameter search. Select factors from observed failure
modes.

### Wave C: Three-Seed Confirmation

Promote the best plain candidate and the best BC candidate:

| Slots | Candidate | Seeds |
|---|---|---|
| 1-3 | Best plain TPPO | 7, 17, 29 |
| 4-6 | Best BC TPPO | 7, 17, 29 |

Each run contains:

- three full-year training episodes;
- one frozen deterministic full-year evaluation episode;
- final KPI export from evaluation only.

If BC did not improve on plain TPPO in Wave A or B, do not allocate three
final slots to it. Use those slots for extra plain TPPO seeds or a justified
finalist.

## Runtime Diagnostics

Every TPPO training run must expose enough evidence to validate learning:

- exact update count and rollout size;
- pre-update policy ratio error;
- approximate KL divergence;
- policy clip fraction;
- actor and critic gradient norms;
- value loss and explained variance;
- raw and clipped reward quantiles;
- action mean, standard deviation, saturation, and near-zero fractions by
  storage and EV type;
- BC demonstration count, pretraining loss, auxiliary loss, and effective
  weight;
- topology version and rollout-boundary events;
- per-building summaries, including Building 15.

Evaluation logs must state that deterministic mode, frozen parameters, and
teacher-free action execution are active.

## Per-Run Validity Checks

Reject a run before KPI comparison when any condition holds:

- the job did not complete;
- an expected PPO or BC update is missing;
- a loss, gradient, value, action, or probability diagnostic is non-finite;
- the pre-update policy ratio violates tolerance;
- evaluation uses stochastic actions, teacher actions, or active learning;
- commit, image, resolved configuration, seed, or horizon is unknown;
- the KPI export does not correspond to the frozen evaluation phase.

## Promotion Rules

A candidate advances only when:

- all electrical safety gates pass;
- all EV service gates pass;
- Building 15 passes separately;
- battery throughput remains within its limit;
- cost improves over its direct control;
- learning diagnostics show a functioning critic and bounded policy updates.

BC advances only when it improves on repaired plain TPPO. Equal performance
does not justify its extra complexity.

## Architecture Escalation

Permit structural changes only after the repaired implementation:

- passes every correctness gate;
- learns stably across multiple configurations;
- still fails KPI gates for consistent and explainable reasons.

Escalate in this order:

1. Share tokenizer and actor parameters across buildings.
2. Add building identity embeddings.
3. Add a centralized or district-aware critic.
4. Add action masking or a safety projection layer.
5. Add recurrence only when evidence demonstrates partial observability.

Move safety projection earlier only if valid PPO repeatedly creates electrical
violations despite stable learning. Such a change is a new design decision and
requires its own focused experiment.

## Analysis Handoff

For each exported bundle, the analyzing agent produces a compact report with:

1. run identity and validity;
2. training diagnostic summary;
3. KPI gate table;
4. Building 15 analysis;
5. comparison with Smart, BAU, and repaired plain TPPO;
6. promote, reject, or request-targeted-timeseries decision.

The report must distinguish confirmed defects from tuning hypotheses. It must
not recommend another expensive run without naming the question that run will
answer and its pass/fail criterion.

## Expected Number Of Waves

- Best case: Wave A and Wave C.
- Expected case: Wave A, focused Wave B, and Wave C.
- Additional waves: only after evidence supports an architecture escalation.

This structure uses the six-run parallel capacity while keeping each remote
wave reproducible and interpretable.
