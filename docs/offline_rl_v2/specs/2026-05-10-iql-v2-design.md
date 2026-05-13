# IQL-v2 Design Spec

_Author_: collaborative session, 2026-05-10
_Status_: approved, ready for implementation plan
_Predecessor_: `docs/offline_rl_v2/step4_bc_status.md` (BC-v2 met its success criterion)
_Successor_: implementation plan + code under `algorithms/offline_rl_v2/iql_*`, benchmark in `docs/offline_rl_v2/iql_v2_vs_rbc_v2_benchmark.md`

---

## 1. Problem statement (plain English)

BC-v2 has shown the v2 pipeline can faithfully reproduce its behaviour
policy and even slightly beat it (~3% on Building 5 cost). But BC is
imitation, not optimisation — by construction it can't exceed the
behaviour policy in expectation across the full state distribution.

The promise of offline RL is that, given a calibrated reward signal
(`reward_v2`, frozen in Step 3) and the same dataset, a policy
trained to **maximise expected return** can do strictly better than
the behaviour policy on its own success metric. Step 5 tests this
claim end-to-end with **Implicit Q-Learning (IQL)**.

**Goal:** Train an IQL policy on Building 5 such that, evaluated under
the same conditions as BC-v2, it improves on RBC-v2 by **more than 1σ
on at least one of {cost, peak, ramping}** at the district level,
without violating the safety constraint (`unserved_energy = 0`).

**Why IQL specifically:** IQL avoids policy evaluation against
out-of-distribution actions (which is what blows up offline DDPG and
SAC). It does so by treating value estimation as a regression problem
on the dataset only — never querying the policy for `Q(s, π(s))` on
unseen actions during V-update. This makes it a robust default for
offline-RL on small datasets like ours (87k transitions).

---

## 2. Algorithm (one page)

IQL has three networks trained jointly on a fixed offline dataset:

### 2.1 Value network V(s)

Trained by **expectile regression** of the Q-target:

$$
\mathcal{L}_V(\psi) = \mathbb{E}_{(s,a)\sim D}\Big[L_\tau\big(Q_{\bar\theta}(s,a) - V_\psi(s)\big)\Big],
\quad L_\tau(u) = |\tau - \mathbb{1}_{u<0}|\, u^2
$$

with τ = 0.7. Asymmetric loss: under-prediction penalised more than
over-prediction → V approximates the τ-expectile of the action-value
distribution at state s. **Q_bar** is the target Q (frozen copy).

### 2.2 Twin Q networks Q1, Q2

Standard Bellman MSE with V (not Q) as the bootstrap target — this is
the IQL key trick:

$$
\mathcal{L}_Q(\theta_i) = \mathbb{E}_{(s,a,r,s',\text{done})\sim D}\Big[
  \big(r + \gamma\,(1 - \text{done})\,V_{\bar\psi}(s') - Q_{\theta_i}(s,a)\big)^2
\Big],\ i \in \{1,2\}
$$

with γ = 0.99. Twin Q (clipped double Q-learning, taking
`min(Q1, Q2)` for the V-update) reduces over-estimation. Soft target
update: `θ_bar ← (1 - τ_target) θ_bar + τ_target θ` with τ_target =
0.005 each step.

### 2.3 Policy π(a|s)

Advantage-weighted regression (AWR-style):

$$
\mathcal{L}_\pi(\phi) = -\mathbb{E}_{(s,a)\sim D}\Big[
  \min(\exp(\beta \cdot A(s,a)), c) \cdot \log \pi_\phi(a|s)
\Big]
$$

with $A(s,a) = \min(Q_{\bar\theta_1}, Q_{\bar\theta_2})(s,a) - V_\psi(s)$,
β = 3.0, and weight clip c = 100.0 (numerical safety against advantage
outliers; standard in IQL implementations).

Policy is a **diagonal Gaussian with `tanh`-squashed mean** (matches
BC-v2 output range). Log-σ is a per-action **learned** parameter
(initialised at log(0.1), so σ≈0.1), not state-dependent — keeps the
policy network output to just the mean and avoids stochasticity
collapse. For inference (`predict(deterministic=True)`) we return
`tanh(mean)` — no sampling.

### 2.4 Why this works on our dataset

- The expectile loss makes V (and hence the advantage) optimistic
  about *good actions in the dataset* without requiring action samples
  outside the dataset. So Q is never queried at out-of-distribution
  actions during the V-update — the failure mode of vanilla actor-critic
  offline.
- The advantage-weighted policy update steers toward dataset actions
  that look better than average under the learned value function. With
  τ=0.7 the threshold of "above-expectile" is moderate; β=3 makes the
  weighting noticeable but not collapsing toward a single action.
- Twin Q + soft targets are stability scaffolding shared with TD3.

---

## 3. Components & module layout

Mirrors BC-v2 (self-contained, no imports from `algorithms/offline/`):

| File | Role |
|---|---|
| `algorithms/offline_rl_v2/iql_networks_v2.py` | `MLP`, `QNetworkV2`, `ValueNetworkV2`, `GaussianPolicyV2`. All three use the same MLP backbone (hidden=[256,256], dropout=0.1). |
| `algorithms/offline_rl_v2/iql_dataset_v2.py` | `IQLDatasetV2`, extends BC dataset to also yield `(s, a, r, s', done)` quintuples. Reuses `bc_dataset_v2.ObservationStandardiser`. Reads `terminated` (truncated is always 0 in this dataset). |
| `algorithms/offline_rl_v2/iql_trainer_v2.py` | `IQLTrainingConfigV2`, `train_single_seed`, `train_multi_seed`. Best-checkpoint persistence keyed by **deterministic policy MSE on the held-out val split** — same proxy as BC-v2 (compares `tanh(policy_mean(s))` to `dataset_action(s)`). True on-policy return isn't observable offline; this proxy at least catches regression to noise.|
| `algorithms/offline_rl_v2/iql_agent_v2.py` | `IQLAgentV2(BaseAgent)`. Mirrors `BCAgentV2`: controls B5, defers off-target buildings to `RuleBasedPolicyV2`. `update()` is no-op. |
| `scripts/train_iql_v2.py` | Multi-seed driver. CLI mirrors `train_bc_v2.py`. |
| `scripts/benchmark_iql_v2.py` | Mirror `benchmark_bc_v2.py` but baseline is RBC-v2 *and* BC-v2 (3-column report). |
| `tests/offline_rl_v2/test_iql_v2.py` | 8–10 tests (see §6). |
| `docs/offline_rl_v2/iql_v2_vs_rbc_v2_benchmark.md` | Generated. |
| `docs/offline_rl_v2/step5_iql_status.md` | Status doc, written after the benchmark. |

Boundaries:
- Networks know nothing about the dataset format.
- Dataset knows nothing about IQL specifics — it just exposes (s,a,r,s',done) batches.
- Trainer owns the IQL math; consumes networks + dataset.
- Agent is inference-only; loads a trained policy + standardiser, defers off-target buildings.

---

## 4. Hyperparameters (frozen for run-001)

| Parameter | Value | Rationale |
|---|---|---|
| Expectile τ | 0.7 | IQL paper default; moderate |
| Advantage temp β | 3.0 | IQL paper default |
| Advantage clip c | 100 | Standard numerical safety |
| Discount γ | 0.99 | Standard |
| Target soft update τ_target | 0.005 | Standard (TD3-style) |
| Optimiser | Adam | All three networks |
| Learning rate | 3e-4 | All three networks |
| Weight decay | 1e-5 | Same as BC-v2 |
| Gradient clip norm | 1.0 | Same as BC-v2 |
| Batch size | 256 | Same as BC-v2 |
| Hidden layers | [256, 256] | All three networks; matches BC-v2 |
| Dropout | 0.1 | All three networks; matches BC-v2 |
| Gradient steps | 150,000 | ~3× BC's 51k (one pass per network roughly) |
| Train/val split | 90/10 | Same as BC-v2 |
| Train seeds | 100..104 | Same as BC-v2 (reproducibility comparison) |
| Eval seeds | 200..209 | Disjoint from RBC dataset seeds 22..31 |

Frozen for run-001. Sweeps deferred to run-002+ if needed.

---

## 5. Data contract

Input: `datasets/offline_rl_v2/derived/rbc_with_reward_v2.parquet`
(87,590 rows, schema in `docs/offline_rl_v2/dataset_schema.md`).

Columns used:
- `obs_*` (35) and `next_obs_*` (35) — the SARS' transition.
- `action_electrical_storage`, `action_electric_vehicle_storage_charger_5_1` — 2-D action.
- `reward_v2` — scalar reward (frozen Step-3 weights).
- `terminated` — bool. (`truncated` is always 0; we don't read it but assert `truncated.sum() == 0` once at load time.)

Standardisation: identical to BC-v2 (`ObservationStandardiser` fit on
train slice only; same standardiser applied to both `s` and `s'`).

Bellman target uses raw `reward_v2` (no further scaling). If
training is unstable we will inspect reward magnitude and reconsider —
documented as a contingency in §7.

---

## 6. Tests (TDD)

Written before implementation. Eight tests minimum:

1. **expectile loss correctness.** For τ=0.7 and a known input/target,
   `expectile_loss(pred=0, target=1, tau=0.7) == 0.7 * 1`; symmetric
   case `tau=0.5` recovers MSE. Numerical check against hand-computed
   values.
2. **Bellman target with done flag.** For a transition with `done=1`,
   target = r exactly (no V(s') term). For `done=0`, target =
   r + γV(s'). Hand-built tensors.
3. **Advantage-weighted loss sign & magnitude.** For a single (s, a)
   with `A > 0`, loss decreases as `log π(a|s)` increases (i.e.,
   gradient pushes π toward a). For `A < 0`, gradient pushes π away.
   Sanity check on a fixed batch.
4. **Twin Q clipped min.** `Q_min(s,a) == min(Q1, Q2)(s,a)` element-wise
   on a small batch.
5. **Policy output shape & range.** `GaussianPolicyV2` mean is in
   `[-1, 1]` after `tanh`; `predict(deterministic=True)` returns
   2-D action vector per state.
6. **End-to-end smoke train.** 1 seed × 200 steps on tiny network →
   produces all artefacts (`policy.pt`, `q1.pt`, `q2.pt`, `value.pt`,
   `obs_standardiser.npz`, `metrics.jsonl`, `seed_summary.json`,
   `architecture.json`). All scalars in metrics.jsonl finite.
7. **Best-checkpoint persistence.** Same contract test as BC-v2:
   the saved `policy.pt` reproduces `best_val_policy_mse` to 1e-6
   when reloaded.
8. **`IQLAgentV2.predict` returns 17 action vectors.** Mirror BC-v2's
   test: B5 from IQL, others from a fresh RBC-v2.

Plus reuse of existing `_benchmark_common` infrastructure: no new
benchmark tests needed (the benchmark script is a thin orchestrator
already covered structurally by `benchmark_bc_v2`).

---

## 7. Risks & contingencies

| Risk | Likelihood | Mitigation |
|---|---|---|
| Training unstable (Q diverges, policy collapses) | medium | Twin Q + soft targets + grad clip 1.0. If still unstable, halve LR. |
| Advantage saturation (`exp(βA)` → ∞) | medium | Hard clip at c=100. Inspect advantage histogram in metrics.jsonl; if many samples clip, lower β. |
| `reward_v2` magnitude wrong for γ=0.99 (effective horizon ~100 steps; episode ~8760 steps) | low-medium | If V values blow up, scale reward by a constant (does not change π). Document if applied. |
| Policy too close to BC (no improvement) | medium | Try β=10 (sharper), or τ=0.9 (sharper expectile). Sweep deferred to run-002. |
| Policy violates unserved=0 (chooses not to charge EVs) | low | All v1 policies and BC-v2 stay at 0 unserved on this dataset; IQL's reward includes a 50× penalty on unserved (frozen Step-3 weight). Hard fail-stop in benchmark report. |
| Wall-clock too long | low | If >5h on CPU, reduce gradient steps to 100k. |

---

## 8. Definition of done

1. All eight tests in `tests/offline_rl_v2/test_iql_v2.py` green.
2. `runs/offline_iql_v2/run-001/seed_{100..104}/` populated with
   `policy.pt` + companions for each seed.
3. `docs/offline_rl_v2/iql_v2_vs_rbc_v2_benchmark.md` generated from
   50 IQL rollouts × 10 RBC rollouts.
4. **Success criterion met:** IQL-v2 beats RBC-v2 by >1σ on ≥1 of
   {cost_total, all_time_peak_average, ramping_average} at the
   district level, with `annual_normalized_unserved_energy_total = 0`.
   *Or* the benchmark report explicitly documents which criterion
   failed and proposes the next iteration's adjustment (back to
   brainstorming, not silent failure).
5. `docs/offline_rl_v2/step5_iql_status.md` written, mirroring
   `step4_bc_status.md` structure.
6. `OFFLINE_RL_AGENTS.md` § 7 updated to mark step 5 done with a
   one-line headline result.

---

## 9. Out of scope

- Hyperparameter sweeps (deferred to run-002 if run-001 misses bar).
- Online fine-tuning of IQL policy.
- Multi-building rollout.
- Comparison with TD3+BC, CQL, AWAC, etc. (decided: vanilla IQL only).
- Behaviour-policy swap (Step 6) — depends on Step 5 outcome.
