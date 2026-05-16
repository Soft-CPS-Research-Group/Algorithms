# IQL Reference

Implicit Q-Learning (IQL) for Building 5 offline RL. Covers algorithm design,
module layout, hyperparameters, and run commands.

---

## 1. Problem statement

BC showed the pipeline can reproduce its behaviour policy and slightly beat it
(~3% on Building 5 cost). But BC is imitation, not optimisation — by
construction it cannot exceed the behaviour policy in expectation across the
full state distribution.

IQL tests whether a policy trained to **maximise expected return** can do
strictly better than the behaviour policy on its own success metric, using the
same calibrated reward signal and dataset.

**Goal**: train an IQL policy on Building 5 such that, under the same
evaluation conditions as BC, it beats RBC by **more than 1σ on at least one of
{cost, peak, ramping}** at the district level, with `unserved_energy = 0`.

---

## 2. Algorithm

IQL has three networks trained jointly on a fixed offline dataset.

### 2.1 Value network V(s)

Trained by expectile regression of the Q-target:

$$
\mathcal{L}_V(\psi) = \mathbb{E}_{(s,a)\sim D}\Big[L_\tau\big(Q_{\bar\theta}(s,a) - V_\psi(s)\big)\Big],
\quad L_\tau(u) = |\tau - \mathbb{1}_{u<0}|\, u^2
$$

with τ = 0.7. The asymmetric loss makes V approximate the τ-expectile of the
action-value distribution at state s. `Q_bar` is the frozen target Q.

### 2.2 Twin Q networks Q1, Q2

Standard Bellman MSE with V (not Q) as the bootstrap target — the IQL key
trick that avoids querying out-of-distribution actions:

$$
\mathcal{L}_Q(\theta_i) = \mathbb{E}_{(s,a,r,s',\text{done})\sim D}\Big[
 \big(r + \gamma\,(1 - \text{done})\,V_{\bar\psi}(s') - Q_{\theta_i}(s,a)\big)^2
\Big],\ i \in \{1,2\}
$$

with γ = 0.99. Twin Q (clipped double Q-learning) reduces over-estimation.
Soft target update: τ_target = 0.005 each step.

### 2.3 Policy π(a|s)

Advantage-weighted regression (AWR-style):

$$
\mathcal{L}_\pi(\phi) = -\mathbb{E}_{(s,a)\sim D}\Big[
  \min(\exp(\beta \cdot A(s,a)), c) \cdot \log \pi_\phi(a|s)
\Big]
$$

with $A(s,a) = \min(Q_{\bar\theta_1}, Q_{\bar\theta_2})(s,a) - V_\psi(s)$,
β = 3.0, and weight clip c = 100.0.

Policy is a diagonal Gaussian with tanh-squashed mean. Log-σ is a per-action
learned parameter (init log(0.1)). For inference (`predict(deterministic=True)`)
returns `tanh(mean)` — no sampling.

---

## 3. Module layout

| File | Role |
|---|---|
| `algorithms/offline_rl/iql_networks.py` | `MLP`, `QNetwork`, `ValueNetwork`, `GaussianPolicy`. Hidden=[256,256], dropout=0.1. |
| `algorithms/offline_rl/iql_dataset.py` | `IQLDataset`, `IQLSplit`, `load_iql_split`. Reuses `bc_dataset.ObservationStandardiser`. Yields (s, a, r, s', done) quintuples. |
| `algorithms/offline_rl/iql_trainer.py` | `IQLTrainingConfig`, `train_single_seed`, `train_multi_seed`, `expectile_loss`, `bellman_target`. Best-checkpoint keyed by deterministic policy MSE on val split. |
| `algorithms/offline_rl/iql_agent.py` | `IQLAgent(BaseAgent)`. Controls B5, defers off-target buildings to `OfflineRBC`. |
| `scripts/train_iql.py` | Multi-seed training driver. |
| `scripts/benchmark_iql.py` | 3-column report (RBC \| BC \| IQL). Raw KPIs to `datasets/offline_rl/benchmarks/<run-id>/`. |
| `tests/offline_rl/test_iql.py` | 8 tests (see §6). |

Boundaries: networks ↔ dataset ↔ trainer are independent; agent is
inference-only.

---

## 4. Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Expectile τ | 0.7 | IQL paper default; moderate |
| Advantage temp β | 3.0 | IQL paper default |
| Advantage clip c | 100 | Standard numerical safety |
| Discount γ | 0.99 | Standard |
| Target soft update τ_target | 0.005 | Standard (TD3-style) |
| Optimiser | Adam | All three networks |
| Learning rate | 3e-4 | All three networks |
| Weight decay | 1e-5 | Same as BC |
| Gradient clip norm | 1.0 | Same as BC |
| Batch size | 256 | Same as BC |
| Hidden layers | [256, 256] | Matches BC |
| Dropout | 0.1 | Matches BC |
| Gradient steps | 150 000 | ~3× BC's one-epoch pass |
| Train/val split | 90/10 | Same as BC |
| Train seeds | 100..104 | Same as BC |
| Eval seeds | 200..209 | Disjoint from RBC dataset seeds 22..31 |

---

## 5. Data contract

Input: `datasets/offline_rl/derived/rbc_with_reward.parquet` (87 590 rows).

Columns used:
- `obs_*` (35) and `next_obs_*` (35) — the SARS' transition.
- `action_electrical_storage`, `action_electric_vehicle_storage_charger_5_1` — 2-D action.
- `reward` — scalar (frozen Step-2 weights).
- `terminated` — bool. (`truncated` always 0; asserted at load.)

Standardisation: `ObservationStandardiser` fit on train slice only; applied to
both s and s'.

---

## 6. Tests

File: `tests/offline_rl/test_iql.py` (8 tests).

1. **expectile loss correctness** — τ=0.5 recovers MSE/2 weighting; τ=0.7
   hand-computed values checked.
2. **Bellman target with done flag** — `done=1` → target = r; `done=0` → r + γV(s').
3. **Advantage-weighted loss sign** — for A > 0, gradient pushes policy toward
   action; for A < 0, gradient pushes away.
4. **Twin Q clipped min** — element-wise `min(Q1, Q2)` correct on small batch.
5. **Policy output shape & range** — `predict_deterministic` shape `(B, 2)`,
   values in `[-1, 1]`.
6. **Smoke train artefacts finite** — 1 seed, 200 steps, tiny nets; all
   artefacts present; metrics.jsonl values finite.
7. **Best-checkpoint round trip** — saved `policy.pt` reproduces
   `best_val_policy_mse` to 1e-6 on reload.
8. **`IQLAgent.predict` returns 17 action vectors** — B5 from IQL, others from
   fresh `OfflineRBC`.

---

## 7. Run commands

### Smoke run

```bash
.venv/bin/python -m scripts.train_iql \
    --output runs/offline_iql/smoke \
    --seeds 100 \
    --gradient-steps 200 \
    --hidden-layers 64,64 \
    --eval-every 50
```

### Full run

```bash
nohup .venv/bin/python -m scripts.train_iql \
    --output runs/offline_iql/run-001 \
    --seeds 100,101,102,103,104 \
    --gradient-steps 150000 \
    > runs/offline_iql/run-001/train.log 2>&1 &
```

### Benchmark

```bash
.venv/bin/python -m scripts.benchmark_iql \
    --iql-root runs/offline_iql/run-001 \
    --bc-root runs/offline_bc/run-001 \
    --run-id iql_run001 \
    --output docs/offline_rl/benchmarks.md
```

---

## 8. Risks and contingencies

| Risk | Likelihood | Mitigation |
|---|---|---|
| Q divergence / policy collapse | medium | Twin Q + soft targets + grad clip 1.0. Halve LR if unstable. |
| Advantage saturation | medium | Hard clip at c=100. If many samples clip, lower β. |
| Reward magnitude wrong for γ=0.99 | low-medium | Scale reward by constant if V values blow up. |
| Policy too close to BC | medium | Try β=10 or τ=0.9 in a sweep (run-002+). |
| Unserved energy violation | low | Reward includes 50× penalty on unserved (frozen). |
| Wall-clock too long | low | If >5h CPU, reduce gradient steps to 100k. |

---

## 9. Results summary

See `pipeline_status.md` §4–5 and `benchmarks.md` §2–3 for full numbers.

**run-001** (trained on RBC data):
- best_val_policy_mse: 0.002182 ± 0.000078
- Building 5 cost: 2.634 ± 0.051 (RBC: 2.730 ± 0.081) — **IQL better** >1σ

**run-002** (trained on IQL-generated data, behaviour-policy swap):
- best_val_policy_mse: 0.000158 ± 0.000010 (13.8× lower than run-001)
- Building 5 cost: 2.666 ± 0.153 — slight regression vs run-001, wider std
- District `all_time_peak_average`: 1.8005 ± 0.0056 — **>1σ vs RBC** (1.8247)

**Conclusion**: run-001 (RBC data) achieves the Building 5 success criterion.
The iterative behaviour-policy swap (run-002) did not compound the improvement;
distributional narrowing from IQL-generated data reduces eval robustness.
