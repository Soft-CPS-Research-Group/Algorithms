# IQL Implementation Plan (run-001)

Frozen build plan for Step 5. Source spec: `docs/offline_rl/specs/iql_design.md` (approved). Mirrors the BC build (`algorithms/offline_rl/bc_*`) for consistency.

---

## Order

TDD with one module green before the next:

1. `tests/offline_rl/test_iql.py` — write all 8 tests first (fail).
2. `algorithms/offline_rl/iql_networks.py` — `MLP`, `QNetwork`, `ValueNetwork`, `GaussianPolicy`. Tests 1, 4, 5 green.
3. `algorithms/offline_rl/iql_dataset.py` — `IQLDataset`, `load_iql_split`. Reuses `bc_dataset.ObservationStandardiser`. (No new tests; covered by trainer smoke.)
4. `algorithms/offline_rl/iql_trainer.py` — `IQLTrainingConfig`, `train_single_seed`, `train_multi_seed`, `expectile_loss`, `bellman_target`. Tests 2, 3, 6, 7 green.
5. `algorithms/offline_rl/iql_agent.py` — `IQLAgent(BaseAgent)`, mirrors `BCAgent`. Test 8 green.
6. `scripts/train_iql.py` — multi-seed driver (mirror `train_bc.py`).
7. `scripts/benchmark_iql.py` — 3-column report (Random | RBC | BC | IQL → trim per spec to RBC | BC | IQL); reuses `_benchmark_common.render_kpi_table_four`.
8. Smoke run → full run → benchmark → status doc.

Pytest must remain green after each module; final target = **121/121 passing** (113 BC/RBC/reward + 8 IQL).

---

## Module surface

### `algorithms/offline_rl/iql_networks.py`

```python
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 256), dropout=0.1): ...

class QNetwork(nn.Module):
    """Q(s, a) → scalar. MLP over concat(s, a)."""
    def __init__(self, obs_dim, action_dim, hidden, dropout): ...
    def forward(self, obs, action) -> Tensor: ...  # (B,)

class ValueNetwork(nn.Module):
    """V(s) → scalar."""
    def __init__(self, obs_dim, hidden, dropout): ...
    def forward(self, obs) -> Tensor: ...  # (B,)

class GaussianPolicy(nn.Module):
    """Diagonal Gaussian over actions, tanh-squashed mean.
    log_std is a learned per-action nn.Parameter, init log(0.1).
    """
    def __init__(self, obs_dim, action_dim, hidden, dropout, log_std_init=-2.302585): ...
    def forward(self, obs) -> Tuple[Tensor, Tensor]:
        # returns (tanh_mean (B, A), log_std (A,))
    def log_prob(self, obs, action) -> Tensor:
        # diag-Gaussian log-prob of dataset action under N(mean, exp(log_std)^2)
        # action passed in is in [-1, 1] (dataset values); we score under the
        # Gaussian on the pre-tanh "mean" head — i.e. AWR objective regresses
        # the tanh(mean) toward action with isotropic variance (standard IQL
        # impl shortcut; avoids tanh-jacobian correction noise).
    def predict_deterministic(self, obs) -> Tensor:
        # tanh(mean), shape (B, A)
    def architecture_summary(self) -> dict: ...
```

### `algorithms/offline_rl/iql_dataset.py`

```python
class IQLDataset(torch.utils.data.Dataset):
    """Holds (s, a, r, s', done) tensors in memory."""
    def __init__(self, obs_std, actions, rewards, next_obs_std, dones): ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx): ...
    def sample(self, batch_size, generator=None) -> tuple[Tensor, ...]: ...

@dataclass
class IQLSplit:
    train: IQLDataset
    val: IQLDataset                        # for policy MSE proxy
    standardiser: ObservationStandardiser
    train_indices: np.ndarray
    val_indices: np.ndarray
    obs_feature_names: list[str]
    action_feature_names: list[str]

def load_iql_split(parquet_path, *, val_fraction=0.1, seed) -> IQLSplit:
    # Read OBS, NEXT_OBS, ACTION, REWARD, terminated columns.
    # Assert truncated.sum() == 0 (fail-fast).
    # Fit ObservationStandardiser on TRAIN obs only; apply to both s and s'.
```

### `algorithms/offline_rl/iql_trainer.py`

Pure functions for losses (testable without a network):

```python
def expectile_loss(diff: Tensor, tau: float) -> Tensor:
    weight = torch.where(diff < 0, tau, 1.0 - tau)
    # Wait — convention check: L_tau(u) = |tau - 1{u<0}| u^2.
    # If u < 0: weight = |tau - 1| = 1 - tau  (under-prediction → small u<0 if pred>target, etc.)
    # IQL paper: u = Q_target - V_pred. We want under-prediction (V too low) penalised more, so tau=0.7
    # gives weight 0.7 when u > 0 (V < Q), weight 0.3 when u < 0 (V > Q).
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return weight * diff.pow(2)

def bellman_target(reward, gamma, next_v, done) -> Tensor:
    return reward + gamma * (1.0 - done) * next_v
```

`IQLTrainingConfig`: dataclass mirroring `BCTrainingConfig` with extra fields:
`tau_expectile=0.7`, `beta_advantage=3.0`, `advantage_clip=100.0`, `gamma=0.99`,
`tau_target=0.005`, `gradient_steps=150_000`, `eval_every_n_steps=2500`,
`hidden_layers=[256,256]`, `dropout=0.1`, `lr=3e-4`, `weight_decay=1e-5`,
`batch_size=256`, `gradient_clip_norm=1.0`, `val_fraction=0.1`, `device="cpu"`.

`train_single_seed(parquet, output_dir, *, seed, config) -> dict`:
- Build `IQLSplit`; build `policy, q1, q2, v, q1_target, q2_target` (no V target net).
- Optimisers: separate Adam for policy / Q (joint Q1+Q2) / V.
- Loop `gradient_steps`:
  1. Sample batch (s, a, r, s', done).
  2. **V update**: `q_target_min = min(q1_target(s,a), q2_target(s,a)).detach()`; `v_pred = v(s)`; `v_loss = expectile_loss(q_target_min - v_pred, tau).mean()`; backward, clip, step.
  3. **Q update**: `with torch.no_grad(): tgt = bellman_target(r, gamma, v(s'), done)`; `q1_loss = mse(q1(s,a), tgt)`; ditto q2; sum, backward, clip, step.
  4. **Policy update**: `with torch.no_grad(): adv = q_target_min - v(s)`; `weight = clamp(exp(beta * adv), max=advantage_clip)`; `lp = policy.log_prob(s, a)`; `policy_loss = -(weight * lp).mean()`; backward, clip, step.
  5. **Soft update**: `q_target ← (1-tau_target)*q_target + tau_target*q` for both Q.
  6. Every `eval_every_n_steps`: compute val policy MSE (deterministic); log metrics.jsonl line `{step, v_loss, q_loss, policy_loss, val_policy_mse, adv_mean, adv_std, adv_clip_frac}`; update best snapshot.
- Persist `policy.pt` (best snapshot), `q1.pt`, `q2.pt`, `value.pt`, `obs_standardiser.npz`, `architecture.json`, `seed_summary.json`.

`train_multi_seed(...)` mirrors BC.

### `algorithms/offline_rl/iql_agent.py`

```python
class IQLAgent(BaseAgent):
    """Same shape as BCAgent — controls B5 with the trained Gaussian
    policy (deterministic = tanh(mean)); defers other 16 buildings to OfflineRBC.
    """
    @classmethod
    def from_seed_dir(cls, seed_dir, *, rbc_config=None, device="cpu") -> "IQLAgent": ...
```

Identical control flow to `BCAgent.predict`, but uses `policy.predict_deterministic(obs)`.

---

## Tests (8)

Final list (file: `tests/offline_rl/test_iql.py`):

1. `test_expectile_loss_correctness` — for `tau=0.5` recovers MSE/2 weighting; for `tau=0.7`, hand-computed values: `expectile_loss(diff=+1, tau=0.7) == 0.7`, `expectile_loss(diff=-1, tau=0.7) == 0.3`.
2. `test_bellman_target_with_done` — `done=1` → target = r exactly; `done=0` → target = r + γ·V(s').
3. `test_advantage_weighted_loss_sign` — fix policy + obs + action; for synthetic batch where `adv > 0`, gradient of policy_loss wrt mean parameters pushes `tanh(mean)` toward action.
4. `test_twin_q_clipped_min` — element-wise `min(Q1, Q2)` correct on a small batch.
5. `test_policy_output_shape_and_range` — `predict_deterministic` shape `(B, 2)` and values in `[-1, 1]`.
6. `test_smoke_train_artifacts_finite` — 1 seed, 200 steps, tiny nets — produces all artefacts; metrics.jsonl values finite.
7. `test_best_checkpoint_round_trip` — saved `policy.pt` reproduces `best_val_policy_mse` to `1e-6`.
8. `test_iql_agent_predict_returns_17` — mirror BC test 6: B5 from IQL, others from fresh `OfflineRBC` (identical).

---

## Smoke run command

```bash
.venv/bin/python -m scripts.train_iql \
    --output runs/offline_iql/smoke \
    --seeds 100 \
    --gradient-steps 200 \
    --hidden-layers 64,64 \
    --eval-every 50
```

Expected: ~30s, all artefacts written, `metrics.jsonl` has 4 lines with finite scalars.

## Full run command

```bash
nohup .venv/bin/python -m scripts.train_iql \
    --output runs/offline_iql/run-001 \
    --seeds 100,101,102,103,104 \
    --gradient-steps 150000 \
    > runs/offline_iql/run-001/train.log 2>&1 &
```

Expected wall-clock: ~30–40 min/seed × 5 ≈ 3 h on CPU.

## Benchmark command

```bash
.venv/bin/python -m scripts.benchmark_iql \
    --iql-root runs/offline_iql/run-001 \
    --bc-root runs/offline_bc/run-001 \
    --output docs/offline_rl/iql_vs_rbc_benchmark.md
```

Renders RBC | BC | IQL (3 columns, Δ = IQL − RBC, verdict on IQL vs RBC).

---

## Definition of done

- pytest 121/121 green.
- `runs/offline_iql/run-001/seed_{100..104}/` populated with `policy.pt`, `q1.pt`, `q2.pt`, `value.pt`, `obs_standardiser.npz`, `metrics.jsonl`, `seed_summary.json`, `architecture.json`.
- `docs/offline_rl/iql_vs_rbc_benchmark.md` generated.
- `docs/offline_rl/step5_iql_status.md` written.
- `docs/offline_rl/pipeline_status.md` updated with IQL section.
- Commit: `[offline-rl] Step 5: IQL implementation + benchmark`.
