# TD3+BC Implementation Guide

## Overview

This guide outlines how to implement TD3+BC (Twin Delayed DDPG + Behavior Cloning) as your first Offline RL algorithm, leveraging the existing repository architecture.

---

## Implementation Feasibility: ✅ Fully Possible

The architecture is well-designed for extension. The existing MADDPG implementation provides a solid foundation to build upon.

---

## Files to Create

| File | Purpose |
|------|---------|
| `algorithms/agents/td3bc_agent.py` | TD3+BC implementation |
| `algorithms/utils/offline_replay_buffer.py` | Buffer that loads from CSV/static data |

## Files to Modify

| File | Change |
|------|--------|
| `algorithms/registry.py` | Add `"TD3BC": TD3BC` to `ALGORITHM_REGISTRY` |
| `configs/config.yaml` | Add TD3BC hyperparameters section |
| `utils/config_schema.py` | Add validation for TD3BC config (optional but recommended) |

---

## Key Implementation Differences (Online vs Offline)

| Aspect | MADDPG (Online) | TD3BC (Offline) |
|--------|-----------------|-----------------|
| **Data source** | `replay_buffer.push()` from env | Pre-loaded from CSV at init |
| **`update()` method** | Receives live transitions | Samples from static buffer only |
| **Exploration** | Gaussian noise in `predict()` | **None** — uses dataset actions for BC |
| **Actor loss** | `-Q(s,a)` | `-Q(s,a) + α * BC_loss` |

---

## TD3BC Actor Loss (The Core Change)

```python
# MADDPG actor loss (line 233 in maddpg_agent.py):
actor_loss = -critic(global_state, global_predicted_actions).mean()

# TD3BC actor loss:
q_loss = -critic(global_state, predicted_actions).mean()
bc_loss = F.mse_loss(predicted_actions, dataset_actions)
actor_loss = q_loss + self.alpha * bc_loss  # α ≈ 2.5
```

The BC regularization term penalizes the actor for deviating from the actions present in the dataset, preventing out-of-distribution action selection.

---

## What You Can Reuse from MADDPG

| Component | Reusable? | Notes |
|-----------|-----------|-------|
| `Actor` / `Critic` networks | ✅ Directly | Same architecture works |
| `_soft_update()` | ✅ Directly | Target network updates |
| `export_artifacts()` | ✅ Copy and adapt | ONNX export logic |
| `save_checkpoint()` / `load_checkpoint()` | ✅ Copy and adapt | Checkpointing logic |
| `_initialize_networks()` | ✅ With minor changes | Network initialization |
| Replay buffer | ❌ Need new | Offline-specific buffer required |

---

## Skeleton Structure

```python
from algorithms.agents.base_agent import BaseAgent

class TD3BC(BaseAgent):
    """TD3+BC: Offline RL with Behavior Cloning regularization."""
    
    def __init__(self, config: dict):
        super().__init__()
        # Load hyperparams (alpha, gamma, tau, batch_size)
        # Initialize networks (same as MADDPG)
        # Load dataset into OfflineReplayBuffer
        
    def predict(self, observations, deterministic=True):
        """Always deterministic for offline (no exploration)."""
        # Return actor output without noise
        
    def update(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        terminated,
        truncated,
        *,
        update_target_step,
        global_learning_step,
        update_step,
        initial_exploration_done,
    ):
        """Train from static buffer, ignore incoming observations."""
        # Ignore incoming observations (offline doesn't use them)
        # Sample batch from static buffer
        # Compute TD3+BC losses (critic + actor with BC term)
        # Update networks
        # Soft update targets if update_target_step
        
    def export_artifacts(self, output_dir, context):
        """Export ONNX models (same pattern as MADDPG)."""
        # Same as MADDPG (export ONNX)
```

---

## Critical Design Decision: Dataset Format

You need to define how your `OfflineReplayBuffer` loads data. Two options:

### Option 1: Pre-collected Trajectories (Recommended to start)
Run RBC/MADDPG, save transitions to CSV/pickle, load at init.

```python
class OfflineReplayBuffer:
    def __init__(self, dataset_path: str, batch_size: int):
        # Load pre-saved transitions
        self.data = self._load_dataset(dataset_path)
        
    def sample(self, batch_size: int):
        # Random sample from static data
        indices = np.random.choice(len(self.data), batch_size)
        return self.data[indices]
```

### Option 2: Generate On-the-fly
Load building CSVs + run RBC in simulation mode to generate actions.

**Recommendation:** Start with Option 1 — create a script that runs your existing RBC/MADDPG, saves transitions, then load that file in TD3BC.

---

## Hyperparameters

| Parameter | Typical Range | Recommended Start |
|-----------|---------------|-------------------|
| `alpha` (BC weight) | 0.1 - 10.0 | **2.5** |
| `gamma` (discount) | 0.95 - 0.99 | 0.99 |
| `tau` (soft update) | 0.001 - 0.01 | 0.005 |
| `batch_size` | 128 - 1024 | 256 |
| `lr_actor` | 1e-5 - 1e-3 | 3e-4 |
| `lr_critic` | 1e-5 - 1e-3 | 3e-4 |

---

## Registration Example

In `algorithms/registry.py`:

```python
from algorithms.agents.td3bc_agent import TD3BC

ALGORITHM_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "MADDPG": MADDPG,
    "RuleBasedPolicy": RuleBasedPolicy,
    "TD3BC": TD3BC,  # Add this line
}
```

---

## Next Steps After TD3BC

Once TD3BC is working:

1. **Validate** — Compare vs RBC baseline on same building
2. **Implement IQL** — Better for zero-shot generalization (avoids OOD actions entirely)
3. **Add feature engineering** — Static fingerprints, temporal encodings
4. **Experiment with data augmentation** — TimeGAN, meteorological perturbation

---

## References

- [TD3+BC Paper](https://arxiv.org/abs/2106.06860) — Fujimoto & Gu, 2021
- Existing MADDPG implementation: `algorithms/agents/maddpg_agent.py`
- Base agent contract: `algorithms/agents/base_agent.py`
