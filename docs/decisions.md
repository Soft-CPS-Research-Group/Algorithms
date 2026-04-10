# Decision Log

Architectural and implementation decisions for the TransformerPPO agent.

---

## Decision #1 — Action Distribution: Squashed Gaussian

**Date:** 2026-04-06
**Status:** Accepted

### Context

PPO requires a continuous probability distribution over actions to compute log-probabilities for the clipped surrogate objective. Actions are scalar values in `[-1, 1]` (one per controllable asset).

### Options Considered

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **Squashed Gaussian** | Sample from `Normal(μ, σ)`, apply `tanh` to bound to `[-1, 1]` | Industry standard (CleanRL, SB3); well-documented; easy to debug | Requires log-prob correction for the tanh transform; can saturate at boundaries |
| **Beta distribution** | Naturally bounded to `[0, 1]`, rescaled to `[-1, 1]` | No squashing correction needed; naturally respects bounds | Less common; fewer reference implementations; parameterization can be unstable |

### Decision

**Squashed Gaussian.** It is the standard approach for continuous-action PPO, with extensive reference implementations and literature. The tanh log-prob correction is well-understood and adds minimal complexity.

### Revisit If

Action saturation at `±1` boundaries becomes a recurring problem during training, or if the log-prob correction introduces numerical instability.

---

## Decision #2 — Cross-Topology Transfer: Requires Actual Topology Presence

**Date:** 2026-04-08
**Status:** Accepted

### Context

Phase A (pre-allocation) aims to enable cross-topology checkpoint transfer by allocating projections for all CA/SRO types in the config vocabulary. However, global vocabulary computation uses actual observed feature dimensions from the buildings present at `attach_environment()` time. If a CA type (e.g., `ev_charger`) is not observed in any building, the system uses a fallback dimension estimate based on the number of feature patterns in the config.

The fallback estimate (e.g., 7 dims for EV chargers) does NOT match the actual encoded dims when that CA type is present in a real building (e.g., 61 dims for EV chargers in Building_1). This causes weight shape mismatches when loading checkpoints across topologies where one topology used fallback estimates and the other used actual dims.

### Options Considered

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **Use fallback estimates** (current) | When a CA type is unseen, estimate dims from config patterns | Simple; doesn't require scanning full dataset | Fallback dims don't match actual dims; checkpoint transfer fails |
| **Require full topology at train time** | Train on a dataset that includes all CA types to observe actual dims | Accurate dims; checkpoint transfer works | Requires careful dataset selection; can't train on minimal subsets |
| **Pre-compute global vocabulary from dataset metadata** | Scan entire dataset schema at startup to get true dims for all types | Most accurate; works for any training subset | Requires dataset-level metadata; adds complexity; breaks encapsulation |

### Decision

**Require full topology at train time.** For cross-topology transfer to work reliably, the training dataset must include at least one building with each CA type that the model might encounter at inference time. This ensures all projection layers are sized correctly based on actual encoded dims, not fallback estimates.

### Implementation Notes

- The global vocabulary computation (`_compute_global_vocabulary`) uses actual observed dims when CA types are present.
- Fallback estimates remain as a safety mechanism but are not sufficient for checkpoint compatibility.
- For the validation phases, we will train on Building_1 (which has battery + EV + washing machine) to ensure all CA types are observed, then load into Building_2 (battery only), Building_4 (battery + EV), and Building_15 (battery + 2 EVs).

### Revisit If

Users require training on minimal single-CA datasets and transferring to multi-CA topologies. In that case, we would need to implement dataset-level metadata scanning or require explicit CA type dimension configuration in the config file.
