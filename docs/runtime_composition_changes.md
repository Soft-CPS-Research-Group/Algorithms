# Runtime Composition Changes — Production Handling

## Overview

This document explains how the TransformerPPO agent handles changes in asset availability during runtime, which is critical for production deployment.

## Architecture Constraints

The agent operates on **encoded observations** from the wrapper, not raw sensor data. The flow is:

```
Raw sensors → Wrapper (encoding) → Agent (tokenization) → Actions
```

**Key Constraint**: The tokenizer's index buffers (`_ca_idx_i`, `_sro_idx_i`) are computed once during `attach_environment()` based on the initial observation space.

## What "Runtime Composition Changes" Means

There are two distinct scenarios:

### Scenario 1: Different Buildings (Checkpoint Transfer) ✅ SUPPORTED
**Example**: Train on Building_4 (2 CAs), deploy to Building_2 (1 CA)

**Solution**: Load checkpoint with cross-topology transfer
- Global vocabulary includes all CA types seen during training
- Missing projections are pre-allocated with checkpoint dimensions
- Optimizer state is skipped for cross-topology transfers

**Status**: ✅ Fully implemented and tested (Phase 2c validation)

### Scenario 2: Same Building, Asset Goes Offline Mid-Run ⚠️  PARTIALLY SUPPORTED
**Example**: EV charger disconnects during operation

**Current Behavior**:
- Wrapper continues to provide encoded observations with the same dimension
- EV features become zeros (connected_state=0, soc=0, etc.)
- Agent receives zero-valued features and predicts actions
- EV action output is present but should be masked in production

**Production Handling**:
1. **Observation encoding**: Wrapper fills missing features with zeros
2. **Action masking**: Check `connected_state` or similar flags before applying actions
3. **Graceful degradation**: Agent continues operating with available assets

## Production Deployment Recommendations

### 1. Asset Availability Monitoring

```python
def apply_actions_with_masking(observations, actions, asset_metadata):
    """Apply actions only to available assets."""
    masked_actions = actions.copy()
    
    # Example: Check EV connected_state
    ev_connected_idx = asset_metadata["observation_indices"]["ev_connected_state"]
    if observations[ev_connected_idx] < 0.5:  # EV disconnected
        ev_action_idx = asset_metadata["action_indices"]["ev_charger"]
        masked_actions[ev_action_idx] = 0.0  # Disable charging
    
    return masked_actions
```

### 2. Handling New Assets (Not Seen During Training)

**Option A**: Re-deploy with new checkpoint trained on updated topology
- Train on building with new asset configuration
- Deploy new model

**Option B**: Default behavior for unknown assets
- New asset observations will be ignored (no projection exists)
- Action for new asset will be zero (no corresponding CA token)

**Recommendation**: Use Option A for production - retrain and redeploy when asset composition changes permanently.

### 3. Temporary Asset Outages

**Best Practice**:
- Keep agent running with same model
- Wrapper provides zero-filled observations for offline assets
- Apply action masking based on availability flags
- Agent continues to optimize available assets

**Example**:
```python
# Battery fails, EV charger continues
observations = {
    "battery_soc": 0.0,  # Offline - zero-filled
    "ev_connected": 1.0,
    "ev_soc": 0.3,
    ...
}

actions = agent.predict(observations)

# Mask battery action
if battery_offline:
    actions["battery"] = 0.0  # No charge/discharge
```

### 4. Validation Strategy

**Phase 2d Test Plan**:
1. Train on Building_4 (battery + 1 EV)
2. Simulate EV disconnection by zeroing EV observations
3. Verify agent continues to optimize battery
4. Compare KPIs: full operation vs. partial operation

## Limitations

### What CANNOT Be Handled Mid-Run

1. **Adding new CA types not in global vocabulary**
   - If training never saw a washing machine, can't handle it at runtime
   - Would need to redeploy with updated model

2. **Changing observation encoding schema**
   - Index buffers are fixed after `attach_environment()`
   - Changing feature order/encoding requires rebuild

3. **Variable observation dimensions**
   - Encoded observation must have fixed size
   - Cannot dynamically resize mid-episode

### What CAN Be Handled

1. **Zero-valued observations** ✅
   - Agent processes zeros gracefully
   - Produces actions (may be suboptimal for offline assets)

2. **NaN observations** ⚠️
   - PyTorch propagates NaN through network
   - **Production recommendation**: Replace NaN with zeros or last-known values before calling agent

3. **Cross-topology deployment** ✅
   - Load checkpoint trained on different building
   - Global vocabulary enables weight sharing

## Implementation Notes

### Global Vocabulary System

The agent pre-allocates projections for all CA types in the tokenizer config, even if not present in the current building:

```python
# Building_4 trains with: battery, ev_charger
# Building_2 has only: battery
# Global vocab includes: battery, ev_charger, washing_machine (from config)

# When deployed to Building_2:
# - battery projection loaded from checkpoint ✅
# - ev_charger projection exists but unused ✅ 
# - washing_machine projection exists but unused ✅
```

### Checkpoint CA Dimension Extraction

When loading a checkpoint for cross-topology transfer:

```python
# Extract actual trained dimensions from checkpoint
checkpoint_ca_dims = agent.extract_checkpoint_ca_dims(checkpoint_path)
# {'battery': 1, 'ev_charger': 61, 'washing_machine': 2}

# Use these instead of fallback estimates during attach_environment
# Prevents shape mismatches when loading weights
```

## Testing

### Unit Tests
- `test_runtime_adaptability.py`: Pre-allocation, checkpoint compatibility
- `test_runtime_composition_changes.py`: Asset offline scenarios (documents current behavior)

### Integration Tests (Validation Phases)
- **Phase 1**: Baseline on Building_4
- **Phase 2a**: Checkpoint saving on Building_4
- **Phase 2b/c**: Cross-topology transfer to Building_2/15 ✅ PASSED
- **Phase 3**: KPI comparison vs baselines
- **Phase 4**: Multi-building scale test

## Summary

| Scenario | Supported | Requires Action |
|----------|-----------|-----------------|
| Different building topology (checkpoint) | ✅ Yes | Load with cross-topology transfer |
| Asset goes offline temporarily | ⚠️ Partial | Apply action masking in production |
| Asset comes back online | ✅ Yes | Remove action mask |
| New asset type (not in training) | ❌ No | Retrain and redeploy |
| NaN observations | ⚠️ Propagates | Replace with zeros before agent |
| Zero observations | ✅ Yes | Agent handles gracefully |

**Production Recommendation**: Use checkpoint transfer for different buildings, action masking for temporary outages, and retraining for permanent composition changes.
