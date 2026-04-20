# Plan D: Wrapper/Encoding Integration Fixes

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix end-to-end wrapper/encoding integration gaps so Transformer agents receive properly enriched and encoded observations at runtime, with all token-type markers preserved through the encoding pipeline.

**Architecture:** The wrapper must (1) detect Transformer agents at `set_model()` time (not just `__init__`), (2) rebuild encoders to include marker encoders with NoNormalization, and (3) ensure the enriched observation names drive encoder construction rather than raw names.

**Tech Stack:** Python, NumPy, CityLearn wrapper, existing ObservationEnricher

---

## Problem Statement

Plans A/B/C delivered the TransformerPPO components (enricher, tokenizer, backbone, PPO, agent). Unit tests pass. However, **the runtime integration path has three blocking gaps**:

### Gap 1: Model Attachment Timing

`run_experiment.py` creates the wrapper **without** a model, then calls `wrapper.set_model(agent)`:

```python
# run_experiment.py:518-541
wrapper = Wrapper(env=env, config=config, job_id=job_id, ...)
...
agent = create_agent(config=config)
wrapper.set_model(agent)  # Agent attached AFTER wrapper init
```

But the wrapper's Transformer setup only runs in `__init__` when model is already present:

```python
# wrapper_citylearn.py:124-128
if hasattr(self.model, 'is_transformer_agent') and self.model.is_transformer_agent:
    self._setup_transformer_enrichers(self.model.tokenizer_config)
    for i in range(len(self.observation_names)):
        self._enrich_observation_names(i)
```

**Result:** `_is_transformer_agent` remains `False`, enrichers are never created.

### Gap 2: Encoder Mismatch

Even if enrichers were initialized, encoders are built from **raw** observation names:

```python
# wrapper_citylearn.py:617-651
def set_encoders(self) -> List[List[Encoder]]:
    ...
    for observation_group, space in zip(self.observation_names, self.observation_space):
        for index, name in enumerate(observation_group):
            # Uses self.observation_names which is RAW names
```

Enriched names (with markers like `__marker_1001__`) never drive encoder construction. Marker names have no matching encoder rule, so encoding would fail or skip them.

### Gap 3: Marker Values Lost

Without proper marker encoders (NoNormalization), marker values like `1001.0` would be:
- Normalized (destroying marker semantics)
- Or missing from the encoded output entirely

The tokenizer scans for marker values in specific ranges (1001-1999, 2001-2999, 3001). If markers are normalized or missing, tokenization fails silently (produces zero CA tokens).

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `utils/wrapper_citylearn.py` | Modify | Fix `set_model()` to initialize enrichers; add encoder rebuilding |
| `tests/test_wrapper_transformer.py` | Modify | Add tests for `set_model()` path and encoder rebuilding |
| `tests/test_wrapper_integration_e2e.py` | Modify | Add smoke test verifying markers survive through encoding |
| `configs/encoders/default.json` | Modify | Add marker pattern rules |

---

## Task 1: Add Marker Encoder Rule to Config

Add an encoder rule that matches marker names and applies NoNormalization.

**Files:**
- Modify: `configs/encoders/default.json`
- Test: Manual inspection (rule syntax only)

- [ ] **Step 1.1: Read current encoder rules**

Verify structure of existing rules before adding marker rule.

- [ ] **Step 1.2: Add marker rule as first rule**

Open `configs/encoders/default.json` and insert this rule at the beginning of the `rules` array:

```json
{
  "match": {
    "prefixes": ["__marker_"]
  },
  "encoder": {
    "type": "NoNormalization"
  }
}
```

The full file should look like:

```json
{
  "rules": [
    {
      "match": {
        "prefixes": ["__marker_"]
      },
      "encoder": {
        "type": "NoNormalization"
      }
    },
    {
      "match": {"equals": ["month", "hour"]},
      "encoder": {
        "type": "PeriodicNormalization",
        "params": {"x_max": "space_high"}
      }
    },
    ... (remaining existing rules)
  ]
}
```

The rule must be **first** because rules are matched in order, and we want marker names to match before any other pattern. Note: the existing default rule at the end also uses NoNormalization, but explicit matching is clearer and ensures markers don't accidentally match other patterns.

- [ ] **Step 1.3: Verify rule file is valid JSON**

Run: `python -c "import json; json.load(open('configs/encoders/default.json'))"`
Expected: No error

- [ ] **Step 1.4: Commit**

```bash
git add configs/encoders/default.json
git commit -m "feat(config): add NoNormalization encoder rule for marker names"
```

---

## Task 2: Fix set_model() to Initialize Transformer Enrichers

Move Transformer agent detection and enricher setup to `set_model()` path.

**Files:**
- Modify: `utils/wrapper_citylearn.py:206-226`
- Test: `tests/test_wrapper_transformer.py`

- [ ] **Step 2.1: Write failing test for set_model enricher initialization**

Add to `tests/test_wrapper_transformer.py`:

```python
def test_set_model_initializes_enrichers_for_transformer_agent(
    mock_env, transformer_agent_config, tmp_path
):
    """When set_model() receives a Transformer agent, enrichers should initialize."""
    from utils.wrapper_citylearn import Wrapper_CityLearn
    from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO
    import json
    
    # Create tokenizer config
    tokenizer_config = {
        "marker_values": {"ca_base": 1000, "sro_base": 2000, "nfc": 3001},
        "ca_types": {
            "battery": {
                "features": ["electrical_storage_soc"],
                "action_name": "electrical_storage",
                "input_dim": 1,
            }
        },
        "sro_types": {"temporal": {"features": ["month"], "input_dim": 2}},
        "nfc": {
            "demand_features": ["non_shiftable_load"],
            "generation_features": [],
            "extra_features": [],
            "input_dim": 1,
        },
    }
    tokenizer_path = tmp_path / "tokenizer.json"
    with open(tokenizer_path, "w") as f:
        json.dump(tokenizer_config, f)
    
    agent_config = {
        "algorithm": {
            "name": "AgentTransformerPPO",
            "tokenizer_config_path": str(tokenizer_path),
            "transformer": {"d_model": 32, "nhead": 2, "num_layers": 1},
            "hyperparameters": {
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_eps": 0.2,
                "ppo_epochs": 2,
                "minibatch_size": 4,
                "entropy_coeff": 0.01,
                "value_coeff": 0.5,
                "max_grad_norm": 0.5,
                "hidden_dim": 32,
                "rollout_length": 8,
            },
        }
    }
    
    # Create wrapper WITHOUT model (mimics run_experiment.py)
    wrapper_config = {
        "training": {},
        "simulator": {},
        "checkpointing": {},
        "tracking": {},
        "runtime": {},
    }
    wrapper = Wrapper_CityLearn(
        env=mock_env,
        model=None,  # No model at init
        config=wrapper_config,
        job_id="test"
    )
    
    # Verify no enrichers yet
    assert not getattr(wrapper, '_is_transformer_agent', False)
    
    # Create agent and call set_model
    agent = AgentTransformerPPO(agent_config)
    wrapper.set_model(agent)
    
    # Verify enrichers now initialized
    assert wrapper._is_transformer_agent is True
    assert len(wrapper._enrichers) == 1
    assert wrapper._enrichers[0] is not None
```

- [ ] **Step 2.2: Run test to verify it fails**

Run: `pytest tests/test_wrapper_transformer.py::test_set_model_initializes_enrichers_for_transformer_agent -v`
Expected: FAIL (enrichers not initialized via set_model path)

- [ ] **Step 2.3: Implement enricher initialization in set_model()**

Modify `utils/wrapper_citylearn.py`:

```python
def set_model(self, model: BaseAgent):
    """
    Set the model after initialization.
    """
    self.model = model
    
    # Initialize enrichers for Transformer agents (deferred from __init__)
    if hasattr(self.model, 'is_transformer_agent') and self.model.is_transformer_agent:
        self._setup_transformer_enrichers(self.model.tokenizer_config)
        for i in range(len(self.observation_names)):
            self._enrich_observation_names(i)
        # Rebuild encoders with enriched names
        self._rebuild_encoders_for_transformer()
    
    metadata = {
        "seconds_per_time_step": getattr(self.env, "seconds_per_time_step", None),
        "building_names": getattr(self.env, "building_names", None),
    }
    try:
        self.model.attach_environment(
            observation_names=self.observation_names,
            action_names=self.action_names,
            action_space=self.action_space,
            observation_space=self.observation_space,
            metadata=metadata,
        )
    except AttributeError:
        # Older agents may not implement attach_environment.
        pass
```

- [ ] **Step 2.4: Run test to verify it passes**

Run: `pytest tests/test_wrapper_transformer.py::test_set_model_initializes_enrichers_for_transformer_agent -v`
Expected: PASS

- [ ] **Step 2.5: Commit**

```bash
git add utils/wrapper_citylearn.py tests/test_wrapper_transformer.py
git commit -m "fix(wrapper): initialize transformer enrichers in set_model path"
```

---

## Task 3: Implement Encoder Rebuilding for Enriched Names

Add method to rebuild encoders using enriched observation names (with markers).

**Files:**
- Modify: `utils/wrapper_citylearn.py`
- Test: `tests/test_wrapper_transformer.py`

- [ ] **Step 3.1: Write failing test for encoder rebuilding**

Add to `tests/test_wrapper_transformer.py`:

```python
def test_encoders_include_marker_encoders_after_set_model(
    mock_env, tmp_path
):
    """Encoders should include NoNormalization for marker names after set_model."""
    from utils.wrapper_citylearn import Wrapper_CityLearn
    from utils.preprocessing import NoNormalization
    from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO
    import json
    
    tokenizer_config = {
        "marker_values": {"ca_base": 1000, "sro_base": 2000, "nfc": 3001},
        "ca_types": {
            "battery": {
                "features": ["electrical_storage_soc"],
                "action_name": "electrical_storage",
                "input_dim": 1,
            }
        },
        "sro_types": {"temporal": {"features": ["month"], "input_dim": 2}},
        "nfc": {
            "demand_features": ["non_shiftable_load"],
            "generation_features": [],
            "extra_features": [],
            "input_dim": 1,
        },
    }
    tokenizer_path = tmp_path / "tokenizer.json"
    with open(tokenizer_path, "w") as f:
        json.dump(tokenizer_config, f)
    
    agent_config = {
        "algorithm": {
            "name": "AgentTransformerPPO",
            "tokenizer_config_path": str(tokenizer_path),
            "transformer": {"d_model": 32, "nhead": 2, "num_layers": 1},
            "hyperparameters": {
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_eps": 0.2,
                "ppo_epochs": 2,
                "minibatch_size": 4,
                "entropy_coeff": 0.01,
                "value_coeff": 0.5,
                "max_grad_norm": 0.5,
                "hidden_dim": 32,
                "rollout_length": 8,
            },
        }
    }
    
    wrapper_config = {
        "training": {},
        "simulator": {},
        "checkpointing": {},
        "tracking": {},
        "runtime": {},
    }
    wrapper = Wrapper_CityLearn(
        env=mock_env,
        model=None,
        config=wrapper_config,
        job_id="test"
    )
    
    # Count encoders before set_model
    original_encoder_count = len(wrapper.encoders[0])
    
    agent = AgentTransformerPPO(agent_config)
    wrapper.set_model(agent)
    
    # Encoders should have more entries (markers added)
    new_encoder_count = len(wrapper.encoders[0])
    assert new_encoder_count > original_encoder_count, (
        f"Expected more encoders after adding markers, got {new_encoder_count} vs {original_encoder_count}"
    )
    
    # Check that some encoders are NoNormalization (for markers)
    marker_encoder_count = sum(
        1 for enc in wrapper.encoders[0] if isinstance(enc, NoNormalization)
    )
    assert marker_encoder_count >= 3, (
        f"Expected at least 3 marker encoders (CA, SRO, NFC), got {marker_encoder_count}"
    )
```

- [ ] **Step 3.2: Run test to verify it fails**

Run: `pytest tests/test_wrapper_transformer.py::test_encoders_include_marker_encoders_after_set_model -v`
Expected: FAIL (encoder count unchanged, no marker encoders)

- [ ] **Step 3.3: Implement _rebuild_encoders_for_transformer method**

Add to `utils/wrapper_citylearn.py`:

```python
def _rebuild_encoders_for_transformer(self) -> None:
    """Rebuild encoders using enriched observation names (Transformer agents only).
    
    This replaces the encoders built from raw observation names with encoders
    built from enriched names (which include marker placeholders). Marker names
    get NoNormalization encoders so their values pass through unchanged.
    """
    if not getattr(self, '_is_transformer_agent', False):
        return
    
    if not hasattr(self, '_enriched_observation_names'):
        return
    
    rules = _load_encoder_rules()
    new_encoders: List[List[Encoder]] = []
    
    for building_idx, space in enumerate(self.observation_space):
        enriched_names = self._enriched_observation_names.get(building_idx, [])
        if not enriched_names:
            # Fallback to existing encoders if no enrichment
            new_encoders.append(self.encoders[building_idx])
            continue
        
        group_encoders: List[Encoder] = []
        
        for index, name in enumerate(enriched_names):
            # Find matching rule
            rule = next((r for r in rules if _matches_rule(name, r.get("match", {}))), None)
            
            if rule is None:
                raise ValueError(
                    f"No encoder rule defined for observation '{name}'. "
                    f"Add a rule to configs/encoders/default.json."
                )
            
            if rule.get("warn_on_use"):
                logger.warning("Encoder rule warning for observation '{}'", name)
            
            # Build encoder - for markers, space bounds don't matter (NoNormalization ignores them)
            encoder = _build_encoder(rule, space, min(index, len(space.high) - 1))
            group_encoders.append(encoder)
        
        new_encoders.append(group_encoders)
    
    self.encoders = new_encoders
    logger.debug("Encoders rebuilt for Transformer agent with {} names per building", 
                 [len(enc) for enc in new_encoders])
```

- [ ] **Step 3.4: Run test to verify it passes**

Run: `pytest tests/test_wrapper_transformer.py::test_encoders_include_marker_encoders_after_set_model -v`
Expected: PASS

- [ ] **Step 3.5: Commit**

```bash
git add utils/wrapper_citylearn.py tests/test_wrapper_transformer.py
git commit -m "feat(wrapper): rebuild encoders with marker support for transformer agents"
```

---

## Task 4: Fix get_encoded_observations for Enriched Names

Ensure encoding uses enriched names count, not raw observation count.

**Files:**
- Modify: `utils/wrapper_citylearn.py:507-522`
- Test: `tests/test_wrapper_transformer.py`

- [ ] **Step 4.1: Write failing test for encoding dimension match**

Add to `tests/test_wrapper_transformer.py`:

```python
def test_encoded_observations_match_enriched_dimension(
    mock_env, tmp_path
):
    """Encoded observations should have length matching enriched names (including markers)."""
    from utils.wrapper_citylearn import Wrapper_CityLearn
    from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO
    import json
    
    tokenizer_config = {
        "marker_values": {"ca_base": 1000, "sro_base": 2000, "nfc": 3001},
        "ca_types": {
            "battery": {
                "features": ["electrical_storage_soc"],
                "action_name": "electrical_storage",
                "input_dim": 1,
            }
        },
        "sro_types": {"temporal": {"features": ["month"], "input_dim": 2}},
        "nfc": {
            "demand_features": ["non_shiftable_load"],
            "generation_features": [],
            "extra_features": [],
            "input_dim": 1,
        },
    }
    tokenizer_path = tmp_path / "tokenizer.json"
    with open(tokenizer_path, "w") as f:
        json.dump(tokenizer_config, f)
    
    agent_config = {
        "algorithm": {
            "name": "AgentTransformerPPO",
            "tokenizer_config_path": str(tokenizer_path),
            "transformer": {"d_model": 32, "nhead": 2, "num_layers": 1},
            "hyperparameters": {
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_eps": 0.2,
                "ppo_epochs": 2,
                "minibatch_size": 4,
                "entropy_coeff": 0.01,
                "value_coeff": 0.5,
                "max_grad_norm": 0.5,
                "hidden_dim": 32,
                "rollout_length": 8,
            },
        }
    }
    
    wrapper_config = {
        "training": {},
        "simulator": {},
        "checkpointing": {},
        "tracking": {},
        "runtime": {},
    }
    wrapper = Wrapper_CityLearn(env=mock_env, model=None, config=wrapper_config, job_id="test")
    
    agent = AgentTransformerPPO(agent_config)
    wrapper.set_model(agent)
    
    # Raw observation values (3 features)
    raw_obs = [0.5, 100.0, 6.0]
    
    # Encode
    encoded = wrapper.get_encoded_observations(0, raw_obs)
    
    # Should have markers (3) + encoded features
    # Markers: CA(1) + SRO(1) + NFC(1) = 3
    # Features: soc(1) + month(2 for periodic) + load(1) = 4
    # Total: 3 + 4 = 7 minimum (exact depends on encoding rules)
    assert len(encoded) >= 6, f"Expected at least 6 encoded values, got {len(encoded)}"
    
    # Verify marker values are present (1001, 2001, 3001)
    assert 1001.0 in encoded, f"CA marker 1001 not found in {encoded}"
    assert 2001.0 in encoded, f"SRO marker 2001 not found in {encoded}"
    assert 3001.0 in encoded, f"NFC marker 3001 not found in {encoded}"
```

- [ ] **Step 4.2: Run test to verify it fails**

Run: `pytest tests/test_wrapper_transformer.py::test_encoded_observations_match_enriched_dimension -v`
Expected: FAIL (markers not present in encoded output)

- [ ] **Step 4.3: Verify get_encoded_observations works with enriched values**

The current implementation at `wrapper_citylearn.py:507-522` should already work IF:
1. Enrichers are set up (Task 2)
2. Encoders are rebuilt (Task 3)

The key change is ensuring `_enrich_observation_values` is called BEFORE encoding, and the encoder list length matches the enriched observation count.

Check current flow:

```python
def get_encoded_observations(self, index: int, observations: List[float]) -> np.ndarray:
    # Enrich observations for Transformer agents
    if getattr(self, '_is_transformer_agent', False):
        observations = self._enrich_observation_values(index, observations)  # <-- adds markers

    obs_array = np.array(observations, dtype=np.float64)

    # Apply encoding transformation correctly
    encoded = np.hstack([
        encoder.transform(obs) if hasattr(encoder, "transform") else encoder * obs
        for encoder, obs in zip(self.encoders[index], obs_array)  # <-- must zip enriched
    ])
```

If `self.encoders[index]` was rebuilt with enriched names, the zip should work correctly.

- [ ] **Step 4.4: Run test to verify it passes**

Run: `pytest tests/test_wrapper_transformer.py::test_encoded_observations_match_enriched_dimension -v`
Expected: PASS

- [ ] **Step 4.5: Commit**

```bash
git add tests/test_wrapper_transformer.py
git commit -m "test(wrapper): verify encoded observations include markers"
```

---

## Task 5: Add E2E Smoke Test for Marker Preservation

Verify markers survive through the full predict() path.

**Files:**
- Modify: `tests/test_wrapper_integration_e2e.py`
- Test: Self-contained

- [ ] **Step 5.1: Add smoke test for marker preservation through predict**

Add to `tests/test_wrapper_integration_e2e.py`:

```python
def test_markers_survive_through_predict_path(
    self, integration_setup: Dict[str, Any], tmp_path: Path
) -> None:
    """Markers should survive from raw observation through encoding to agent."""
    from utils.wrapper_citylearn import Wrapper_CityLearn
    import json
    
    agent = integration_setup["agent"]
    tokenizer_config = integration_setup["tokenizer_config"]
    
    # Minimal mock env matching the tokenizer config
    class MinimalEnv:
        def __init__(self):
            # 3 raw observations: soc, load, month
            self.observation_names = [["electrical_storage_soc", "non_shiftable_load", "month"]]
            self.action_names = [["electrical_storage"]]
            self.observation_space = [
                type("space", (), {
                    "high": np.array([1.0, 1000.0, 12.0]),
                    "low": np.array([0.0, 0.0, 1.0]),
                })()
            ]
            self.action_space = [
                type("space", (), {
                    "high": np.array([1.0]),
                    "low": np.array([-1.0]),
                })()
            ]
            self.reward_function = type("reward", (), {"__dict__": {}})()
            self.time_steps = 8760
            self.seconds_per_time_step = 3600
            self.time_step_ratio = 1.0
            self.random_seed = 0
            self.episode_tracker = type("tracker", (), {"episode_time_steps": self.time_steps})()
            self.unwrapped = self
        
        def get_metadata(self):
            return {"buildings": [{}]}
    
    mock_env = MinimalEnv()
    
    wrapper_config = {
        "environment": {"buildings": ["Building_1"]},
        "algorithm": integration_setup["agent_config"]["algorithm"],
        "training": {},
        "simulator": {},
        "checkpointing": {},
        "tracking": {},
        "runtime": {},
    }
    
    # Create wrapper WITHOUT model, then set_model (matches run_experiment.py)
    wrapper = Wrapper_CityLearn(
        env=mock_env,
        model=None,
        config=wrapper_config,
        job_id="test"
    )
    wrapper.set_model(agent)
    
    # Verify setup
    assert wrapper._is_transformer_agent is True
    
    # Raw observations
    raw_obs = [[0.5, 100.0, 6.0]]  # soc=0.5, load=100, month=6
    
    # Get encoded observations (what predict() does internally)
    encoded = wrapper.get_all_encoded_observations(raw_obs)
    
    # Verify markers present
    encoded_flat = encoded[0]
    assert 1001.0 in encoded_flat, f"CA marker missing from encoded: {encoded_flat}"
    assert 2001.0 in encoded_flat, f"SRO marker missing from encoded: {encoded_flat}"
    assert 3001.0 in encoded_flat, f"NFC marker missing from encoded: {encoded_flat}"
    
    # Call predict - should produce valid actions
    actions = wrapper.predict(raw_obs, deterministic=True)
    
    assert len(actions) == 1
    assert actions[0].shape[-1] == 1  # One CA = one action
    assert (actions[0] >= -1.0).all() and (actions[0] <= 1.0).all()
```

- [ ] **Step 5.2: Run test to verify it passes**

Run: `pytest tests/test_wrapper_integration_e2e.py::TestWrapperAgentIntegration::test_markers_survive_through_predict_path -v`
Expected: PASS

- [ ] **Step 5.3: Commit**

```bash
git add tests/test_wrapper_integration_e2e.py
git commit -m "test(e2e): verify markers survive through predict path"
```

---

## Task 6: Run Full Test Suite and Verify No Regressions

**Files:**
- None (verification only)

- [ ] **Step 6.1: Run Transformer-related tests**

Run: `pytest tests/ -k "transformer or wrapper or enricher or tokenizer" -v`
Expected: All tests pass

- [ ] **Step 6.2: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass (no regressions in MADDPG, RBC, etc.)

- [ ] **Step 6.3: Run a short training smoke test (optional)**

Run:
```bash
source .venv/bin/activate
python run_experiment.py --config configs/templates/transformer_ppo.yaml --job_id test_plan_d
```

Expected: Training starts without errors, produces valid actions

- [ ] **Step 6.4: Commit any final fixes**

If any issues discovered, fix and commit before proceeding.

---

## Task 7: Clean Up and Final Commit

- [ ] **Step 7.1: Remove any debug prints or temporary code**

Search for debug statements:
```bash
grep -rn "print(" utils/wrapper_citylearn.py algorithms/
```

Remove any temporary debugging code.

- [ ] **Step 7.2: Update AGENTS.md if needed**

If any new developer-facing changes were made, update documentation.

- [ ] **Step 7.3: Final commit**

```bash
git add -A
git commit -m "feat(plan-d): complete wrapper/encoding integration for transformer agents"
```

---

## Verification Checklist

After completing all tasks, verify:

- [ ] `wrapper.set_model(agent)` initializes enrichers for TransformerPPO agents
- [ ] Encoders are rebuilt to include marker encoders (NoNormalization)
- [ ] Encoded observations contain marker values (1001, 2001, 3001)
- [ ] Agent produces valid actions (not all zeros, not all same value)
- [ ] Full test suite passes (no regressions)
- [ ] Training runs without errors for at least 10 steps

---

## Rollback Plan

If issues are discovered after deployment:

1. The changes are isolated to wrapper behavior for Transformer agents only
2. Non-Transformer agents (MADDPG, RBC) are unaffected
3. Revert commits in reverse order if needed:
   ```bash
   git revert HEAD~N..HEAD
   ```

---

## Dependencies

This plan has no external dependencies. All changes are within the existing codebase.

**Prerequisite:** Plans A/B/C completed (TransformerPPO agent, enricher, tokenizer exist)
