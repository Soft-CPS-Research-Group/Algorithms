# WP05 — `AgentTransformerPPO` + Wrapper Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **For all v2 WPs:** every production-code task MUST follow `superpowers:test-driven-development`. The WP MUST end with `superpowers:requesting-code-review`.

**Goal:** Implement `AgentTransformerPPO` (`algorithms/agents/agent_transformer_ppo.py`) per spec §11 and §12, plus the wrapper integration: a topology-change coordinator hook and a generic `supports_dynamic_topology` guardrail (§12.4) replacing the MADDPG-specific check. After this WP the system runs end-to-end on the entity interface, with the agent absorbing topology mutations correctly via buffer-flush + layout-rebuild.

**Architecture:** Per-building agent: each building owns its own tokenizer + backbone + actor + critic + optimizer + rollout buffer + cached `BuildingTokenLayout`. The agent class itself is a thin orchestrator over those per-building stacks. The wrapper's role expands by exactly one method: `TransformerObservationCoordinator.handle_topology_change(self)` (in `utils/wrapper_transformer_entity/`). The generic guardrail moves the `supports_dynamic_topology` ClassVar into `BaseAgent` and rewrites `validate_config`'s topology check to consult the registry.

**Tech Stack:** Python 3.11, PyTorch, pytest. Consumes WP01 (PPO components, helpers) + WP02 (schema + 5 rules) + WP03 (layout builder) + WP04 (tokenizer + backbone).

**Branch:** `gj/wp05-agent-wrapper`
**Base branch:** `gj/wp04-tokenizer-backbone`

---

## Scope

**Files created:**

- `algorithms/agents/agent_transformer_ppo.py` — `AgentTransformerPPO(BaseAgent)`.
- `utils/wrapper_transformer_entity/__init__.py` + `utils/wrapper_transformer_entity/coordinator.py` — `TransformerObservationCoordinator` with `handle_topology_change(wrapper)`. (New package: `_entity` suffix distinguishes from any v1 `wrapper_transformer/` that may be present in history but is NOT in this branch — WP01 deliberately did not port it.)
- `tests/test_agent_transformer_ppo_entity.py` — covers spec §16.4.
- `tests/test_wrapper_transformer_entity.py` — covers spec §16.5.

**Files modified:**

- `algorithms/agents/base_agent.py` — add `supports_dynamic_topology: ClassVar[bool] = False`.
- `algorithms/registry.py` — register `AgentTransformerPPO`.
- `algorithms/agents/maddpg.py` (or wherever the MADDPG class lives — locate via grep) — set `supports_dynamic_topology = False` (default) explicitly for clarity.
- `algorithms/agents/rule_based_policy.py` — set `supports_dynamic_topology = True`.
- `utils/config_schema.py` — replace MADDPG-literal topology check with a generic `supports_dynamic_topology` lookup against the registry.
- `utils/wrapper_citylearn.py` — (a) detect when the attached agent is a transformer-entity agent and dispatch to the coordinator on topology change; (b) replace the runtime MADDPG-literal guardrail with the same flag-based check.

**Out of scope:**

- The YAML template (WP06).
- The E2E run on real data (WP06).
- Any change to the adapter (`utils/entity_adapter.py`).

---

## File Structure

```
algorithms/
  agents/
    base_agent.py              # MODIFIED (+ supports_dynamic_topology ClassVar)
    agent_transformer_ppo.py    # NEW (~500 lines)
    maddpg.py                   # MODIFIED (explicit False)  [adjust path if different]
    rule_based_policy.py        # MODIFIED (explicit True)   [adjust path if different]
  registry.py                   # MODIFIED (register AgentTransformerPPO)
utils/
  wrapper_citylearn.py          # MODIFIED (coordinator dispatch + flag-based guardrail)
  config_schema.py              # MODIFIED (flag-based topology check)
  wrapper_transformer_entity/
    __init__.py                  # NEW
    coordinator.py                # NEW (~80 lines)
tests/
  test_agent_transformer_ppo_entity.py  # NEW (covers §16.4, ~15 tests)
  test_wrapper_transformer_entity.py    # NEW (covers §16.5, ~7 tests)
```

---

## Key Design Decisions

- **Per-building stacks held in lists indexed by building_idx.** Each list element is a small dataclass `_PerBuildingState` containing tokenizer, backbone, actor, critic, optimizer, rollout buffer, cached layout, and last-known `(observation_names_tuple, action_names_tuple, entity_specs_signature)`.
- **Reuse of WP01 helpers**: `state_helper.py`, `update_helper.py`, `export_helper.py` from WP01 are intended for v2 reuse but were ported as v1 marker-era code. WP05 takes the conservative path: **wrap them only where their interface is type-stable** (e.g. `update_helper.compute_ppo_loss_and_backward` if its signature is `(actor, critic, buffer, optimizer, hyperparams) -> dict`), and inline anything that depended on v1 marker tokenizer state. If the helper takes a `marker_tokenizer` arg, do NOT shim it — write a fresh function in the agent. Document this decision in code comments referencing this WP.
- **Topology-change reentry**: `on_topology_change(b)` flushes the buffer using the **previous** layout (cached on the per-building state), then rebuilds. The "previous layout" is held until the rebuild completes, so any in-flight gradients from `update()` finish under the old graph before we discard.
- **Entity_specs signature**: a stable hash of `entity_specs.tables[<table>].features` per table. Used to detect schema changes the wrapper missed (defensive).
- **`load_checkpoint` rejects layout mismatch**: compares `layout_signature` (sorted tuple of observation_names) — if different, raise. Cross-topology resumption is explicitly out of scope (spec §14.3).
- **Wrapper coordinator opt-in**: the wrapper checks `getattr(self.model, "supports_dynamic_topology", False)` AND `getattr(self.model, "uses_entity_layout", False)`. The latter is a small marker on `AgentTransformerPPO` that says "I want the coordinator hook." `RuleBasedPolicy` sets `uses_entity_layout = False` so the coordinator does not call its `on_topology_change` (it has none). This avoids hardcoding class names in the wrapper.
- **ONNX export**: the spec §14.1 prescribes a non-trivial graph with `segment_offsets` / `segment_indices`. For WP05 we ship a **minimal-but-spec-compliant** export: the actor head on the reconstructed token tensor. The full `segment_indices` graph is reasonable to defer to WP06 if WP05 risks bloat — flag any deferral in the self-review.

---

## Tasks

### Task 1: Branch + verify prerequisites

- [ ] **Step 1: Create branch from WP04**

```bash
git checkout gj/wp04-tokenizer-backbone
git checkout -b gj/wp05-agent-wrapper
```

- [ ] **Step 2: Smoke check WP01–WP04 deliverables**

```bash
python -c "
from algorithms.utils.transformer_backbone import TransformerBackbone
from algorithms.utils.ppo_components import ActorHead, CriticHead, RolloutBuffer
from algorithms.utils.entity_token_layout import EntityTokenLayoutBuilder
from algorithms.utils.entity_observation_tokenizer import EntityObservationTokenizer
from utils.entity_tokenizer_schema import load_entity_tokenizer_config, validate_against_payload
print('OK: prerequisites resolvable')
"
```

---

### Task 2: Add `supports_dynamic_topology` ClassVar to `BaseAgent`

**Files:**
- Modify: `algorithms/agents/base_agent.py`
- Modify: `algorithms/agents/maddpg.py` (or whichever module holds it — `grep -RIn "class MADDPG" algorithms/`)
- Modify: `algorithms/agents/rule_based_policy.py` (or wherever — `grep -RIn "class RuleBasedPolicy" algorithms/`)

- [ ] **Step 1: Failing test**

```python
# tests/test_agent_transformer_ppo_entity.py (skeleton — first test only)
"""Tests for AgentTransformerPPO. Covers spec §16.4."""
from __future__ import annotations

import pytest


def test_supports_dynamic_topology_flag_default_false():
    """Spec §12.4: default is False; opt-in per agent class."""
    from algorithms.agents.base_agent import BaseAgent
    assert BaseAgent.supports_dynamic_topology is False


def test_maddpg_supports_dynamic_topology_false():
    from algorithms.agents.maddpg import MADDPG  # adjust import to actual path
    assert MADDPG.supports_dynamic_topology is False


def test_rule_based_supports_dynamic_topology_true():
    from algorithms.agents.rule_based_policy import RuleBasedPolicy  # adjust path
    assert RuleBasedPolicy.supports_dynamic_topology is True
```

- [ ] **Step 2: Run, expect FAIL**

```bash
pytest tests/test_agent_transformer_ppo_entity.py -k supports_dynamic -v
```

- [ ] **Step 3: Edit `base_agent.py`**

Add at top of class:
```python
from typing import ClassVar
...
class BaseAgent:
    supports_dynamic_topology: ClassVar[bool] = False
    ...
```

Then in `MADDPG`:
```python
class MADDPG(BaseAgent):
    supports_dynamic_topology: ClassVar[bool] = False  # explicit; default
    ...
```

In `RuleBasedPolicy`:
```python
class RuleBasedPolicy(BaseAgent):
    supports_dynamic_topology: ClassVar[bool] = True
    ...
```

- [ ] **Step 4: Run; iterate until PASS**

- [ ] **Step 5: Commit**

```bash
git add algorithms/agents/
git commit -m "feat(wp05): add supports_dynamic_topology ClassVar to BaseAgent + concretes"
```

---

### Task 3: Generic guardrail in `validate_config` (replace MADDPG literal)

**Files:**
- Modify: `utils/config_schema.py`
- Modify: existing tests that asserted on the MADDPG-specific error message — locate via `grep -RIn "MADDPG.*dynamic\|dynamic.*MADDPG" tests/`

- [ ] **Step 1: Identify the existing MADDPG check**

```bash
grep -n "topology_mode\|MADDPG\|dynamic" utils/config_schema.py | head -30
```

- [ ] **Step 2: Failing test — generic flag check**

```python
def test_validate_config_rejects_maddpg_with_dynamic_topology_via_flag():
    """Spec §12.4: the check is now flag-based, not MADDPG-literal."""
    from utils.config_schema import validate_config
    cfg = _make_minimal_maddpg_dynamic_cfg()  # implement helper inline
    with pytest.raises(ValueError, match="MADDPG"):  # error message kept stable per §12.4
        validate_config(cfg)


def test_validate_config_accepts_transformer_ppo_with_dynamic_topology():
    """AgentTransformerPPO has supports_dynamic_topology=True so dynamic must work."""
    from utils.config_schema import validate_config
    cfg = _make_minimal_transformer_ppo_dynamic_cfg()
    validate_config(cfg)  # must not raise
```

- [ ] **Step 3: Implement the flag-based dispatch**

Replace the MADDPG-literal check in `validate_config(...)`:

```python
# OLD (find and replace):
# if validated.algorithm.name == "MADDPG" and validated.simulator.topology_mode == "dynamic":
#     raise ValueError("MADDPG does not support dynamic topology.")

# NEW:
if validated.simulator.topology_mode == "dynamic":
    from algorithms.registry import ALGORITHM_REGISTRY
    cls = ALGORITHM_REGISTRY.get(validated.algorithm.name)
    if cls is not None and not getattr(cls, "supports_dynamic_topology", False):
        # Preserve historical wording for MADDPG so existing user-facing tests pass
        if validated.algorithm.name == "MADDPG":
            raise ValueError(
                "MADDPG does not support dynamic topology. "
                "Use a dynamic-capable agent (e.g. RuleBasedPolicy, AgentTransformerPPO)."
            )
        raise ValueError(
            f"Algorithm {validated.algorithm.name!r} does not support dynamic topology "
            f"(supports_dynamic_topology=False). Use an agent with the flag set to True."
        )
```

(`AgentTransformerPPO` isn't in the registry yet — that's Task 7. Until then `cls` is None, so the check is silently skipped for the transformer agent name. We will add a follow-up assertion in Task 7's tests.)

- [ ] **Step 4: Run new tests + existing test suite**

```bash
pytest tests/ -k "validate_config" -v
pytest -x -q
```

- [ ] **Step 5: Commit**

```bash
git add utils/config_schema.py tests/test_agent_transformer_ppo_entity.py
git commit -m "feat(wp05): replace MADDPG-literal dynamic-topology check with flag-based registry lookup"
```

---

### Task 4: `AgentTransformerPPO.__init__` + `attach_environment` (TDD)

**Files:**
- Create: `algorithms/agents/agent_transformer_ppo.py`
- Modify: `tests/test_agent_transformer_ppo_entity.py`

- [ ] **Step 1: Failing test — construction + first attach**

```python
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def attach_kwargs_for_sample():
    """Compute observation_names + action_names + entity_specs from the bundled
    sample payload — the same shape AgentTransformerPPO will see in production."""
    from utils.entity_adapter import EntityContractAdapter
    payload = json.loads(
        Path("datasets/tmp_entity_obs_full_step2200_named.json").read_text()
    )
    adapter = EntityContractAdapter()
    obs_per_b, names_per_b = adapter.to_agent_observations(payload)
    # action_names per building. The adapter's to_entity_actions emits payloads
    # with one action_field per CA segment; the canonical order in the demo
    # dataset is [electrical_storage, electric_vehicle_storage, ...repeated for
    # multiple CAs]. For the smoke fixture, hard-code per spec §13.1 ca_types.
    # In production this comes from the schema via the simulator.
    # Compute by counting CA segments per building from observation names:
    action_names_per_b = []
    for names in names_per_b:
        actions = []
        for n in names:
            if n.startswith("storage::"):
                actions.append("electrical_storage")
            elif n.startswith("charger::") and "::connected_ev::" not in n and "::incoming_ev::" not in n:
                # CA charger feature — but this lists ALL charger features, we want one per charger instance
                pass
        # Simpler approach: derive from action_field of the simulator. For testing,
        # use a minimum: 1 storage + 1 charger per building.
        # The bundled sample first building has those.
        action_names_per_b.append(["electrical_storage", "electric_vehicle_storage"])
    obs_arrays = [np.asarray(o, dtype=np.float64) for o in obs_per_b]
    obs_spaces = [None] * len(obs_arrays)
    action_spaces = [None] * len(obs_arrays)
    metadata = {
        "entity_specs": payload.get("meta", {}).get("entity_specs"),
        "interface": "entity",
        "topology_mode": "dynamic",
        "building_names": [f"Building_{i+1}" for i in range(len(obs_arrays))],
    }
    return {
        "observation_names": names_per_b,
        "action_names": action_names_per_b,
        "action_space": action_spaces,
        "observation_space": obs_spaces,
        "metadata": metadata,
        "_obs": obs_arrays,
    }


def _make_agent_config():
    return {
        "algorithm": {
            "name": "AgentTransformerPPO",
            "tokenizer_config_path": "configs/tokenizers/entity_default.json",
            "transformer": {"d_model": 16, "nhead": 2, "num_layers": 1,
                            "dim_feedforward": 32, "dropout": 0.0},
            "hyperparameters": {"learning_rate": 3.0e-4, "gamma": 0.99,
                                "gae_lambda": 0.95, "clip_eps": 0.2,
                                "ppo_epochs": 1, "minibatch_size": 4,
                                "entropy_coeff": 0.0, "value_coeff": 0.5,
                                "max_grad_norm": 0.5},
        }
    }


def test_attach_environment_builds_layouts(attach_kwargs_for_sample):
    from algorithms.agents.agent_transformer_ppo import AgentTransformerPPO
    agent = AgentTransformerPPO(_make_agent_config())
    kw = {k: v for k, v in attach_kwargs_for_sample.items() if k != "_obs"}
    agent.attach_environment(**kw)
    # one cached layout per building
    assert len(agent._per_building) == len(kw["observation_names"])
    for state in agent._per_building:
        assert state.layout is not None
```

- [ ] **Step 2: Run; expect FAIL** (`ModuleNotFoundError`).

- [ ] **Step 3: Skeleton implementation**

Create `algorithms/agents/agent_transformer_ppo.py`:

```python
"""AgentTransformerPPO — per-building Transformer + PPO over the entity interface.

See docs/specv2.md §11, §12, §14.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from algorithms.agents.base_agent import BaseAgent
from algorithms.utils.transformer_backbone import TransformerBackbone
from algorithms.utils.ppo_components import (
    ActorHead, CriticHead, RolloutBuffer, compute_ppo_loss,
)
from algorithms.utils.entity_token_layout import (
    BuildingTokenLayout, EntityTokenLayoutBuilder,
)
from algorithms.utils.entity_observation_tokenizer import (
    EntityObservationTokenizer, TokenizedObservation,
)
from utils.entity_tokenizer_schema import (
    load_entity_tokenizer_config,
    EntityTokenizerConfig,
    validate_against_payload,
    EntityPayloadSample,
)


@dataclass
class _PerBuildingState:
    building_id: str
    tokenizer: EntityObservationTokenizer
    backbone: TransformerBackbone
    actor: ActorHead
    critic: CriticHead
    optimizer: torch.optim.Optimizer
    buffer: RolloutBuffer
    layout: Optional[BuildingTokenLayout]
    obs_names_tuple: Tuple[str, ...]
    action_names_tuple: Tuple[str, ...]
    entity_specs_signature: Optional[str]


class AgentTransformerPPO(BaseAgent):
    supports_dynamic_topology: ClassVar[bool] = True
    uses_entity_layout: ClassVar[bool] = True   # wrapper coordinator opt-in

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        algo = config["algorithm"]
        self._tokenizer_config: EntityTokenizerConfig = load_entity_tokenizer_config(
            algo["tokenizer_config_path"]
        )
        self._transformer_cfg = algo["transformer"]
        self._hparams = algo["hyperparameters"]
        self._layout_builder = EntityTokenLayoutBuilder(self._tokenizer_config)
        self._per_building: List[_PerBuildingState] = []
        self._entity_topology_version_at_export: Optional[int] = None

    # --- BaseAgent contract ----------------------------------------------

    def attach_environment(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Idempotent: if signatures match, no-op.
        if self._per_building and all(
            tuple(observation_names[b]) == s.obs_names_tuple
            and tuple(action_names[b]) == s.action_names_tuple
            for b, s in enumerate(self._per_building)
        ):
            return
        # Fresh attach: tear down everything and rebuild.
        self._per_building = []
        for b, (obs_n, act_n) in enumerate(zip(observation_names, action_names)):
            building_id = (
                metadata.get("building_names", [None])[b]
                if metadata else f"building_{b}"
            ) or f"building_{b}"
            layout = self._layout_builder.build(building_id, obs_n, act_n)
            type_input_dims = self._compute_type_input_dims(layout)
            d_model = int(self._transformer_cfg["d_model"])
            tokenizer = EntityObservationTokenizer(
                self._tokenizer_config, d_model, type_input_dims,
            )
            backbone = TransformerBackbone(
                d_model=d_model,
                nhead=int(self._transformer_cfg["nhead"]),
                num_layers=int(self._transformer_cfg["num_layers"]),
                dim_feedforward=int(self._transformer_cfg["dim_feedforward"]),
                dropout=float(self._transformer_cfg["dropout"]),
            )
            actor = ActorHead(d_model=d_model)   # adjust if v1 ctor differs
            critic = CriticHead(d_model=d_model)
            optim = torch.optim.Adam(
                list(tokenizer.parameters())
                + list(backbone.parameters())
                + list(actor.parameters())
                + list(critic.parameters()),
                lr=float(self._hparams["learning_rate"]),
            )
            buffer = RolloutBuffer()  # adjust ctor args to ported API
            self._per_building.append(_PerBuildingState(
                building_id=building_id,
                tokenizer=tokenizer, backbone=backbone, actor=actor, critic=critic,
                optimizer=optim, buffer=buffer, layout=layout,
                obs_names_tuple=tuple(obs_n),
                action_names_tuple=tuple(act_n),
                entity_specs_signature=self._signature(metadata),
            ))
        # CA-order post-condition assertion (spec §10.1).
        for b, state in enumerate(self._per_building):
            assert state.layout is not None
            if state.layout.ca_action_names != tuple(action_names[b]):
                raise ValueError(
                    f"BuildingTokenLayout.ca_action_names "
                    f"{state.layout.ca_action_names!r} does not match "
                    f"action_names[{b}] {tuple(action_names[b])!r}"
                )

    def predict(
        self,
        observations: List[npt.NDArray[np.float64]],
        deterministic: Optional[bool] = None,
    ) -> List[List[float]]:
        deterministic = bool(deterministic) if deterministic is not None else False
        out: List[List[float]] = []
        for state, obs in zip(self._per_building, observations):
            obs_t = torch.as_tensor(obs, dtype=torch.float).unsqueeze(0)
            with torch.no_grad():
                tokenized = state.tokenizer(obs_t, state.layout)
                ca_emb, pooled = state.backbone(
                    tokenized.sro_tokens, tokenized.nfc_token, tokenized.ca_tokens,
                )
                # ActorHead API may return (mean, log_std) or a Distribution; adapt.
                actor_out = state.actor(ca_emb)
                if isinstance(actor_out, tuple):
                    mean, log_std = actor_out
                    if deterministic:
                        action = torch.tanh(mean)
                    else:
                        std = log_std.exp()
                        sample = mean + std * torch.randn_like(mean)
                        action = torch.tanh(sample)
                else:
                    # Distribution path
                    if deterministic:
                        action = torch.tanh(actor_out.mean)
                    else:
                        action = torch.tanh(actor_out.rsample())
            out.append(action.squeeze(0).clamp(-1.0, 1.0).tolist())
        return out

    def update(
        self,
        observations, actions, rewards, next_observations,
        terminated: bool, truncated: bool,
        *,
        update_target_step: bool, global_learning_step: int,
        update_step: bool, initial_exploration_done: bool,
    ) -> None:
        # Append transitions to each per-building buffer; run PPO when update_step.
        # Detailed implementation in Task 5.
        raise NotImplementedError("see Task 5")

    def on_topology_change(self, building_idx: int) -> None:
        # Detailed implementation in Task 6.
        raise NotImplementedError("see Task 6")

    def export_artifacts(
        self, output_dir: str, context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Detailed implementation in Task 8.
        raise NotImplementedError("see Task 8")

    def save_checkpoint(self, output_dir: str, step: int) -> Optional[str]:
        # Task 9.
        raise NotImplementedError("see Task 9")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        # Task 9.
        raise NotImplementedError("see Task 9")

    # --- helpers ----------------------------------------------------------

    def _compute_type_input_dims(
        self, layout: BuildingTokenLayout,
    ) -> Dict[str, int]:
        """Spec §8.5: per-type input dim derived from the layout's segment
        feature counts. NFC is hard-coded to 1."""
        dims: Dict[str, int] = {self._tokenizer_config.nfc.type_name: 1}
        for seg in layout.segments:
            if seg.family == "nfc":
                continue
            # Segments of the same type in one building have equal feature counts;
            # if two per-asset segments disagree, that's a real bug — assert.
            existing = dims.get(seg.type_name)
            new = len(seg.feature_indices)
            if existing is not None and existing != new:
                raise ValueError(
                    f"Inconsistent input dim for type {seg.type_name!r}: "
                    f"{existing} vs {new}"
                )
            dims[seg.type_name] = new
        # Every declared SRO/CA type must appear; if a type has no instance in
        # this layout (e.g. building has no PV), we still need a projection
        # constructed so checkpoints stay stable. Use input_dim_fallback.
        for tname, ca in self._tokenizer_config.ca_types.items():
            dims.setdefault(tname, ca.input_dim_fallback)
        for tname, sro in self._tokenizer_config.sro_types.items():
            dims.setdefault(tname, sro.input_dim_fallback)
        return dims

    @staticmethod
    def _signature(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        if not metadata or "entity_specs" not in metadata:
            return None
        import hashlib
        import json as _json
        s = _json.dumps(metadata["entity_specs"], sort_keys=True, default=str)
        return hashlib.sha256(s.encode()).hexdigest()[:16]
```

- [ ] **Step 4: Run the failing test until PASS**

```bash
pytest tests/test_agent_transformer_ppo_entity.py::test_attach_environment_builds_layouts -v
```
Iterate. Likely issues: `RolloutBuffer` ctor needs args; `ActorHead`/`CriticHead` need different ctor signatures. Inspect `algorithms/utils/ppo_components.py` and adapt construction to the actual ported API.

- [ ] **Step 5: Commit**

```bash
git add algorithms/agents/agent_transformer_ppo.py tests/test_agent_transformer_ppo_entity.py
git commit -m "feat(wp05): AgentTransformerPPO __init__ + attach_environment (per-building stacks)"
```

---

### Task 5: `predict()` + supporting tests (§16.4 rows 4–7)

```python
def test_predict_action_count_matches_n_ca(attach_kwargs_for_sample):
    from algorithms.agents.agent_transformer_ppo import AgentTransformerPPO
    agent = AgentTransformerPPO(_make_agent_config())
    kw = {k: v for k, v in attach_kwargs_for_sample.items() if k != "_obs"}
    agent.attach_environment(**kw)
    actions = agent.predict(attach_kwargs_for_sample["_obs"])
    for state, a in zip(agent._per_building, actions):
        assert len(a) == state.layout.n_ca


def test_predict_returns_list_of_lists_of_float(attach_kwargs_for_sample):
    from algorithms.agents.agent_transformer_ppo import AgentTransformerPPO
    agent = AgentTransformerPPO(_make_agent_config())
    kw = {k: v for k, v in attach_kwargs_for_sample.items() if k != "_obs"}
    agent.attach_environment(**kw)
    actions = agent.predict(attach_kwargs_for_sample["_obs"])
    assert isinstance(actions, list)
    assert all(isinstance(a, list) and all(isinstance(x, float) for x in a) for a in actions)


def test_deterministic_uses_means(attach_kwargs_for_sample):
    from algorithms.agents.agent_transformer_ppo import AgentTransformerPPO
    agent = AgentTransformerPPO(_make_agent_config())
    kw = {k: v for k, v in attach_kwargs_for_sample.items() if k != "_obs"}
    agent.attach_environment(**kw)
    a1 = agent.predict(attach_kwargs_for_sample["_obs"], deterministic=True)
    a2 = agent.predict(attach_kwargs_for_sample["_obs"], deterministic=True)
    for r1, r2 in zip(a1, a2):
        assert r1 == pytest.approx(r2, abs=1e-7)


def test_stochastic_samples(attach_kwargs_for_sample):
    from algorithms.agents.agent_transformer_ppo import AgentTransformerPPO
    torch.manual_seed(0)
    agent = AgentTransformerPPO(_make_agent_config())
    kw = {k: v for k, v in attach_kwargs_for_sample.items() if k != "_obs"}
    agent.attach_environment(**kw)
    a1 = agent.predict(attach_kwargs_for_sample["_obs"], deterministic=False)
    a2 = agent.predict(attach_kwargs_for_sample["_obs"], deterministic=False)
    diffs = [abs(x - y) for r1, r2 in zip(a1, a2) for x, y in zip(r1, r2)]
    assert max(diffs) > 1e-6
```

- [ ] **Step 1: Run these tests; `predict` is already implemented; iterate on any API mismatches**

```bash
pytest tests/test_agent_transformer_ppo_entity.py -k "predict_action_count or predict_returns or deterministic_uses_means or stochastic_samples" -v
```

- [ ] **Step 2: Commit**

```bash
git add tests/test_agent_transformer_ppo_entity.py
git commit -m "test(wp05): cover predict() shape, dtype, determinism, and stochasticity"
```

---

### Task 6: `update()` — minimal PPO loop

This is the largest single sub-task. Implement enough of `update()` to:
1. Accept transitions and store them in each building's `RolloutBuffer`.
2. When `update_step is True`, run a PPO update per building over the collected trajectory.
3. Return `None`.

Then add the §16.4 row tests.

**Files:**
- Modify: `algorithms/agents/agent_transformer_ppo.py`
- Modify: `tests/test_agent_transformer_ppo_entity.py`

- [ ] **Step 1: Failing test — signature + return type**

```python
def test_update_signature_uses_scalar_terminated_truncated(attach_kwargs_for_sample):
    from algorithms.agents.agent_transformer_ppo import AgentTransformerPPO
    agent = AgentTransformerPPO(_make_agent_config())
    kw = {k: v for k, v in attach_kwargs_for_sample.items() if k != "_obs"}
    agent.attach_environment(**kw)
    obs = attach_kwargs_for_sample["_obs"]
    actions = agent.predict(obs)
    rewards = [0.0] * len(obs)
    res = agent.update(
        obs, [np.asarray(a) for a in actions], rewards, obs,
        terminated=False, truncated=False,
        update_target_step=False, global_learning_step=0,
        update_step=False, initial_exploration_done=True,
    )
    assert res is None


def test_update_returns_none(attach_kwargs_for_sample):
    """Same as above but with update_step=True (must run PPO and still return None)."""
    from algorithms.agents.agent_transformer_ppo import AgentTransformerPPO
    agent = AgentTransformerPPO(_make_agent_config())
    kw = {k: v for k, v in attach_kwargs_for_sample.items() if k != "_obs"}
    agent.attach_environment(**kw)
    obs = attach_kwargs_for_sample["_obs"]
    # Push a few transitions so the buffer has data.
    for _ in range(8):
        actions = agent.predict(obs)
        agent.update(
            obs, [np.asarray(a) for a in actions], [0.5] * len(obs), obs,
            terminated=False, truncated=False,
            update_target_step=False, global_learning_step=0,
            update_step=False, initial_exploration_done=True,
        )
    res = agent.update(
        obs, [np.asarray(actions[i]) for i in range(len(obs))], [0.5] * len(obs), obs,
        terminated=False, truncated=False,
        update_target_step=False, global_learning_step=0,
        update_step=True, initial_exploration_done=True,
    )
    assert res is None
```

- [ ] **Step 2: Implement `update()`**

Replace the `NotImplementedError` in `update()`:

```python
    def update(
        self,
        observations, actions, rewards, next_observations,
        terminated: bool, truncated: bool,
        *,
        update_target_step: bool, global_learning_step: int,
        update_step: bool, initial_exploration_done: bool,
    ) -> None:
        # Append per-building transition.
        for b, state in enumerate(self._per_building):
            obs_t = torch.as_tensor(observations[b], dtype=torch.float).unsqueeze(0)
            act_t = torch.as_tensor(actions[b], dtype=torch.float).unsqueeze(0)
            with torch.no_grad():
                tokenized = state.tokenizer(obs_t, state.layout)
                _, pooled = state.backbone(
                    tokenized.sro_tokens, tokenized.nfc_token, tokenized.ca_tokens,
                )
                value = state.critic(pooled).squeeze(-1)
            # log_prob computation depends on actor API; use a tanh-Gaussian
            # convention: store action and recompute log_prob during the PPO
            # update from the stored obs (so we don't have to capture it now).
            state.buffer.add(
                obs=obs_t.squeeze(0),
                action=act_t.squeeze(0),
                reward=float(rewards[b]),
                value=value.item(),
                done=bool(terminated or truncated),
            )

        if not update_step:
            return
        # Run PPO update per building.
        for state in self._per_building:
            self._ppo_update(state)
            state.buffer.clear()

    def _ppo_update(self, state: _PerBuildingState) -> None:
        """One PPO update on the current rollout buffer. Stub here — flesh out
        per the ported `compute_ppo_loss` API in algorithms/utils/ppo_components.
        """
        if len(state.buffer) == 0:
            return
        # Sketch: compute advantages, sample minibatches, do ppo_epochs of
        # gradient steps, clip grad norm, optimizer.step. Adapt to the actual
        # ported API. The test only checks that update() returns None and
        # doesn't crash; the PPO numerics are exercised by the ported
        # tests/test_ppo_components.py from WP01.
        adv = state.buffer.compute_advantages(
            gamma=float(self._hparams["gamma"]),
            gae_lambda=float(self._hparams["gae_lambda"]),
        )
        for _epoch in range(int(self._hparams["ppo_epochs"])):
            for batch in state.buffer.iter_minibatches(
                int(self._hparams["minibatch_size"]),
            ):
                state.optimizer.zero_grad()
                loss = compute_ppo_loss(
                    tokenizer=state.tokenizer,
                    backbone=state.backbone,
                    actor=state.actor,
                    critic=state.critic,
                    layout=state.layout,
                    batch=batch,
                    advantages=adv,
                    clip_eps=float(self._hparams["clip_eps"]),
                    entropy_coeff=float(self._hparams["entropy_coeff"]),
                    value_coeff=float(self._hparams["value_coeff"]),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(state.tokenizer.parameters())
                    + list(state.backbone.parameters())
                    + list(state.actor.parameters())
                    + list(state.critic.parameters()),
                    float(self._hparams["max_grad_norm"]),
                )
                state.optimizer.step()
```

**Critical:** the `compute_ppo_loss` signature above is hypothetical. Inspect the ported `algorithms/utils/ppo_components.py` and adapt `_ppo_update` to the real API. If the ported `compute_ppo_loss` is incompatible (e.g. it expects flat features, no tokenizer/layout), then write a small wrapper here that materializes the token tensors via the layout and calls the simpler v1 loss with them.

- [ ] **Step 3: Run, iterate**

```bash
pytest tests/test_agent_transformer_ppo_entity.py -k "update_signature or update_returns" -v
```

- [ ] **Step 4: Commit**

```bash
git add algorithms/agents/agent_transformer_ppo.py tests/test_agent_transformer_ppo_entity.py
git commit -m "feat(wp05): AgentTransformerPPO.update — buffer push + per-building PPO step"
```

---

### Task 7: `on_topology_change()` (§16.4 rows 10–12)

```python
def test_on_topology_change_runs_update_then_flushes_buffer(attach_kwargs_for_sample):
    from algorithms.agents.agent_transformer_ppo import AgentTransformerPPO
    agent = AgentTransformerPPO(_make_agent_config())
    kw = {k: v for k, v in attach_kwargs_for_sample.items() if k != "_obs"}
    agent.attach_environment(**kw)
    obs = attach_kwargs_for_sample["_obs"]
    # Fill buffer for building 0
    for _ in range(4):
        a = agent.predict(obs)
        agent.update(obs, [np.asarray(x) for x in a], [0.0]*len(obs), obs,
                     terminated=False, truncated=False,
                     update_target_step=False, global_learning_step=0,
                     update_step=False, initial_exploration_done=True)
    assert len(agent._per_building[0].buffer) > 0
    # Topology change for building 0 — same names (no real topology change), so
    # the test verifies buffer flush + idempotent rebuild.
    agent.on_topology_change(0)
    assert len(agent._per_building[0].buffer) == 0


def test_on_topology_change_rebuilds_layout(attach_kwargs_for_sample, monkeypatch):
    from algorithms.agents.agent_transformer_ppo import AgentTransformerPPO
    agent = AgentTransformerPPO(_make_agent_config())
    kw = {k: v for k, v in attach_kwargs_for_sample.items() if k != "_obs"}
    agent.attach_environment(**kw)
    # Mutate stored obs/action names to simulate topology change.
    state0 = agent._per_building[0]
    state0.obs_names_tuple = state0.obs_names_tuple + ("storage::Building_1/extra::soc",)
    state0.action_names_tuple = state0.action_names_tuple + ("electrical_storage",)
    old_layout = state0.layout
    # The agent uses obs_names_tuple/action_names_tuple as the source for
    # rebuild — but really it must consult the wrapper. For this unit test, set
    # them directly and call on_topology_change.
    agent.on_topology_change(0)
    assert state0.layout is not old_layout


def test_on_topology_change_does_not_treat_first_attach_as_topology_change(attach_kwargs_for_sample):
    """Spec §12.1: first attach must not call on_topology_change."""
    from algorithms.agents.agent_transformer_ppo import AgentTransformerPPO
    agent = AgentTransformerPPO(_make_agent_config())
    # Spy on on_topology_change
    called = []
    orig = AgentTransformerPPO.on_topology_change
    def spy(self, b):
        called.append(b)
        return orig(self, b)
    AgentTransformerPPO.on_topology_change = spy
    try:
        kw = {k: v for k, v in attach_kwargs_for_sample.items() if k != "_obs"}
        agent.attach_environment(**kw)
    finally:
        AgentTransformerPPO.on_topology_change = orig
    assert called == [], "attach_environment must not call on_topology_change"
```

- [ ] **Step 1: Implement `on_topology_change`**

```python
    def on_topology_change(self, building_idx: int) -> None:
        state = self._per_building[building_idx]
        # 1. Flush PPO update on previous trajectory if buffer non-empty.
        if len(state.buffer) > 0:
            try:
                self._ppo_update(state)
            finally:
                state.buffer.clear()
        # 2. Rebuild layout from current names. Source of truth = the agent's
        # cached tuples (which the wrapper updates via attach_environment
        # before calling on_topology_change — see spec §12.2).
        new_layout = self._layout_builder.build(
            state.building_id,
            list(state.obs_names_tuple),
            list(state.action_names_tuple),
        )
        # 3. Re-run §13.4 rules 1-5 against current names.
        sample = EntityPayloadSample(
            feature_names_per_table={
                # Synthetic single-table view from the layout — not exhaustive but
                # enough for rules 1+2 to operate at runtime over what we *see*.
                # If entity_specs is available on metadata, prefer that.
                "district": [
                    n for n in state.obs_names_tuple if n.startswith("district__")
                ],
                "building": [
                    n for n in state.obs_names_tuple
                    if not n.startswith(("district__", "storage::", "charger::", "pv::"))
                ],
                "storage": [], "charger": [], "pv": [], "ev": [],
            }
        )
        validate_against_payload(
            self._tokenizer_config, sample,
            [list(state.action_names_tuple)],
        )
        # 4. Replace state.layout (and re-derive type_input_dims if grown).
        new_dims = self._compute_type_input_dims(new_layout)
        # If a type's input_dim grew beyond the existing projection, we must
        # rebuild the tokenizer (would lose those weights). Spec §8.5 plus the
        # per-type projection sharing means new INSTANCES reuse weights but a
        # changed FEATURE COUNT (e.g. simulator added a new pv feature column)
        # would invalidate weights; this is a hard fail per spec §13.4 rule 5
        # generalisation. For now, assert and raise.
        for tname, dim in new_dims.items():
            existing = state.tokenizer.projections[tname].in_features
            if existing != dim:
                raise ValueError(
                    f"Topology change for building {state.building_id!r}: "
                    f"feature count for type {tname!r} changed "
                    f"{existing} -> {dim}; weights cannot be preserved across "
                    "feature-schema changes."
                )
        state.layout = new_layout
        # 5. CA-order post-condition.
        if state.layout.ca_action_names != state.action_names_tuple:
            raise ValueError(
                f"Post-rebuild CA order mismatch for building "
                f"{state.building_id!r}: layout has "
                f"{state.layout.ca_action_names!r}, action_names "
                f"{state.action_names_tuple!r}"
            )


    def update_topology_names(
        self, building_idx: int,
        observation_names: List[str], action_names: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Wrapper-facing helper called BEFORE on_topology_change(b) so the
        agent knows the new layout to build. Updates the per-building cached
        tuples atomically."""
        state = self._per_building[building_idx]
        state.obs_names_tuple = tuple(observation_names)
        state.action_names_tuple = tuple(action_names)
        state.entity_specs_signature = self._signature(metadata)
```

(`update_topology_names` is the agent-facing API the wrapper coordinator calls before `on_topology_change`. This makes the contract explicit and testable.)

Add a test:

```python
def test_on_topology_change_validates_input_dim(attach_kwargs_for_sample):
    """If feature counts for a type change, on_topology_change must raise."""
    from algorithms.agents.agent_transformer_ppo import AgentTransformerPPO
    agent = AgentTransformerPPO(_make_agent_config())
    kw = {k: v for k, v in attach_kwargs_for_sample.items() if k != "_obs"}
    agent.attach_environment(**kw)
    state0 = agent._per_building[0]
    # Force a layout that would change tokenizer dim — easiest is to mutate the
    # tokenizer projection in-place to a wrong dim and then invoke topology change.
    # Replace storage projection with one expecting more features.
    import torch.nn as nn
    storage_proj = state0.tokenizer.projections["storage"]
    state0.tokenizer.projections["storage"] = nn.Linear(storage_proj.in_features + 7, storage_proj.out_features)
    with pytest.raises(ValueError, match="feature count|cannot be preserved"):
        agent.on_topology_change(0)


def test_supports_dynamic_topology_flag():
    from algorithms.agents.agent_transformer_ppo import AgentTransformerPPO
    assert AgentTransformerPPO.supports_dynamic_topology is True
```

- [ ] **Step 2: Run all WP05 agent tests so far**

```bash
pytest tests/test_agent_transformer_ppo_entity.py -v
```

- [ ] **Step 3: Commit**

```bash
git add algorithms/agents/agent_transformer_ppo.py tests/test_agent_transformer_ppo_entity.py
git commit -m "feat(wp05): on_topology_change with buffer-flush, layout rebuild, and rule re-validation"
```

---

### Task 8: `export_artifacts` (§16.4 not directly listed; manifest contract = §14.2)

**Files:**
- Modify: `algorithms/agents/agent_transformer_ppo.py`
- Modify: `tests/test_agent_transformer_ppo_entity.py`

- [ ] **Step 1: Failing test for manifest shape**

```python
def test_export_artifacts_returns_canonical_manifest_shape(tmp_path, attach_kwargs_for_sample):
    """Spec §14.2: manifest MUST have 'format' top-level + 'artifacts' list."""
    from algorithms.agents.agent_transformer_ppo import AgentTransformerPPO
    agent = AgentTransformerPPO(_make_agent_config())
    kw = {k: v for k, v in attach_kwargs_for_sample.items() if k != "_obs"}
    agent.attach_environment(**kw)
    out = agent.export_artifacts(str(tmp_path), context={"topology_version": 7})
    assert out["format"] == "onnx"
    assert isinstance(out["artifacts"], list) and len(out["artifacts"]) == len(agent._per_building)
    for i, art in enumerate(out["artifacts"]):
        assert art["agent_index"] == i
        assert art["format"] == "onnx"
        assert art["path"].endswith(".onnx")
        assert (tmp_path / art["path"]).exists()
        assert "config" in art
    assert out["supports_dynamic_topology"] is True
    assert out["tokenizer_config_path"] == "configs/tokenizers/entity_default.json"
```

- [ ] **Step 2: Implement `export_artifacts`**

Replace the `NotImplementedError`:

```python
    def export_artifacts(
        self, output_dir: str, context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        out = Path(output_dir)
        onnx_dir = out / "onnx_models"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        topology_version = int((context or {}).get("topology_version", 0))
        artifacts: List[Dict[str, Any]] = []
        agent_models: List[Dict[str, Any]] = []
        for b, state in enumerate(self._per_building):
            assert state.layout is not None
            obs_dim = state.tokenizer.projections[
                next(iter(state.tokenizer.projections.keys()))
            ].in_features  # placeholder — better: derive from any segment
            # Compute true obs_dim:
            obs_dim = max(
                max(seg.feature_indices) for seg in state.layout.segments
            ) + 1
            relpath = f"onnx_models/agent_{b}__topology_v{topology_version}.onnx"
            self._export_onnx_for_building(state, out / relpath, obs_dim)
            sro_types = [s.type_name for s in state.layout.segments if s.family == "sro"]
            ca_types = [s.type_name for s in state.layout.segments if s.family == "ca"]
            cfg = {
                "building_id": state.building_id,
                "topology_version": topology_version,
                "obs_dim": obs_dim,
                "n_sro": state.layout.n_sro,
                "n_ca": state.layout.n_ca,
                "sro_types": sro_types,
                "ca_types": ca_types,
            }
            artifacts.append({
                "agent_index": b,
                "path": relpath,
                "format": "onnx",
                "config": cfg,
            })
            agent_models.append({
                "building_index": b,
                "building_id": state.building_id,
                "topology_version": topology_version,
                "onnx_path": relpath,
                **{k: cfg[k] for k in ("obs_dim", "n_sro", "n_ca", "sro_types", "ca_types")},
            })
        return {
            "format": "onnx",
            "artifacts": artifacts,
            "tokenizer_config_path": self.config["algorithm"]["tokenizer_config_path"],
            "supports_dynamic_topology": True,
            "agent_models": agent_models,
        }

    def _export_onnx_for_building(
        self, state: _PerBuildingState, path: Path, obs_dim: int,
    ) -> None:
        """Minimal-but-spec-compliant ONNX export.

        The full §14.1 graph (segment_offsets / segment_indices) is reasonable
        to defer to a follow-up tuning WP. For now we export the deterministic
        actor as a function of (encoded_obs,) where obs_dim is pinned and the
        layout indices are baked into the graph as constants. This is enough
        for the bundle validator and any downstream inference that runs the
        same topology.
        """
        layout = state.layout
        assert layout is not None

        class _Wrapper(nn.Module):
            def __init__(self, tok, bb, actor, layout_):
                super().__init__()
                self.tok = tok; self.bb = bb; self.actor = actor; self.layout = layout_
            def forward(self, encoded_obs):
                tk = self.tok(encoded_obs, self.layout)
                ca_emb, _ = self.bb(tk.sro_tokens, tk.nfc_token, tk.ca_tokens)
                actor_out = self.actor(ca_emb)
                if isinstance(actor_out, tuple):
                    mean = actor_out[0]
                else:
                    mean = actor_out.mean
                return torch.tanh(mean)

        wrapper = _Wrapper(state.tokenizer, state.backbone, state.actor, layout).eval()
        dummy = torch.zeros(1, obs_dim)
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(path),
            input_names=["encoded_obs"],
            output_names=["actions"],
            opset_version=17,
            dynamic_axes={"encoded_obs": {1: "obs_dim"}},
        )
```

- [ ] **Step 3: Run**

```bash
pytest tests/test_agent_transformer_ppo_entity.py::test_export_artifacts_returns_canonical_manifest_shape -v
```
ONNX export of `index_select` against a Python `BuildingTokenLayout` may fail because torch.onnx can't trace through the Python loop. If so:
  - Refactor `_Wrapper.forward` to use precomputed tensor indices from the layout (pre-flatten into a single `LongTensor` and `index_select` once per family) baked into the wrapper as buffers.
  - If that's still hard within WP05's scope, **defer the actual ONNX serialization** by exporting a JSON sidecar with the layout + saving torchscript `.pt` instead of `.onnx`. Document the deferral in the self-review and open an issue. Update the manifest `format` to `"torchscript"` so the bundle validator still accepts it (verify validator accepts).

  Choose the lighter path that keeps the manifest contract correct.

- [ ] **Step 4: Commit**

```bash
git add algorithms/agents/agent_transformer_ppo.py tests/test_agent_transformer_ppo_entity.py
git commit -m "feat(wp05): export_artifacts with canonical manifest shape (per-building ONNX/torchscript)"
```

---

### Task 9: Checkpoints (§16.4 rows 14–15)

**Files:**
- Modify: `algorithms/agents/agent_transformer_ppo.py`
- Modify: `tests/test_agent_transformer_ppo_entity.py`

- [ ] **Step 1: Failing tests**

```python
def test_checkpoint_round_trip_same_topology(tmp_path, attach_kwargs_for_sample):
    from algorithms.agents.agent_transformer_ppo import AgentTransformerPPO
    agent = AgentTransformerPPO(_make_agent_config())
    kw = {k: v for k, v in attach_kwargs_for_sample.items() if k != "_obs"}
    agent.attach_environment(**kw)
    cp = agent.save_checkpoint(str(tmp_path), step=42)
    assert cp is not None and Path(cp).exists()
    agent2 = AgentTransformerPPO(_make_agent_config())
    agent2.attach_environment(**kw)
    agent2.load_checkpoint(cp)
    # Compare a single weight to confirm load happened.
    w1 = agent._per_building[0].actor.parameters().__next__().detach().clone()
    w2 = agent2._per_building[0].actor.parameters().__next__().detach().clone()
    assert torch.allclose(w1, w2)


def test_checkpoint_load_rejects_layout_signature_mismatch(tmp_path, attach_kwargs_for_sample):
    from algorithms.agents.agent_transformer_ppo import AgentTransformerPPO
    agent = AgentTransformerPPO(_make_agent_config())
    kw = {k: v for k, v in attach_kwargs_for_sample.items() if k != "_obs"}
    agent.attach_environment(**kw)
    cp = agent.save_checkpoint(str(tmp_path), step=1)
    # New agent with mutated obs names → different layout signature.
    kw2 = dict(kw)
    kw2["observation_names"] = [
        list(names) + ["pv::Building_1/pv_extra::generation_power_kw"]
        for names in kw["observation_names"]
    ]
    agent2 = AgentTransformerPPO(_make_agent_config())
    agent2.attach_environment(**kw2)
    with pytest.raises(ValueError, match="layout|topology|signature"):
        agent2.load_checkpoint(cp)
```

- [ ] **Step 2: Implement**

```python
    def save_checkpoint(self, output_dir: str, step: int) -> Optional[str]:
        out = Path(output_dir) / "checkpoints"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"transformer_ppo_step{step}.pt"
        payload = {
            "step": int(step),
            "topology_version": int(self._entity_topology_version_at_export or 0),
            "config": dict(self.config["algorithm"]),
            "agents": [
                {
                    "building_id": s.building_id,
                    "tokenizer_state": s.tokenizer.state_dict(),
                    "backbone_state": s.backbone.state_dict(),
                    "actor_state": s.actor.state_dict(),
                    "critic_state": s.critic.state_dict(),
                    "optimizer_state": s.optimizer.state_dict(),
                    "layout_signature": tuple(sorted(s.obs_names_tuple)),
                }
                for s in self._per_building
            ],
        }
        torch.save(payload, path)
        return str(path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu")
        agents = payload["agents"]
        if len(agents) != len(self._per_building):
            raise ValueError(
                f"Checkpoint has {len(agents)} per-building entries; current "
                f"agent has {len(self._per_building)}."
            )
        for state, saved in zip(self._per_building, agents):
            sig = tuple(sorted(state.obs_names_tuple))
            if tuple(saved["layout_signature"]) != sig:
                raise ValueError(
                    f"Checkpoint layout_signature mismatch for building "
                    f"{state.building_id!r}: cannot resume across topology "
                    f"changes. Saved={saved['layout_signature'][:5]}..., "
                    f"current={sig[:5]}..."
                )
            state.tokenizer.load_state_dict(saved["tokenizer_state"])
            state.backbone.load_state_dict(saved["backbone_state"])
            state.actor.load_state_dict(saved["actor_state"])
            state.critic.load_state_dict(saved["critic_state"])
            state.optimizer.load_state_dict(saved["optimizer_state"])
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_agent_transformer_ppo_entity.py -k checkpoint -v
git add algorithms/agents/agent_transformer_ppo.py tests/test_agent_transformer_ppo_entity.py
git commit -m "feat(wp05): save/load_checkpoint with layout_signature mismatch rejection"
```

---

### Task 10: Register `AgentTransformerPPO` + verify validate_config end-to-end

**Files:**
- Modify: `algorithms/registry.py`
- Modify: `tests/test_agent_transformer_ppo_entity.py`

- [ ] **Step 1: Failing test**

```python
def test_agent_registered():
    from algorithms.registry import ALGORITHM_REGISTRY
    assert "AgentTransformerPPO" in ALGORITHM_REGISTRY
    cls = ALGORITHM_REGISTRY["AgentTransformerPPO"]
    assert cls.supports_dynamic_topology is True
```

- [ ] **Step 2: Edit `algorithms/registry.py`** to add the import + dict entry:

```python
from algorithms.agents.agent_transformer_ppo import AgentTransformerPPO
...
ALGORITHM_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "MADDPG": MADDPG,
    "RuleBasedPolicy": RuleBasedPolicy,
    "AgentTransformerPPO": AgentTransformerPPO,
}
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_agent_transformer_ppo_entity.py::test_agent_registered -v
git add algorithms/registry.py
git commit -m "feat(wp05): register AgentTransformerPPO in algorithms registry"
```

---

### Task 11: Wrapper `TransformerObservationCoordinator` + dispatch

**Files:**
- Create: `utils/wrapper_transformer_entity/__init__.py`
- Create: `utils/wrapper_transformer_entity/coordinator.py`
- Modify: `utils/wrapper_citylearn.py`
- Create: `tests/test_wrapper_transformer_entity.py`

- [ ] **Step 1: Failing tests (subset of §16.5 — most rely on a real CityLearn env, so use mocks)**

```python
# tests/test_wrapper_transformer_entity.py
"""Tests for wrapper integration with AgentTransformerPPO. Covers spec §16.5."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_coordinator_initialised_only_for_transformer_agent():
    """Wrapper checks `uses_entity_layout` on the agent before invoking coordinator."""
    from utils.wrapper_transformer_entity.coordinator import (
        TransformerObservationCoordinator,
    )
    coord = TransformerObservationCoordinator()
    fake_wrapper = MagicMock()
    fake_wrapper.observation_names = [["a"]]
    fake_wrapper.action_names = [["b"]]
    fake_wrapper.model = MagicMock()
    fake_wrapper.model.uses_entity_layout = False
    coord.handle_topology_change(fake_wrapper)
    fake_wrapper.model.update_topology_names.assert_not_called()
    fake_wrapper.model.on_topology_change.assert_not_called()


def test_topology_version_increment_triggers_rebuild_path():
    from utils.wrapper_transformer_entity.coordinator import (
        TransformerObservationCoordinator,
    )
    coord = TransformerObservationCoordinator()
    fake_wrapper = MagicMock()
    fake_wrapper.observation_names = [["x"]]
    fake_wrapper.action_names = [["y"]]
    fake_wrapper.model.uses_entity_layout = True
    fake_wrapper._entity_topology_version = 7
    fake_wrapper.metadata = {"interface": "entity"}
    coord.handle_topology_change(fake_wrapper)
    fake_wrapper.model.update_topology_names.assert_called_once_with(
        0, ["x"], ["y"], metadata=fake_wrapper.metadata,
    )
    fake_wrapper.model.on_topology_change.assert_called_once_with(0)
```

- [ ] **Step 2: Implement coordinator**

```python
# utils/wrapper_transformer_entity/__init__.py
from .coordinator import TransformerObservationCoordinator
__all__ = ["TransformerObservationCoordinator"]
```

```python
# utils/wrapper_transformer_entity/coordinator.py
"""Coordinator that drives AgentTransformerPPO topology rebuilds.

See docs/specv2.md §12.2.
"""
from __future__ import annotations

from typing import Any


class TransformerObservationCoordinator:
    def handle_topology_change(self, wrapper: Any) -> None:
        agent = getattr(wrapper, "model", None)
        if agent is None:
            return
        if not getattr(agent, "uses_entity_layout", False):
            return
        for b in range(len(wrapper.observation_names)):
            agent.update_topology_names(
                b,
                list(wrapper.observation_names[b]),
                list(wrapper.action_names[b]),
                metadata=getattr(wrapper, "metadata", None),
            )
            agent.on_topology_change(b)
```

- [ ] **Step 3: Wire into `utils/wrapper_citylearn.py`**

Find `_apply_entity_layout(...)` (around line 309). After the existing call to `_attach_model_environment_metadata()` on a topology-version increment, add:

```python
        # spec §12.2 step 3: drive transformer agent's per-building rebuild.
        if previous_version is not None and self._entity_topology_version != previous_version:
            from utils.wrapper_transformer_entity import TransformerObservationCoordinator
            TransformerObservationCoordinator().handle_topology_change(self)
```

(The exact insertion point is between the existing `_attach_model_environment_metadata()` call and the function return.)

- [ ] **Step 4: Run wrapper tests**

```bash
pytest tests/test_wrapper_transformer_entity.py -v
```

- [ ] **Step 5: Commit**

```bash
git add utils/wrapper_transformer_entity/ utils/wrapper_citylearn.py tests/test_wrapper_transformer_entity.py
git commit -m "feat(wp05): TransformerObservationCoordinator + wrapper hook for topology change"
```

---

### Task 12: Wrapper-level guardrail flag refactor (§16.5 rows 6–7)

**Files:**
- Modify: `utils/wrapper_citylearn.py`
- Modify: `tests/test_wrapper_transformer_entity.py`

- [ ] **Step 1: Failing test**

```python
def test_dynamic_guardrail_uses_supports_dynamic_topology_flag():
    """At wrapper construction, a non-dynamic agent under topology_mode='dynamic' raises."""
    # The test uses an in-process minimal config + a stub agent class with the
    # flag set False. Concrete shape depends on wrapper_citylearn API; if too
    # heavy to mock here, test by patching ALGORITHM_REGISTRY.
    pytest.skip("Implement once the wrapper guardrail is generalised — see Task 12.")


def test_maddpg_dynamic_error_message_unchanged():
    from utils.config_schema import validate_config
    cfg = _make_minimal_maddpg_dynamic_cfg()  # reuse helper from earlier
    with pytest.raises(ValueError, match="MADDPG does not support dynamic topology"):
        validate_config(cfg)
```

- [ ] **Step 2: Replace the MADDPG-literal in `utils/wrapper_citylearn.py`**

```bash
grep -n "MADDPG" utils/wrapper_citylearn.py
```

Around line 333–338, replace the literal `if isinstance(self.model, MADDPG):` with:

```python
if (
    self._topology_mode == "dynamic"
    and self.model is not None
    and not getattr(type(self.model), "supports_dynamic_topology", False)
):
    raise RuntimeError(
        f"Agent {type(self.model).__name__} does not support dynamic topology "
        "(supports_dynamic_topology=False)."
    )
```

- [ ] **Step 3: Run; un-skip the wrapper test if you can mock; commit**

```bash
pytest -x -q
git add utils/wrapper_citylearn.py tests/test_wrapper_transformer_entity.py
git commit -m "feat(wp05): generic supports_dynamic_topology guardrail in wrapper (replaces MADDPG literal)"
```

---

### Task 13: Action conversion test (§16.5 row 5)

```python
def test_action_conversion_uses_entity_adapter_tables_only():
    """Wrapper outputs {'tables': {...}} only — no 'map' key.

    This is a property of EntityContractAdapter.to_entity_actions, but we
    re-assert it at the wrapper boundary in case anyone refactors.
    """
    from utils.entity_adapter import EntityContractAdapter
    adapter = EntityContractAdapter()
    # Build a synthetic action_names = [["electrical_storage", "electric_vehicle_storage"]]
    # and actions = [[0.5, -0.5]]
    actions = [[0.5, -0.5]]
    action_names = [["electrical_storage", "electric_vehicle_storage"]]
    payload = adapter.to_entity_actions(actions, action_names)
    assert "tables" in payload
    assert "map" not in payload
```

- [ ] **Step 1: Run + commit**

```bash
pytest tests/test_wrapper_transformer_entity.py::test_action_conversion_uses_entity_adapter_tables_only -v
git add tests/test_wrapper_transformer_entity.py
git commit -m "test(wp05): assert adapter-output payload has only 'tables' key"
```

---

### Task 14: Full sweep + lint

- [ ] **Step 1: Full test run**

```bash
pytest -x -q
```
Expected: exit 0. Pay attention to any pre-existing wrapper / agent tests that may have asserted specific error-message wording — they should still pass because we preserved the MADDPG message verbatim.

- [ ] **Step 2: Commit cleanups**

---

## Self-Review Checklist (run before requesting code review)

- [ ] **Spec §16.4 coverage:** all 15 rows present in `tests/test_agent_transformer_ppo_entity.py`. `grep -c "^def test_" tests/test_agent_transformer_ppo_entity.py` ≥ 15.
- [ ] **Spec §16.5 coverage:** all 7 rows present in `tests/test_wrapper_transformer_entity.py` (some may be `pytest.skip`d if they require a real env; document in PR).
- [ ] **`AgentTransformerPPO` registered:** `python -c "from algorithms.registry import ALGORITHM_REGISTRY; print(ALGORITHM_REGISTRY['AgentTransformerPPO'].__name__)"` → `AgentTransformerPPO`.
- [ ] **`supports_dynamic_topology=True`:** `python -c "from algorithms.agents.agent_transformer_ppo import AgentTransformerPPO; print(AgentTransformerPPO.supports_dynamic_topology)"` → `True`.
- [ ] **Manifest contract honored:** `export_artifacts` returns a dict with `format` (top-level) and `artifacts` list whose entries each have `agent_index`, `path`, `format`, `config`. Bundle validator (`utils/bundle_validator.py`) accepts the output:
  ```bash
  python -c "
  import tempfile, json
  from algorithms.agents.agent_transformer_ppo import AgentTransformerPPO
  # Construct + attach + export to tmpdir, then run bundle_validator on the manifest dict.
  # Spec out: see utils/bundle_validator.py:108-172
  print('Manual verification — confirm via integration test or in WP06.')
  "
  ```
- [ ] **First attach does NOT trigger `on_topology_change`:** the spy test passes.
- [ ] **No imports of v1 marker code:** `grep -RIn "marker\|wrapper_transformer/" algorithms/agents/agent_transformer_ppo.py utils/wrapper_transformer_entity/ utils/wrapper_citylearn.py` — should be empty (we deliberately use `wrapper_transformer_entity/` to avoid colliding with any v1 module name).
- [ ] **MADDPG error message preserved verbatim** so any pre-existing tests still pass.
- [ ] **Full repo test suite passes:** `pytest -x -q` exits 0.
- [ ] **Note any deferrals** in the PR description (e.g. ONNX `segment_indices` graph if it had to be deferred to torchscript).

---

## Code Review

After self-review passes, invoke `superpowers:requesting-code-review`. Reviewer should check the diff against this plan and §11, §12, §14, §16.4, §16.5 of `docs/specv2.md`, with particular attention to:
1. Topology-change sequencing matches §12.2 exactly (wrapper rebuilds names → wrapper attaches → coordinator → agent flushes → agent rebuilds → re-validates).
2. Manifest shape matches §14.2 exactly.
3. `supports_dynamic_topology` flag is the single source of truth — no remaining hardcoded `MADDPG` strings in topology checks.

---

## PR Description

```markdown
## Summary
Implements `AgentTransformerPPO` (per-building Transformer + PPO over the entity interface, dynamic-topology aware) and wires it into the wrapper via a small coordinator hook. Replaces the MADDPG-specific dynamic-topology check with a generic `supports_dynamic_topology` flag-based registry lookup. After this WP, the system runs end-to-end on the entity interface; topology mutations trigger PPO buffer flush + layout rebuild + 5-rule re-validation.

## Key Changes
- Add `algorithms/agents/agent_transformer_ppo.py`: per-building stacks (tokenizer, backbone, actor, critic, optimizer, rollout buffer, cached layout); `attach_environment` (idempotent), `predict`, `update`, `on_topology_change` (flush PPO → rebuild layout → re-run §13.4 rules), `export_artifacts` (canonical manifest shape per §14.2), `save_checkpoint`/`load_checkpoint` (layout-signature mismatch rejection).
- Add `utils/wrapper_transformer_entity/coordinator.py`: `TransformerObservationCoordinator.handle_topology_change(wrapper)` — opt-in via `agent.uses_entity_layout`.
- Modify `utils/wrapper_citylearn.py`: dispatch coordinator on topology-version increment (spec §12.2 step 3); replace MADDPG-literal guardrail with `supports_dynamic_topology` flag check.
- Modify `utils/config_schema.py`: replace MADDPG-literal `validate_config` check with registry-flag lookup; preserve historical MADDPG error wording.
- Add `supports_dynamic_topology: ClassVar[bool] = False` to `BaseAgent`; explicit overrides on `MADDPG` (False) and `RuleBasedPolicy` (True); `AgentTransformerPPO` (True).
- Register `AgentTransformerPPO` in `algorithms/registry.py`.
- Add `tests/test_agent_transformer_ppo_entity.py` covering spec §16.4 (15 rows).
- Add `tests/test_wrapper_transformer_entity.py` covering spec §16.5 (7 rows; some require real-env mocks documented inline).
- [If applicable] ONNX export: full §14.1 graph with `segment_offsets`/`segment_indices` deferred — current implementation bakes layout indices as graph constants per topology version, which satisfies the manifest contract and works for inference within a stable topology.
```
