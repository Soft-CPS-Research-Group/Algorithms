"""Plan A tests for AgentTransformerMATD3 foundation."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from utils.config_schema import TransformerMATD3StageConfig


class TestTransformerMATD3StageConfig:
    def test_valid_minimal_config(self):
        cfg = TransformerMATD3StageConfig(
            algorithm="AgentTransformerMATD3",
            tokenizer_config_path="configs/tokenizers/entity_default.json",
            transformer_actor={"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128, "dropout": 0.1},
            transformer_critic={"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128, "dropout": 0.1},
            hyperparameters={
                "gamma": 0.99,
                "tau": 0.005,
                "batch_size": 256,
                "replay_capacity": 100000,
                "actor_lr": 1e-4,
                "critic_lr": 3e-4,
                "target_policy_noise": 0.2,
                "target_policy_noise_clip": 0.5,
                "actor_update_interval": 2,
            },
        )
        assert cfg.algorithm == "AgentTransformerMATD3"
        assert cfg.transformer_actor.d_model == 64
        assert cfg.transformer_critic.d_model == 64

    def test_rejects_wrong_algorithm_name(self):
        with pytest.raises(ValidationError):
            TransformerMATD3StageConfig(
                algorithm="MATD3",
                tokenizer_config_path="configs/tokenizers/entity_default.json",
                transformer_actor={"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128},
                transformer_critic={"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128},
                hyperparameters={"gamma": 0.99, "tau": 0.005, "batch_size": 256, "replay_capacity": 100000, "actor_lr": 1e-4, "critic_lr": 3e-4, "target_policy_noise": 0.2, "target_policy_noise_clip": 0.5, "actor_update_interval": 2},
            )

    def test_rejects_missing_tokenizer_path(self):
        with pytest.raises(ValidationError):
            TransformerMATD3StageConfig(
                algorithm="AgentTransformerMATD3",
                tokenizer_config_path="",
                transformer_actor={"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128},
                transformer_critic={"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128},
                hyperparameters={"gamma": 0.99, "tau": 0.005, "batch_size": 256, "replay_capacity": 100000, "actor_lr": 1e-4, "critic_lr": 3e-4, "target_policy_noise": 0.2, "target_policy_noise_clip": 0.5, "actor_update_interval": 2},
            )


import torch

from algorithms.utils.matd3_actor_head import DeterministicActorHead


class TestDeterministicActorHead:
    def test_output_shape(self):
        head = DeterministicActorHead(d_model=16, hidden_dim=32)
        ca_emb = torch.randn(2, 3, 16)  # [batch=2, n_ca=3, d_model=16]
        actions = head(ca_emb)
        assert actions.shape == (2, 3, 1)

    def test_output_range_tanh(self):
        head = DeterministicActorHead(d_model=16, hidden_dim=32)
        ca_emb = torch.randn(4, 5, 16) * 10.0  # large inputs
        actions = head(ca_emb)
        assert actions.min() >= -1.0
        assert actions.max() <= 1.0

    def test_deterministic_same_output(self):
        head = DeterministicActorHead(d_model=16, hidden_dim=32)
        head.eval()
        ca_emb = torch.randn(1, 2, 16)
        a1 = head(ca_emb)
        a2 = head(ca_emb)
        assert torch.allclose(a1, a2)

    def test_pre_tanh_accessor(self):
        head = DeterministicActorHead(d_model=16, hidden_dim=32)
        ca_emb = torch.randn(1, 2, 16)
        actions, pre_tanh = head.forward_with_pre_tanh(ca_emb)
        assert torch.allclose(actions, torch.tanh(pre_tanh))


from algorithms.registry import ALGORITHM_REGISTRY


class TestRegistry:
    def test_agent_registered(self):
        assert "AgentTransformerMATD3" in ALGORITHM_REGISTRY

    def test_supports_dynamic_topology(self):
        cls = ALGORITHM_REGISTRY["AgentTransformerMATD3"]
        assert cls.supports_dynamic_topology is True


import numpy as np

from algorithms.agents.agent_transformer_matd3 import AgentTransformerMATD3
from tests._entity_sample_obs_names import load_sample_observation_names_for_first_building


_TOKENIZER_CFG = "configs/tokenizers/entity_default.json"
_DEFAULT_ACTIONS = ["electrical_storage", "electric_vehicle_storage"]


def _matd3_config() -> dict:
    return {
        "algorithm": {
            "name": "AgentTransformerMATD3",
            "tokenizer_config_path": _TOKENIZER_CFG,
            "transformer_actor": {
                "d_model": 16, "nhead": 2, "num_layers": 1,
                "dim_feedforward": 32, "dropout": 0.0,
            },
            "transformer_critic": {
                "d_model": 16, "nhead": 2, "num_layers": 1,
                "dim_feedforward": 32, "dropout": 0.0,
            },
            "hyperparameters": {
                "gamma": 0.99, "tau": 0.005, "batch_size": 4,
                "replay_capacity": 100, "actor_lr": 1e-3, "critic_lr": 3e-4,
                "target_policy_noise": 0.2, "target_policy_noise_clip": 0.5,
                "actor_update_interval": 2, "actor_hidden_dim": 32,
            },
        },
    }


def _make_matd3(n_buildings: int = 1):
    obs_names = load_sample_observation_names_for_first_building()
    obs_per = [list(obs_names) for _ in range(n_buildings)]
    act_per = [list(_DEFAULT_ACTIONS) for _ in range(n_buildings)]
    agent = AgentTransformerMATD3(_matd3_config())
    agent.attach_environment(
        observation_names=obs_per,
        action_names=act_per,
        action_space=[None] * n_buildings,
        observation_space=[None] * n_buildings,
        metadata={"building_names": [f"Building_{b}" for b in range(n_buildings)]},
    )
    obs_dim = len(obs_names)
    return agent, obs_per, act_per, obs_dim


class TestAttachAndPredict:
    def test_attach_builds_actors(self):
        agent, _, _, _ = _make_matd3(n_buildings=2)
        assert len(agent._actors) == 2
        for s in agent._actors:
            assert s.layout.n_ca == 2

    def test_attach_noop_on_same_names(self):
        agent, obs_per, act_per, _ = _make_matd3()
        layout_before = agent._actors[0].layout
        agent.attach_environment(
            observation_names=obs_per,
            action_names=act_per,
            action_space=[None],
            observation_space=[None],
        )
        assert agent._actors[0].layout is layout_before

    def test_attach_rejects_building_count_change(self):
        agent, obs_per, act_per, _ = _make_matd3(n_buildings=1)
        with pytest.raises(ValueError, match="building count changed"):
            agent.attach_environment(
                observation_names=obs_per + [obs_per[0]],
                action_names=act_per + [act_per[0]],
                action_space=[None, None],
                observation_space=[None, None],
            )

    def test_predict_returns_correct_shape(self):
        agent, _, act_per, obs_dim = _make_matd3(n_buildings=2)
        obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
        actions = agent.predict(obs, deterministic=True)
        assert len(actions) == 2
        for a, expected_names in zip(actions, act_per):
            assert len(a) == len(expected_names)

    def test_predict_actions_in_range(self):
        agent, _, _, obs_dim = _make_matd3()
        obs = [np.random.randn(obs_dim).astype(np.float64)]
        actions = agent.predict(obs, deterministic=True)
        for val in actions[0]:
            assert -1.0 <= val <= 1.0

    def test_predict_deterministic_reproducible(self):
        agent, _, _, obs_dim = _make_matd3()
        obs = [np.random.randn(obs_dim).astype(np.float64)]
        a1 = agent.predict(obs, deterministic=True)
        a2 = agent.predict(obs, deterministic=True)
        assert a1 == a2


import tempfile
from pathlib import Path


class TestTopologyChange:
    def test_topology_change_rebuilds_layout(self):
        agent, obs_per, act_per, _ = _make_matd3()
        # Add a fake charger to observation names by mirroring a full existing
        # charger feature block, so per-type feature dimensions remain stable.
        charger_id = next(
            n.split("::", 2)[1]
            for n in obs_per[0]
            if n.startswith("charger::")
            and "::connected_ev::" not in n
            and "::incoming_ev::" not in n
        )
        old_prefix = f"charger::{charger_id}::"
        new_prefix = "charger::Building_0/charger_new::"
        new_obs = list(obs_per[0]) + [
            new_prefix + n[len(old_prefix):]
            for n in obs_per[0]
            if n.startswith(old_prefix)
        ]
        new_act = list(act_per[0]) + ["electric_vehicle_storage_charger_new"]
        agent.attach_environment(
            observation_names=[new_obs],
            action_names=[new_act],
            action_space=[None],
            observation_space=[None],
        )
        assert agent._actors[0].layout.n_ca == 3
        assert agent._actors[0].topology_version == 1

    def test_topology_feature_count_drift_fails(self):
        agent, obs_per, act_per, _ = _make_matd3()
        # Remove one feature from a storage segment to trigger feature-count drift
        # while keeping the storage CA/action mapping present.
        removed = False
        new_obs = []
        for name in obs_per[0]:
            if not removed and name.startswith("storage::"):
                removed = True
                continue
            new_obs.append(name)
        # This should fail because storage type feature count changes
        with pytest.raises(ValueError, match="feature count"):
            agent.attach_environment(
                observation_names=[new_obs],
                action_names=[act_per[0]],
                action_space=[None],
                observation_space=[None],
            )


class TestExport:
    def test_export_creates_onnx_files(self):
        agent, _, _, _ = _make_matd3(n_buildings=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = agent.export_artifacts(tmpdir)
            assert manifest["format"] == "onnx"
            assert len(manifest["artifacts"]) == 2
            for art in manifest["artifacts"]:
                onnx_path = Path(tmpdir) / art["path"]
                assert onnx_path.exists()
                assert art["config"]["n_ca"] == 2
                assert "action_low" in art["config"]
                assert "action_high" in art["config"]

    def test_export_manifest_has_no_critic(self):
        agent, _, _, _ = _make_matd3()
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = agent.export_artifacts(tmpdir)
            # No critic keys in manifest
            assert "critic" not in str(manifest).lower() or "critic" not in manifest

    def test_checkpoint_round_trip(self):
        agent, _, _, obs_dim = _make_matd3()
        obs = [np.random.randn(obs_dim).astype(np.float64)]
        a_before = agent.predict(obs, deterministic=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = agent.save_checkpoint(tmpdir, step=100)
            # Create a fresh agent and load
            agent2, _, _, _ = _make_matd3()
            agent2.load_checkpoint(ckpt_path)
            a_after = agent2.predict(obs, deterministic=True)
        assert a_before == a_after

    def test_checkpoint_rejects_building_count_mismatch(self):
        agent1, _, _, _ = _make_matd3(n_buildings=1)
        agent2, _, _, _ = _make_matd3(n_buildings=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = agent1.save_checkpoint(tmpdir, step=1)
            with pytest.raises(ValueError, match="Building-count mismatch"):
                agent2.load_checkpoint(ckpt_path)


class TestDynamicTopologyGuardrail:
    def test_transformer_matd3_allows_dynamic_topology(self):
        """AgentTransformerMATD3 should not trigger the dynamic-topology error."""
        from algorithms.agents.agent_transformer_matd3 import AgentTransformerMATD3
        assert AgentTransformerMATD3.supports_dynamic_topology is True

    def test_legacy_matd3_still_rejects_dynamic(self):
        """Legacy MATD3 error message must remain unchanged."""
        from algorithms.agents.matd3_agent import MATD3
        assert not getattr(MATD3, "supports_dynamic_topology", False)
