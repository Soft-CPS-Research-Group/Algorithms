from __future__ import annotations

from pathlib import Path

import pytest
import torch

from algorithms.agents.maddpg_agent import MADDPG
from algorithms.agents.matd3_agent import MATD3


class _ReplayBufferProbe:
    def __init__(self):
        self.loaded_state = None
        self.pushed_done = None

    def set_state(self, state):
        self.loaded_state = state

    def push(self, _states, _actions, _rewards, _next_states, done):
        self.pushed_done = done

    def __len__(self):
        return 0


def _build_checkpoint_payload():
    actor = torch.nn.Linear(2, 1)
    critic = torch.nn.Linear(3, 1)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    actor_loss = actor(torch.ones(1, 2)).sum()
    actor_loss.backward()
    actor_optimizer.step()
    actor_optimizer.zero_grad(set_to_none=True)

    critic_loss = critic(torch.ones(1, 3)).sum()
    critic_loss.backward()
    critic_optimizer.step()
    critic_optimizer.zero_grad(set_to_none=True)

    return {
        "actor_state_dict_0": actor.state_dict(),
        "critic_state_dict_0": critic.state_dict(),
        "actor_optimizer_state_dict_0": actor_optimizer.state_dict(),
        "critic_optimizer_state_dict_0": critic_optimizer.state_dict(),
        "replay_buffer": {"entries": 7},
        "exploration_state": {"sigma": 0.123, "exploration_step": 42},
        "reward_normalization_state": {"count": 9, "mean": 3.0, "m2": 4.0},
    }


def _build_agent_for_load() -> MADDPG:
    agent = MADDPG.__new__(MADDPG)
    agent.device = torch.device("cpu")
    agent.num_agents = 1
    agent.actors = [torch.nn.Linear(2, 1)]
    agent.critics = [torch.nn.Linear(3, 1)]
    agent.actor_optimizers = [torch.optim.Adam(agent.actors[0].parameters(), lr=1e-3)]
    agent.critic_optimizers = [torch.optim.Adam(agent.critics[0].parameters(), lr=1e-3)]
    agent.replay_buffer = _ReplayBufferProbe()
    agent.fine_tune = False
    agent.reset_replay_buffer = False
    agent.freeze_pretrained_layers = False
    agent.sigma = 0.9
    agent.exploration_step = 0
    agent.reward_norm_count = 0
    agent.reward_norm_mean = 0.0
    agent.reward_norm_m2 = 0.0
    return agent


def _build_matd3_checkpoint_payload():
    payload = _build_checkpoint_payload()
    critic_2 = torch.nn.Linear(3, 1)
    critic_optimizer_2 = torch.optim.Adam(critic_2.parameters(), lr=1e-3)
    critic_loss = critic_2(torch.ones(1, 3)).sum()
    critic_loss.backward()
    critic_optimizer_2.step()
    critic_optimizer_2.zero_grad(set_to_none=True)
    payload["critic_2_state_dict_0"] = critic_2.state_dict()
    payload["critic_optimizer_2_state_dict_0"] = critic_optimizer_2.state_dict()
    return payload


def _build_matd3_agent_for_load() -> MATD3:
    agent = MATD3.__new__(MATD3)
    agent.device = torch.device("cpu")
    agent.num_agents = 1
    agent.actors = [torch.nn.Linear(2, 1)]
    agent.actor_targets = [torch.nn.Linear(2, 1)]
    agent.critics = [torch.nn.Linear(3, 1)]
    agent.critics_2 = [torch.nn.Linear(3, 1)]
    agent.critic_targets = [torch.nn.Linear(3, 1)]
    agent.critic_targets_2 = [torch.nn.Linear(3, 1)]
    agent.actor_aux_heads = []
    agent.actor_optimizers = [torch.optim.Adam(agent.actors[0].parameters(), lr=1e-3)]
    agent.critic_optimizers = [torch.optim.Adam(agent.critics[0].parameters(), lr=1e-3)]
    agent.critic_optimizers_2 = [torch.optim.Adam(agent.critics_2[0].parameters(), lr=1e-3)]
    agent.replay_buffer = _ReplayBufferProbe()
    agent.fine_tune = False
    agent.reset_replay_buffer = False
    agent.freeze_pretrained_layers = False
    agent.sigma = 0.9
    agent.exploration_step = 0
    agent.reward_norm_count = 0
    agent.reward_norm_mean = 0.0
    agent.reward_norm_m2 = 0.0
    return agent


def test_maddpg_load_checkpoint_restores_weights_optimizers_and_replay(tmp_path):
    payload = _build_checkpoint_payload()
    checkpoint_path = tmp_path / "resume_checkpoint.pth"
    torch.save(payload, checkpoint_path)

    agent = _build_agent_for_load()
    agent.load_checkpoint(str(checkpoint_path))

    expected_actor_state = payload["actor_state_dict_0"]
    expected_critic_state = payload["critic_state_dict_0"]
    for key, value in expected_actor_state.items():
        assert torch.equal(agent.actors[0].state_dict()[key], value)
    for key, value in expected_critic_state.items():
        assert torch.equal(agent.critics[0].state_dict()[key], value)
    assert len(agent.actor_optimizers[0].state_dict()["state"]) > 0
    assert len(agent.critic_optimizers[0].state_dict()["state"]) > 0
    assert agent.replay_buffer.loaded_state == {"entries": 7}
    assert agent.sigma == 0.123
    assert agent.exploration_step == 42


def test_maddpg_load_checkpoint_respects_fine_tune_and_freeze_flags(tmp_path):
    payload = _build_checkpoint_payload()
    checkpoint_path = tmp_path / "resume_checkpoint.pth"
    torch.save(payload, checkpoint_path)

    agent = _build_agent_for_load()
    agent.fine_tune = True
    agent.freeze_pretrained_layers = True

    freeze_calls = []
    agent.freeze_layers = lambda freeze_actor=True, freeze_critic=False: freeze_calls.append(  # type: ignore[method-assign]
        (freeze_actor, freeze_critic)
    )
    actor_optimizer_state_before = agent.actor_optimizers[0].state_dict()

    agent.load_checkpoint(str(checkpoint_path))

    assert agent.actor_optimizers[0].state_dict()["state"] == actor_optimizer_state_before["state"]
    assert freeze_calls == [(True, False)]


def test_maddpg_explicit_restore_flags_override_legacy_fine_tune_mix(tmp_path):
    payload = _build_checkpoint_payload()
    checkpoint_path = tmp_path / "resume_checkpoint.pth"
    torch.save(payload, checkpoint_path)

    agent = _build_agent_for_load()
    agent.fine_tune = True
    agent.reset_replay_buffer = True
    agent.restore_optimizers = True
    agent.restore_replay_buffer = False
    agent.restore_exploration_state = False
    agent.restore_reward_normalizer = False

    agent.load_checkpoint(str(checkpoint_path))

    assert len(agent.actor_optimizers[0].state_dict()["state"]) > 0
    assert agent.replay_buffer.loaded_state is None
    assert agent.sigma == 0.9
    assert agent.exploration_step == 0
    assert agent.reward_norm_count == 0
    assert agent.reward_norm_mean == 0.0
    assert agent.reward_norm_m2 == 0.0


def test_matd3_explicit_restore_flags_can_reset_continuation_state(tmp_path):
    payload = _build_matd3_checkpoint_payload()
    checkpoint_path = tmp_path / "matd3_resume_checkpoint.pth"
    torch.save(payload, checkpoint_path)

    agent = _build_matd3_agent_for_load()
    agent.restore_optimizers = False
    agent.restore_replay_buffer = False
    agent.restore_exploration_state = False
    agent.restore_reward_normalizer = False

    agent.load_checkpoint(str(checkpoint_path))

    for key, value in payload["actor_state_dict_0"].items():
        assert torch.equal(agent.actors[0].state_dict()[key], value)
    for key, value in payload["critic_state_dict_0"].items():
        assert torch.equal(agent.critics[0].state_dict()[key], value)
    for key, value in payload["critic_2_state_dict_0"].items():
        assert torch.equal(agent.critics_2[0].state_dict()[key], value)
    assert agent.actor_optimizers[0].state_dict()["state"] == {}
    assert agent.critic_optimizers[0].state_dict()["state"] == {}
    assert agent.critic_optimizers_2[0].state_dict()["state"] == {}
    assert agent.replay_buffer.loaded_state is None
    assert agent.sigma == 0.9
    assert agent.exploration_step == 0
    assert agent.reward_norm_count == 0
    assert agent.reward_norm_mean == 0.0
    assert agent.reward_norm_m2 == 0.0


def test_matd3_explicit_restore_flags_override_legacy_fine_tune_mix(tmp_path):
    payload = _build_matd3_checkpoint_payload()
    checkpoint_path = tmp_path / "matd3_resume_checkpoint.pth"
    torch.save(payload, checkpoint_path)

    agent = _build_matd3_agent_for_load()
    agent.fine_tune = True
    agent.reset_replay_buffer = True
    agent.restore_optimizers = True
    agent.restore_replay_buffer = True
    agent.restore_exploration_state = True
    agent.restore_reward_normalizer = True

    agent.load_checkpoint(str(checkpoint_path))

    assert len(agent.actor_optimizers[0].state_dict()["state"]) > 0
    assert len(agent.critic_optimizers[0].state_dict()["state"]) > 0
    assert len(agent.critic_optimizers_2[0].state_dict()["state"]) > 0
    assert agent.replay_buffer.loaded_state == {"entries": 7}
    assert agent.sigma == 0.123
    assert agent.exploration_step == 42
    assert agent.reward_norm_count == 9
    assert agent.reward_norm_mean == 3.0
    assert agent.reward_norm_m2 == 4.0


def test_matd3_inference_checkpoint_contains_only_frozen_actor_state(tmp_path):
    source = _build_matd3_agent_for_load()
    source.checkpoint_mode = "inference"
    source.observation_dimension = [2]
    source.action_dimension = [1]
    source.checkpoint_artifact = "matd3_leaf.pth"
    with torch.no_grad():
        source.actors[0].weight.fill_(0.75)
        source.actors[0].bias.fill_(-0.25)
        source.actor_targets[0].load_state_dict(source.actors[0].state_dict())

    checkpoint_path = source.save_checkpoint(str(tmp_path), step=42)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert payload["checkpoint_mode"] == "inference"
    assert payload["num_agents"] == 1
    assert "actor_state_dict_0" in payload
    assert "critic_state_dict_0" not in payload
    assert "critic_2_state_dict_0" not in payload
    assert "actor_optimizer_state_dict_0" not in payload
    assert "replay_buffer" not in payload

    restored = _build_matd3_agent_for_load()
    restored.frozen = True
    critic_before = {
        key: value.clone() for key, value in restored.critics[0].state_dict().items()
    }
    restored.load_checkpoint(checkpoint_path)

    for key, value in source.actors[0].state_dict().items():
        assert torch.equal(restored.actors[0].state_dict()[key], value)
        assert torch.equal(restored.actor_targets[0].state_dict()[key], value)
    for key, value in critic_before.items():
        assert torch.equal(restored.critics[0].state_dict()[key], value)


def test_matd3_inference_checkpoint_rejects_trainable_stage(tmp_path):
    source = _build_matd3_agent_for_load()
    source.checkpoint_mode = "inference"
    source.observation_dimension = [2]
    source.action_dimension = [1]
    checkpoint_path = source.save_checkpoint(str(tmp_path), step=1)

    trainable = _build_matd3_agent_for_load()
    trainable.frozen = False
    with pytest.raises(RuntimeError, match="only into a frozen pipeline stage"):
        trainable.load_checkpoint(checkpoint_path)


def test_maddpg_update_uses_terminated_or_truncated_for_done():
    agent = MADDPG.__new__(MADDPG)
    agent.replay_buffer = _ReplayBufferProbe()
    agent.batch_size = 10

    agent.update(
        observations=[torch.zeros(2)],
        actions=[torch.zeros(1)],
        rewards=[0.0],
        next_observations=[torch.zeros(2)],
        terminated=False,
        truncated=True,
        update_target_step=False,
        global_learning_step=1,
        update_step=True,
        initial_exploration_done=True,
    )

    assert agent.replay_buffer.pushed_done is True
