from __future__ import annotations

from pathlib import Path

import torch

from algorithms.agents.maddpg_agent import MADDPG


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
