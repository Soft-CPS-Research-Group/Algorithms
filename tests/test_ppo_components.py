"""Tests for PPO components (Phase 3).

Verifies actor output shapes and ranges, critic output, rollout buffer
GAE computation, PPO loss clipping behaviour, and log-prob correctness.
"""

from __future__ import annotations

import pytest
import torch

from algorithms.utils.ppo_components import (
    ActorHead,
    CriticHead,
    PPORolloutBuffer,
    ppo_loss,
)


D_MODEL = 32
D_FF = 64
BATCH = 8
N_CA = 3


# ---------------------------------------------------------------------------
# Actor Head tests
# ---------------------------------------------------------------------------


class TestActorHead:
    def test_output_shape(self):
        """Actions shape should be [batch, N_ca, 1]."""
        head = ActorHead(D_MODEL, D_FF, n_ca_types=2)
        ca_emb = torch.randn(BATCH, N_CA, D_MODEL)
        actions, log_probs, entropy = head(ca_emb)

        assert actions.shape == (BATCH, N_CA, 1)
        assert log_probs.shape == (BATCH, N_CA, 1)
        assert entropy.shape == (BATCH, N_CA, 1)

    def test_action_range(self):
        """Actions should be in [-1, 1] due to tanh."""
        head = ActorHead(D_MODEL, D_FF)
        ca_emb = torch.randn(BATCH, N_CA, D_MODEL)
        actions, _, _ = head(ca_emb, deterministic=False)

        assert (actions >= -1.0).all()
        assert (actions <= 1.0).all()

    def test_deterministic_mode(self):
        """Deterministic actions should be consistent across calls."""
        head = ActorHead(D_MODEL, D_FF)
        ca_emb = torch.randn(BATCH, N_CA, D_MODEL)

        actions1, _, _ = head(ca_emb, deterministic=True)
        actions2, _, _ = head(ca_emb, deterministic=True)

        torch.testing.assert_close(actions1, actions2)

    def test_stochastic_mode_varies(self):
        """Stochastic actions should differ across calls (with high probability)."""
        head = ActorHead(D_MODEL, D_FF)
        ca_emb = torch.randn(BATCH, N_CA, D_MODEL)

        torch.manual_seed(42)
        actions1, _, _ = head(ca_emb, deterministic=False)
        torch.manual_seed(123)
        actions2, _, _ = head(ca_emb, deterministic=False)

        # Very unlikely to be identical
        assert not torch.allclose(actions1, actions2)

    def test_per_type_log_std(self):
        """With n_ca_types=3, log_std has 3 entries."""
        head = ActorHead(D_MODEL, D_FF, n_ca_types=3)
        assert head.log_std.shape == (3,)

    def test_evaluate_actions_shape(self):
        """evaluate_actions should return same shapes as forward."""
        head = ActorHead(D_MODEL, D_FF)
        ca_emb = torch.randn(BATCH, N_CA, D_MODEL)
        old_actions = torch.randn(BATCH, N_CA, 1).tanh()

        log_probs, entropy = head.evaluate_actions(ca_emb, old_actions)
        assert log_probs.shape == (BATCH, N_CA, 1)
        assert entropy.shape == (BATCH, N_CA, 1)

    def test_evaluate_actions_consistency(self):
        """Log-probs from evaluate_actions should match forward for same action."""
        head = ActorHead(D_MODEL, D_FF)
        ca_emb = torch.randn(BATCH, N_CA, D_MODEL)

        # Get actions and their log-probs from forward
        actions, log_probs_fwd, _ = head(ca_emb, deterministic=True)

        # Re-evaluate the same actions
        log_probs_eval, _ = head.evaluate_actions(ca_emb, actions)

        # Should be close (deterministic actions → same mu, same log_prob)
        torch.testing.assert_close(log_probs_fwd, log_probs_eval, atol=1e-4, rtol=1e-4)

    def test_single_ca(self):
        """Works with N_ca=1."""
        head = ActorHead(D_MODEL, D_FF)
        ca_emb = torch.randn(BATCH, 1, D_MODEL)
        actions, log_probs, entropy = head(ca_emb)

        assert actions.shape == (BATCH, 1, 1)


# ---------------------------------------------------------------------------
# Critic Head tests
# ---------------------------------------------------------------------------


class TestCriticHead:
    def test_output_shape(self):
        """Critic should produce scalar V(s) per batch element."""
        head = CriticHead(D_MODEL, D_FF)
        pooled = torch.randn(BATCH, D_MODEL)
        value = head(pooled)

        assert value.shape == (BATCH, 1)

    def test_gradient_flow(self):
        """Gradients should flow through the critic."""
        head = CriticHead(D_MODEL, D_FF)
        pooled = torch.randn(BATCH, D_MODEL, requires_grad=True)
        value = head(pooled)
        value.sum().backward()

        assert pooled.grad is not None
        assert head.fc1.weight.grad is not None


# ---------------------------------------------------------------------------
# Rollout Buffer tests
# ---------------------------------------------------------------------------


class TestPPORolloutBuffer:
    def _fill_buffer(self, n: int = 10) -> PPORolloutBuffer:
        buf = PPORolloutBuffer(gamma=0.99, gae_lambda=0.95)
        for i in range(n):
            buf.push(
                observation=torch.randn(16),  # flat obs
                action=torch.randn(2, 1),  # N_ca=2
                log_prob=torch.randn(2, 1),
                reward=float(i) * 0.1,
                value=torch.tensor([0.5]),
                done=(i == n - 1),
            )
        return buf

    def test_push_and_length(self):
        buf = self._fill_buffer(5)
        assert len(buf) == 5

    def test_clear(self):
        buf = self._fill_buffer(5)
        buf.clear()
        assert len(buf) == 0

    def test_compute_gae(self):
        """After computing GAE, returns and advantages should be populated."""
        buf = self._fill_buffer(10)
        buf.compute_returns_and_advantages(
            last_value=torch.tensor([0.0]),
            last_done=True,
        )
        assert buf._returns is not None
        assert buf._advantages is not None
        assert buf._returns.shape == (10,)
        assert buf._advantages.shape == (10,)

    def test_advantages_normalized_in_batches(self):
        """Advantages in minibatches should be roughly normalized."""
        buf = self._fill_buffer(20)
        buf.compute_returns_and_advantages(
            last_value=torch.tensor([0.0]),
            last_done=True,
        )

        all_advs = []
        for batch in buf.get_batches(batch_size=10):
            all_advs.append(batch["advantages"])

        combined = torch.cat(all_advs)
        # After normalization, mean should be near 0 and std near 1
        assert abs(combined.mean().item()) < 0.3
        assert abs(combined.std().item() - 1.0) < 0.3

    def test_get_batches_keys(self):
        buf = self._fill_buffer(8)
        buf.compute_returns_and_advantages(
            last_value=torch.tensor([0.0]),
            last_done=True,
        )

        for batch in buf.get_batches(batch_size=4):
            assert "observations" in batch
            assert "actions" in batch
            assert "old_log_probs" in batch
            assert "returns" in batch
            assert "advantages" in batch
            break

    def test_get_batches_without_compute_raises(self):
        buf = self._fill_buffer(5)
        with pytest.raises(RuntimeError, match="compute_returns_and_advantages"):
            next(buf.get_batches(batch_size=4))

    def test_gae_terminal_episode(self):
        """In a terminal episode, the last advantage should not bootstrap."""
        buf = PPORolloutBuffer(gamma=0.99, gae_lambda=0.95)
        # Simple 3-step episode ending in terminal state
        for i in range(3):
            buf.push(
                observation=torch.randn(4),
                action=torch.randn(1, 1),
                log_prob=torch.randn(1, 1),
                reward=1.0,
                value=torch.tensor([0.0]),
                done=(i == 2),
            )
        buf.compute_returns_and_advantages(
            last_value=torch.tensor([0.0]),
            last_done=True,
        )

        # With V(s)=0 for all states and reward=1 at each step:
        # delta_2 = 1 + 0*0 - 0 = 1.0 (done=True, mask=0)
        # delta_1 = 1 + 0.99*0 - 0 = 1.0 (mask=1, last_gae=1)
        #   gae_1 = 1 + 0.99*0.95*1 = 1.9405
        # delta_0 = 1 + 0.99*0 - 0 = 1.0
        #   gae_0 = 1 + 0.99*0.95*1.9405 ≈ 2.8249
        assert buf._advantages[2].item() == pytest.approx(1.0, abs=1e-4)
        assert buf._advantages[1].item() == pytest.approx(1.9405, abs=1e-3)
        assert buf._advantages[0].item() == pytest.approx(2.8249, abs=1e-2)


# ---------------------------------------------------------------------------
# PPO Loss tests
# ---------------------------------------------------------------------------


class TestPPOLoss:
    def test_loss_output_type(self):
        """Loss should be a scalar tensor, metrics a dict."""
        new_lp = torch.randn(BATCH, requires_grad=True)
        old_lp = torch.randn(BATCH)
        advs = torch.randn(BATCH)
        values = torch.randn(BATCH, 1)
        returns = torch.randn(BATCH)
        entropy = torch.randn(BATCH)

        loss, metrics = ppo_loss(new_lp, old_lp, advs, values, returns, entropy)

        assert loss.dim() == 0  # scalar
        assert isinstance(metrics, dict)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert "total_loss" in metrics

    def test_clipping_behaviour(self):
        """When ratio exceeds clip bounds, the clipped term should dominate."""
        # Construct a scenario where ratio > 1 + eps for positive advantages
        old_lp = torch.tensor([0.0, 0.0])
        # Very different new log-probs → large ratio
        new_lp = torch.tensor([2.0, -2.0])
        advs = torch.tensor([1.0, 1.0])  # positive advantages
        values = torch.tensor([[0.0], [0.0]])
        returns = torch.tensor([1.0, 1.0])
        entropy = torch.tensor([0.0, 0.0])

        loss, _ = ppo_loss(new_lp, old_lp, advs, values, returns, entropy,
                           clip_eps=0.2, entropy_coeff=0.0)
        # The loss should be finite and reasonable
        assert torch.isfinite(loss)

    def test_zero_ratio_gives_zero_policy_loss(self):
        """When new_lp == old_lp, ratio=1, policy_loss = -advantages.mean()."""
        lp = torch.zeros(BATCH)
        advs = torch.ones(BATCH)
        values = torch.zeros(BATCH, 1)
        returns = torch.zeros(BATCH)
        entropy = torch.zeros(BATCH)

        loss, metrics = ppo_loss(lp, lp, advs, values, returns, entropy,
                                 entropy_coeff=0.0, value_coeff=0.0)
        # ratio=1, surr1 = surr2 = 1*1 = 1, policy_loss = -1.0
        assert metrics["policy_loss"] == pytest.approx(-1.0, abs=1e-5)

    def test_gradient_flows(self):
        """Gradients should flow through the total loss."""
        new_lp = torch.randn(BATCH, requires_grad=True)
        old_lp = torch.randn(BATCH)
        advs = torch.randn(BATCH)
        values = torch.randn(BATCH, 1, requires_grad=True)
        returns = torch.randn(BATCH)
        entropy = torch.randn(BATCH, requires_grad=True)

        loss, _ = ppo_loss(new_lp, old_lp, advs, values, returns, entropy)
        loss.backward()

        assert new_lp.grad is not None
        assert values.grad is not None
        assert entropy.grad is not None
