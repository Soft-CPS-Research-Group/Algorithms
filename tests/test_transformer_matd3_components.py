from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from algorithms.transformer_shared.entity_token_layout import (
    BuildingTokenLayout,
    NfcExpression,
    TokenSegment,
)


D_MODEL = 8


def _tokenizer_config() -> SimpleNamespace:
    return SimpleNamespace(
        nfc=SimpleNamespace(type_name="building_nfc"),
        sro_types={"weather": SimpleNamespace()},
        ca_types={"storage": SimpleNamespace()},
    )


def _layout(building_id: str) -> BuildingTokenLayout:
    return BuildingTokenLayout(
        building_id=building_id,
        segments=(
            TokenSegment("sro", "weather", None, (0, 1), ("a", "b")),
            TokenSegment(
                "nfc",
                "building_nfc",
                building_id,
                (2, 3),
                ("load", "solar"),
                NfcExpression("subtract", 0, 1),
            ),
            TokenSegment("ca", "storage", "battery", (4, 5), ("soc", "power")),
        ),
        n_sro=1,
        n_ca=1,
        ca_action_names=("electrical_storage",),
        excluded_feature_names=(),
    )


def _critic():
    from algorithms.transformer_matd3.components import CentralizedCritic

    return CentralizedCritic(
        d_model=D_MODEL,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        hidden_dim=12,
        dropout=0.0,
        tokenizer_config=_tokenizer_config(),
        type_input_dims={"weather": 2, "building_nfc": 1, "storage": 2},
    )


def test_should_return_one_raw_action_per_ca() -> None:
    from algorithms.transformer_matd3.components import DeterministicActorHead

    actor = DeterministicActorHead(d_model=D_MODEL, hidden_dim=12)
    embeddings = torch.randn(3, 4, D_MODEL)

    first = actor(embeddings)
    second = actor(embeddings, deterministic=True)

    assert first.shape == (3, 4, 1)
    assert torch.equal(first, second)


def test_should_inject_each_action_without_detaching_its_gradient() -> None:
    from algorithms.transformer_matd3.components import ActionInjectionMLP

    injector = ActionInjectionMLP(d_model=D_MODEL, hidden_dim=12)
    embeddings = torch.randn(2, 3, D_MODEL)
    actions = torch.randn(2, 3, requires_grad=True)

    output = injector(embeddings, actions)
    output.sum().backward()

    assert output.shape == (2, 3, D_MODEL)
    assert actions.grad is not None
    assert actions.grad.abs().sum() > 0


def test_should_reject_action_width_that_does_not_match_ca_count() -> None:
    from algorithms.transformer_matd3.components import ActionInjectionMLP

    injector = ActionInjectionMLP(d_model=D_MODEL, hidden_dim=12)

    with pytest.raises(ValueError, match="action shape"):
        injector(torch.randn(2, 3, D_MODEL), torch.randn(2, 2))


def test_should_return_one_q_value_per_batch_item() -> None:
    critic = _critic()
    observations = [torch.randn(4, 6), torch.randn(4, 6)]
    layouts = [_layout("building-1"), _layout("building-2")]
    actions = [torch.randn(4, 1), torch.randn(4, 1)]

    q_values = critic(observations, layouts, actions)

    assert q_values.shape == (4, 1)


def test_should_propagate_critic_gradient_to_joint_actions() -> None:
    critic = _critic()
    observations = [torch.randn(2, 6), torch.randn(2, 6)]
    layouts = [_layout("building-1"), _layout("building-2")]
    actions = [
        torch.randn(2, 1, requires_grad=True),
        torch.randn(2, 1, requires_grad=True),
    ]

    critic(observations, layouts, actions).sum().backward()

    assert all(action.grad is not None for action in actions)
    assert all(action.grad.abs().sum() > 0 for action in actions)


def test_should_be_permutation_invariant_across_buildings() -> None:
    critic = _critic()
    critic.eval()
    observations = [torch.randn(3, 6), torch.randn(3, 6)]
    layouts = [_layout("building-1"), _layout("building-2")]
    actions = [torch.randn(3, 1), torch.randn(3, 1)]

    original = critic(observations, layouts, actions)
    permuted = critic(
        list(reversed(observations)),
        list(reversed(layouts)),
        list(reversed(actions)),
    )

    assert torch.allclose(original, permuted, atol=1e-6)


def test_should_isolate_gradients_between_independent_critics() -> None:
    critic_1 = _critic()
    critic_2 = _critic()
    observations = [torch.randn(2, 6), torch.randn(2, 6)]
    layouts = [_layout("building-1"), _layout("building-2")]
    actions = [torch.randn(2, 1), torch.randn(2, 1)]

    critic_1(observations, layouts, actions).sum().backward()

    assert any(parameter.grad is not None for parameter in critic_1.parameters())
    assert all(parameter.grad is None for parameter in critic_2.parameters())


def test_should_initialize_twin_critics_independently() -> None:
    torch.manual_seed(7)
    critic_1 = _critic()
    critic_2 = _critic()
    observations = [torch.randn(2, 6)]
    layouts = [_layout("building-1")]
    actions = [torch.randn(2, 1)]

    first_q = critic_1(observations, layouts, actions)
    second_q = critic_2(observations, layouts, actions)

    assert not torch.equal(first_q, second_q)


def test_should_reject_mixed_community_batch_sizes() -> None:
    critic = _critic()

    with pytest.raises(ValueError, match="batch size"):
        critic(
            [torch.randn(2, 6), torch.randn(3, 6)],
            [_layout("building-1"), _layout("building-2")],
            [torch.randn(2, 1), torch.randn(3, 1)],
        )
