"""Shared inference preserves asset bindings and passive members on export."""

import json

import numpy as np
import pytest
import torch

from algorithms.offline_rl.bc_dataset import ObservationStandardiser
from algorithms.offline_rl.entity_layout import canonicalize_entity_names
from algorithms.offline_rl.iql_entity_agent import IQLEntityAgent
from algorithms.offline_rl.iql_networks import GaussianPolicy


def _shared_agent(tmp_path):
    observations = ["charger::A/CHG-A::soc", "charger::A/CHG-B::soc"]
    actions = ["electric_vehicle_storage_CHG-A", "electric_vehicle_storage_CHG-B"]
    canonical_obs, canonical_actions = canonicalize_entity_names(observations, actions)
    torch.manual_seed(123)
    policy = GaussianPolicy(2, 2, [8], dropout=0.0)
    torch.save(policy.state_dict(), tmp_path / "policy.pt")
    arch = policy.architecture_summary()
    arch.update(obs_dim=2, action_dim=2, obs_names=canonical_obs, action_names=canonical_actions)
    (tmp_path / "architecture.json").write_text(json.dumps(arch))
    ObservationStandardiser(np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32), canonical_obs).save(
        tmp_path / "obs_standardiser.npz"
    )
    return IQLEntityAgent.from_model_dir(tmp_path), observations, actions


def test_shared_policy_binding_and_passive_export(tmp_path):
    agent, observations, actions = _shared_agent(tmp_path)
    other_obs = ["charger::B/CHG-D::soc", "charger::B/CHG-C::soc"]
    other_actions = ["electric_vehicle_storage_CHG-D", "electric_vehicle_storage_CHG-C"]
    # Passive member intentionally has the same observation dimension as the actor.
    agent.attach_environment(
        observation_names=[["passive_0", "passive_1"], observations, other_obs],
        action_names=[[], actions, other_actions], action_space=[], observation_space=[],
    )
    states = [np.array([0.2, 0.3]), np.array([0.1, 0.9]), np.array([0.9, 0.1])]
    predicted = agent.predict(states)
    assert predicted[0] == []
    np.testing.assert_allclose(predicted[1], predicted[2][::-1])
    exported = agent.export_artifacts(str(tmp_path / "export"))
    assert [a["agent_index"] for a in exported["artifacts"]] == [0, 1, 2]
    assert exported["artifacts"][0]["config"]["auxiliary_paths"] == []
    ort = pytest.importorskip("onnxruntime")
    for index, artifact in enumerate(exported["artifacts"]):
        session = ort.InferenceSession(str(tmp_path / "export" / artifact["path"]), providers=["CPUExecutionProvider"])
        actual = session.run(None, {"observations": states[index].astype(np.float32)[None]})[0]
        np.testing.assert_allclose(actual, np.asarray(predicted[index])[None], rtol=1e-5, atol=1e-6)


def test_matching_dimensions_do_not_override_feature_semantics(tmp_path):
    agent, observations, actions = _shared_agent(tmp_path)
    observations[0] = observations[0].replace("::soc", "::capacity")
    with pytest.raises(ValueError, match="observation names differ"):
        agent.attach_environment(observation_names=[observations], action_names=[actions], action_space=[], observation_space=[])
