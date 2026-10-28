"""Regression gates for implementations recovered during branch consolidation."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from gymnasium.spaces import Box

from algorithms.registry import build_execution_unit
from utils.bundle_validator import BundleValidationError, validate_bundle_contract
from tests.test_bundle_validator import _base_manifest


def _check_bundle(agent, root, num_agents):
    metadata = agent.export_artifacts(str(root))
    manifest = _base_manifest()
    manifest['topology']['num_agents'] = num_agents
    manifest['agent'] = metadata
    if not any(item.get("config", {}).get("scope") == "joint" for item in metadata["artifacts"]):
        validate_bundle_contract(manifest, root)
    if metadata["format"] in {"research_torch", "ti_marl_torch"}:
        assert metadata['deployable'] is False
        assert all(item['deployable'] is False for item in metadata['artifacts'])
    else:
        from algorithms.export.verification import onnx_session
        for item in metadata["artifacts"]:
            onnx_session(root / item["path"])
    return metadata


def _build(name, model_dir):
    from utils.config_schema import OfflineAgentStageConfig
    stage = OfflineAgentStageConfig(algorithm=name, hyperparameters={'model_dir': str(model_dir)})
    return build_execution_unit({'pipeline': [stage.model_dump()]})


def _prepare(name, source, output, parameters, groups=None):
    from utils.config_schema import OfflineAgentStageConfig
    preparation = {'dataset_path': str(source), 'parameters': parameters}
    if groups:
        preparation['groups'] = groups
    stage = OfflineAgentStageConfig(algorithm=name, hyperparameters={'offline_training': preparation})
    agent = build_execution_unit({'pipeline': [stage.model_dump()], 'training': {'seed': 17}})
    agent.prepare_training(str(output))
    weights = next(output.rglob('policy.pt'))
    modified = weights.stat().st_mtime_ns
    agent.prepare_training(str(output))
    assert weights.stat().st_mtime_ns == modified
    return agent


@pytest.mark.parametrize('name', ['BCAgent', 'IQLAgent'])
def test_flat_offline_train_registry_export_reload(name, tmp_path):
    from algorithms.offline_rl import schema as schema
    from algorithms.offline_rl.bc_trainer import BCTrainingConfig, train_single_seed as train_bc
    from algorithms.offline_rl.iql_trainer import IQLTrainingConfig, train_single_seed as train_iql
    rng = np.random.default_rng(13)
    columns = [*schema.OBS_COLUMNS, *schema.NEXT_OBS_COLUMNS, *schema.ACTION_COLUMNS]
    frame = pd.DataFrame({key: rng.random(40) for key in columns})
    frame['reward'] = -0.1
    frame['terminated'] = 0
    frame['truncated'] = 0
    source = tmp_path / 'data.parquet'
    frame.to_parquet(source)
    model = tmp_path / 'trained'
    parameters = {'hidden_layers': [8], 'batch_size': 8}
    parameters.update({'epochs': 1} if name == 'BCAgent' else {'gradient_steps': 4, 'eval_every_n_steps': 2})
    agent = _prepare(name, source, model, parameters)
    count = 5  # The recovered flat policies explicitly target Building_5.
    obs_names = [list(schema.OBSERVATION_NAMES) for _ in range(count)]
    act_names = [list(schema.ACTION_NAMES) for _ in range(count)]
    attachment = dict(observation_names=obs_names, action_names=act_names,
                      observation_space=[Box(0, 1, (len(obs_names[0]),))] * count,
                      action_space=[Box(-1, 1, (len(act_names[0]),))] * count, metadata={})
    agent.attach_environment(**attachment)
    observations = [np.zeros(len(obs_names[0])) for _ in range(count)]
    before = agent.predict(observations, True, context=None)
    export = tmp_path / 'export'
    _check_bundle(agent, export, count)
    reloaded = _build(name, export)
    reloaded.attach_environment(**attachment)
    np.testing.assert_allclose(reloaded.predict(observations, True), before)

    from algorithms.export.bundle import export_tree
    common = export_tree(agent, tmp_path / 'common-bundle', {'config': {}})
    assert common['deployment']['eligible'] is False
    assert common['format'] == 'research_torch'


@pytest.mark.parametrize('name', ['IQLEntityAgent', 'CQLEntityAgent'])
def test_entity_offline_train_registry_feature_order_export_reload(name, tmp_path):
    from tests.offline_rl.test_iql_entity_resume import _make_synthetic_df
    from algorithms.offline_rl.iql_trainer import IQLTrainingConfig
    from algorithms.offline_rl.iql_entity_trainer import train_entity_single_seed
    from algorithms.offline_rl.cql_entity_trainer import CQLTrainingConfig, train_cql_single_seed
    data = tmp_path / 'data'
    data.mkdir()
    for seed in (22, 23):
        _make_synthetic_df(n_steps=30, seed=seed, groups=[(4, 2)]).to_parquet(data / f'seed_{seed}.parquet')
    trainer, config_cls = (train_entity_single_seed, IQLTrainingConfig) if name == 'IQLEntityAgent' else (train_cql_single_seed, CQLTrainingConfig)
    model = tmp_path / 'model'
    agent = _prepare(name, data, model, {'hidden_layers': [8], 'gradient_steps': 4,
        'eval_every_n_steps': 2, 'batch_size': 8}, groups=[{'obs_dim': 4, 'action_dim': 2}])
    obs = [np.array([0.1, 0.2, 0.4, 0.8], dtype=np.float32)]
    reference = agent.predict(obs, True)
    attachment = dict(observation_names=[[f'obs_feat_{i}' for i in reversed(range(4))]],
                      action_names=[['act_1', 'act_0']], observation_space=[Box(-1, 1, (4,))],
                      action_space=[Box(-1, 1, (2,))], metadata={})
    agent.attach_environment(**attachment)
    reordered = agent.predict([obs[0][::-1]], True)
    np.testing.assert_allclose(reordered, [reference[0][::-1]])
    export = tmp_path / 'export'
    metadata = _check_bundle(agent, export, 1)
    restored = _build(name, export)
    restored.attach_environment(**attachment)
    np.testing.assert_allclose(restored.predict([obs[0][::-1]], True), reordered)
    # Sidecars are required for restoration, not merely optional files.
    sidecar = export / metadata['artifacts'][0]['config']['auxiliary_paths'][0]
    sidecar.unlink()
    manifest = _base_manifest()
    manifest['agent'] = metadata
    with pytest.raises(BundleValidationError, match='sidecar|auxiliary_paths'):
        validate_bundle_contract(manifest, export)
    with pytest.raises(ValueError, match='action dimension'):
        restored.attach_environment(**{**attachment, 'action_names': [['act_0']]})


def test_transformer_registry_predict_export_reload_and_heterogeneous_guard(tmp_path):
    from utils.config_schema import TransformerMADDPGAlgorithmConfig
    from tests.test_transformer_maddpg import make_observation_names
    stage = TransformerMADDPGAlgorithmConfig(algorithm='TransformerMADDPG',
        networks={'transformer': {'d_model': 8, 'nhead': 2, 'num_layers': 1, 'dim_feedforward': 16}},
        replay_buffer={'capacity': 16, 'batch_size': 2})
    config = {'pipeline': [stage.model_dump(exclude_none=True)]}
    agent = build_execution_unit(config)
    names = make_observation_names(2)
    attachment = dict(observation_names=names, action_names=[['storage']] * 2,
                      observation_space=[Box(-1, 1, (9,))] * 2,
                      action_space=[Box(-1, 1, (1,))] * 2, metadata={})
    agent.attach_environment(**attachment)
    observations = [np.zeros(9), np.ones(9)]
    expected = agent.predict(observations, True, context=None)
    metadata = _check_bundle(agent, tmp_path, 2)
    restored = build_execution_unit(config)
    restored.attach_environment(**attachment)
    restored.load_checkpoint(str(tmp_path / metadata['checkpoint_path']))
    np.testing.assert_allclose(restored.predict(observations, True), expected)
    with pytest.raises(ValueError, match='equal, nonzero'):
        restored.attach_environment(**{**attachment, 'action_space': [Box(-1, 1, (1,)), Box(-1, 1, (2,))]})


def test_capacity_variant_uses_metadata_and_checkpoint_roundtrip(tmp_path):
    from algorithms.agents.cc_level2_capacity_agent import _CC_LEVEL2_DISTRICT_FEATURES
    config = {'pipeline': [{'algorithm': 'CCLevel2Capacity', 'hyperparameters': {
        'num_buildings': 2, 'hidden_dims': [8, 8], 'num_steps': 4, 'mini_batch_size': 2,
        'bc_pretrain_enabled': False, 'cc_action_interval': 1, 'cc_capacity_reference_kwh': 100.0}}]}
    agent = build_execution_unit(config)
    names = [list(_CC_LEVEL2_DISTRICT_FEATURES) + [f'storage::B{i}/battery::soc'] for i in range(2)]
    attachment = dict(observation_names=names, action_names=[[], []], observation_space=[None] * 2,
                      action_space=[None] * 2, metadata={'building_battery_capacity_kwh': [10, 40]})
    agent.attach_environment(**attachment)
    observations = [np.ones(len(n)) * 0.5 for n in names]
    context = agent._build_context(observations)
    assert context[17 + 6] == pytest.approx(0.1)
    assert context[17 + 7 + 6] == pytest.approx(0.4)
    expected = agent.predict(observations, True)
    metadata = _check_bundle(agent, tmp_path, 2)
    restored = build_execution_unit(config)
    restored.attach_environment(**attachment)
    restored.load_checkpoint(str(tmp_path / metadata['artifacts'][0]['path']))
    np.testing.assert_allclose(restored.predict(observations, True), expected)


def test_hierarchical_price_coupling_is_opt_in_and_does_not_mutate_observations():
    from algorithms.agents.maddpg_agent import MADDPG
    from algorithms.agents.baseline_policies import SignalAwareRBCCommunity
    agent = MADDPG.__new__(MADDPG)
    torch.nn.Module.__init__(agent)
    agent.actor_community_observation_masking = True
    agent._masked_obs_indices = [[1]]
    agent._price_obs_indices = [0]
    agent.signal_aware_actor_obs_gain = 0.5
    observations = [np.array([0.2, 0.8, 0.3])]
    masked = agent._apply_actor_observation_mask(observations)
    adjusted = agent._apply_manager_price_to_actor_obs(masked, 1.2)
    np.testing.assert_allclose(adjusted, [[0.3, 0.0, 0.3]])
    np.testing.assert_allclose(observations, [[0.2, 0.8, 0.3]])
    base = SignalAwareRBCCommunity({'algorithm': {'hyperparameters': {}}})
    agent._warm_start_policy = base
    agent._apply_manager_price_signal([0.8, 1.2])
    base._current_agent_idx = 1
    assert base._effective_price_multiplier() == pytest.approx(1.2)
    agent._apply_manager_price_signal(None)
    assert base._effective_price_multiplier() == 1.0


def test_pipeline_export_keeps_research_sidecars_and_non_deployability(tmp_path):
    from utils.artifact_manifest import _normalize_agent_metadata
    root = tmp_path / 'stage_0'
    root.mkdir()
    (root / 'weights.pt').write_bytes(b'weights')
    (root / 'architecture.json').write_text('{}')
    metadata = _normalize_agent_metadata({'format': 'pipeline', 'stages': [{
        'format': 'research_torch', 'deployable': False,
        'artifacts': [{'agent_index': 0, 'path': 'weights.pt', 'format': 'research_torch',
                       'deployable': False, 'config': {'loader': 'test', 'auxiliary_paths': ['architecture.json']}}]
    }]})
    manifest = _base_manifest()
    manifest['agent'] = metadata
    validate_bundle_contract(manifest, tmp_path)
    assert metadata['deployable'] is False
    (root / 'architecture.json').unlink()
    with pytest.raises(BundleValidationError, match='sidecar|auxiliary_paths'):
        validate_bundle_contract(manifest, tmp_path)
