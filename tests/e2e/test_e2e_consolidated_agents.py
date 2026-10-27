"""Real jobs for the recovered flat Transformer and hierarchical coordinators."""
import json
from pathlib import Path
import pytest
from run_experiment import run_experiment
from utils.bundle_validator import validate_bundle_contract

@pytest.mark.parametrize('recipe', ['transformer_maddpg_smoke', 'cc_capacity_smoke', 'cc_directional_smoke'])
def test_recovered_agent_job_exports_final_results_and_model(recipe, tmp_path):
    run_experiment(f'configs/templates/experimental/{recipe}.yaml', recipe, tmp_path)
    manifests = list(tmp_path.rglob('artifact_manifest.json'))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    validate_bundle_contract(manifest, manifests[0].parent)
    results = list(tmp_path.rglob('result.json'))
    assert results
    result = json.loads(results[0].read_text())
    assert result['status'] == 'completed'
    assert result['kpi_source'] == 'simulator_export'
    episodes = result['episode_execution']
    assert episodes[-1]['deterministic'] is True
    assert episodes[-1]['export_enabled'] is True
    assert not any(episode['export_enabled'] for episode in episodes[:-1])
    kpis = list(Path(result['simulation_data_dir']).rglob('exported_kpis.csv'))
    assert kpis and kpis[0].stat().st_size > 0
    assert manifest['agent']['artifacts']

@pytest.mark.parametrize('name', ['IQLEntityAgent', 'CQLEntityAgent'])
def test_offline_training_runs_inside_the_common_job(name, tmp_path):
    import numpy as np
    import pandas as pd
    import yaml
    from citylearn.citylearn import CityLearnEnv
    from run_experiment import _resolve_citylearn_schema_input
    from utils.entity_adapter import EntityContractAdapter
    config = yaml.safe_load(Path('configs/templates/experimental/transformer_maddpg_smoke.yaml').read_text())
    config['simulator']['building_ids'] = ['Building_2']
    config['simulator']['episodes'] = 1
    env = CityLearnEnv(schema=_resolve_citylearn_schema_input(config['simulator']['dataset_path'], config['simulator']),
                      central_agent=False, interface='entity', episode_time_steps=2)
    try:
        payload, _ = env.reset()
        adapter = EntityContractAdapter(env, normalization_enabled=True, clip=True)
        _, names, _ = adapter.to_agent_encoded_observations(payload)
        actions = env.action_names[0]
    finally:
        env.close()
    data = tmp_path / 'transitions'
    data.mkdir()
    for seed in (22, 23):
        rng = np.random.default_rng(seed)
        frame = pd.DataFrame({
            'seed': [seed] * 24, 'episode': [0] * 24, 'timestep': range(24),
            'agent_idx': [0] * 24, 'obs_dim': [len(names[0])] * 24, 'action_dim': [len(actions)] * 24,
            'reward': [-0.1] * 24, 'terminated': [0] * 23 + [1], 'truncated': [0] * 24,
            **{f'obs__{feature}': rng.random(24) for feature in names[0]},
            **{f'next_obs__{feature}': rng.random(24) for feature in names[0]},
            **{f'action__{feature}': rng.random(24) for feature in actions},
        })
        frame.to_parquet(data / f'seed_{seed}.parquet')
    config['pipeline'] = [{'algorithm': name, 'hyperparameters': {'offline_training': {
        'dataset_path': str(data), 'parameters': {'hidden_layers': [8], 'gradient_steps': 2,
                                                'eval_every_n_steps': 1, 'batch_size': 4}}}}]
    source = tmp_path / 'config.yaml'
    source.write_text(yaml.safe_dump(config))
    run_experiment(str(source), name, tmp_path / 'jobs-root')
    job = tmp_path / 'jobs-root/jobs' / name
    assert list((job / 'offline_training').rglob('policy.pt'))
    manifest_path = next(job.rglob('artifact_manifest.json'))
    manifest = json.loads(manifest_path.read_text())
    validate_bundle_contract(manifest, manifest_path.parent)
    assert manifest['agent']['deployable'] is False
    result = json.loads(next(job.rglob('result.json')).read_text())
    assert result['status'] == 'completed'
    assert result['kpi_source'] == 'simulator_export'
    episodes = result['episode_execution']
    assert episodes[-1]['deterministic'] is True
    assert episodes[-1]['export_enabled'] is True
    assert not any(episode['export_enabled'] for episode in episodes[:-1])
    kpis = list(Path(result['simulation_data_dir']).rglob('exported_kpis.csv'))
    assert kpis and kpis[0].stat().st_size > 0
