import copy
import hashlib
import json

import pytest

from utils.artifact_manifest import build_manifest
from utils.training_provenance import seal_manifest


@pytest.fixture
def inputs():
    config = {'metadata': {'experiment_name': 'houses', 'run_name': 'test'},
              'simulator': {'central_agent': False}, 'topology': {'num_agents': 2},
              'tracking': {'tags': {'energaize_training': {
                  'workspace_id': 'community:rec', 'community_id': 'rec', 'job_id': 'job',
                  'datasets': [{'resource_id': 'dataset', 'version': 3, 'manifest_digest': 'snapshot',
                                'mapping': [{'building_id': 'b1', 'site_id': 'site-1'}, {'building_id': 'b2', 'site_id': 'site-2'}]}]}}}}
    environment = {'building_names': ['b2', 'b1'], 'topology_mode': 'dynamic', 'topology_version': 7}
    agent = {'format': 'onnx', 'artifacts': [{'agent_index': i, 'path': f'agent_{i}.onnx'} for i in range(2)]}
    return config, environment, agent


def test_site_ids_follow_runtime_order_and_survive_labels_and_reordering(inputs):
    config, env, agent = inputs
    initial = build_manifest(config, env, agent)['provenance']
    assert initial['mapping_status'] == 'verified'
    assert [b['site_ids'] for b in initial['bindings']] == [['site-2'], ['site-1']]
    env['building_names'].reverse()
    config['metadata']['community_name'] = 'A new display name'
    reordered = build_manifest(config, env, agent)['provenance']
    assert initial['bindings'][0]['agent_id'] == reordered['bindings'][1]['agent_id']
    assert reordered['topology']['version'] == 7


def test_central_model_preserves_all_sites_and_actual_entity_topology(inputs):
    config, env, agent = inputs
    config['simulator']['central_agent'] = True
    config['topology']['num_agents'] = 1
    agent['artifacts'] = agent['artifacts'][:1]
    env['building_names'] = ['misleading display labels']
    env['entity_specs'] = {'tables': {'building': {'ids': ['b2', 'b1']}}}
    provenance = build_manifest(config, env, agent)['provenance']
    assert provenance['mapping_status'] == 'verified'
    assert provenance['bindings'][0]['scope'] == 'community'
    assert provenance['bindings'][0]['site_ids'] == ['site-2', 'site-1']
    assert provenance['topology']['entity_specs_digest']


@pytest.mark.parametrize('change', ['unknown_building', 'duplicate_mapping', 'bad_index', 'missing_mapping'])
def test_ambiguous_identity_never_claims_verified_mapping(inputs, change):
    config, env, agent = inputs
    dataset = config['tracking']['tags']['energaize_training']['datasets'][0]
    if change == 'unknown_building': env['building_names'][0] = 'new-house'
    if change == 'duplicate_mapping': dataset['mapping'].append({'building_id': 'b1', 'site_id': 'foreign'})
    if change == 'bad_index': agent['artifacts'][0]['agent_index'] = 5
    if change == 'missing_mapping': dataset['mapping'] = []
    provenance = build_manifest(config, env, agent)['provenance']
    assert provenance['mapping_status'] == 'unmapped'
    assert provenance['mapping_issues']


def test_pipeline_coordinator_and_house_models_have_distinct_agent_ids(inputs):
    config, env, agent = inputs
    pipeline = {'format': 'pipeline', 'stages': [
        {'stage_index': 0, 'format': 'onnx', 'artifacts': [agent['artifacts'][0]]},
        {'stage_index': 1, **agent}]}
    provenance = build_manifest(config, env, pipeline)['provenance']
    assert provenance['mapping_status'] == 'verified'
    assert [b['scope'] for b in provenance['bindings']] == ['community', 'site', 'site']
    assert len({b['agent_id'] for b in provenance['bindings']}) == 3


def test_seal_pins_exact_config_and_exported_files(inputs, tmp_path):
    config, env, agent = inputs
    root = tmp_path / 'bundle'; root.mkdir()
    for artifact in agent['artifacts']: (root / artifact['path']).write_bytes(b'model')
    saved = tmp_path / 'config.resolved.yaml'; saved.write_text('simulator: {}\n')
    manifest = build_manifest(config, env, agent)
    seal_manifest(manifest, root, saved, 'job')
    assert manifest['provenance']['resolved_config_sha256'] == hashlib.sha256(saved.read_bytes()).hexdigest()
    assert manifest['files']['agent_0.onnx'] == hashlib.sha256(b'model').hexdigest()
    assert manifest['provenance']['identity_sha256']
    with pytest.raises(ValueError, match='another job'): seal_manifest(manifest, root, saved, 'other')


def test_legacy_lab_remains_explicitly_unmapped(inputs):
    config, env, agent = inputs
    config.pop('tracking')
    provenance = build_manifest(config, env, agent)['provenance']
    assert provenance['workspace_id'] == 'laboratory'
    assert provenance['mapping_status'] == 'unmapped'
    assert all(b['agent_id'] is None for b in provenance['bindings'])


def test_v2_nodes_keep_identity_when_local_agent_indices_repeat(inputs):
    config, env, agent = inputs
    agent['artifacts'] = [
        {'agent_index': 0, 'node_id': 'stage_0', 'path': 'manager.onnx', 'output_signal': 'price'},
        {'agent_index': 0, 'node_id': 'stage_1/agent_0', 'member_index': 0, 'path': 'one.onnx'},
        {'agent_index': 0, 'node_id': 'stage_2/agent_0', 'member_index': 0, 'path': 'two.onnx'},
    ]
    provenance = build_manifest(config, env, agent)['provenance']
    assert provenance['mapping_status'] == 'verified'
    bindings = provenance['bindings']
    assert [b['site_ids'] for b in bindings] == [['site-2', 'site-1'], ['site-2'], ['site-2']]
    assert len({b['agent_id'] for b in bindings}) == 3
