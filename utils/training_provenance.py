"""Versioned links between a frozen dataset, runtime topology and model artifacts.

Human names are never identity. Only exact simulator building keys from the
dataset snapshot may be joined to platform site IDs. Legacy runs remain usable
but cannot claim a verified community mapping.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def build_provenance(config, environment, agent):
    source = deepcopy(((config.get('tracking') or {}).get('tags') or {}).get('energaize_training') or {})
    source.pop('configuration', None)
    source.setdefault('workspace_id', 'laboratory')
    source.setdefault('community_id', None)
    source.setdefault('datasets', [])
    problems = []
    # Entity IDs are authoritative when available; flat runtimes expose the
    # schema keys in building_names, never display labels from the REC Manager.
    specs = environment.get('entity_specs') or {}
    buildings = ((specs.get('tables') or {}).get('building') or {}).get('ids')
    if buildings is None:
        buildings = environment.get('building_names') or []
    buildings = [str(value) for value in buildings]
    if not buildings or len(set(buildings)) != len(buildings):
        problems.append('Runtime building identities are missing or duplicated')
    sites = {}
    for dataset in source['datasets']:
        for entry in dataset.get('mapping') or []:
            building, site = entry.get('building_id'), entry.get('site_id')
            if not building or not site:
                problems.append('Dataset contains an incomplete site mapping')
                continue
            if building in sites:
                problems.append('Multiple dataset mappings for building ' + str(building))
            sites[building] = {**entry, 'dataset_resource_id': dataset['resource_id'], 'dataset_version': dataset['version']}
    if any(building not in sites for building in buildings):
        problems.append('Runtime buildings have no frozen platform site mapping')
    selected_sites = [sites[building] for building in buildings if building in sites]
    if len({entry['site_id'] for entry in selected_sites}) != len(selected_sites):
        problems.append('Multiple runtime buildings map to the same site')
    count = (config.get('topology') or {}).get('num_agents')
    central = bool((config.get('simulator') or {}).get('central_agent'))
    artifacts = agent.get('artifacts') or []
    groups = {}
    for artifact in artifacts:
        groups.setdefault(artifact.get('stage_index'), []).append(artifact)
    last_stage = max((k for k in groups if k is not None), default=None)
    bindings = []
    for artifact in artifacts:
        stage = artifact.get('stage_index')
        group = groups[stage]
        index = artifact.get('agent_index')
        indices = [item.get('agent_index') for item in group]
        shared = artifact.get('format', agent.get('format')) == 'ti_marl_torch'
        coordinator = stage is not None and stage != last_stage and len(group) == 1 and index == 0
        if buildings and len(group) == 1 and index == 0 and (central and count == 1 or shared or coordinator):
            controlled = buildings
            scope = 'community'
        elif not central and count == len(buildings) and all(isinstance(i, int) for i in indices) and sorted(indices) == list(range(len(buildings))) and isinstance(index, int):
            controlled = [buildings[index]]
            scope = 'site'
        else:
            controlled, scope = [], 'unmapped'
            problems.append('Artifact has no unambiguous runtime agent binding: ' + str(artifact.get('path')))
        site_ids = [sites[b]['site_id'] for b in controlled if b in sites]
        verified = bool(controlled) and len(site_ids) == len(controlled)
        bindings.append({'artifact_path': artifact['path'], 'agent_index': index,
                         'stage_index': stage, 'scope': scope, 'building_ids': controlled,
                         'site_ids': site_ids if verified else [],
                         'agent_id': 'agent_' + digest([source['workspace_id'], stage, scope, sorted(site_ids)])[:24] if verified else None})
    provenance = {**source, 'version': 1, 'mapping_status': 'verified' if not problems and source['community_id'] else 'unmapped',
                  'mapping_issues': sorted(set(problems)), 'sites': selected_sites, 'bindings': bindings,
                  'topology': {'num_agents': count, 'central_agent': central, 'building_ids': buildings,
                               'mode': environment.get('topology_mode', 'static'),
                               'version': environment.get('topology_version'),
                               'entity_specs_digest': digest(specs) if specs else None}}
    return provenance


def seal_manifest(manifest, artifacts_root, resolved_config_path, job_id):
    """Hash the exact saved runtime config and every exported bundle file."""
    provenance = manifest['provenance']
    if provenance.get('job_id') not in (None, job_id):
        raise ValueError('Training provenance belongs to another job')
    provenance['job_id'] = job_id
    provenance['resolved_config_sha256'] = file_digest(resolved_config_path)
    root = Path(artifacts_root).resolve()
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name != 'artifact_manifest.json':
            if not path.resolve().is_relative_to(root):
                raise ValueError('Bundle file escapes its artifact directory')
            files[path.relative_to(root).as_posix()] = file_digest(path)
    manifest['files'] = files
    provenance['identity_sha256'] = digest({k: v for k, v in provenance.items() if k != 'identity_sha256'})
