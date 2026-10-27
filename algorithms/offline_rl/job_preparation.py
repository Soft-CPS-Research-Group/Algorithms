"""Offline optimization as an optional phase of the shared job lifecycle.

The runner owns directories, simulation, evaluation, KPIs and manifests. This
mixin only trains the agent's policy from its explicit transition dataset.
"""
from copy import deepcopy
from pathlib import Path


class OfflineTrainingPreparation:
    requires_final_pipeline_stage = True

    def _defer_offline_training(self, config):
        hyper = (config or {}).get('algorithm', {}).get('hyperparameters', {})
        if hyper.get('offline_training') is None:
            return False
        if hyper.get('model_dir'):
            raise ValueError('Choose model_dir or offline_training, not both')
        self._offline_job_config = deepcopy(config)
        return True

    def prepare_training(self, output_dir, context=None):
        config = getattr(self, '_offline_job_config', None)
        if config is None:
            return
        from algorithms.offline_rl.bc_trainer import BCTrainingConfig, train_single_seed as train_bc
        from algorithms.offline_rl.iql_trainer import IQLTrainingConfig, train_single_seed as train_iql
        from algorithms.offline_rl.iql_entity_trainer import train_entity_single_seed
        from algorithms.offline_rl.cql_entity_trainer import CQLTrainingConfig, train_cql_single_seed

        hyper = config['algorithm']['hyperparameters']
        preparation = hyper['offline_training']
        source = Path(preparation['dataset_path']).expanduser().resolve()
        output = Path(output_dir).resolve()
        seed = int(config.get('training', {}).get('seed', 22))
        params = dict(preparation.get('parameters') or {})
        params.setdefault('device', hyper.get('device', 'cpu'))
        name = type(self).__name__
        if name in {'BCAgent', 'IQLAgent'}:
            trainer, settings = (train_bc, BCTrainingConfig) if name == 'BCAgent' else (train_iql, IQLTrainingConfig)
            if not source.is_file():
                raise ValueError('BC/IQL offline_training.dataset_path must be a transition parquet file')
            trainer(source, output, seed=seed, config=settings(**params))
        else:
            if not source.is_dir():
                raise ValueError('Entity offline_training.dataset_path must be a directory of transition parquets')
            groups = preparation.get('groups')
            if not groups:
                topology = config.get('topology', {})
                groups = [{'obs_dim': obs, 'action_dim': act} for obs, act in sorted(set(zip(
                    topology.get('observation_dimensions', []), topology.get('action_dimensions', [])
                ))) if act > 0]
            if not groups:
                raise ValueError('Entity offline training needs groups or an attached job topology')
            trainer, settings = (train_cql_single_seed, CQLTrainingConfig) if name == 'CQLEntityAgent' else (train_entity_single_seed, IQLTrainingConfig)
            for group in groups:
                obs, act = int(group['obs_dim']), int(group['action_dim'])
                trainer(source, output / f'obs{obs}_act{act}', obs_dim=obs, action_dim=act,
                        seed=seed, config=settings(**params), val_seeds=preparation.get('val_seeds'))
        # Construct the same public agent class with the newly trained policy.
        prepared_config = deepcopy(config)
        prepared_config['algorithm']['hyperparameters'] = {
            'model_dir': str(output), 'device': hyper.get('device', 'cpu')
        }
        self._offline_job_config = None
        was_frozen = self.frozen
        self.__init__(config=prepared_config)
        self.frozen = was_frozen
        self.prepared_model_dir = str(output)
