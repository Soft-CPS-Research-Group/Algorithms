"""Reproduce the small isolated CPU tracking comparison (requires the tracking extra).

Run from the repository root. File MLflow is forced: no production service is used.
The same real-job fixture used in CI supplies data, networks and training settings.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--worker-config', type=Path)
    parser.add_argument('--worker-name')
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ['MLFLOW_TRACKING_URI'] = f'file:{root}/mlruns'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    if args.worker_config:
        from algorithms.runtime.job import run_experiment
        cfg = json.loads(args.worker_config.read_text())
        start, cpu = time.perf_counter(), time.process_time()
        run_experiment(cfg, args.worker_name, root)
        job = root / 'jobs' / args.worker_name
        times = {}
        metrics = job / 'logs/metrics.jsonl'
        if metrics.exists():
            for line in metrics.read_text().splitlines():
                for key, value in json.loads(line)['metrics'].items():
                    if key.startswith('Runtime/'):
                        times[key] = times.get(key, 0) + value
        report = {'mode': args.worker_name, 'wall_seconds': time.perf_counter()-start,
                  'cpu_seconds': time.process_time()-cpu,
                  'peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  'gpu': 'not used', 'job_bytes': sum(p.stat().st_size for p in job.rglob('*') if p.is_file()),
                  'tracking': json.loads((job/'results/tracking.json').read_text()),
                  'finalization': json.loads((job/'results/finalization.json').read_text()),
                  'sampled_runtime_totals': times}
        (root/f'{args.worker_name}.metrics.json').write_text(json.dumps(report, indent=2))
        return
    from tests.e2e.test_common_job_profiles import config_for
    base = config_for('MADDPG', root)
    base['job'] = {'profile': 'train'}
    base['training'].update(episodes=2, seed=22)
    base['evaluation'] = {'episodes': 1}
    base['simulator']['episode_time_steps'] = 17
    base['outputs'] = {'model_artifacts': 'both'}
    base['tracking'].update(runtime_profiling_enabled=True, runtime_profiling_interval=1, log_frequency=1)
    modes = [('warmup', False, False, 'minimal', 10)]
    for repeat in range(args.repeats):
        modes += [(f'{mode}_{repeat}', mode!='off', mode.startswith('mlflow'),
                   'full' if mode=='mlflow_full' else 'minimal', 1 if mode=='mlflow_full' else 10)
                  for mode in ['off', 'local', 'mlflow_minimal', 'mlflow_full']]
    for name, local, remote, profile, interval in modes:
        config = deepcopy(base)
        config['tracking'].update(metrics_enabled=local, mlflow_enabled=remote,
            mlflow_artifacts_profile=profile, mlflow_step_sample_interval=interval,
            mlflow_upload_models=profile=='full', mlflow_upload_checkpoints=profile=='full')
        path = root/f'{name}.json'; path.write_text(json.dumps(config))
        with (root/f'{name}.log').open('w') as log:
            subprocess.run([sys.executable, __file__, '--output', str(root), '--worker-config', str(path), '--worker-name', name],
                check=True, stdout=log, stderr=log,
                env={**os.environ, 'PYTHONPATH': str(Path.cwd())+os.pathsep+os.environ.get('PYTHONPATH',''), 'OPEVA_STARTUP_TRACE':'0'})
    results = [json.loads(p.read_text()) for p in sorted(root.glob('*.metrics.json')) if p.name!='warmup.metrics.json']
    report = {'method': 'One unmeasured warmup; three independent processes per mode by default. Same MADDPG 2 training + 1 evaluation, 17 state points, pinned CPU runtime; file MLflow, no network latency. Local detailed profiling enabled deliberately; totals overlap and must not be added. Peak RSS includes imports; wall time excludes imports but includes job setup, validation and output. Full uploads all artifacts and logs every step; minimal samples every 10 steps.', 'results': results,
              'mlflow_store_bytes_total': sum(p.stat().st_size for p in (root/'mlruns').rglob('*') if p.is_file())}
    (root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(root/'summary.json')


if __name__ == '__main__':
    main()
