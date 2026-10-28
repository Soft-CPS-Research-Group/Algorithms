# Installation profiles

Python 3.10 is the tested runtime. `pyproject.toml` provides an installable wheel
and the `algorithms-job` command. No editable checkout is required to validate
configuration or use the packaged encoders/tokenizers. Datasets are explicit
external job inputs, not included in the wheel.

```bash
python -m venv .venv
. .venv/bin/activate
pip install .
algorithms-job --config configs/config.yaml --validate
```

The default installation contains only configuration/catalogue dependencies.
Select extras for the work being performed:

| Extra | Purpose |
| --- | --- |
| `training` | Simulator 1.9.1, PyTorch, numerical runtime and telemetry |
| `export` | ONNX serialization |
| `tracking` | MLflow 2.19 |
| `offline` | Parquet transition datasets |
| `solver` | SciPy MILP/LP research oracles |
| `dev` | Tests, ONNX runtime verification, Ruff, mypy and wheel build |

For CPU training, select CPU wheels before installing the runtime:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install '.[training,export,offline,solver]' -c constraints/runtime-py310.txt
pip check
```

The exact dependency snapshot verified by the clean Linux x86_64 / Python 3.10
CPU job is [lock-cpu-py310-linux-x86_64.txt](../../requirements/lock-cpu-py310-linux-x86_64.txt).
To reproduce that dependency selection, install this file, then `pip install . --no-deps`.
The lock pins transitive versions but has no wheel hashes; it is platform-specific.

MLflow is absent from this installation. Install `.[tracking]` to enable it.
Rule-based agents still require the numerical simulator runtime because the
existing shared agent base and simulator use PyTorch. The simulator also brings
building-simulation dependencies; this extraction does not rewrite that package.

`constraints/runtime-py310.txt` reuses the existing deployed dependency pins,
excluding accelerator-specific torch wheels. It is a tested constraints set,
**not** a cross-platform hashed lock of every possible transitive dependency.
`requirements.txt` and the Docker/Jetson/Union definitions retain their platform
wheel choices; do not apply x86 CPU/CUDA wheel choices to Jetson. Changing the
published image's dependency footprint is a separate operational decision.

For development with the existing deployed environment:

```bash
pip install -r requirements-test.txt
pip install -e '.[dev]'
ruff check
ruff format --check
mypy
pytest
pytest -o addopts='' tests/e2e/test_e2e_consolidated_agents.py tests/e2e/test_common_job_profiles.py
```

CI checks installation without numerical packages outside the checkout, the
common-core lint/format/type boundaries, the full default test suite and real
representative jobs. CUDA-specific tests still require CUDA; a CPU pass is not
GPU certification.
