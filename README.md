# Algorithms

**English** · [Português](docs/README.md)

Algorithms trains and evaluates controllers for energy communities. To run a job you choose a dataset, a time window and a policy. The same runner can evaluate a charging rule or train a multi-agent reinforcement learning policy. You can also run an offline oracle to estimate the lowest achievable community electricity cost for a week.

## How a job works

1. **Configure the experiment.** The YAML file sets the simulation window, seed, training and evaluation episodes, and requested outputs. `simulator.dataset_path` points to the simulator schema. That schema describes the buildings and devices and references time series such as demand, solar generation and prices.
2. **Run the simulator.** The runner opens the dataset with SoftCPS REC Simulator (`softcpsrecsimulator==2.1.0`). At each interval the agent receives observations and returns actions, such as how much power to request for an EV. The simulator applies the physical limits and returns the next observations and rewards.
3. **Train and evaluate.** During training, the runner passes transitions to the agent according to its update schedule. The agent owns the learning rule. The final deterministic evaluation produces the comparable KPIs. The runner writes the results and any requested checkpoint or model bundle.

**Change the controller in `pipeline`.** Each stage selects a registered `algorithm` and its own parameters. For example, the Basic and Smart Core30 templates use the same simulator and dataset. To compare the two policies, keep the window, seed and evaluation settings matched and use the stage from the corresponding template. Run `--dry-run` to check that an agent supports the chosen interface and topology. [Basic](configs/templates/baselines/core30/basic.yaml) · [Smart](configs/templates/baselines/core30/smart.yaml)

**Add a new agent.** Implement its decision, learning and export methods in a `BaseAgent` subclass. Register its implementation, parameter schema and supported capabilities with an `AgentSpec` in `algorithms/catalogue.py`. Once registered, it uses the same runner, simulator loop and job directory as the built-in agents. [Agent development guide](docs/runtime/agents.md)

## Run a first job

Start at the root of a clone with Python 3.10 on Linux x86-64. This example uses two buildings from the bundled demo dataset. The commands install the CPU runtime and export dependencies.

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements/lock-cpu-py310-linux-x86_64.txt
python -m pip install '.[export]'

config=configs/templates/experimental/transformer_maddpg_smoke.yaml
algorithms-job --config "$config" --validate
algorithms-job --config "$config" --dry-run
algorithms-job --config "$config" --profile smoke --job_id first-job
```

The first check validates the YAML. The second loads and resets the environment. The last command runs one short training episode and one deterministic evaluation. Open `runs/jobs/first-job/results/result.json` and look for `"status": "completed"`. In the same job directory, `config.resolved.yaml` records the effective settings, `results/kpis.json` contains the evaluation metrics and `bundle/artifact_manifest.json` describes the exported policy. For Core30, supply the separate dataset and set its schema path in the config. [Installation on other platforms](docs/runtime/installation.md) · [Configuration reference](docs/runtime/configuration.md)

## Available today and remaining work

- **Controllers.** The catalogue includes MADDPG, MATD3, Transformer agents, TI-MARL, hierarchical coordinators and offline learners. Run `algorithms-job --list-agents` to see their registered names and capabilities. [Agent catalogue](docs/agent_catalogue.md)
- **Comparison.** Core30 uses two BAUs (charge to the user target or to 100%), Basic, Smart, Community and the weekly cost oracle. Savings use the user-target BAU as the reference. The long-run comparison of the MARL candidates against this set is still pending. [Comparison protocol (PT)](docs/baseline_comparison_core30.md) · [Weekly oracle](docs/oracles/total_energy_milp.md)
- **Inference.** A job can export a policy after evaluation. Supported neural policies use ONNX and supported rule-based policies use executable JSON. Check deployment eligibility for the specific bundle. TI-MARL and complete coordinator pipelines still require specialized inference adapters. [Bundle formats and support](docs/inference_bundle.md#delivery-formats)

For remote jobs, see the [platform guide](docs/platform_guide.md). For context when choosing an algorithm, see the [lessons from long runs (PT)](docs/learning-experience.md).

The current baseline evaluation starts again with Simulator 2.1.0 and the
Portugal `portugal-realism-4` / `portugal-ev-mobility-1` inputs. Previous scores
are not comparable. First audit EV/appliance service and battery energy over
14 days of 2023, revalidate the rules across the four reserved 2023 fortnights,
then check the weekly oracle's state transfer and solver bounds. Freeze the
policies before the common full-2024 evaluation and subsequent MARL comparison.
Short integration jobs verify execution and exports only. Remote runs require
the new dataset snapshot and matching image; the old server copy is unsuitable.
The [working protocol (PT)](docs/baseline_comparison_core30.md) contains the dates,
checks, market assumptions and acceptance sequence.
