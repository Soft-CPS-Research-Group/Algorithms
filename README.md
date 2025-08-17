

mlflow ui --backend-store-uri file:./mlruns




# Algorithms for Energy Flexibility Optimization


### Prerequisites
- **Python 3.8+**
- **PyTorch** (for training and inference)
- **Other Dependencies** (listed in `requirements.txt`)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/my_marl_algorithm.git
   cd my_marl_algorithm

2. Set up a virtual environment:

    ```bash
    python3 -m venv env
    source env/bin/activate

3. Install dependencies:

    ```bash
    pip install -r requirements.txt```

4. Install the package:

    ```bash
    pip install -e .

### 🧩 Quick Start

1. Run a Rule-Based Baseline
To test a rule-based algorithm in the simulator:

```bash
python examples/rule_based_demo.py --config configs/default_config.yaml
```

2. Train or Evaluate Agents in Docker

```bash
python docker_run.py --config configs/config.yaml --job_id maddpg_run
python docker_run.py --config configs/config_rbc.yaml --job_id rbc_run
```

3. Export a Trained Model
Exported models are logged to MLflow as ONNX artifacts.

### 🚢 Docker Usage

To train inside a Docker container, build the image and launch the entry script:

```bash
docker build -t citylearn-marl .
docker run --rm \
  -v /path/to/persistent/data:/data \
  -v $(pwd)/configs/config.yaml:/app/config.yaml \
  citylearn-marl \
  --config /app/config.yaml \
  --job_id my_run \
  --log-level INFO \
  --resume --checkpoint-run-id <previous_run_id>
```

Select the learning algorithm through `algorithm.class` in the config file.
Supported options are `MADDPG`, `RBC`, `GNN`, and `Transformer`.

### Resuming Training

When `experiment.save_checkpoints` is enabled, the agent periodically stores its
state as an MLflow artifact. To continue a run:

1. Obtain the previous MLflow run ID.
2. Re‑launch the container with `--resume --checkpoint-run-id <run_id>` or set
   `experiment.resume_training` and `experiment.checkpoint_run_id` in the config.

All agents implement `save_checkpoint`, `load_checkpoint`, and
`export_to_onnx` to provide a consistent interface for persistence and
deployment. Exported ONNX models can be served via:

```bash
python scripts/serve_onnx.py --run-id <mlflow_run_id>
```

### 📚 Documentation

- [Wiki's Repository](https://github.com/Soft-CPS-Research-Group/.opeva_wiki)

### 🧪 Testing
Run unit and integration tests:

    ```bash
    pytest tests/

### 🤝 Contributing
- Tiago Fonseca calof@isep.ipp.pt

### 📜 License

This project is licensed under the MIT License.

