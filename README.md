

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

2. Train a MARL Agent
To train a MADDPG agent:

    ```bash
    python examples/train_maddpg.py --config configs/maddpg_config.yaml

3. Export a Trained Model
To export a trained model for deployment:

    ```bash
    python marl/export/torchscript_export.py --model-path models/maddpg_agent.pth --output-path models/maddpg_scripted.pt

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

