## Background: The problem are we solving

We have a simulator called **CityLearn** that models 17 buildings consuming and producing electricity over a year (8760 hours). Building 5 has an **electric vehicle charger** that can both charge an EV and send energy back to the grid (V2G — vehicle-to-grid). At every hour, someone has to decide: "should I charge the EV now, sit idle, or discharge it to the grid?"

The traditional way to make these decisions is a **rule-based controller (RBC)** — a hand-written set of if-then rules like "if it's morning, charge fully." It works, but it's rigid: it can't adapt to weather, electricity prices, or unusual EV usage patterns.

**The big idea**: train a *machine learning model* to make these decisions instead. Specifically, use **Offline Reinforcement Learning (Offline RL)** — learn from a fixed dataset of past decisions, without ever touching the simulator during training. This is realistic because in the real world we often have historical logs but can't risk experimenting on a live system.

But to do Offline RL, we need two things:
1. **A dataset** of (situation → action → outcome) tuples.
2. **An algorithm** that learns a policy from that dataset.

That's exactly what we built, in two milestones.

---

## Building the offline dataset

### The recording strategy

We don't have real-world EV data, so we **simulate it**. We let the existing RBC drive Building 5 through a full year, three times, and record every single decision it makes along with everything the simulator told it.

Think of it like installing a black box in a car driven by an experienced driver: we capture what they saw (observations), what they did (actions), and what happened next (rewards and next observations).

### What was built

#### 1. The recording agent — [`EVDataCollectionRBC`](../../algorithms/agents/ev_data_collection_agent.py)

This is the agent that drives the simulator. Two responsibilities:

**(a) Decide actions** using a fixed schedule based on the **hour of the day**:

| Time | EV action | Battery action |
|---|---|---|
| Midnight–6 AM | charge moderately (+0.4) | slow charge (+0.091) |
| 7–9 AM | charge fully (+1.0) | slow discharge (−0.08) |
| 10 AM–2 PM | discharge to grid (−1.0) | slow discharge |
| 3–7 PM | discharge moderately (−0.6) | slow discharge |
| 8 PM–11 PM | charge heavily (+0.8) | slow charge |

Actions are between −1 and +1, where the sign means "charge or discharge" and the magnitude means "how aggressively." The simulator translates these into actual kilowatts (e.g., +1.0 on the EV charger = 7.4 kW of charging power).

**(b) Record everything** at every hour into an in-memory list:

```python
row = {
    "episode": ..., "timestep": ...,
    "obs_<feature>": <value>,        # 35 features about what the agent saw
    "action_<name>": <value>,        # 2 actions it took
    "reward": <value>,               # CityLearn's reward signal
    "next_obs_<feature>": <value>,   # what the simulator looked like 1 hour later
    "done": <bool>,                  # was this the last step of the year?
}
```

> **Important**: the agent doesn't compute the reward. **CityLearn** computes it via its built-in `RewardFunction`, which is essentially `−1 × net electricity consumption`. The agent just receives the number and writes it down.

#### 2. The execution chain

When you run [`run_experiment.py`](../../run_experiment.py), this happens:

```
run_experiment.py
   ↓ (reads YAML config)
CityLearnEnv ←──── the physics simulator (loads Building_5.csv + 16 others)
   ↓
Wrapper_CityLearn ← the training loop manager
   ↓
EVDataCollectionRBC ← our recording agent
   ↓
For 3 episodes × 8759 hours each:
   obs ──► agent.predict() ──► actions
   actions ──► env.step() ──► next_obs, reward
   agent.update() ──► append row to memory
   ↓
agent.export_artifacts() ──► writes offline_dataset.csv
```

### The result: [`datasets/offline_rl/offline_dataset.csv`](../../datasets/offline_rl/offline_dataset.csv)

- **26,277 rows** (3 episodes × 8,759 hours each)
- **76 columns**: `episode` + `timestep` + 35 observation columns + 2 action columns + `reward` + 35 next-observation columns + `done`
- Observations cover: month, hour, weather (temperature, humidity, solar irradiance — current + forecasts), energy (PV generation, building load, battery SOC, grid draw, carbon intensity), pricing (current + 3 forecasts), and 7 EV-specific signals (is a car connected? what's its SOC? when does it leave? what SOC does it need at departure?)

---

## Training the Behavioral Cloning policy

### What is Behavioral Cloning?

The simplest possible Offline RL technique:

> **"Show a neural network what the expert did in each situation, and ask it to copy."**

Mathematically, given pairs $(s_i, a_i)$ where $s_i$ is the situation and $a_i$ is the expert's action, train a network $\pi_\theta$ to minimize:
$$\mathcal{L}(\theta) = \frac{1}{N} \sum_i \|\pi_\theta(s_i) - a_i\|^2$$

That's literally all it is — **supervised regression**. No environment interaction, no reward modeling, no fancy RL math.

**Why start here?** Two reasons:
1. **Sanity check**: if a neural network *can't* even imitate a deterministic schedule, something is fundamentally broken in our pipeline.
2. **Baseline**: more advanced offline RL algorithms (CQL, IQL) need a baseline to compare against. BC is the natural one.

The ceiling: BC can never *beat* the expert it copies. It just imitates. Future iterations will use smarter algorithms that try to *exceed* the RBC.

### The model — [`algorithms/offline/bc_policy.py`](../../algorithms/offline/bc_policy.py)

A small neural network: **35 inputs → 256 → 256 → 2 outputs**, with ReLU activations on the hidden layers and Tanh on the output.

| Choice | Reason |
|---|---|
| 2 hidden layers | One layer can't learn interactions like "if EV connected AND solar high AND morning"; three would overfit on 26K samples. |
| 256 neurons | Enough capacity (~9K params/layer) for our 26K samples, well below the overfitting risk threshold. |
| ReLU activation | Cheap, doesn't suffer from vanishing gradients, works almost everywhere. |
| Tanh output | Squashes outputs into [−1, 1], **exactly matching our action range**. Without it, the network could output 47.3, which would be silently clipped. |

### The data preparation — [`algorithms/offline/data_loader.py`](../../algorithms/offline/data_loader.py)

Three crucial steps before training:

**(a) Episode-level train/validation split** (not random!)

Episodes 0 and 1 → training (~17,500 rows). Episode 2 → validation (~8,700 rows).

> Why not shuffle randomly? Because hour $t$ and hour $t+1$ are nearly identical. If we randomly put $t$ in training and $t+1$ in validation, the model would "cheat" by memorizing rather than truly learning to generalize. Splitting by episode forces it to generalize to a fresh year.

**(b) Standardization** (subtract mean, divide by std, per feature)

The 35 features have wildly different scales: `electrical_storage_soc` is in [0, 1], `solar_generation` ranges 0–5000, `electricity_pricing` is around 0.05–0.35.

If we feed raw values into a neural network, two things go wrong:
- **Domination**: large-scale features (solar) generate huge activations and dominate gradient updates; small-scale features (SOC — which is critical for EV decisions!) are essentially ignored.
- **Slow convergence**: the loss landscape becomes a long narrow ravine, and gradient descent bounces around without making progress.

The fix: for each feature, compute mean $\mu$ and std $\sigma$ from the **training set only**, then transform every value as $x_{norm} = (x - \mu) / \sigma$. Now every feature has mean 0, std 1 — balanced, well-behaved.

> **Crucial detail**: stats are computed from training only (computing them from the whole dataset would leak validation info into training). And we **save** these stats to disk so that at inference time the model can re-apply the exact same transformation.

**(c) Batching**: PyTorch `DataLoader`s wrap the data into mini-batches of 256, ready for gradient descent.

### The training loop — [`algorithms/offline/bc_trainer.py`](../../algorithms/offline/bc_trainer.py)

Standard supervised learning:

```python
for epoch in range(50):
    for (obs_batch, action_batch) in train_loader:
        predicted = policy(obs_batch)
        loss = MSE(predicted, action_batch)
        loss.backward()
        optimizer.step()
    val_loss = evaluate(policy, val_loader)
    log_to_mlflow(epoch, train_loss, val_loss)
```

The hyperparameter choices and their reasoning:

| Hyperparameter | Value | Why |
|---|---|---|
| Optimizer | Adam | Adapts learning rate per parameter automatically; the "just works" default. |
| Learning rate | 3e-4 | The famous "Karpathy constant" — safest default for Adam. |
| Batch size | 256 | Big enough for stable gradients (~103 updates per epoch on our data), small enough to fit easily in memory. |
| Epochs | 50 | Enough for loss to plateau; more would risk overfitting. |
| Loss | MSE | Standard for continuous regression; penalizes large errors more than small ones. |

### The training entrypoint — [`scripts/train_offline_bc.py`](../../scripts/train_offline_bc.py)

A standalone CLI script (not part of [`run_experiment.py`](../../run_experiment.py)) — because training BC is pure offline supervised learning, no simulator involved. Reads the CSV, calls the trainer, saves five artifacts to `runs/offline_bc/<run_id>/`:

| Artifact | Purpose |
|---|---|
| `model.pth` | Trained network weights |
| `normalization_stats.json` | μ and σ per feature (essential for inference!) |
| `training_metadata.json` | Hyperparameters used, final losses, dataset path |
| `loss_history.json` | Per-epoch train and val loss |
| `loss_curve.png` | Visual plot of training progress |

### Training phase

- 50 epochs, ~4 seconds on CPU.
- Train loss: 0.092 → 0.00083
- Validation loss: 0.051 → **0.00077** ← finishes lower than train loss!

A monotonically decreasing loss with val ≤ train is the **textbook ideal training run**. It tells us:
- The model **is learning** (loss decreased by 100×).
- It's **not overfitting** (val ≤ train).
- The RBC is a deterministic hour-based map, so there's a clean function to learn — and the network found it.

### The inference agent — [`algorithms/agents/offline_bc_agent.py`](../../algorithms/agents/offline_bc_agent.py)

Now we need to **use** the trained model inside CityLearn. We wrap it as another `BaseAgent` so it plugs into the existing platform without changes:

- `__init__`: loads `model.pth` + `normalization_stats.json`
- `predict()`: takes raw observations → standardizes them with the saved stats → forward pass through the network → returns actions
- `update()`: no-op (BC doesn't learn online — it's a frozen policy)
- `export_artifacts()`: copies the model files into the job output directory for traceability

---

## BC vs RBC (CityLearn)

### The benchmark script — [`scripts/benchmark_bc_vs_rbc.py`](../../scripts/benchmark_bc_vs_rbc.py)

Strategy: run both controllers under **identical** CityLearn conditions (same schema, same reward function, same episode length), compute the official KPIs, and write a markdown report.

### Critical fairness fix

The RBC drives **all 17 buildings**, but BC only knows how to drive Building 5 (it's only seen Building 5's data). A naive comparison would have BC compete against an RBC that has 17× more "control surface."

**The fix**: we mask out non-target buildings to zero actions for **both** controllers, so each one is effectively only controlling Building 5. Now any difference in the metrics is genuinely attributable to the policy controlling Building 5.

```python
def _mask_non_target_actions(actions):
    return [a if i == TARGET_BUILDING_INDEX else [0.0] * len(a)
            for i, a in enumerate(actions)]
```

### CityLearn's KPIs

CityLearn provides built-in evaluation via `env.evaluate()`, which returns metrics in a **normalized form**:

> **`1.0` = no-control baseline** (every device idle). **Lower than 1.0 = improvement**. **Higher than 1.0 = worse than doing nothing**.

The headline KPIs (lower = better in all cases):

| KPI | What it measures | Why it matters |
|---|---|---|
| `electricity_consumption_total` | Total grid kWh drawn | The headline efficiency number |
| `carbon_emissions_total` | Grid draw × hourly carbon factor | Captures *when* you use energy, not just how much |
| `cost_total` | Grid draw × tariff (incl. peak pricing) | Direct economic impact |
| `all_time_peak_average` | Highest single-hour grid draw | Grid stress / infrastructure cost proxy |
| `daily_peak_average` | Average daily peak | Smoothness of demand |
| `ramping_average` | Step-to-step load change | Penalizes "spiky" control |
| `daily_one_minus_load_factor_average` | `1 − mean/peak` per day | Lower = flatter load profile |
| `annual_normalized_unserved_energy_total` | EV demand the controller failed to satisfy | **Constraint violation indicator** — must stay ~0 |
| `zero_net_energy` | Imbalance between consumption and PV | Self-sufficiency proxy |

Plus the **raw cumulative reward** from the simulator, broken down for the target building and the district.

### The report — [`docs/offline_rl/bc_vs_rbc_benchmark.md`](bc_vs_rbc_benchmark.md)

The headline result for Building 5:

| KPI | RBC | BC | Δ |
|---|---:|---:|---:|
| `electricity_consumption_total` | 4.0203 | 3.9866 | **−0.84%** 🟢 |
| `carbon_emissions_total` | 3.8319 | 3.8051 | **−0.70%** 🟢 |
| `cost_total` | 4.4853 | 4.4445 | **−0.91%** 🟢 |
| `annual_normalized_unserved_energy_total` | 0.0000 | 0.0000 | tie |

All differences are **less than 1%**. This is **exactly the intended outcome** for Milestone 2: BC successfully imitates the RBC.

---

## General View

```
                  ┌──────────────────────────────────┐
                  │  CityLearn Simulator              │
                  │  (17 buildings, 1 year, EV data) │
                  └──────────────┬───────────────────┘
                                 │
       ┌─────────────────────────┴─────────────────────────┐
       │                                                   │
       ▼                                                   ▼
┌──────────────────┐                                ┌─────────────────────┐
│ MILESTONE 1      │                                │ MILESTONE 2.5       │
│ Data collection  │                                │ Benchmark           │
│                  │                                │                     │
│ EVDataCollection │                                │ benchmark_bc_vs_rbc │
│ RBC drives the   │                                │ runs both agents    │
│ simulator and    │                                │ under identical     │
│ logs everything  │                                │ conditions          │
└────────┬─────────┘                                └──────────┬──────────┘
         │                                                     │
         ▼                                                     ▼
┌──────────────────┐                                ┌─────────────────────┐
│ offline_dataset  │                                │ bc_vs_rbc_benchmark │
│ .csv             │                                │ .md                 │
│ 26,277 rows      │                                │ KPI comparison      │
│ 76 columns       │                                │ |Δ| < 1%            │
└────────┬─────────┘                                └─────────────────────┘
         │                                                     ▲
         ▼                                                     │
┌──────────────────┐    ┌────────────────────┐                │
│ MILESTONE 2      │    │ Trained artifacts  │                │
│ BC training      │    │ • model.pth        │                │
│                  │───►│ • norm_stats.json  │────────────────┘
│ data_loader →    │    │ • metadata.json    │   (loaded by
│ bc_policy →      │    │ • loss_history     │    OfflineBC
│ bc_trainer       │    │ • loss_curve.png   │    Agent)
└──────────────────┘    └────────────────────┘
```

---

## Concepts

| Concept | One-line summary |
|---|---|
| **Simulator (CityLearn)** | A physics model of buildings + EVs that responds to actions and returns rewards. |
| **Rule-Based Controller (RBC)** | A hand-written set of decisions based on simple rules (e.g., hour of day). |
| **Observation** | What the agent sees at a given time (35 numbers per step for our setup). |
| **Action** | What the agent does (2 numbers per step: battery + EV charger commands). |
| **Reward** | A score the simulator returns; for us, `−1 × net grid consumption`. |
| **Transition** | A single tuple (obs, action, reward, next_obs, done) — one row in our dataset. |
| **Offline RL** | Training a policy from a *fixed* dataset, never interacting with the simulator. |
| **Behavioral Cloning** | The simplest offline RL: supervised regression to imitate the expert's actions. |
| **Episode** | One complete run from reset to terminal state (a full simulated year for us). |
| **Standardization** | Rescaling features to mean 0, std 1 — required for stable neural network training. |
| **Train/val split** | Held-out data to detect overfitting; we split *by episode* to prevent leakage. |
| **Loss function (MSE)** | The thing the network minimizes: average squared error between predicted and true actions. |
| **Epoch** | One complete pass through the training data. |
| **Optimizer (Adam)** | The algorithm that adjusts network weights to reduce the loss. |
| **CityLearn KPIs** | Normalized metrics where 1.0 = baseline, lower = better. |