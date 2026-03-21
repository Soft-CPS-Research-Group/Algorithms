# Thesis Implementation Plan
## Offline Reinforcement Learning and Zero-Shot Generalization in Renewable Energy Communities

**Author:** Guilherme Barbosa de Sousa  
**Deadline:** August 2026  
**Created:** March 21, 2026

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Definition of Done](#2-definition-of-done)
3. [Repository Orientation](#3-repository-orientation)
4. [Algorithm Selection Framework](#4-algorithm-selection-framework)
5. [Implementation Roadmap](#5-implementation-roadmap)
6. [Validation Strategy](#6-validation-strategy)
7. [Thesis Integration Guide](#7-thesis-integration-guide)
8. [Risk Mitigation](#8-risk-mitigation)

---

## 1. Executive Summary

### 1.1 Research Goal
Develop an offline reinforcement learning solution that:
- Learns energy management policies from historical data (no online exploration)
- Generalizes to unseen buildings within the same community (intra-community)
- Transfers knowledge across different communities (inter-community)
- Handles building-level storage and EV flexibility

### 1.2 Scientific Contribution Criteria
Your work will be scientifically valuable if it demonstrates:

| Criterion | What It Means | How to Measure |
|-----------|---------------|----------------|
| **Offline Learning Efficacy** | Policy learns from static data without environment interaction | Compare vs. behavioral policy (RBC) on same data |
| **Zero-Shot Generalization** | Policy works on unseen buildings without retraining | Test on held-out buildings, report performance gap |
| **Scalability** | Approach extends to larger communities | Measure training time vs. agent count |
| **Safety Guarantee** | No out-of-distribution actions harm physical assets | Track constraint violations, action bounds |
| **Sample Efficiency** | Less data needed than online methods | Learning curves vs. online baselines |

### 1.3 Timeline Overview

```
March 2026     April 2026      May 2026        June 2026       July 2026       August 2026
|--------------|---------------|---------------|---------------|---------------|
  Phase 1         Phase 2          Phase 3         Phase 4          Phase 5
  Foundation     Algorithm &       Experiments &   EV Extension    Writing &
  & Design       Implementation    Validation      & Scaling       Defense Prep
```

---

## 2. Definition of Done

### 2.1 Minimum Viable Thesis (Must Have)
A thesis is **complete** when:

- [ ] **Offline RL agent implemented** extending `BaseAgent` in this repo
- [ ] **Training pipeline** works with CityLearn datasets
- [ ] **Intra-community generalization** demonstrated (train on buildings 1-5, test on 6-10)
- [ ] **Quantitative comparison** vs. RBC baseline and online MADDPG
- [ ] **Exported ONNX models** compatible with inference pipeline
- [ ] **Reproducible experiments** with config files and documented seeds

### 2.2 Target Thesis (Should Have)
- [ ] **Inter-community transfer** tested across both available datasets
- [ ] **Multiple offline RL algorithms** compared (BCQ, CQL, IQL, TD3+BC)
- [ ] **EV charging optimization** integrated
- [ ] **Ablation studies** on key design choices
- [ ] **Statistical significance** (multiple seeds, confidence intervals)

### 2.3 Stretch Goals (Nice to Have)
- [ ] **Data augmentation** techniques explored
- [ ] **Cluster-based policy mapping** for new building classification
- [ ] **Hybrid offline-to-online** fine-tuning demonstrated
- [ ] **Scaled experiments** on server with hyperparameter search

---

## 3. Repository Orientation

### 3.1 Critical Files (Read These First)

| Priority | File | Purpose | Time Estimate |
|----------|------|---------|---------------|
| 🔴 1 | `algorithms/agents/base_agent.py` | Contract your agent must fulfill | 30 min |
| 🔴 2 | `algorithms/registry.py` | How agents are registered and instantiated | 15 min |
| 🔴 3 | `algorithms/agents/maddpg_agent.py` | Reference online RL implementation | 1 hour |
| 🟠 4 | `utils/wrapper_citylearn.py` | Training loop, observations, rewards | 1 hour |
| 🟠 5 | `configs/config.yaml` | Configuration schema and options | 30 min |
| 🟠 6 | `utils/config_schema.py` | Validation rules for config | 30 min |
| 🟡 7 | `reward_function/` | Reward shaping implementations | 30 min |
| 🟡 8 | `tests/test_agents.py` | Test patterns for agents | 30 min |

### 3.2 Dataset Structure

**Primary dataset:** `datasets/citylearn_challenge_2022_phase_all_plus_evs/`

```
schema.json          → Environment configuration
Building_*.csv       → 17 buildings with consumption/generation
charger_*_*.csv      → 8 EV chargers with sessions
weather.csv          → Temperature, humidity, solar irradiance
pricing.csv          → Electricity prices
carbon_intensity.csv → Grid carbon signals
```

**Key columns in building CSVs:**
- `Equipment Electric Power [kWh]` → Non-shiftable load
- `DHW Heating [kWh]`, `Cooling Load [kWh]`, `Heating Load [kWh]` → Flexible loads
- `Solar Generation [kW/kWp]` → PV output (needs scaling by capacity)

**Key columns in charger CSVs:**
- `required_soc_departure` → Target state-of-charge
- `estimated_departure_time` → Departure constraint
- `charger_rated_capacity` → Max charging power

### 3.3 Understanding the Training Flow

```
1. run_experiment.py loads config
2. Config validated by config_schema.py
3. Wrapper_CityLearn builds CityLearn environment
4. Agent instantiated via registry.py
5. Training loop:
   - wrapper calls agent.predict(observations)
   - environment steps
   - wrapper calls agent.update(transitions)
6. agent.export_artifacts() saves ONNX + manifest
```

**Key insight:** Your offline RL agent must work differently:
- It receives a **static dataset** (not streaming observations)
- `update()` performs batch training on replay buffer
- `predict()` uses the learned policy for evaluation

---

## 4. Algorithm Selection Framework

### 4.1 Decision Matrix

Select your algorithm based on these criteria:

| Algorithm | Distribution Shift Handling | Implementation Complexity | Multi-Agent Ready | Best For |
|-----------|---------------------------|--------------------------|-------------------|----------|
| **BCQ** | Action constraint (VAE) | Medium | With modifications | Continuous control, SoC tasks |
| **CQL** | Conservative Q-values | Medium | Yes | Safety-critical, HVAC |
| **IQL** | Expectile regression | Low | Yes | Zero-shot transfer |
| **TD3+BC** | BC regularization | Low | Yes | Simple baseline |

### 4.2 Selection Process

**Step 1: Prototype with TD3+BC** (simplest)
- Why: Minimal changes to existing MADDPG critic structure
- Deliverable: Working offline agent in 1-2 weeks

**Step 2: Implement IQL** (recommended for generalization)
- Why: Best documented for zero-shot transfer (ORCHID paper)
- Why: Avoids querying OOD actions (safer for real deployment)

**Step 3: Optionally add CQL** (for comparison)
- Why: Strong baseline in literature
- Why: Different conservatism mechanism provides contrast

### 4.3 Key Features Your Algorithm MUST Provide

1. **Batch Learning Interface**
   ```python
   def update(self, dataset: ReplayBuffer, batch_size: int) -> dict:
       """Learn from static dataset, return metrics."""
   ```

2. **Deterministic Evaluation Mode**
   ```python
   def predict(self, obs, deterministic=True) -> np.ndarray:
       """No exploration noise during deployment."""
   ```

3. **ONNX Export**
   ```python
   def export_artifacts(self, output_dir, context) -> dict:
       """Export policy network for inference."""
   ```

4. **Multi-Agent Support**
   - Independent learners (simplest) or parameter sharing
   - Per-building observation/action spaces

---

## 5. Implementation Roadmap

### Phase 1: Foundation & Design (March 21 - April 4)

**Week 1-2 Goals:**
- [ ] Complete repository orientation (Section 3)
- [ ] Set up development environment, run existing tests
- [ ] Create skeleton `OfflineBaseAgent` class
- [ ] Document dataset observation/action spaces

**Deliverables:**
- `algorithms/agents/offline_base_agent.py` (interface)
- `docs/observation_action_spec.md` (dataset analysis)

**Validation:**
```bash
pytest tests/  # All existing tests pass
python run_experiment.py --config configs/config.yaml --job_id test-run
```

---

### Phase 2: Algorithm Implementation (April 5 - May 2)

**Week 3-4: TD3+BC Baseline**
- [ ] Implement TD3+BC extending `OfflineBaseAgent`
- [ ] Add to registry, create config template
- [ ] Train on single building, verify loss convergence
- [ ] Compare vs RBC on training building

**Week 5-6: IQL Implementation**
- [ ] Implement IQL with expectile regression
- [ ] Add temperature/expectile hyperparameters to schema
- [ ] Validate on same single-building setup

**Deliverables:**
- `algorithms/agents/td3bc_agent.py`
- `algorithms/agents/iql_agent.py`
- `configs/templates/offline_*.yaml`

**Validation Checkpoint:**
```
Training building performance:
- TD3+BC cost reduction vs RBC: ≥0% (matches behavioral)
- IQL cost reduction vs RBC: ≥0%
- Actor loss converges within 50k steps
```

---

### Phase 3: Generalization Experiments (May 3 - May 30)

**Week 7-8: Intra-Community Generalization**
- [ ] Define train/test splits (e.g., buildings 1-10 train, 11-17 test)
- [ ] Train single policy on multiple buildings
- [ ] Evaluate zero-shot on held-out buildings
- [ ] Document per-building performance breakdown

**Week 9-10: Inter-Community Transfer**
- [ ] Train on `citylearn_challenge_2022_phase_all_plus_evs`
- [ ] Test on `citylearn_three_phase_electrical_service_demo`
- [ ] Measure transfer gap, identify failure modes

**Deliverables:**
- Experiment configs for each split
- Results CSV with per-building metrics
- Figures: training curves, generalization gap plots

**Validation Checkpoint:**
```
Intra-community:
- Zero-shot performance ≥ 80% of training performance
- Statistical test (p < 0.05) vs random policy

Inter-community:
- Document transfer gap (expected: larger than intra)
- Identify which building types transfer well/poorly
```

---

### Phase 4: EV Extension & Scaling (June 1 - June 27)

**Week 11-12: EV Charger Integration**
- [ ] Extend observation space with charger state
- [ ] Add V2G actions to action space
- [ ] Implement EV-specific reward terms
- [ ] Validate on charger-equipped buildings

**Week 13-14: Scaling & Ablations**
- [ ] Run experiments on server (if available)
- [ ] Hyperparameter sensitivity analysis
- [ ] Ablation: with/without carbon signal, price signal, etc.

**Deliverables:**
- Updated agent supporting EV actions
- Hyperparameter sensitivity plots
- Ablation study results table

---

### Phase 5: Writing & Defense Prep (June 28 - August)

**Week 15-17: Thesis Writing**
- [ ] Update Chapter 5 (Experimentation) with new results
- [ ] Write detailed Methods section for chosen algorithm
- [ ] Create publication-quality figures
- [ ] Complete related work positioning

**Week 18-20: Polish & Defense**
- [ ] Internal review with supervisors
- [ ] Prepare defense slides
- [ ] Practice presentation
- [ ] Final document formatting

---

## 6. Validation Strategy

### 6.1 Metrics to Track

| Metric | What It Measures | Target |
|--------|------------------|--------|
| **Cost Reduction (%)** | (RBC_cost - Agent_cost) / RBC_cost | > 5% |
| **Carbon Reduction (%)** | Same formula for emissions | > 0% |
| **Peak Shaving (%)** | Reduction in max demand | > 0% |
| **Constraint Violations** | Actions outside [-1, 1] bounds | 0 |
| **Generalization Gap** | Test perf / Train perf | > 0.8 |

### 6.2 Experiment Tracking

Use MLflow (already integrated):
```python
# In your agent's update method
mlflow.log_metrics({
    "actor_loss": actor_loss,
    "critic_loss": critic_loss,
    "q_value_mean": q_values.mean(),
}, step=global_step)
```

### 6.3 Statistical Rigor

- **Seeds:** Run each experiment with 3-5 different seeds
- **Confidence intervals:** Report mean ± std
- **Significance tests:** Paired t-test for comparing algorithms

---

## 7. Thesis Integration Guide

### 7.1 Chapter Mapping

| Thesis Chapter | Implementation Artifact |
|----------------|------------------------|
| Ch. 4 Methods | Algorithm pseudocode, architecture diagrams |
| Ch. 5 Experimentation | Experiment configs, MLflow logs |
| Ch. 5 Results | Figures from `runs/jobs/*/results/` |
| Ch. 6 Discussion | Ablation studies, failure analysis |

### 7.2 Figure Generation Workflow

```bash
# After experiments complete
python scripts/generate_figures.py --job_id <your_job> --output_dir thesis_figures/

# Figures to generate:
# 1. Training curves (loss, Q-values)
# 2. Episode reward evolution
# 3. Generalization gap bar charts
# 4. Per-building performance heatmap
# 5. Action distribution histograms
```

### 7.3 Writing Tips (Following Your Style)

1. **Structure:** Problem → Approach → Results → Analysis
2. **Tables:** Use for algorithm comparisons, hyperparameters
3. **Figures:** Training curves, bar charts for KPIs, time-series for behavior
4. **Citations:** Connect each design decision to literature (your systematic review)

---

## 8. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Algorithm doesn't converge | Medium | High | Start with TD3+BC (simplest), tune learning rates |
| Poor generalization | Medium | High | Use data augmentation, cluster-based policies |
| Compute bottleneck | Low | Medium | Prioritize local experiments, scale later |
| Dataset issues | Low | High | Validate data preprocessing early |
| Time overrun | Medium | High | Prioritize MVT (Section 2.1), cut stretch goals |

### 8.1 Fallback Plan

If Phase 3 experiments show poor generalization:
1. Reduce scope to intra-community only
2. Focus on building-type clustering
3. Frame as "understanding limitations" (still valuable)

---

## Appendix A: Quick Start Commands

```bash
# Run existing MADDPG to understand the flow
python run_experiment.py --config configs/config.yaml --job_id understand-flow

# Run tests
pytest tests/ -v

# View MLflow results
mlflow ui --backend-store-uri file:./runs/mlflow/mlruns
```

## Appendix B: Key Hyperparameters to Tune

| Parameter | Typical Range | Start With |
|-----------|---------------|------------|
| Learning rate (actor) | 1e-5 to 1e-3 | 3e-4 |
| Learning rate (critic) | 1e-5 to 1e-3 | 3e-4 |
| Batch size | 128 to 1024 | 256 |
| Discount (γ) | 0.95 to 0.99 | 0.99 |
| **IQL expectile (τ)** | 0.7 to 0.9 | 0.8 |
| **TD3+BC α** | 0.1 to 2.5 | 2.5 |
| **CQL α** | 0.1 to 10 | 1.0 |

---

*This plan is a living document. Update it as you progress.*
