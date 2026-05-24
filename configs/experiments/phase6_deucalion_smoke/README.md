# Phase 6 Deucalion Smoke Configs

Small configs for validating a freshly built SIF on Deucalion before launching
long jobs.

Run from the repository root so the relative dataset paths resolve:

```bash
python run_experiment.py \
  --config configs/experiments/phase6_deucalion_smoke/remote_20260524_deucalion_smoke_2022_maddpg_v3_direct_cpu.yaml

python run_experiment.py \
  --config configs/experiments/phase6_deucalion_smoke/remote_20260524_deucalion_smoke_2022_rbc_smart_cpu.yaml
```

The MADDPG smoke intentionally disables warm-start, wrapper reward and action
diagnostics so the direct `entity payload -> maddpg_v3` path is active. The
RBCSmart smoke intentionally keeps the raw-observation path active.
