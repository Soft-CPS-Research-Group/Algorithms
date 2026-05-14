# Step 6 — IQL Behaviour-Policy Swap

## What was done

Step 6 replaced the RBC behaviour policy with the trained IQL policy (best
checkpoint: run-001 / seed 101) and re-collected a dataset under the same
frozen reward weights. The goal was to test whether the distribution shift
toward IQL-generated data compounds the Building 5 improvement.

---

## Collection

`scripts/collect_iql_dataset.py` rolled out IQLAgent for seeds 32–41
(disjoint from RBC seeds 22–31 and eval seeds 200–209).

| Quantity | Value |
|---|---|
| Behaviour policy | IQL run-001 / seed 101 |
| Seeds | 10 (32..41) |
| Steps per rollout | 8 759 |
| Total transitions | 87 590 |
| EV action std (per seed) | ≈ 0.229 |
| Storage action std (per seed) | ≈ 0.001 |

IQL controls both action dimensions, but in practice the stationary-battery
output is very close to zero (std ≈ 0.001 > 1e-6 tolerance, so the
zero-variance fail-fast does not trip). This is not surprising: IQL was
trained on RBC data where storage was always zero, so it learned to leave it
near zero.

Reward was applied via `scripts/apply_reward.py` using the unchanged frozen
weights from Step 2:

```
cost = 0.050, carbon = 0.056, peak = 0.025, ramp = 0.0017, unserved = 50.0
```

Augmented dataset written to
`datasets/offline_rl/iql_derived/iql_with_reward.parquet` (87 590 rows, 0
non-finite reward values).

---

## Training (run-002)

Same IQL config as run-001: 5 seeds × 150 000 gradient steps, same
hyperparameters.

| Seed | best_val_policy_mse | best_step | duration |
|---|---:|---:|---:|
| 100 | 0.000142 | 141 000 | 840 s |
| 101 | 0.000167 | 89 000 | 850 s |
| 102 | 0.000152 | 149 000 | 840 s |
| 103 | 0.000165 | 114 000 | 840 s |
| 104 | 0.000167 | 96 000 | 843 s |

Mean ± std: **0.000158 ± 0.000010**. For comparison run-001 achieved
0.002182 ± 0.000078 — run-002 is **13.8× lower proxy MSE**. This is expected:
training on IQL-generated data means the training distribution is more similar
to what the policy actually produces, narrowing the imitation gap.

---

## Benchmark results (run-002 vs run-001 vs RBC)

Full benchmark: 5 run-002 seeds × 10 eval seeds = 50 rollouts; 50 BC rollouts;
10 RBC rollouts. Eval seeds 200–209.

Full report: `iql_run002_vs_rbc_benchmark.md`.

### District level (17 buildings)

| KPI | RBC | BC | IQL run-002 | Δ (IQL−RBC) | Verdict |
|---|---:|---:|---:|---:|:---|
| `cost_total` | 2.2289 ± 0.0750 | 2.2239 ± 0.0707 | 2.2251 ± 0.0724 | −0.0037 | within noise |
| `all_time_peak_average` | 1.8247 ± 0.0228 | 1.8026 ± 0.0095 | 1.8005 ± 0.0056 | −0.0242 | **IQL better** |
| `ramping_average` | 1.5780 ± 0.0673 | 1.5707 ± 0.0631 | 1.5704 ± 0.0629 | −0.0076 | within noise |
| `annual_normalized_unserved_energy_total` | 0 | 0 | 0 | 0 | safe ✅ |

### Building 5 (IQL's controlled building)

| KPI | RBC | BC | IQL run-001 | IQL run-002 | Δ (run-002 − RBC) | Verdict |
|---|---:|---:|---:|---:|---:|:---|
| `cost_total` | 2.730 ± 0.081 | 2.645 ± 0.059 | 2.634 ± 0.051 | **2.666 ± 0.153** | −0.064 | within noise |
| `annual_normalized_unserved_energy_total` | 0 | 0 | 0 | 0 | 0 | safe ✅ |

---

## Interpretation

**Policy MSE improved dramatically** (13.8×) when training on IQL-generated
data. The policy is very good at predicting what IQL would do.

**Eval performance regressed slightly** relative to run-001: run-002 B5 cost
is 2.666 ± 0.153 vs run-001's 2.634 ± 0.051. The mean is worse and the
standard deviation is 3× wider. This suggests:

1. **Distribution narrowing**: the IQL behaviour data covers a narrower state
   space than RBC (IQL is more deterministic). Run-002 is well-fit to the IQL
   distribution but generalises less robustly across the wider eval range.

2. **Overfitting to behaviour distribution**: the run-001 policy learned a
   general strategy from diverse RBC data. Run-002 converged tightly to the
   IQL policy's deterministic manifold, losing some of the robustness that
   diversity provides.

3. **RBC data value**: the diverse RBC dataset (18% PV-bonus, 1% emergency,
   81% idle across varied weather and EV arrival patterns) is informationally
   richer than IQL-generated data where the policy consistently chooses similar
   actions.

The district `all_time_peak_average` improvement is notable: run-002 achieves
1.8005 ± 0.0056 vs RBC 1.8247 ± 0.0228 (Δ = −0.0242, >1σ). This exceeds
run-001's 1.8087 on that KPI.

---

## Conclusion on iterative swap

The behaviour-policy swap did not compound the Building 5 improvement. The
underlying reason is that a narrower, more deterministic dataset reduces
out-of-distribution generalisation even as it reduces training loss.

**Recommendation**: return to the RBC dataset for further training (higher β
or τ hyperparameter sweep) rather than continuing the iterative swap loop with
IQL-generated data.

---

## Process health

- 132/132 tests green (+11 from Step 6 TDD tests).
- No safety violations: unserved energy = 0 across all 50 rollouts.
- Reward column: 0 non-finite values in iql_with_reward.parquet.
- Both scripts (collect_iql_dataset.py, apply_reward.py) tested via TDD.
