# MATD3 V3 improvement campaign

This campaign fixes two structural limitations found in the recovered annual
MATD3 run before spending more compute on hyperparameter search:

1. Derived price/load/PV/net forecasts used simulator bounds of
   `[-1e6, 1e6]`, collapsing realistic encoded values to approximately `0.5`.
   V3 scales price forecasts with the dataset tariff envelope and power
   forecasts with stable local/community power references without changing the
   observation names or dimensions. The fix is opt-in through
   `maddpg_v4_operational`, preserving checkpoint compatibility for the legacy
   V3 profile.
2. `critic_update_mode: joint_mean` aggregated optimization work but did not
   make the critic target cooperative. V3 adds `critic_team_reward_mix`. It is
   disabled by default for backward compatibility and blends the per-building
   reward with the mean community reward when explicitly configured.

All candidates use the exact hyperparameters of the current annual SMART
reference as `RBCSmartPolicy` teacher, two full training years and a final
deterministic annual evaluation. Settlement remains enabled.

## Controlled candidates

| Candidate | Isolated question |
|---|---|
| `smart_anchor` | How much was lost by learning around the weaker `RBCCommunityPolicy` teacher? |
| `cooperative_team70` | Does 70% team / 30% local critic credit improve community cost without losing local service? |
| `cooperative_storage_open` | Can wider battery authority and lighter regularization turn the cooperative signal into useful dispatch? |
| `cooperative_scorecard` | Can stronger peak and action-smoothness guards improve physical KPIs while retaining cost gains? |
| `cooperative_cost_first` | What is the attainable settled-cost gain when battery cycling is deliberately cheap and authority is widest? |

The sequence is intentionally progressive. A candidate can be rejected while
still identifying whether the teacher, credit assignment, or control authority
was the limiting factor.

## Promotion protocol

Use the current annual settled SMART replay as the paired reference. Do not
promote from training reward or a shortened smoke.

- Hard gates: zero electrical violations and the established Phase-6 EV
  feasibility/precision gates.
- Primary objective: annual settled cost below SMART. The first milestone is a
  reliable SMART win; the stretch target is the current PPO result.
- Physical guards: inspect community import, daily and all-time peak ratios,
  ramping, solar self-consumption, export and load factor.
- Operational diagnostics: battery throughput, V2G export, action saturation,
  critic gap/TD error and the learned residual magnitude.
- Fairness: report per-building deltas and the number of buildings improved;
  aggregate improvement alone is insufficient evidence.

Run a real 4096-step smoke after image CI, then launch the five annual seed-789
candidates in parallel. Only the best valid annual recipe advances to three
seeds. The generated files are reproducible with:

```bash
python scripts/generate_matd3_v3_campaign.py
```
