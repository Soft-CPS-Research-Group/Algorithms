# CC-PPO causal online V5.3

This protocol removes the annual-trace schedule used by V5/V5.1. At each
hourly decision boundary, `CausalPriceSignal` reads only the current pre-action
observation and discounts the strict-local residual base when both conditions
hold:

1. current community export power is positive;
2. the current tariff is cheap relative to its three available forecasts.

The decision persists for four 15-minute steps. The frozen PPO actors retain
their original building-local observations and never receive community state.
The two annual configurations differ only in residual-base charge rate:

- `0.45`: balanced incumbent-strength candidate;
- `0.60`: cost-first candidate.

Generate the committed annual configurations with:

```bash
.venv/bin/python scripts/generate_cc_ppo_causal_online_v5p3.py
```

Use `--smoke --output-dir <ignored-dir>` for functional smokes.
