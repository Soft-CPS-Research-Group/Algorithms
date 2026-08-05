# CC causal price control V4

This settlement-enabled campaign fixes the causal price channel before making
another CC+PPO claim and gives CC-SMART a stronger cost-oriented optimisation
budget.

## CC-PPO gate

The previous CC-PPO changed the encoded current-price feature consumed by an
already frozen actor. That was out-of-distribution inference and regressed the
annual settled cost by EUR 225.18. V4 leaves every PPO actor observation
unchanged. The CC multiplier is delivered only to the strict-local SMART
residual base beneath the PPO correction.

The original SMART residual base disables price-driven grid charging. V4 adds
an explicit `signal_price_charge_rate=0.6` that is active only when the CC
sends a discount below `1.00`. At `1.00` it remains disabled, preserving exact
parity; above `1.00` the existing price-sensitive discharge logic remains in
control. This gives the scalar channel bidirectional battery authority without
changing the PPO actor or exposing community observations to the leaf.

The annual fixed grid `0.90/0.95/1.00/1.05/1.10/1.20/1.30` is run before a
learned coordinator. `1.00` is the neutral causal control and must reproduce
the accepted PPO. A learned CC is created only around a fixed multiplier that
beats this control. Promotion requires lower annual settled cost, all hard
gates, and the complete peak/ramping/import/emissions/solar/fairness scorecard.

If no fixed multiplier beats `1.00`, the scalar channel has no demonstrated
authority over this frozen leaf. The next experiment is then a new local PPO
trained with randomized effective-price multipliers, followed by a fresh
neutral replay and fixed response grid. The PPO remains building-local and
community-blind; only the coordinator observes the community.

## CC-SMART candidates

All three learned candidates start from the fixed `1.30` incumbent and have an
effective range of `0.90--1.30`, ten annual episodes and a seven-day PPO
horizon. The ablation isolates:

- cost-only hourly control;
- cost-only 15-minute control;
- 15-minute control with small peak and ramp terms.

The 15-minute discount factor `0.99875` is approximately equivalent to `0.995`
per hour. Action regularisation is reduced to `w_factor=0.01` and
`w_smoothness=0.005`; it remains non-zero only to avoid economically
meaningless price jitter. The cost-only candidate decides whether extra
temporal authority can improve the primary KPI. The small peak/ramp candidate
tests whether a visible physical scorecard gain is available without giving
up that cost focus.

Generate the committed templates with:

```bash
.venv/bin/python scripts/generate_cc_causal_price_control_v4.py
```

Generate short functional smokes outside the committed config directory with:

```bash
.venv/bin/python scripts/generate_cc_causal_price_control_v4.py \
  --smoke \
  --output-dir runs/local_configs/cc_causal_price_control_v4_smokes
```
