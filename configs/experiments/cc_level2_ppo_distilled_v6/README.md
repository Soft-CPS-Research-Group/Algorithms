# CC-L2 PPO distilled V6

V6 gives the frozen PPO leaf a measured bidirectional price channel and gives
the Level-2 actor a strong training-only teacher. It addresses the V5 result in
which autonomous CC-L2 was safe but improved annual settled cost by only
EUR 4--5.

## Causal action contract

- price range: `[0.70, 1.30]`;
- `1.00`: exact neutral PPO path;
- `< 1.00`: continuous discretionary battery charging authority;
- `> 1.00`: continuous local-import battery discharge/conservation authority;
- full configured response at `0.85` and `1.15`, capped at 1.5 times that
  response at the outer range;
- the leaf remains frozen, building-local, V2G-off and unaware of community
  state.

Future price forecasts remain real and unmodified. Only the current effective
price and the strict-local residual base receive the CC signal, preserving the
best matched V5 price-path contract.

## Teacher protocol

The first repeated year executes the exact neutral vector while collecting
policy contexts. Each context receives a label, but that label is not executed:

- `causal_teacher_cost_seed123`: causal cheap-and-export labels;
- `milp_cost_seed456`: fixed-service battery cost-oracle labels;
- `milp_scorecard_seed789`: globally shaped fixed-service scorecard labels.

For MILP labels, positive battery power is mapped below one and negative power
above one. Four 15-minute schedule steps are averaged into each hourly CC
decision. The coordinator is then supervised on those price labels and keeps a
small decaying BC anchor during PPO training, analogous to the successful
MATD3 distillation protocol.

The MILP is perfect-foresight and is therefore a training instrument only. It
is not deployable, is unavailable during validation/evaluation, and does not
support an optimality claim for the resulting CC policy.

## Safety and selection

The second year is an exact neutral PPO baseline. Two stochastic training years
alternate with deterministic validation years. A validation candidate is
promoted only when its full-year objective exceeds the incumbent. Rejected
candidates roll back. If no learned policy improves the neutral baseline, the
final year restores the immutable neural state that emits exactly
`[1.0] * 17`, not the teacher-pretrained state.

Promotion still requires a separate paired KPI comparison against
`cc_l2_v6_paired_neutral_annual.yaml`, with EV and electrical hard gates before
cost, peak, ramping, emissions, solar, throughput and fairness are interpreted.

## Causal map

`scripts/generate_cc_level2_bidirectional_map_v6.py` builds deterministic
matched probes at `0.70`, `0.85`, `1.00`, `1.15` and `1.30`. The first stage
measures the global response curve on seasonal windows. The optional member
stage changes one building at a time (68 non-neutral probes) and identifies
where each sign and magnitude has real physical/economic authority. These
measurements can filter or weight future teacher labels instead of assuming
that every building responds equally.

Constant signals also change battery state and eventually saturate it, so the
generator supports a stricter pulse experiment. A run remains exactly neutral
until the intervention, applies one global or per-building vector price for one
CC interval, then returns to neutral. The paired trajectories are identical up
to the pulse; their immediate action and energy deltas therefore measure the
causal response of the selected building without conflating it with days of
state-of-charge drift. `FixedPriceSignal.vector_schedule` provides the
auditable per-member intervention contract used by this test.

The local 384-step functional check confirmed the intended direction: at the
same one-hour intervention, the aggregate battery changed from `-3.65 kWh` at
neutral to `+6.29 kWh` under `0.70`, and to `-6.32 kWh` under `1.30`. A pulse
sent only to Building 8 changed Building 8 while the other 16 battery
trajectories stayed numerically identical. These are implementation checks,
not annual performance claims; annual/seasonal remote probes remain required
before promoting a teacher-trained policy.
