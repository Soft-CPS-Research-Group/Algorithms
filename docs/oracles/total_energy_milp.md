# Weekly community cost oracle

The oracle plans one week of battery flows, EV charging and appliance starts using the realized demand, PV, prices and arrivals. It minimizes EV shortfall first, then the community energy bill at that service level. The reported peak is an outcome. A separate research controller, `MILP`, replans with at most 24 hours of forecasts.

## Prepare a week

Start with a [Core30 evaluation configuration](../../configs/templates/baselines/core30/milp_community.yaml). Point `simulator.dataset_path` at the evaluation schema. Set `simulation_end_time_step` to the start plus 672 and `episode_time_steps` to 673 (672 transitions of 15 minutes). Use one deterministic evaluation episode, zero training episodes and the same market and seed as the candidate.

```bash
python scripts/validate_core30_milp_reference.py \
  --config week.yaml --output-dir runs/oracle-week \
  --time-limit-seconds 900 --mip-relative-gap 0.000001 --solver-threads 1

python scripts/score_milp_reference.py \
  --reference-dir runs/oracle-week --job-dir runs/oracle-week/replay/jobs/oracle-week
```

Choose a new output directory for each solve. The first command builds the physical problem, saves the solver result and schedule, and replays it through the common job runner. The second compares the planned states and flows with the simulator and checks cost, EV service and electrical limits. `--solve-only` is useful to measure planning time. A replay is required for a simulator comparison.

## Conditions and interpretation

The `simulator_linear` physics profile supports constant battery/charger efficiencies, concave SOC-dependent power limits, zero degradation and charge-only EVs with zero standby loss and discharge reserve. Standby loss for stationary storage is scaled by timestep duration. A charger's minimum applies to the command, so actual delivery can be smaller at saturation. The builder checks supported physics, known initial EV states and complete appliance service windows.

The weekly reference sets a terminal minimum of 50% for stationary batteries. An EV still connected at the end is given its announced target on a best-effort basis. Count actual departures separately. Compare the terminal stored energy when interpreting the cost of another policy.

For a pure cost objective, the solver's dual bound and its feasible plan bracket the optimum **of the formulated model**, subject to numerical tolerances. Report both values, solver status, wall time and independent simulator cost. The time budget applies per solver phase. A small residual interval after a time limit can be adequate for a monetary comparison. The offline solve time is measured per week. Online latency is measured per decision in the runner.
