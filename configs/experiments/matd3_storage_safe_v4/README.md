# MATD3 storage-safe V4

V3's closest candidate was EUR 71.25 above the current SMART baseline. Its
training contract still leaked exploration and TD3 target smoothing into EV
and deferrable actions after the residual mask, and its critic fitted large EV
penalties even when the desired improvement was stationary-battery control.

V4 closes both leaks. SMART remains the exact service teacher; EV and
deferrable residual authority are zero in deterministic action construction,
environment exploration, and target-policy smoothing. MATD3 learns only a
bounded stationary-storage correction. The V55 reward retains exact member
settlement, battery/grid safety, peak, export and throughput terms, while
removing EV/deferrable penalties that the actor is contractually unable to
change.

The zero-residual gate must reproduce SMART. The measured pilot operating
point is promoted as the conservative `0.06` residual-authority contract.
Building 15 is the only member with configured three-phase electrical-service
limits in this dataset; its learned residual gain is therefore exactly zero
and it keeps the audited SMART storage action. The other sixteen actors retain
storage authority. Local physical projection remains a second guardrail with
0.10 kW reserve.

`0.10` and `0.16` are explicit wider-authority projected ablations. A separate
`0.06` smoothness variant tests whether actor-delta regularization improves
ramping without giving up the cost gain. The `net_smooth` ablation changes only
the shared import/peak penalty from gross positive member imports to the net
community exchange measured at the scorecard's grid boundary. The
`net_context_smooth` ablation additionally exposes current causal community
net/import/export/PV and mean storage-SOC features to the actors; this tests
whether decentralized local observations were hiding the signal required for
community peak and ramp control. Two- and four-year candidates remain bounded
at `0.08` and the 25% team-credit candidate is kept separate. Short pilots
reach their final configured authority instead of silently exercising only the
beginning of the annual ramp. Promotion requires cost below the paired SMART
run, the Phase-6 hard gates, and the full physical and per-building scorecard.

The follow-up storage-focused ablations isolate four remaining bottlenecks.
`net_context_team25` adds cooperative critic credit after removing the
uncontrollable service penalties. `net_context_replay` stops prioritizing EV
teacher actions and samples PV/price/import/export events relevant to the only
actions the residual can change. `net_context_accelerated` exposes the actor
earlier, trains during the safe teacher warm-up and modestly increases policy
authority. `net_context_temporal` adds a one-hour frame stack and direct net
ramp pressure. Finally, `net_context_b15_guarded` restores only 25% residual
authority to Building 15 behind a 0.75 kW headroom reserve, testing whether its
useful storage flexibility can be recovered without reopening the electrical
violations seen in unconstrained pilots.
