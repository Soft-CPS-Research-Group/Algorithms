# CC-L2 PPO causal-guard V4

This campaign preserves the frozen seed-789 local PPO leaf and the exact
settlement-on comparison surface. It responds to the V3 matched pilot, where
member-specific advantages produced distinct prices but increased cost,
imports, peaks and ramping.

V4 does not let a broad neural policy perturb the leaf everywhere. The
coordinator emits exactly `1.0` unless the current tariff is cheap relative to
its three available forecasts and the community is currently exporting. Only
inside that causal interval does it learn a price per building, initialized at
the deployable Level-1 incumbent `0.90`. Actor gradients are masked outside the
active region; the centralized member critics still learn the state value.

The leaf response is no longer binary. The residual SMART charge authority is
linear in the discount around the exactly preserved `0.90` incumbent, and the
same multiplier changes only the frozen PPO actor's current-price coordinate.
The three price forecasts stay real and unmodified: the paired response
ablation showed that persisting the current multiplier into forecasts was
worse. This gives the coordinator a causal, continuous and empirically
selected actuator.

The candidates compare hourly versus 30-minute decisions, cost-first versus
scorecard credit, and a deeper active-only discount range. A conservative
variant uses lower exploration, tighter KL, raw member credit and twelve
episodes to avoid the moving scale introduced by online per-member reward
normalization. Raw credit and cooperative mixing are independent controls:
`reward_normalization: none` keeps the physical member-reward scale but still
applies `team_reward_mix`. The causal initializer can also be a complete
per-building vector, allowing learning to begin at the deterministic V5
coordinate-search incumbent instead of returning to a weaker global `0.90`.
The trainable variants evaluate the causal gate from physical tariff,
forecast and export observations, exactly like that incumbent. They also use
separate actor and value encoders so critic updates on inactive timesteps do
not silently move the price policy.
The guarded vector variant further learns only a 20% residual inside the
available distance from the measured vector to each price bound. It starts
exactly at the incumbent and cannot jump back into the broad, destructive
search surface observed in the first learned L2 pilots.
Pilot
selection has two explicit comparators: the exact neutral PPO and the exact
global causal incumbent. Promotion requires lower settled cost than PPO,
preservation or improvement of the incumbent, hard EV/network gates, and the
full community plus per-building scorecard.
