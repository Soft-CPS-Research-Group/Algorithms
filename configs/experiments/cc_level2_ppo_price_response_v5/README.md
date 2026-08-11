# CC-L2 frozen-PPO price-response ablation V5

The frozen PPO leaf is a storage residual over `SignalAwareRBCSmartLocal`.
Consequently, a CC price can act on the residual SMART base, on the PPO actor's
price observation, or on both. These five paired recipes use the same causal
`0.90` cheap-and-export signal and differ only in that response path.

`real_unmodified` changes only the price known at the current decision.
`persist_current` also applies the current multiplier to the three forecast
coordinates; it is causal, but explicitly represents the assumption that the
current CC decision persists across the forecast horizon. The actor and PPO
checkpoint remain frozen in all recipes.

This is a mechanism ablation, not a promotion campaign. The selected response
path must first beat the residual-only incumbent on the matched scorecard. It
is then used by the per-building vector search and, only after that, by a
trainable CC-L2 residual policy.
