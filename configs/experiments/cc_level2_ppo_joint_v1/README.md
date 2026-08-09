# CC-L2 + price-conditioned PPO joint V1

The seed-789 PPO checkpoint is restored, but it is no longer treated as a
price-aware frozen policy. During CC BC warm-up the local actors start adapting
to their effective prices; afterwards both CC-L2 and the local PPOs update.
PPO observations remain building-local, and community state reaches a leaf
only through its scalar price multiplier.

The manager's BC teacher uses physical tariff/import/export units, while its
policy keeps normalized inputs and explicitly observes community grid
headroom. This fixes the previous normalized-versus-kWh teacher mismatch
without exposing community state directly to any PPO leaf.

The campaign compares current-price conditioning, current-plus-forecast
conditioning, and a guarded V2G variant. Promotion requires two annual replays
from the winning checkpoint: the learned CC-L2 vector and a neutral all-ones
vector. That pair separates the value of the adapted PPO from the incremental
effect of the coordinator.
