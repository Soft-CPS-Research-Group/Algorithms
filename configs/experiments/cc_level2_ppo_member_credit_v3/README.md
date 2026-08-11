# CC-L2 PPO member-credit V3

This campaign keeps the accepted seed-789 local PPO leaves frozen and changes
only the Level-2 coordinator learning contract.

V2 returned the same community reward to all 17 price factors and used one
scalar critic. Its deterministic policy collapsed to an almost uniform
multiplier around 0.99 and made every building more expensive than neutral
PPO. V3 uses exact member settlement/service reward components and one
centralized value/advantage per building. A configurable team-reward mixture
retains coordination without erasing local causal credit.

The BC teacher also follows the actual `SignalAwareRBCSmartLocal` price
semantics: a full stationary battery raises the virtual price to encourage
discharge; an empty battery lowers it to encourage charging. The price range
is `[0.55, 1.15]`; the known V5.2 diagnostic used a `0.90` discount multiplier
and a separate `0.60` storage charge rate. The frozen PPO actor remains
community-blind and unchanged.

The four annual candidates isolate cost-first hourly control, a 30-minute
decision interval, a scorecard-aware objective, and a causal BC teacher that
reproduces the successful `cheap AND community export` intervention without
using future outcome traces. Promotion still requires a paired replay below
the exact PPO baseline, hard EV/network gates, and the full physical and
per-building scorecard.
