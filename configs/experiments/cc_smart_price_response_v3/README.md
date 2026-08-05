# CC-SMART price-response V3

This annual, settlement-enabled identification campaign separates two
questions that the cost-focus V2 campaign mixed together:

1. how much control authority the global scalar price channel has over the
   frozen `SignalAwareRBC` leaf;
2. whether the learned CC improves when it receives more PPO updates rather
   than only longer rollouts.

The fixed probes use multipliers 0.7, 0.9, 1.1 and 1.3.  The existing neutral
SMART replay at 1.0 is reused as the matching baseline.  These are response
probes, not candidates selected after seeing the annual outcome.

`legacy_update_dense` preserves the complete V1 reward, BC teacher,
regularization and 96-decision rollout, but uses eight annual episodes.  With
one BC episode and one deterministic final episode, the six PPO episodes yield
approximately 547 updates, versus approximately 182 in V1 and 156 in the V2
336-decision recipes.

All rows use the same full-year dataset, community settlement, frozen SMART
leaf, price range 0.5--1.3 and hard-gate scorecard.  A learned policy must be
compared both with neutral SMART and with the best fixed probe; otherwise a
gain caused by a constant bias would be misattributed to state-dependent CC
coordination.

## Post-sweep adaptive follow-up

The fixed sweep selected 1.3 as the best constant multiplier on the annual
development surface.  `incumbent_residual_update_dense` is therefore labelled
post-sweep rather than a pre-registered probe.  It starts from 1.3 and maps the
policy's original 0.5--1.3 output through a residual scale of 0.5, giving an
effective range of 0.9--1.3.  The policy can retain the best fixed action or
apply selective discounts during cheap, PV-surplus and local-matching periods;
it cannot explore the clearly harmful low-price region below 0.9.

Promotion requires beating both neutral SMART and fixed 1.3 with hard gates,
then confirmation on another seed or a held-out temporal surface.  The first
annual run is development evidence because its bounds were chosen after the
fixed sweep.
