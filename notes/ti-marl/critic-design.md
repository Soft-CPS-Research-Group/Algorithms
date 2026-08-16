# TI-MARL central critic design

The first critic is a permutation-equivariant `CentralSetCritic`. For every
active agent it receives the local compiled latent, raw/final bundle summary,
health/availability summary and explicitly authorised shared-resource context.

The critic applies shared per-agent encoding, permutation-invariant attention
pooling and a per-agent value head conditioned on local plus pooled context.
It supports changing population cardinality without parameter reconstruction.

The actor never receives critic-only joint data. Critic input construction is
kept in the TI-MAPPO learning adapter rather than the TIC or local policy.

Required invariants:

- input permutation permutes output values identically;
- joining agents have no predecessor transition;
- departing agents terminate individually;
- survivors bootstrap across topology changes;
- padding is an implementation detail with an explicit validity mask, not a
  universal fixed agent schema.

A resource graph critic is a later ablation and is only promoted when the
Simulator exposes meaningful resource topology beyond one community aggregate.

