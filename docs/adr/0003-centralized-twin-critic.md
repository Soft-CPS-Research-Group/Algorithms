# ADR-0003 — Centralized twin-critic architecture

Status: accepted
Date: 2026-08-18
Depends on: ADR-0002
Related: ADR-0004, ADR-0005, ADR-0008

## Context

MATD3 requires two independent centralized critics per agent. TPPO's
`CriticHead` is a per-building V(s) scalar; not reusable. The shared
backbone's pooled output is per-building state, not a joint community
representation.

Four sub-decisions:

- 3a — critic multiplicity.
- 3b — centralized critic architecture.
- 3c — how the CA action attaches to CA features.
- 3d — whether the critic reuses the actor's feature stack.

## Plain-language

The critic must fold a variable-size community into one Q scalar. Two
strategies: (S1) put every building's tokens in one big attention room
with name tags, or (S2) let each building meet internally, elect a
summary vector, and have all summaries meet in a smaller room.

Fairness note for aggregation: Deep Sets mean gives every building
weight 1/N and thereby guarantees every per-building actor receives
proportional critic gradient. A learned attention aggregator can
concentrate on a subset of buildings and starve the others' actor
gradients.

## Decisions

### 3a — critic multiplicity: M1

One critic pair `(Q1, Q2)` per agent, matching the established MATD3
ownership model. `2N` critic instances total. Every agent
holds its own critic pair and its own target critic pair.

### 3b — critic architecture: S2b Deep Sets

Each critic processes the community as:

1. For every building, run the critic's tokenizer + backbone (P1 from
   3d) to obtain per-CA embeddings `[N_ca_b, d_model]` and a pooled
   representation `[d_model]`.
2. Fold the CA action scalars into their CA embeddings via a small
   post-tokenizer injection MLP (3c).
3. Compute a per-building "building embedding" from the pooled
   representation and action-conditioned CA summary.
4. Aggregate across buildings with Deep Sets:
   `Q = MLP_out(mean_b(MLP_in(building_embedding_b)))`.

Backbone upgrades are not required. `num_buildings` is fixed within a
topology, so the aggregator sees fixed-shape input.

### 3c — action injection: post-tokenizer MLP

For every CA token, a small MLP maps
`(ca_embedding: [d_model], action_scalar: [1]) -> [d_model]`. Tokenizer
input widths remain identical to the actor's tokenizer.

### 3d — critic feature stack: P1 independent

Every critic (all `2N` instances) owns its own tokenizer + backbone.
Twin critics are fully independent from each other and from the
actor. Gradients never cross role boundaries.

## Consequences

- Parameter count is significantly higher than TPPO's V-head critic:
  `2N × (tokenizer + backbone + action injection MLP + building
  embedding MLP + Deep Sets aggregator + Q head)`.
- Per-actor gradient magnitude from a critic is `~1/num_buildings`
  through the Deep Sets mean. This is the expected trade for
  fairness-by-construction.
- Two independent critic optimizers per agent.
- Target networks mirror online modules for actor, critic 1, and
  critic 2.

## Evidence

- `AgentTransformerMATD3._learn` owns independent critic updates and holds
  other actors' actions detached during each actor update.
- The shared Transformer entity contract defines a per-building pooled output,
  not a community representation.
- TPPO `CriticHead` is a per-building V(s) scalar.

## Future improvements

- 3b promotion path: Deep Sets → Transformer aggregator (attention
  over building embeddings) → joint attention (S1). Promotion is a
  swap of the aggregator module; requires backbone upgrades only at
  the S1 step.
- 3d P2 alternative: reuse the actor's tokenizer + backbone as a
  detached feature extractor. Fewer parameters; couples critic quality
  to actor feature quality. Adopt only if parameter budget becomes
  binding.
- 3a M2 alternative: one shared critic pair across all agents.
  Removes a factor of N from critic compute. Small deliberate
  deviation from current MATD3; adopt if compute is binding.
