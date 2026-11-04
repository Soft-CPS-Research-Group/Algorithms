# Hierarchical community control

A community coordinator sets a signal and downstream policies act on the physical assets. In the price-conditioned configuration the coordinator supplies a multiplier and each building policy sees an effective local price. The simulator settles electricity at the actual tariff. Other registered coordinators have different action contracts. Check the [catalogue](agent_catalogue.md) and the selected stage schema before composing a pipeline.

## Compare coordinator designs

Begin with the downstream controller on its own. Record the base policy, actor weights, preprocessing, safety projection and output layout. Insert the coordinator with a neutral signal. For a price multiplier, use 1. Verify that the same observations lead to the same building actions and KPIs.

Freeze the downstream policies while varying the coordinator. Record the proposed signal, effective price, building commands and the energy actually delivered. This separates a weak coordination signal from a leaf that cannot respond, or a command clipped by the electrical limits. If both levels are trained jointly, run a separate comparison with the same base policies, warmup and reward assignment.

An aggregate community observation is cheaper to serve; per-building context can identify which member has flexibility. Exposing additional context changes the information available to the coordinator. Keep that choice explicit when comparing variants.

See [lessons from annual hierarchical evaluations](learning-experience.md) and the [pipeline contract](runtime/agents.md).
