# CC-L2 causal coordinate search V5

The vector signal path first has to reproduce the deployable global causal
incumbent exactly: all 17 active-event multipliers are `0.90`, prices are
neutral outside current cheap-and-export decisions, the frozen PPO leaf is
unchanged, and community settlement remains enabled.

A paired mechanism ablation selected the leaf response contract: the current
price coordinate of the frozen PPO actor and the SMART residual base both see
the multiplier, while all three actor forecast coordinates remain real and
unmodified. Applying the current multiplier to those forecasts was worse on
the matched scorecard and is not used by the coordinate search.

Initial `±0.005` and `±0.05` sensitivity checks produced bitwise-identical
physical trajectories. The audit exposed a stronger issue than a narrow
deadband: the residual SMART base treated every multiplier below `1.0` as a
binary switch for one fixed charging rate. V5 uses the opt-in
`linear_discount` response instead. It preserves the measured action at
`0.90`, gives half that authority at `0.95`, and caps deeper-discount authority
at 1.5 times the reference. The coarse campaign moves exactly one building by
`±0.05` around `0.90`, producing 34 paired probes plus the parity reference.
Only hard-gate-passing changes that reduce settled community cost are combined.
The combined vector is then replayed on multiple windows and the full year;
coordinate interactions are rechecked rather than assumed additive.

This deterministic search is not the final learning claim. It establishes a
measured CC-L2 incumbent and a supervised teacher. A later contextual policy
may learn small residuals around that vector. `CCLevel2` therefore supports a
full `causal_initial_multipliers` vector as well as the legacy scalar initial
multiplier: its deterministic policy starts exactly at the measured vector
inside active intervals and remains exactly neutral outside them. Member/team
credit is applied independently from reward normalization, so disabling the
running z-score no longer silently disables cooperative credit. A learned
policy cannot be promoted unless it beats both the neutral PPO and this vector
incumbent on the matched scorecard.
