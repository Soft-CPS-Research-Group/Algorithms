# CC-L2 SMART trainable V2

This campaign replaces the non-learning V1 boundary initialization. The V1
annual logs stayed at `mult_mean=1.300` and `mult_spread=0.000`: its absolute
`tanh` mean was initialized at the upper bound and BC was disabled.

V2 keeps the matched settled SMART leaf and the measured 1.30 incumbent, but
uses `centered_residual`, causal BC and a lower-variance PPO update. The BC
teacher reads physical tariff/import/export values while the policy keeps its
normalized inputs, and the compact policy context now includes community grid
headroom. The two cost runs differ only by seed. The scorecard run adds small
peak and export terms. All final claims must use the annual deterministic
episode and the matched settled SMART baseline; a short smoke is functional
evidence only.
