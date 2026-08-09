# CC-PPO causal online V5.3b

V5.3b preserves the deployable, current-observation-only `cheap AND export`
rule and removes the weakest part of V5.3: its 0.95 intervention. The matched
V5.2 ablation identified 0.90, so this campaign tests that discount with the
balanced charge rate (0.45), the cost-first rate (0.60), and 15-minute versus
hourly decisions.

The rule never reads an annual trace, a next observation, or a realized future
outcome. The PPO leaf is frozen and community-blind. Results are compared to
the exact settled neutral PPO replay on the full-year scorecard.
