# TransformerPPO No-BC 15-Minute Template Design

## Goal

Add one full-year `AgentTransformerPPO` template for the dynamic 15-minute
dataset with no behavior cloning or warm-start teacher.

## Template

The template is named
`configs/templates/dynamic/transformer_ppo_entity_dynamic_15min_year.yaml`.
It matches the existing BC year template's dataset, dynamic entity interface,
35,040-step horizon, model, PPO hyperparameters, seed, exports, diagnostics,
and runtime safeguards. This makes the runs directly comparable.

The template omits the entire `behavior_cloning` block. It also removes
teacher-specific tracking tags, uses no-BC experiment and session names, and
sets `tracking.mlflow_enabled: false`.

## Scope

The work adds one configuration and focused template validation. It does not
change the PPO algorithm, tokenizer, dataset, reward, or BC implementation.
