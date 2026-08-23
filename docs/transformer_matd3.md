# AgentTransformerMATD3

`AgentTransformerMATD3` is an off-policy controller for the entity interface.
It uses one Transformer actor per building and centralized twin critics. It
supports dynamic asset topology. It must be the final pipeline stage.

## Supported workflow

Start with one template under `configs/templates/dynamic/`:

- `transformer_matd3_entity_dynamic.yaml` uses the core MATD3 path.
- `transformer_matd3_entity_dynamic_residual.yaml` adds an RBC residual base.
- `transformer_matd3_entity_dynamic_bc.yaml` enables replay and demonstration
  behavior cloning plus the runtime local-action safety guard.

The default template disables residual control, local action safety, local
price conditioning, and both behavior-cloning paths. Enable each path only when
its runtime inputs are available.

The BC template applies EV deadline-feasibility and local electrical-headroom
constraints to executed actions. It is experiment-only: the exported ONNX actor
requires the same safety guard in the serving runtime.

Dynamic topology changes preserve compatible neural and optimizer state.
Feature-width drift or a new entity type fails atomically. A building-count
change rebuilds the agent and clears replay. Replay batches never mix layout
signatures.

Checkpoint format 5 supports `full` and `inference` modes. Restore validation is
strict. ONNX export writes one opset-17 actor per building and topology version.
Residual, safety, and price logic stay outside ONNX. Their runtime-only export
flags must be explicit.

## Traceability

Each accepted architecture consequence maps to named automated tests.

| ADR | Consequence coverage |
|---|---|
| 0001 shared package | `test_no_transformer_compatibility_shim_or_cross_algorithm_import_remains`; `test_no_external_imports`; `test_projection_is_per_type_no_new_params_on_topology_grow`; `test_variable_token_count_supported`; `test_normalizer_state_round_trip_is_device_agnostic` |
| 0002 actor ownership | `test_actor_and_targets_update_only_on_delayed_due_step`; `test_building_count_change_applies_reset_full`; `test_export_writes_one_topology_versioned_opset17_model_per_building` |
| 0003 centralized twin critics | `test_should_be_permutation_invariant_across_buildings`; `test_should_initialize_twin_critics_independently`; `test_should_isolate_gradients_between_independent_critics`; `test_learning_uses_twin_target_minimum` |
| 0004 unchanged backbone | `test_variable_token_count_supported`; `test_forward_returns_ca_and_pooled_with_correct_shapes`; `test_should_sample_only_one_layout_signature` |
| 0005 replay representation | `test_should_sample_only_one_layout_signature`; `test_compatible_topology_commit_preserves_neural_optimizer_and_history`; `test_building_count_change_applies_reset_full` |
| 0006 batching policy | `test_should_evict_oldest_transition_globally`; `test_should_reject_underfilled_sample_request`; `test_underfull_update_emits_explicit_finite_skip_metrics` |
| 0007 behavior cloning | `test_bc_b_pretraining_changes_only_actor_stack`; `test_bc_a_extra_update_changes_only_actor_stack`; `test_bc_b_zero_weight_short_circuits_before_actor_forward`; `test_bc_a_external_targets_use_lazy_cloning_replay_field`; `test_bc_b_reservoir_capacity_applies_per_building`; `test_bc_b_teacher_is_rebuilt_for_topology_attachment`; `test_bc_a_disabled_allocates_no_optimizer_or_optional_replay_fields`; `test_bc_b_pretraining_fails_before_rl_when_building_has_no_usable_demo`; `test_missing_warm_start_bc_a_context_does_not_block_replay`; `test_bc_a_and_bc_b_use_separate_optimizers` |
| 0008 residual policy | `test_residual_composition_uses_ca_order_span_and_authority`; `test_warm_start_teacher_width_is_validated_before_composition`; `test_target_smoothing_uses_residual_authority`; `test_export_guards_fail_before_writing_files` |
| 0009 local price adapter | `test_price_conditioning_rewrites_a_copy_before_tokenization`; `test_neutral_price_context_is_an_exact_copy`; `test_schema_requires_entity_interface_and_minmax_price_profile`; `test_export_guards_fail_before_writing_files` |
| 0010 checkpoint | `test_full_format_5_round_trip_restores_training_replay_queue_and_rng`; `test_inference_round_trip_restores_actor_stack_and_operational_step`; all strict rejection and atomic-restore cases in `test_agent_transformer_matd3_checkpoint.py` |
| 0011 ONNX export | `test_export_writes_one_topology_versioned_opset17_model_per_building`; `test_exported_model_matches_deterministic_actor`; `test_runtime_only_export_marks_manifest_and_bundle_non_deployable` |
| 0012 schema and wrapper | `test_schema_accepts_matd3_and_defaults_n_step_gamma`; `test_schema_rejects_invalid_matd3_stage`; `test_registry_constructs_transformer_matd3`; `test_registry_exposes_dynamic_topology_hooks` |

The end-to-end test
`test_e2e_learning_mutation_checkpoint_and_export` runs the normal experiment
entry point. It proves learning, topology mutation, checkpoint output, ONNX
export, and bundle validation together.

## Known limits

- Only the entity interface is supported.
- Checkpoints cannot restore across layouts or building counts.
- Export contains only the current topology.
- Residual, safety, and price processing require an external ONNX runtime path.
- BC-A samples only the current layout signature.
- Layout-homogeneous batches are required.
