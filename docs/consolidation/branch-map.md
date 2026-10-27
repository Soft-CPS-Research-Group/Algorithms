# Branch reconciliation — 26–27 October 2026

Snapshot: **69 remote branches**, including main. **25** tips (including main) are ancestors of main; **44** are not. A non-ancestral tip is not proof of missing code: squashed/restacked PRs require content comparison.

The machine-readable [branch map](branch-map.json) preserves full SHAs, merge bases, patch-only commits, changed files, PR associations and decisions. Archives use `archive/2026-10-26/<original-branch>` and must be verified remotely before removing the corresponding branch. New or advanced heads after the snapshot are not eligible for removal.

| Branch | Snapshot | Decision | Archive |
|---|---|---|---|
| `GustavoJ6-patch-1` | `fa03b9c47dcb` | superseded | `archive/2026-10-26/GustavoJ6-patch-1` |
| `backup/matd3-cost4-faithful-pre-reconcile-20260824` | `46aa570cecfd` | superseded | `archive/2026-10-26/backup/matd3-cost4-faithful-pre-reconcile-20260824` |
| `backup/matd3-documentation-final-pre-cost4-reconcile-20260824` | `8298f3164244` | superseded | `archive/2026-10-26/backup/matd3-documentation-final-pre-cost4-reconcile-20260824` |
| `backup/matd3-pre-main-restack-20260824-pr41` | `eb71c9685497` | superseded | `archive/2026-10-26/backup/matd3-pre-main-restack-20260824-pr41` |
| `backup/matd3-pre-main-restack-20260824-pr42` | `81958e9e2ac9` | superseded | `archive/2026-10-26/backup/matd3-pre-main-restack-20260824-pr42` |
| `backup/matd3-pre-main-restack-20260824-pr43` | `f83ba4e6a5c5` | superseded | `archive/2026-10-26/backup/matd3-pre-main-restack-20260824-pr43` |
| `backup/matd3-pre-main-restack-20260824-pr44` | `c1dd9a7ab712` | superseded | `archive/2026-10-26/backup/matd3-pre-main-restack-20260824-pr44` |
| `backup/matd3-pre-main-restack-20260824-pr45` | `5810d1ff34f0` | superseded | `archive/2026-10-26/backup/matd3-pre-main-restack-20260824-pr45` |
| `backup/matd3-pre-main-restack-20260824-pr46` | `4500e718e817` | superseded | `archive/2026-10-26/backup/matd3-pre-main-restack-20260824-pr46` |
| `backup/matd3-pre-main-restack-20260824-pr47` | `10d068a658cd` | superseded | `archive/2026-10-26/backup/matd3-pre-main-restack-20260824-pr47` |
| `backup/matd3-pre-main-restack-20260824-pr48` | `916a84cd6554` | superseded | `archive/2026-10-26/backup/matd3-pre-main-restack-20260824-pr48` |
| `backup/matd3-pre-main-restack-20260824-pr49` | `61e4c9113fec` | superseded | `archive/2026-10-26/backup/matd3-pre-main-restack-20260824-pr49` |
| `codex/matd3-documentation-final` | `3d034907a6fc` | already_integrated | `archive/2026-10-26/codex/matd3-documentation-final` |
| `codex/tppo-architecture-cleanup` | `18690095ad02` | already_integrated | `archive/2026-10-26/codex/tppo-architecture-cleanup` |
| `codex/tppo-bc-integration-clean` | `95a81569e003` | already_integrated | `archive/2026-10-26/codex/tppo-bc-integration-clean` |
| `codex/tppo-bc-primitives` | `dbd86e0198b3` | already_integrated | `archive/2026-10-26/codex/tppo-bc-primitives` |
| `codex/tppo-documentation` | `eb0f006dbd4b` | superseded | `archive/2026-10-26/codex/tppo-documentation` |
| `codex/tppo-on-policy-correctness` | `7bbbfe92621b` | already_integrated | `archive/2026-10-26/codex/tppo-on-policy-correctness` |
| `codex/tppo-transactional-recovery` | `b714afae9c68` | already_integrated | `archive/2026-10-26/codex/tppo-transactional-recovery` |
| `feat/matd3-pr1-shared-package-extraction` | `630a1126b899` | already_integrated | `archive/2026-10-26/feat/matd3-pr1-shared-package-extraction` |
| `feat/matd3-pr2a-tppo-cutover` | `d502252b4cda` | already_integrated | `archive/2026-10-26/feat/matd3-pr2a-tppo-cutover` |
| `feat/matd3-pr2b-foundations` | `3e45dc40b43f` | already_integrated | `archive/2026-10-26/feat/matd3-pr2b-foundations` |
| `feat/matd3-pr3-core-learning-loop` | `0a114b36d0b5` | already_integrated | `archive/2026-10-26/feat/matd3-pr3-core-learning-loop` |
| `feat/matd3-pr4-residual-bca-safety` | `7c5192920fde` | already_integrated | `archive/2026-10-26/feat/matd3-pr4-residual-bca-safety` |
| `feat/matd3-pr5-demonstration-bcb` | `b90188d4655f` | already_integrated | `archive/2026-10-26/feat/matd3-pr5-demonstration-bcb` |
| `feat/matd3-pr6-dynamic-checkpoints` | `39efb4e4f159` | already_integrated | `archive/2026-10-26/feat/matd3-pr6-dynamic-checkpoints` |
| `feat/matd3-pr7-integration` | `27be2594198a` | already_integrated | `archive/2026-10-26/feat/matd3-pr7-integration` |
| `feat/matd3-pr8-export-manifest` | `8861d47b47e4` | already_integrated | `archive/2026-10-26/feat/matd3-pr8-export-manifest` |
| `feat/matd3-pr9-docs-templates-e2e` | `8054890908e2` | already_integrated | `archive/2026-10-26/feat/matd3-pr9-docs-templates-e2e` |
| `feat/tppo-cuda-timeout` | `7604a168dc65` | superseded | `archive/2026-10-26/feat/tppo-cuda-timeout` |
| `feature/cc-improvements` | `a7a395d743c9` | recovered | `archive/2026-10-26/feature/cc-improvements` |
| `feature/cc-level1` | `b3db7bfd868e` | already_integrated | `archive/2026-10-26/feature/cc-level1` |
| `feature/cc-level2` | `e435c4685227` | superseded | `archive/2026-10-26/feature/cc-level2` |
| `feature/create-hiro-agent` | `6445b90d17e5` | already_integrated | `archive/2026-10-26/feature/create-hiro-agent` |
| `feature/entity-iql-pipeline` | `5e86c4ea5dd7` | recovered | `archive/2026-10-26/feature/entity-iql-pipeline` |
| `feature/make-wrapper-hierarchical` | `2b02b898aa04` | already_integrated | `archive/2026-10-26/feature/make-wrapper-hierarchical` |
| `feature/new-architecture` | `1aa46e020f14` | superseded | `archive/2026-10-26/feature/new-architecture` |
| `feature/offline-agents-implementation` | `69316e574145` | recovered | `archive/2026-10-26/feature/offline-agents-implementation` |
| `gj/investigate-tmatd3-speed` | `76070832c958` | already_integrated | `archive/2026-10-26/gj/investigate-tmatd3-speed` |
| `gj/investigate_tmatd3_speed` | `5955016f9eaf` | already_integrated | `archive/2026-10-26/gj/investigate_tmatd3_speed` |
| `gj/master` | `ce5bd4fa16eb` | superseded | `archive/2026-10-26/gj/master` |
| `gj/matd3-bc-dynamic-demonstrations` | `bcbe31d85fc4` | already_integrated | `archive/2026-10-26/gj/matd3-bc-dynamic-demonstrations` |
| `gj/matd3-cost4-faithful` | `0dd471af1fcb` | already_integrated | `archive/2026-10-26/gj/matd3-cost4-faithful` |
| `gj/matd3-review-followup` | `75dd49f4f63d` | already_integrated | `archive/2026-10-26/gj/matd3-review-followup` |
| `gj/matd3-transformer-opeva-union-smoke` | `f83ba4e6a5c5` | superseded | `archive/2026-10-26/gj/matd3-transformer-opeva-union-smoke` |
| `gj/plan-a` | `1b3690cffc00` | superseded | `archive/2026-10-26/gj/plan-a` |
| `gj/plan-b` | `6e15eec50902` | superseded | `archive/2026-10-26/gj/plan-b` |
| `gj/plan-c` | `3e9a6737d730` | superseded | `archive/2026-10-26/gj/plan-c` |
| `gj/temp-ai-test-tppo-pr32` | `4335b3f2fcc5` | superseded | `archive/2026-10-26/gj/temp-ai-test-tppo-pr32` |
| `gj/tmatd3-dynamic-asset-removal` | `ab084cbbc230` | squash_integrated | `archive/2026-10-26/gj/tmatd3-dynamic-asset-removal` |
| `gj/tmatd3-neutral-residual-safety` | `841f4edc98a7` | squash_integrated | `archive/2026-10-26/gj/tmatd3-neutral-residual-safety` |
| `gj/tmp/investigate_tmatd3_speed` | `916a84cd6554` | superseded | `archive/2026-10-26/gj/tmp/investigate_tmatd3_speed` |
| `gj/tppo-action-projection-v3` | `5eeeffe72151` | superseded | `archive/2026-10-26/gj/tppo-action-projection-v3` |
| `gj/tppo-action-projection-v3-fix` | `db11cdd3f5e3` | superseded | `archive/2026-10-26/gj/tppo-action-projection-v3-fix` |
| `gj/tppo-projection-diagnostics` | `cd3ea9565d38` | superseded | `archive/2026-10-26/gj/tppo-projection-diagnostics` |
| `gj/tppo-recovery` | `48884d261ca4` | superseded | `archive/2026-10-26/gj/tppo-recovery` |
| `gj/tppo-tmatd3-deferrable-tokenizer` | `77ae47867289` | squash_integrated | `archive/2026-10-26/gj/tppo-tmatd3-deferrable-tokenizer` |
| `gj/tppo_bclonning` | `cfb4f3f09a33` | superseded | `archive/2026-10-26/gj/tppo_bclonning` |
| `gj/tranformer_ppo` | `dde6f887b9d4` | superseded | `archive/2026-10-26/gj/tranformer_ppo` |
| `gj/transformer_maddpg_integration` | `e4dd4e73b3bf` | recovered | `archive/2026-10-26/gj/transformer_maddpg_integration` |
| `gj/transformers_matd3` | `22c6b7a2197b` | superseded | `archive/2026-10-26/gj/transformers_matd3` |
| `gj/wp06-01-tokenizer-schema` | `60b6063830e4` | superseded | `archive/2026-10-26/gj/wp06-01-tokenizer-schema` |
| `gj/wp06-02-layout-builder` | `497b708897c3` | superseded | `archive/2026-10-26/gj/wp06-02-layout-builder` |
| `gj/wp06-03-torch-blocks` | `1bbb6c98eb24` | superseded | `archive/2026-10-26/gj/wp06-03-torch-blocks` |
| `gj/wp06-04-agent-and-registry` | `219185677760` | superseded | `archive/2026-10-26/gj/wp06-04-agent-and-registry` |
| `gj/wp06-05-wrapper-and-template` | `ed6fb48b9f19` | superseded | `archive/2026-10-26/gj/wp06-05-wrapper-and-template` |
| `gj/wp06-template-and-e2e` | `0412bfd9f80b` | superseded | `archive/2026-10-26/gj/wp06-template-and-e2e` |
| `macos_onboard` | `23d5ddef19be` | archived_recipe | `archive/2026-10-26/macos_onboard` |
| `main` | `9773cf3d9033` | retained | — |

## Evidence by decision

- Earlier Transformer PPO/MATD3 prototypes and implementation plans replaced by current entity-based AgentTransformerPPO/AgentTransformerMATD3. No separate missing algorithm family; exact prototype and documentation preserved in archive.
- Pre-restack MATD3 implementation/backups superseded by merged PRs #34–49; current transformer_shared and transformer_matd3 packages retain the architecture, batching/cache and dynamic behavior. Exact historical heads retained in archive.
- Tip is an ancestor of the audited origin/main; all commits are reachable from main.
- WP06 implementation/documentation delivered in merged PRs #17–21 and #33, then maintained by #26–32 and the shared Transformer extraction #34–35. Current entity contract supersedes preliminary flat layouts.
- TPPO recovery/BC/CUDA work reconciled through #26–32 and maintained in transformer_ppo/transformer_shared. Building-identity correction cd3ea95 is present in current citylearn_local_action_safety.py. Archive retains experimental diagnostics and earlier plans.
- SignalAwareRBCCommunity, SignalDirectedRBC, optional worker coupling, environment-reward option, CCRewardLevel1V2 and named CCLevel2Capacity/CCRewardLevel2Capacity recovered. Current main CCLevel1/CCLevel2 retained. Broad best-effort RNG restoration not adopted; it would weaken current training resume semantics.
- Current main already contains CCLevel1, CCLevel2, CommunityCoordinator, hierarchical PPO, pipeline and reward interfaces. Capacity/directional additions from the later cc-improvements branch are recovered separately.
- BC/IQL/CQL libraries, collectors, training tools and tests recovered from final 69316e5. Normal jobs use prepare_training and the shared runner. PR #13 merged into the offline branch, not main. Historical reports remain in the archive.
- Merged PR #51; current main contains the resulting safety, asset-topology or tokenizer behavior. Squash explains non-ancestral heads.
- Merged PR #50; current main contains the resulting safety, asset-topology or tokenizer behavior. Squash explains non-ancestral heads.
- Merged PR #52; current main contains the resulting safety, asset-topology or tokenizer behavior. Squash explains non-ancestral heads.
- Distinct TransformerMADDPG, tokenizer, replay, networks and tests recovered; schema/registry/checkpoint export adapted. Different from AgentTransformerMATD3.
- Only removes CUDA suffixes from three requirements. Not a distinct agent. Preserve exact onboarding patch in archive; platform dependency profiles remain a later packaging action rather than changing GPU image dependencies during consolidation.
- Authoritative integration base 9773cf3; retained as main.
