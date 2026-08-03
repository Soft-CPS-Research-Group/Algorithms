# Wave A TPPO Recovery

| Config path | UI run name | Purpose | Phases | Required commit/image |
| --- | --- | --- | --- | --- |
| `rbc_smart.yaml` | `tppo-recovery-wa-rbc-smart-s7` | Tuned Smart rule-based reference | deterministic evaluation |  |
| `rbc_community.yaml` | `tppo-recovery-wa-rbc-community-s7` | Community rule-based reference | deterministic evaluation |  |
| `tppo_plain.yaml` | `tppo-recovery-wa-tppo-plain-s7` | Plain TPPO control | PPO, deterministic evaluation |  |
| `tppo_plain_conservative.yaml` | `tppo-recovery-wa-tppo-plain-conservative-s7` | Lower initial TPPO action variance | PPO, deterministic evaluation |  |
| `tppo_bc_pretrain.yaml` | `tppo-recovery-wa-tppo-bc-pretrain-s7` | Smart-teacher pretraining without auxiliary BC | demonstration, PPO, deterministic evaluation |  |
| `tppo_bc_auxiliary.yaml` | `tppo-recovery-wa-tppo-bc-auxiliary-s7` | Smart-teacher auxiliary BC through PPO | demonstration, PPO, deterministic evaluation |  |

Required commit/image: blank until handoff. Use the final handoff Wave A SHA.

Each run must export the log, resolved YAML, KPI JSON/CSV, job ID, and image tag.
