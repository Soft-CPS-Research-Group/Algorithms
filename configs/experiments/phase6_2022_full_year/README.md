# Phase 6 2022 Full-Year Matrix

Corrected remote matrix for the hourly 2022 all-plus-EVs dataset.

These configs intentionally use the full available 2022 window:

- `simulation_start_time_step: 0`
- `simulation_end_time_step: 8759`
- `episode_time_steps: 8760`

The earlier `remote_20260520_*_2022_*` configs used `0..1999` and must be treated as short-window experiments, not annual comparisons.

Remote submission, 2026-05-21, image `sha-969d417`:

| Config | Job ID | Target |
|---|---:|---|
| `remote_20260521_fullyear_2022_random.yaml` | `21a973d1-d711-4936-bd54-022a3a47224a` | server |
| `remote_20260521_fullyear_2022_normal_no_battery.yaml` | `954b6b33-47cf-465a-9f82-3f42a12be321` | server |
| `remote_20260521_fullyear_2022_normal.yaml` | `56ab2d71-2ebd-4258-9e14-084418639bf5` | server |
| `remote_20260521_fullyear_2022_rbc_basic.yaml` | `f66339ec-fcb6-4544-aa0f-da0ee897073b` | deucalion CPU |
| `remote_20260521_fullyear_2022_rbc_smart.yaml` | `4ef488f8-c201-48ab-8f59-e795cbadf20f` | deucalion CPU |
| `remote_20260521_fullyear_2022_maddpg_v48_seed123.yaml` | `605cdb64-a9d6-49a8-9a8c-aca15d420e15` | deucalion GPU |
| `remote_20260521_fullyear_2022_maddpg_v48_seed456.yaml` | `0a313a57-73b6-4243-a9d1-838ab70ababe` | deucalion GPU |
| `remote_20260521_fullyear_2022_maddpg_v48_seed789.yaml` | `46a15c21-23cc-48d3-a84a-685a8424332b` | deucalion GPU |

MADDPG jobs run `6` full-year episodes each. Baselines run one full-year episode.
