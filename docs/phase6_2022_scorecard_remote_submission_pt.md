# Submissao Remota Scorecard 2022 Full-Year

Submissao criada em 2026-05-22 para a matriz oficial inicial do scorecard.

- Dataset: `citylearn_challenge_2022_phase_all_plus_evs_data_2026_05_21`
- Imagem: `sha-f20cc24`
- Janela: steps `0..8759`, `episode_time_steps: 8760`
- Baselines: 1 episodio full-year
- RL/MARL: 6 episodios full-year, seeds `123`, `456`, `789`
- MLflow: desligado
- Checkpoints: desligados
- BAU exports: desligados
- KPI export: por episodio; timeseries so no episodio final

O scorecard de decisao esta em `docs/community_optimization_success_scorecard_pt.md`.

## Jobs

| Job | ID | Host | Perfil |
|---|---|---|---|
| `scorecard-2022-random-server-shaf20cc24` | `e7974fab-5cc6-463e-9750-846355d94ea8` | server | CPU |
| `scorecard-2022-normal_no_battery-server-shaf20cc24` | `66ab1802-4969-42f7-b550-bc36434e4d82` | server | CPU |
| `scorecard-2022-normal-server-shaf20cc24` | `231ac838-6602-4b39-9404-b239c7beba0b` | server | CPU |
| `scorecard-2022-rbc_basic-cpu-shaf20cc24` | `44dd62ba-ad42-4a15-9713-6df29232bebc` | deucalion | `normal-x86`, `08:00:00` |
| `scorecard-2022-rbc_smart-cpu-shaf20cc24` | `5f4e9a33-c5be-469b-9186-0e54db6e871e` | deucalion | `normal-x86`, `08:00:00` |
| `scorecard-2022-maddpg_v48-s123-gpu-shaf20cc24` | `5cb0a237-d94f-47f2-915f-7a11bb139e2b` | deucalion | `normal-a100-80`, `48:00:00` |
| `scorecard-2022-maddpg_v48-s456-gpu-shaf20cc24` | `23ca7931-eace-425c-9b77-08377ffb7068` | deucalion | `normal-a100-80`, `48:00:00` |
| `scorecard-2022-maddpg_v48-s789-gpu-shaf20cc24` | `b24942f3-f518-4ae2-9687-0a666edab4c9` | deucalion | `normal-a100-80`, `48:00:00` |
| `scorecard-2022-matd3_service_storage_guard-s123-gpu-shaf20cc24` | `29611cb4-9b6f-495e-9a7e-3de4af4464eb` | deucalion | `normal-a100-80`, `48:00:00` |
| `scorecard-2022-matd3_service_storage_guard-s456-gpu-shaf20cc24` | `442f5b95-e370-4e83-93b6-a186d73fa33b` | deucalion | `normal-a100-80`, `48:00:00` |
| `scorecard-2022-matd3_service_storage_guard-s789-gpu-shaf20cc24` | `66b9a0af-6568-4ebd-804e-cef5a2ee4bcb` | deucalion | `normal-a100-80`, `48:00:00` |
| `scorecard-2022-matd3_storage_guard-s123-gpu-shaf20cc24` | `6b23db83-6fae-43e1-a9e8-4dc20a1f1188` | deucalion | `normal-a100-80`, `48:00:00` |
| `scorecard-2022-matd3_storage_guard-s456-gpu-shaf20cc24` | `b4c892b3-c5bd-4aa7-9f8f-7789a302fb25` | deucalion | `normal-a100-80`, `48:00:00` |
| `scorecard-2022-matd3_storage_guard-s789-gpu-shaf20cc24` | `0307294e-936f-4b2b-a919-0fb449e4f223` | deucalion | `normal-a100-80`, `48:00:00` |

## Notas

- O orchestrator recebe `image_tag`, nao digest OCI. O tag usado foi `sha-f20cc24`, confirmado como Docker e SIF ready antes da submissao.
- Existem jobs antigos `sha-60ff99c` ainda no orchestrator. Esta submissao e a matriz limpa da imagem atual.
- Os manifests locais completos foram gravados em `runs/remote_configs/phase6_2022_scorecard_2026_05_22/`, mas `runs/` e ignorado pelo git.

## Redistribuicao CPU

Depois da submissao inicial, os seeds secundarios RL/MARL foram retirados da fila
GPU e redistribuidos para aproveitar `server` e Deucalion CPU.

Mantidos em GPU:

| Job | ID |
|---|---|
| `scorecard-2022-maddpg_v48-s123-gpu-shaf20cc24` | `5cb0a237-d94f-47f2-915f-7a11bb139e2b` |
| `scorecard-2022-matd3_service_storage_guard-s123-gpu-shaf20cc24` | `29611cb4-9b6f-495e-9a7e-3de4af4464eb` |
| `scorecard-2022-matd3_storage_guard-s123-gpu-shaf20cc24` | `6b23db83-6fae-43e1-a9e8-4dc20a1f1188` |

Cancelados na fila GPU e substituidos:

| Original GPU cancelado | Replacement |
|---|---|
| `23ca7931-eace-425c-9b77-08377ffb7068` | `a4b78202-a9aa-4b96-bfee-fe08cb785ab9` (`server`, MADDPG V48 seed 456) |
| `442f5b95-e370-4e83-93b6-a186d73fa33b` | `b48265d7-cc70-48ec-bb9b-bf234c85b22e` (`server`, MATD3 service/storage guard seed 456) |
| `b4c892b3-c5bd-4aa7-9f8f-7789a302fb25` | `7aadcab5-cf4b-407f-9d05-af1cb5683899` (`server`, MATD3 storage guard seed 456) |
| `b24942f3-f518-4ae2-9687-0a666edab4c9` | `177c29f7-7b10-4186-b84f-743cbfd8ee08` (`deucalion CPU`, MADDPG V48 seed 789) |
| `66b9a0af-6568-4ebd-804e-cef5a2ee4bcb` | `bbd3f782-bd13-4a56-bc91-89924733250f` (`deucalion CPU`, MATD3 service/storage guard seed 789) |
| `0307294e-936f-4b2b-a919-0fb449e4f223` | `0bec8050-ef1d-4da3-8fcb-d46fa2edffa0` (`deucalion CPU`, MATD3 storage guard seed 789) |

Configs de replacement:

- `configs/experiments/phase6_2022_full_year_scorecard/redistribution_manifest.csv`;
- `configs/experiments/phase6_2022_full_year_scorecard/redistribution_manifest.json`.

Jobs antigos `fresh-2022` com `sha-60ff99c` foram marcados como superseded:

- stop requested: `20b6a1b4-7046-48f3-937b-f74c718d9ffa`, `128fa0b7-b5fc-4e8c-a66e-95b0c2d3ea63`;
- canceled: `27172596-9674-427c-8b34-35402194b043`, `37538282-83e7-4355-9f52-83766a70b701`, `c1f41ec1-8532-4676-a891-dbeff9ee3781`.
