# Phase 10 Scorecard - Baselines e Gate Neural

Data: 2026-05-27.

Fonte local: `runs/remote_results/phase10_wave01_fixpath_20260527/summary.csv`.
Tabela limpa: `docs/phase10_scorecard_clean.csv`.
Imagem remota: `calof/opeva_simulator:sha-f3a1360`. Simulador: `softcpsrecsimulator==1.1.0`.
Dataset: `citylearn_challenge_2022_phase_all_plus_evs`.

## Decisao

`RBCBasicPolicy` e a baseline operacional alvo a bater nesta fase. Foi o melhor compromisso full-year entre custo, servico EV e uso moderado de bateria. `RBCSmartPolicy` fica como baseline alternativa forte, mas nesta execucao ficou ligeiramente pior em custo e com mais throughput de bateria. `RandomPolicy` fica apenas como sanity check, porque baixa custo abusando/aleatorizando acoes e falha muito servico EV.

`cost_ratio_to_bau` e `cost_delta_to_bau_eur` estao indisponiveis nesta scorecard porque `include_business_as_usual=false` foi desligado de proposito para poupar export/runtime. Nesta fase a comparacao principal e contra `RBCBasicPolicy`.

## Baselines Full-Year

| Rank | Key | Algorithm | Job ID | Config | Cost EUR | EV min feasible | EV within tol. feasible | Electrical viol. kWh | Import kWh | Export kWh | Battery throughput kWh | V2G export kWh | Runtime s |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `random` | `RandomPolicy` | `b3e90171-1908-4b58-89cf-bf1cf93781fb` | `phase10_w1_baselines_2022_random_8760steps_sha_f3a1360.yaml` | 10938.6 | 0.157 | 0.070 | 0.0 | 137947 | 72916 | 188413 | 94618 | 362.0 |
| 2 | `rbc_basic` | `RBCBasicPolicy` | `2e8fb34a-4d71-471f-8af8-7bd5af710fdf` | `phase10_w1_baselines_2022_rbc_basic_8760steps_sha_f3a1360.yaml` | 17675.9 | 1.000 | 0.445 | 0.0 | 164005 | 58438 | 11760 | 0 | 396.7 |
| 3 | `rbc_smart` | `RBCSmartPolicy` | `e4ef1302-2f43-4238-aebf-5591df35f75a` | `phase10_w1_baselines_2022_rbc_smart_8760steps_sha_f3a1360.yaml` | 17884.3 | 1.000 | 0.437 | 0.0 | 159158 | 52350 | 24510 | 1 | 400.0 |
| 4 | `rbc_community` | `RBCCommunityPolicy` | `4f721982-520c-4708-a880-4f0452301703` | `phase10_w1_baselines_2022_rbc_community_8760steps_sha_f3a1360.yaml` | 18812.4 | 1.000 | 0.437 | 0.0 | 160391 | 51758 | 46826 | 1 | 412.0 |
| 5 | `normal_no_battery` | `NormalNoBatteryPolicy` | `7bb8df05-1287-47ca-8cd0-f8a4645e6c66` | `phase10_w1_baselines_2022_normal_no_battery_8760steps_sha_f3a1360.yaml` | 21625.7 | 1.000 | 0.052 | 0.0 | 181676 | 56635 | 0 | 0 | 340.2 |
| 6 | `normal` | `NormalPolicy` | `b0806602-4fe7-4cc7-a736-60cdb4c4ea7f` | `phase10_w1_baselines_2022_normal_8760steps_sha_f3a1360.yaml` | 21931.9 | 1.000 | 0.052 | 0.0 | 175128 | 47884 | 24193 | 0 | 369.6 |

## Gate Runs

| Key | Algorithm | Job ID | Device | Steps | Cost EUR | EV min feasible | EV within tol. feasible | Runtime s | Resultado |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `rbc_smart` | `RBCSmartPolicy` | `36550194-3d5d-441d-8d1f-b1b824255524` | `cpu` | 512 | 1271.9 | 1.000 | 0.327 | 66.4 | `finished` |
| `rbc_community` | `RBCCommunityPolicy` | `428092aa-e133-4e98-add6-f35d59cf3a6e` | `cpu` | 512 | 1311.5 | 1.000 | 0.327 | 65.0 | `finished` |
| `maddpg` | `MADDPG` | `021158ef-b590-45fc-a5aa-54c144918829` | `cuda` | 512 | 808.3 | 0.252 | 0.063 | 82.3 | `finished` |

## Wave 2 Neural Smoke

Fonte local: `runs/remote_results/phase10_wave2_neural_smoke_20260527/summary.csv`.

Todos os jobs da Wave 2 terminaram com `exit_code=0`, `slurm_state=COMPLETED`, `simulation_data_available=True`, KPI exportado em `simulation_data/exported_kpis.csv`, artifact sync sem falhas, e logs com CUDA no A100. Estes runs foram `4096` steps e servem para validar infraestrutura/update/export; nao sao comparaveis diretamente ao full-year `RBCBasicPolicy`.

| Key | Algorithm | Job ID | Device | Cost EUR | EV min feasible | EV tol feasible | Battery throughput kWh | V2G export kWh | Runtime s | Steps/s | Smoke verdict |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `maddpg` | `MADDPG` | `9da0836c-8cdc-473c-a1c7-3e22db1cd81f` | `cuda` | 7978.3 | 0.767 | 0.062 | 12787 | 10715 | 633.3 | 6.47 | PASS; viable W3 candidate |
| `matd3` | `MATD3` | `8aad07b6-ff9d-4936-a0a1-8668475d545a` | `cuda` | 7972.8 | 0.767 | 0.060 | 12632 | 10720 | 667.0 | 6.14 | PASS; viable W3 candidate |
| `masac` | `MASAC` | `21d9bb1c-9ea8-4c17-88f3-9eac6fc648f1` | `cuda` | 8298.3 | 0.449 | 0.209 | 89888 | 32253 | 722.8 | 5.67 | PASS infra; weak policy signal |
| `ippo` | `IPPO` | `37d8cca6-bff4-4aba-b2e7-be4534970598` | `cuda` | 7601.3 | 0.336 | 0.117 | 90962 | 35479 | 399.1 | 10.26 | PASS infra; weak policy signal |
| `mappo` | `MAPPO` | `8e8c6476-279a-4ff7-ae00-1503c0cef6fd` | `cuda` | 7390.2 | 0.294 | 0.109 | 91254 | 36696 | 430.7 | 9.51 | PASS infra; weak policy signal |

Wave 2 conclusion: `MADDPG` and `MATD3` are the only clear candidates to scale first. They have the strongest EV service signal and much lower battery throughput/V2G abuse than `MASAC`, `IPPO`, and `MAPPO`. `IPPO` and `MAPPO` are faster, but their first smoke policy heavily abuses storage/V2G and misses too much EV service. `MASAC` is slower and also has high battery/V2G use, so it should not be in the first Wave 3 budget unless used as a secondary diagnostic.

## Wave 3 Recommendation

Priority 1: scale `MADDPG` and `MATD3` to longer partial-year training first.

- Run `MADDPG` and `MATD3` at `8760` steps, `3` seeds each, A100, `64 GB`, `4 CPUs`, `06:00:00`.
- Keep export final only and BAU disabled.
- Keep resource guard enabled; current Wave 2 memory signal is safe.
- Gate criteria: `exit_code=0`, CUDA, export ok, `ev_min_acceptable_feasible_rate >= 0.75` on first full-year run, `electrical_violation_kwh=0`, and no explosive battery/V2G growth versus `RBCBasicPolicy`.

Priority 2: test reward/action guardrails before spending more PPO/SAC budget.

- Add an EV-service-first reward profile or stronger penalties for infeasible departures and V2G/storage abuse.
- Re-run a short `4096`-step smoke for `MASAC`, `IPPO`, `MAPPO` only after reward/action guardrails change.
- Do not judge these by low cost until EV service and V2G are controlled.

Priority 3: if `MADDPG`/`MATD3` still show good EV service after full-year smoke, launch multi-seed training waves with larger step budgets and one conservative hyperparameter sweep.

## Wave 3 Submitted

Submetida em 2026-05-27 com configs locais em `runs/remote_configs/phase10_wave3_candidates_2026_05_27_sha_f3a1360/`.

Recursos por job: Deucalion `normal-a100-80`, `1` A100, `64 GB`, `4 CPUs`, `06:00:00`, imagem `calof/opeva_simulator:sha-f3a1360`, dataset `citylearn_challenge_2022_phase_all_plus_evs`, `8760` steps, BAU off, export final only.

| Key | Algorithm | Seed | Job ID | Job name | Initial status |
|---|---|---:|---|---|---|
| `maddpg` | `MADDPG` | 123 | `883676c1-3a37-4c6c-985d-e03168bdb5e3` | `p10-w3-maddpg-s123-full-shaf3a1360` | `dispatched` |
| `maddpg` | `MADDPG` | 456 | `df4e739f-2e76-46cb-995a-19568331d9f7` | `p10-w3-maddpg-s456-full-shaf3a1360` | `queued` |
| `maddpg` | `MADDPG` | 789 | `5f654836-91bb-4ed9-bfe8-ad6653e98d0b` | `p10-w3-maddpg-s789-full-shaf3a1360` | `queued` |
| `matd3` | `MATD3` | 123 | `2ef5c880-c7a7-46d1-a948-8f85d9d5b4eb` | `p10-w3-matd3-s123-full-shaf3a1360` | `queued` |
| `matd3` | `MATD3` | 456 | `6b7483f9-1f75-4968-86f8-2fe78deafde9` | `p10-w3-matd3-s456-full-shaf3a1360` | `queued` |
| `matd3` | `MATD3` | 789 | `b150218b-4996-49aa-b5a7-3e0349e4a1cc` | `p10-w3-matd3-s789-full-shaf3a1360` | `queued` |

## Wave 3b Guardrails Submitted

Submetida em 2026-05-27 com configs locais em `runs/remote_configs/phase10_wave3b_guardrails_2026_05_27_sha_f3a1360/`.

Objetivo: deixar correr durante a noite uma matriz pequena para perceber se o abuso de V2G/storage e falhas de servico EV respondem a reward/action guardrails antes de gastar budget em treino longo.

Recursos por job: Deucalion `normal-a100-80`, `1` A100, `64 GB`, `4 CPUs`, `06:00:00`, imagem `calof/opeva_simulator:sha-f3a1360`, dataset `citylearn_challenge_2022_phase_all_plus_evs`, BAU off, export final only.

Perfis:

- `service_v2g`: `ev_connected_deficit_penalty=90`, `ev_schedule_deficit_penalty=300`, `ev_departure_deficit_penalty=500`, `ev_departure_missed_penalty=1000`, `ev_v2g_service_penalty=220`, `battery_throughput_penalty=1.0`, `actor_ev_v2g_action_l2_penalty=0.02`, `actor_storage_action_l2_penalty=0.005`, `actor_action_saturation_penalty=0.01`.
- `service_heavy`: `ev_connected_deficit_penalty=180`, `ev_schedule_deficit_penalty=600`, `ev_departure_deficit_penalty=1200`, `ev_departure_missed_penalty=2000`, `ev_v2g_service_penalty=400`, `battery_throughput_penalty=1.2`, `actor_ev_v2g_action_l2_penalty=0.05`, `actor_storage_action_l2_penalty=0.01`, `actor_action_saturation_penalty=0.02`.

| Profile | Algorithm | Seed | Steps | Job ID | Job name | Initial status |
|---|---|---:|---:|---|---|---|
| `service_v2g` | `MADDPG` | 123 | 8760 | `62668f47-5d25-4c5f-ba31-45726ba9631e` | `p10-w3b-maddpg-service-v2g-s123-full-shaf3a1360` | `queued` |
| `service_heavy` | `MADDPG` | 123 | 8760 | `c055bf52-aed2-4bdb-8d5d-26a4160d0ca5` | `p10-w3b-maddpg-service-heavy-s123-full-shaf3a1360` | `queued` |
| `service_v2g` | `MATD3` | 123 | 8760 | `ec5cd702-2c1d-49b6-832a-c7f5a6eeb305` | `p10-w3b-matd3-service-v2g-s123-full-shaf3a1360` | `queued` |
| `service_heavy` | `MATD3` | 123 | 8760 | `42a2d745-a690-4606-9a88-8aad2833568b` | `p10-w3b-matd3-service-heavy-s123-full-shaf3a1360` | `queued` |
| `service_v2g` | `MASAC` | 123 | 4096 | `6ac21239-d3a0-497d-b1b8-e820646157fc` | `p10-w3b-masac-service-v2g-s123-4096-shaf3a1360` | `queued` |
| `service_v2g` | `IPPO` | 123 | 4096 | `7046a6e5-3611-484b-b898-c2f6341061a3` | `p10-w3b-ippo-service-v2g-s123-4096-shaf3a1360` | `queued` |
| `service_v2g` | `MAPPO` | 123 | 4096 | `90321e71-fe66-4bb5-8745-02e46ba1dceb` | `p10-w3b-mappo-service-v2g-s123-4096-shaf3a1360` | `queued` |

## Wave 3 / 3b Results

Fonte local: `runs/remote_results/phase10_wave3_and_3b_analysis_20260527/combined_summary.csv`.

Todos os jobs terminaram com `exit_code=0`, `slurm_state=COMPLETED`, export final only e `simulation_data/exported_kpis.csv` disponivel. A queue ficou vazia antes da Wave 4.

Resumo full-year contra `RBCBasicPolicy` (`17675.9 EUR`, EV min `1.000`, EV tol `0.445`, battery throughput `11760 kWh`, V2G `0 kWh`):

| Wave | Profile | Alg | Seed | Job ID | Cost EUR | Delta vs RBC | EV min | EV tol | Battery kWh | V2G kWh | Runtime s | steps/s |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `w3` | `base` | `MADDPG` | 123 | `883676c1-3a37-4c6c-985d-e03168bdb5e3` | 17361.9 | -314.0 | 0.846 | 0.047 | 12941 | 14930 | 859.0 | 10.20 |
| `w3` | `base` | `MADDPG` | 456 | `df4e739f-2e76-46cb-995a-19568331d9f7` | 19285.3 | +1609.4 | 0.911 | 0.050 | 12932 | 8269 | 877.5 | 9.98 |
| `w3` | `base` | `MATD3` | 456 | `6b7483f9-1f75-4968-86f8-2fe78deafde9` | 19397.4 | +1721.5 | 0.907 | 0.049 | 12848 | 8600 | 1054.9 | 8.30 |
| `w3` | `base` | `MATD3` | 123 | `2ef5c880-c7a7-46d1-a948-8f85d9d5b4eb` | 19675.0 | +1999.1 | 0.894 | 0.048 | 12756 | 10297 | 1090.2 | 8.04 |
| `w3b` | `service_heavy` | `MADDPG` | 123 | `c055bf52-aed2-4bdb-8d5d-26a4160d0ca5` | 20576.9 | +2901.0 | 0.943 | 0.050 | 28167 | 5799 | 959.5 | 9.13 |
| `w3b` | `service_v2g` | `MATD3` | 123 | `ec5cd702-2c1d-49b6-832a-c7f5a6eeb305` | 20783.8 | +3107.9 | 0.943 | 0.049 | 18876 | 5805 | 1051.5 | 8.33 |
| `w3b` | `service_v2g` | `MADDPG` | 123 | `62668f47-5d25-4c5f-ba31-45726ba9631e` | 20787.3 | +3111.4 | 0.943 | 0.050 | 23779 | 5802 | 948.8 | 9.23 |
| `w3` | `base` | `MADDPG` | 789 | `5f654836-91bb-4ed9-bfe8-ad6653e98d0b` | 20864.6 | +3188.7 | 0.934 | 0.051 | 14536 | 6099 | 863.5 | 10.14 |
| `w3b` | `service_heavy` | `MATD3` | 123 | `42a2d745-a690-4606-9a88-8aad2833568b` | 20951.7 | +3275.8 | 0.943 | 0.049 | 24758 | 5797 | 1027.8 | 8.52 |
| `w3` | `base` | `MATD3` | 789 | `b150218b-4996-49aa-b5a7-3e0349e4a1cc` | 21096.8 | +3420.9 | 0.939 | 0.050 | 13713 | 5869 | 1024.9 | 8.55 |

Resumo 4096-step dos algoritmos secundarios com guardrails:

| Profile | Alg | Job ID | Cost EUR | EV min | EV tol | Battery kWh | V2G kWh | Runtime s | steps/s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `service_v2g` | `MASAC` | `6ac21239-d3a0-497d-b1b8-e820646157fc` | 10172.2 | 0.798 | 0.133 | 89275 | 16992 | 574.2 | 7.13 |
| `service_v2g` | `IPPO` | `7046a6e5-3611-484b-b898-c2f6341061a3` | 6900.5 | 0.240 | 0.097 | 91601 | 38909 | 420.8 | 9.73 |
| `service_v2g` | `MAPPO` | `90321e71-fe66-4bb5-8745-02e46ba1dceb` | 6688.3 | 0.221 | 0.097 | 91619 | 39572 | 434.1 | 9.44 |

Conclusao:

- Infra esta boa: CUDA/export/sync estaveis e full-year em A100 ficou na ordem de `8-10 steps/s`.
- Nenhum neural bate ainda `RBCBasicPolicy` em criterio operacional completo. O unico custo abaixo da baseline foi `MADDPG` seed 123, mas com EV min `0.846`, EV tol `0.047` e V2G alto.
- Os guardrails manuais funcionaram parcialmente: EV min subiu para ~`0.943` e V2G caiu para ~`5800 kWh`, mas o custo ficou `+2.9k` a `+3.3k EUR` e o throughput de bateria subiu demasiado.
- O bloqueio principal agora nao e RAM nem velocidade; e reward/servico EV. O `ev_within_tolerance_feasible_rate` continua ~`0.05`, muito abaixo de `RBCBasicPolicy` (`0.445`).
- `MASAC` melhorou EV min no smoke com guardrail, mas continua com battery/V2G abusivo. `IPPO` e `MAPPO` continuam fora para treino serio nesta fase.

## Wave 4 Reward Profiles Submitted

Submetida em 2026-05-27 com configs locais em `runs/remote_configs/phase10_wave4_reward_profiles_2026_05_27_sha_f3a1360/`.

Objetivo: testar os reward profiles registados que atacam diretamente precisao de servico EV, deadline pressure e objetivos de pico/export. Nesta wave, `reward_function_kwargs={}` de proposito, para nao sobrescrever os defaults dos perfis nomeados com os kwargs base antigos.

Recursos por job: Deucalion `normal-a100-80`, `1` A100, `64 GB`, `4 CPUs`, `06:00:00`, imagem `calof/opeva_simulator:sha-f3a1360`, dataset `citylearn_challenge_2022_phase_all_plus_evs`, `8760` steps, BAU off, export final only.

| Profile | Reward function | Algorithm | Seed | Job ID | Job name | Initial status |
|---|---|---|---:|---|---|---|
| `v46_precision` | `CostServiceCommunityFeasiblePrecisionRewardV46` | `MADDPG` | 123 | `12476efd-479f-41fa-8927-4250d0d37221` | `p10-w4-maddpg-v46-precision-s123-full-shaf3a1360` | `dispatched` |
| `v50_deadline` | `CostServiceCommunityDeadlineValueRewardV50` | `MADDPG` | 123 | `26ef0c1a-e286-401a-a709-cad60b3d5515` | `p10-w4-maddpg-v50-deadline-s123-full-shaf3a1360` | `queued` |
| `v52_peak_deadline` | `CostServiceCommunityPeakDeadlineRewardV52` | `MADDPG` | 123 | `682c49db-9c6e-407b-ad9c-db202379e37d` | `p10-w4-maddpg-v52-peak-deadline-s123-full-shaf3a1360` | `queued` |
| `v46_precision` | `CostServiceCommunityFeasiblePrecisionRewardV46` | `MATD3` | 123 | `6a7415a2-a6c9-419d-b254-4fe3090bb2a5` | `p10-w4-matd3-v46-precision-s123-full-shaf3a1360` | `queued` |
| `v50_deadline` | `CostServiceCommunityDeadlineValueRewardV50` | `MATD3` | 123 | `fb9a3c27-1158-4c0e-8d80-32b11c0ccb8e` | `p10-w4-matd3-v50-deadline-s123-full-shaf3a1360` | `queued` |
| `v52_peak_deadline` | `CostServiceCommunityPeakDeadlineRewardV52` | `MATD3` | 123 | `a2879b72-d448-4952-90c0-d8fd84786cf1` | `p10-w4-matd3-v52-peak-deadline-s123-full-shaf3a1360` | `queued` |

## Wave 4 Reward Profiles Results

Fonte local: `runs/remote_results/phase10_wave4_reward_profiles_20260527/summary.csv`.

Todos os jobs terminaram com `exit_code=0`, `slurm_state=COMPLETED`, CUDA, export final only e `simulation_data/exported_kpis.csv` disponivel.

Resumo full-year contra `RBCBasicPolicy` (`17675.9 EUR`, EV min `1.000`, EV tol `0.445`, battery throughput `11760 kWh`, V2G `0 kWh`):

| Profile | Alg | Seed | Job ID | Cost EUR | Delta vs RBC | EV min | EV tol | Battery kWh | V2G kWh | Runtime s | steps/s |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v52_peak_deadline` | `MATD3` | 123 | `a2879b72-d448-4952-90c0-d8fd84786cf1` | 20087.7 | +2411.8 | 0.943 | 0.049 | 13047 | 5815 | 960.5 | 9.12 |
| `v50_deadline` | `MATD3` | 123 | `fb9a3c27-1158-4c0e-8d80-32b11c0ccb8e` | 20088.3 | +2412.4 | 0.943 | 0.049 | 13053 | 5815 | 971.8 | 9.01 |
| `v46_precision` | `MATD3` | 123 | `6a7415a2-a6c9-419d-b254-4fe3090bb2a5` | 20089.1 | +2413.2 | 0.943 | 0.049 | 13071 | 5816 | 986.5 | 8.88 |
| `v46_precision` | `MADDPG` | 123 | `12476efd-479f-41fa-8927-4250d0d37221` | 20099.3 | +2423.4 | 0.942 | 0.049 | 13127 | 5812 | 962.1 | 9.11 |
| `v50_deadline` | `MADDPG` | 123 | `26ef0c1a-e286-401a-a709-cad60b3d5515` | 21076.6 | +3400.7 | 0.942 | 0.049 | 12960 | 5812 | 950.5 | 9.22 |
| `v52_peak_deadline` | `MADDPG` | 123 | `682c49db-9c6e-407b-ad9c-db202379e37d` | 21108.1 | +3432.2 | 0.942 | 0.049 | 12957 | 5812 | 898.2 | 9.75 |

Diagnostico:

- Os reward profiles nomeados melhoraram o throughput de bateria face aos guardrails manuais da Wave 3b: ~`13.0 MWh`, perto dos `12.8-14.5 MWh` da Wave 3 base e bastante melhor que os `18.9-28.2 MWh` da Wave 3b.
- O V2G ficou controlado no mesmo patamar dos melhores runs anteriores, ~`5.8 MWh`, mas ainda longe do `0` da `RBCBasicPolicy`.
- O EV min ficou estavel em ~`0.942-0.943`, mas o `ev_within_tolerance_feasible_rate` continua preso em ~`0.049`, muito abaixo da baseline `RBCBasicPolicy` (`0.445`).
- A falha principal e excesso de servico EV, nao falta de carga: no melhor exemplo inspecionado, `138/2774` eventos viaveis ficaram dentro da tolerancia; o deficit medio foi `0.0275`, mas o surplus medio foi `0.1482`.
- V46/V50/V52 quase nao mudam comportamento dentro de cada algoritmo. Isto indica que mais reward shaping, sozinho, tem pouco leverage neste setup full-year de 1 episodio.

## Community Scorecard

Tabela limpa: `docs/phase10_community_scorecard_clean.csv`.

Esta fase nao deve ser avaliada so por custo e EV. A comparacao comunitaria mostra que `RBCSmartPolicy` e `RBCCommunityPolicy` sao baselines importantes para pico, import/export, autoconsumo e emissoes.

| Run | Cost EUR | Import kWh | Export kWh | Solar self-cons. | Peak daily | Peak all-time | Ramping | Emissions kgCO2 | Deferrable service |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `rbc_basic` | 17675.9 | 164005 | 58438 | 0.456 | 2.095 | 1.689 | 1.697 | 29560 | 1.000 |
| `rbc_smart` | 17884.3 | 159158 | 52350 | 0.483 | 1.990 | 1.584 | 1.680 | 29251 | 1.000 |
| `rbc_community` | 18812.4 | 160391 | 51758 | 0.491 | 1.937 | 1.503 | 1.618 | 30272 | 1.000 |
| `normal` | 21931.9 | 175128 | 47884 | 0.505 | 2.264 | 1.691 | 1.572 | 32235 | 1.000 |
| `matd3_v52_peak_deadline_s123` | 20087.7 | 172636 | 57163 | 0.455 | 2.744 | 1.963 | 1.796 | 32354 | 0.060 |
| `maddpg_v52_peak_deadline_s123` | 21108.1 | 178561 | 57158 | 0.455 | 2.323 | 1.697 | 1.669 | 33118 | 0.984 |

Leitura:

- `RBCBasicPolicy` continua baseline alvo por custo + EV.
- `RBCSmartPolicy` e melhor que `RBCBasicPolicy` em import kWh, export kWh, autoconsumo solar, picos, ramping e emissoes, com custo so `+208 EUR`.
- `RBCCommunityPolicy` e a melhor baseline para peak daily/all-time e autoconsumo, mas paga `+1136 EUR` face a `RBCBasicPolicy`.
- A melhor Wave 4 por custo (`MATD3 v52`) ainda e pior que as RBCs em import, picos, emissoes e deferrable service.
- `MADDPG v52` preserva melhor deferrable service que MATD3, mas fica pior em custo, import e emissoes.

Conclusao Wave 4:

- Infra, velocidade e memoria estao suficientemente boas para continuar.
- `MADDPG` e `MATD3` continuam a ser os candidatos principais, mas nao estao prontos para mais sweep cego.
- O proximo bloqueio e controlo fino de EV: parar de carregar quando ja passou o target/tolerancia e reduzir V2G/over-service sem perder feasibility.

## Proximos Passos Recomendados

1. Auditar localmente o encoding/action mapping EV: confirmar que o actor recebe `connected_ev_soc`, `connected_ev_required_soc_departure`, `connected_ev_soc_deficit`, `connected_ev_hours_until_departure`, `connected_ev_departure_urgency_24h`, `available_charge_action_normalized` e `available_discharge_action_normalized` em escala util.
2. Nao usar hard action guard como solucao principal de treino. Usar guard so como diagnostico, safety layer ou teacher para behavior cloning; caso contrario a rede aprende o mundo filtrado pelo guard em vez de aprender a acao certa.
3. Adicionar/validar features de controlo fino EV: erro assinado `required_soc - soc`, `connected_ev_soc_surplus`, `max_charge_to_target_action_normalized` e talvez flags de banda de tolerancia.
4. Testar warm-start/behavior cloning com `RBCBasicPolicy` ou `RBCSmartPolicy` para MADDPG/MATD3. O codigo ja suporta `initial_exploration_strategy: policy`, `warm_start_policy: RBCBasicPolicy` e `actor_behavior_cloning_source: warm_start_policy`; falta desenhar uma config conservadora e smoke local/remoto.
5. So depois disto lancar Wave 5: `MADDPG` e `MATD3`, seed 123, full-year, com BC/warm-start + `v52_peak_deadline` e sem hard guard no caminho principal. Criterio de avanco: EV tol subir claramente acima de `0.10` sem EV min cair, V2G abaixo de `5.8 MWh`, custo nao piorar face a Wave 4, e import/picos/autoconsumo nao degradarem face as RBCs.
6. Multi-seed e treino multi-episodio so fazem sentido depois de resolver a precisao EV; neste momento mais budget provavelmente so confirma o mesmo modo de falha.

## Criterios Para Wave 2

- Validar que cada algoritmo neural arranca em CUDA, faz updates, termina, exporta artifacts e sincroniza resultados.
- Nao usar custo isolado como criterio de sucesso em smoke. A metrica de gate para treino serio continua a ser servico EV primeiro, especialmente `ev_min_acceptable_feasible_rate`.
- Para runs candidatas, bater `RBCBasicPolicy` significa custo abaixo de `17675.9 EUR`, `ev_min_acceptable_feasible_rate` perto de `1.0`, violacoes eletricas `0`, e throughput/V2G sem abuso operacional.

## Atualizacao Wave 5 / W6

Fonte Wave 5: `runs/remote_results/phase10_wave5_completed_20260528/summary.csv`.
Scorecard W5 contra RBCSmart:

- Tabela limpa: `docs/phase10_wave5_candidate_scorecard_clean.csv`.
- Markdown: `docs/phase10_wave5_candidate_scorecard_pt.md`.

Resultado: nenhuma run W5 passou os gates RBCSmart. O padrao confirma o
diagnostico anterior:

- Sem BC/warm-start, custo pode descer em janelas curtas, mas EV min e EV tol
  falham.
- Com BC/warm-start, EV min sobe ate perto do gate, mas EV tol continua baixa
  e bateria/V2G ficam altos.
- A falha ja nao parece ser mapping/encoding puro; o proximo passo e treino
  guiado + controlo de throughput.

Para W6, o alvo operacional passa a ser `RBCSmartPolicy`, nao por custo puro,
mas porque e a melhor referencia comunitaria equilibrada: menor import/export,
melhor autoconsumo, picos melhores e custo apenas ~`208 EUR` acima de
`RBCBasicPolicy`.

Plano implementado em `docs/phase10_w6_guided_training_plan_pt.md`.

Configs W6 geradas localmente em `runs/generated_configs/phase10_w6/`:

- `w6-smoke-local`: `8` configs, 4 recipes MADDPG com 2 seeds, `256`
  steps.
- `w6a-local`: `40` configs, incluindo RBCSmart/RBCCommunity nas quatro
  janelas e 4 recipes MADDPG com 2 seeds.
- `w6b-remote-smoke`: `4` configs MADDPG, A100, `64 GB`, `4 CPUs`, `4h`.
- `w6c-full-year`: `8` configs MADDPG/MATD3, A100, `96 GB`, `4 CPUs`, `12h`,
  watchdog ligado.

Regra: nao promover full-year W6C ate W6A/W6B mostrarem EV gate e custo de
janela contra RBCSmart.
