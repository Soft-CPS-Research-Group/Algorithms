# MADDPG - Fase 6C

Data: 2026-05-14

## Objetivo

Testar duas correcoes antes de treino longo:

- corrigir baselines RBC para nao pouparem energia a custa de EV service;
- reduzir saturacao do actor MADDPG sem mexer no wrapper.

## Implementado

- `RBCBasicPolicy` e `RBCSmartPolicy` preservam taxa minima de servico EV.
- Baselines RBC ganharam `ev_service_target_soc` configuravel.
- `RBCSmartPolicy` evita V2G quando ainda existe deficit de servico.
- MADDPG ganhou:
  - `actor_update_interval`;
  - `target_policy_smoothing`;
  - `actor_action_l2_penalty`;
  - `actor_action_saturation_penalty`;
  - metricas de regularizacao/saturacao.
- `ev_connected_deficit_penalty` passou a `30.0` nos templates/configs para dar
  sinal denso de deficit EV durante a ligacao, nao apenas na partida.

Nota: os dois full-day benchmarks abaixo foram corridos antes de ativar este
sinal denso nos templates. Os KPIs fisicos mantêm-se validos; as componentes de
reward desses runs nao devem ser usadas para avaliar a nova reward densa.

## Benchmark 15s Full Day

Run:

- `runs/benchmarks/phase6c_15s_full_day_seed123`
- 15s, 1 dia, 5760 steps, seed 123
- 8/8 runs completas, 0 falhas

| policy | cost | EV success | EV deficit | EV charge | V2G | near high | near low |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | 81.517 | 0.000 | 0.727 | 408.233 | 210.929 | 0.012 | 0.008 |
| NormalNoBattery | 250.353 | 0.143 | 0.333 | 1494.876 | 0.000 | 0.194 | 0.038 |
| Normal | 255.665 | 0.143 | 0.333 | 1494.876 | 0.000 | 0.194 | 0.038 |
| RBCBasic rev1 | 167.486 | 0.000 | 0.399 | 932.504 | 0.000 | 0.080 | 0.038 |
| RBCSmart rev1 | 160.997 | 0.000 | 0.399 | 824.322 | 0.000 | 0.080 | 0.038 |
| MADDPG current | 92.081 | 0.000 | 0.603 | 451.748 | 160.803 | 0.164 | 0.263 |
| MADDPG anti_saturation | 55.695 | 0.000 | 0.767 | 163.927 | 96.811 | 0.052 | 0.074 |
| MADDPG anti_saturation_warm_rbc_basic | 119.719 | 0.000 | 0.689 | 614.167 | 77.415 | 0.070 | 0.047 |

Conclusao:

- anti-saturation reduz extremos de acoes;
- mas sozinho reduz tambem energia util de EV e piora service;
- warm-start ajuda carga EV, mas nao chega;
- custo baixo continua a significar under-service.

## Retune RBC EV Service

Run:

- `runs/benchmarks/phase6c_15s_rbc_service_retune_seed123`
- 15s, 1 dia, 5760 steps, seed 123
- 4/4 runs completas, 0 falhas

| policy | cost | EV success | EV deficit | EV charge | V2G |
|---|---:|---:|---:|---:|---:|
| NormalNoBattery | 250.353 | 0.143 | 0.333 | 1494.876 | 0.000 |
| Normal | 255.665 | 0.143 | 0.333 | 1494.876 | 0.000 |
| RBCBasic retuned | 182.894 | 0.000 | 0.384 | 1031.221 | 0.000 |
| RBCSmart retuned | 262.018 | 0.143 | 0.333 | 1488.795 | 0.000 |

Conclusao:

- `RBCSmartPolicy` retuned ja fica ao nivel do `Normal` em EV service e deve ser
  o baseline forte;
- `RBCBasicPolicy` fica como baseline intermédio mais economico, mas nao forte
  em EV service;
- no dataset 15s ha departures que parecem impossiveis mesmo com carga maxima
  desde o inicio, por isso `1/7` pode ser o teto desta janela/configuracao.

## Proximo Passo

Antes de treino longo:

1. repetir uma matriz curta com `RBCSmart retuned` como warm-start;
2. se EV service continuar 0, aumentar foco da reward em deficit conectado e
   penalizar V2G quando ha deficit;
3. so depois escalar para multi-seed/varias janelas.

Smoke pos-alteracao de reward:

- `runs/benchmarks/phase6c_maddpg_dense_reward_smoke`
- MADDPG `anti_saturation`, 192 steps, 1/1 completo, 0 falhas.

O harness tambem ficou preparado para testar `warm_rbc_smart` e
`anti_saturation_warm_rbc_smart`.
