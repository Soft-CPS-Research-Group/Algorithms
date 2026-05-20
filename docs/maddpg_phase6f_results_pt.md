# Fase 6F - Diagnostico MADDPG 0.6.6

Data: 2026-05-17.

Objetivo: voltar ao MADDPG depois de fechar baselines e simulador `0.6.6`,
separando tres problemas:

- se o MADDPG executa updates e altera os actors;
- se a reward esta a dar sinal correto;
- se a exploracao/actor satura acoes antes de aprender uma politica util.

## Runs Executadas

Diagnosticos pequenos:

- `runs/maddpg_diagnostics/phase6f_066_building15_base`
- `runs/maddpg_diagnostics/phase6f_066_building1_building15_base`

Matriz MADDPG antes do fix de reward:

- `runs/benchmarks/phase6f_066_maddpg_15s_variants`

Sanity check de reward:

- `runs/benchmarks/phase6f_066_rewardfix_rbc_sanity_v2`

Matriz MADDPG depois do fix de reward:

- `runs/benchmarks/phase6f_066_maddpg_15s_variants_rewardfix`

Matriz com departures EV reais:

- `runs/benchmarks/phase6f_066_maddpg_15s_departures_rewardfix`
- `runs/benchmarks/phase6f_066_maddpg_15s_ev_priority_bc`
- `runs/benchmarks/phase6f_066_maddpg_15s_ev_service_v2g_guard`
- `runs/benchmarks/phase6f_066_maddpg_15s_ev_service_v2g_guard_prioritized`

## Diagnostico Pequeno

O MADDPG atual executa updates e move os actors:

| Caso | Agentes | Replay | Actor delta L2 | Updates |
|---|---:|---:|---:|---|
| Building 15 | 1 | 2044 | 2.835 | sim |
| Building 1 + Building 15 | 2 | 2044 | 2.577 / 2.211 | sim |

Conclusao: nao ha falha basica de loop, replay, update ou target update. O
problema esta no sinal/escala/estabilidade quando se escala para o caso
completo.

## Problema Encontrado na Reward EV

No dataset `citylearn_three_phase_electrical_service_demo_15s_parquet`, o
charger `Building_10/charger_10_1` aparece ligado com departure desconhecido:

- na tabela entity: `connected_ev_departure_time_step = -1`;
- na observacao simplificada usada pela reward: `hours_until_departure = 0`.

A reward interpretava isto como departure falhado em todos os passos e aplicava
`ev_departure_missed_penalty = 250`. Isto dominava o treino e fazia parecer que
o MADDPG/RBC estavam a falhar EV service mesmo quando nao havia evento de
departure avaliavel.

Fix aplicado:

- departure markers negativos passam a ser tratados como departure desconhecido;
- se a observacao nao traz marker explicito de departure conhecido, a reward nao
  aplica o hard missed penalty so por `hours_until_departure == 0`;
- a reward continua a poder aplicar sinal denso/window quando ha deficit, mas
  deixa de aplicar o castigo terminal falso.

Teste novo:

- `test_cost_hard_constraint_reward_ignores_unknown_ev_departure_sentinel_from_entity_dict`
- `test_cost_hard_constraint_reward_ignores_unknown_ev_departure_sentinel_from_flat_names`

## Sanity Check RBC Depois do Fix

Run: `phase6f_066_rewardfix_rbc_sanity_v2`.

| Policy | Custo | EV missed reward mean | EV service reward mean | Reward total mean |
|---|---:|---:|---:|---:|
| RBCBasic | 45.528 | 0.000 | 0.538 | -0.541 |

Antes do fix, a mesma janela tinha `ev_departure_missed` artificial perto de
`2.03` em media, vindo de quatro snapshots com penalizacao `250`.

## MADDPG Depois do Fix

Run: `phase6f_066_maddpg_15s_variants_rewardfix`.

Janela curta: 15s, 2 episodios, 960 steps por episodio, seed 123. Esta janela
nao tem departures exportaveis nos KPIs (`ev_departure_count = 0`), por isso a
comparacao aqui e de reward/action diagnostics, nao KPI final.

| Variante | Custo | Near low | Near high | Action std | EV service mean | Reward mean |
|---|---:|---:|---:|---:|---:|---:|
| current | 0.702 | 0.409 | 0.012 | 0.359 | 0.0218 | -0.0227 |
| anti_saturation | 2.662 | 0.261 | 0.000 | 0.239 | 0.0183 | -0.0193 |
| anti_saturation_warm_rbc_basic | 0.911 | 0.240 | 0.012 | 0.346 | 0.0159 | -0.0170 |

Leitura:

- `current` tem custo baixo nesta janela mas satura demasiado no limite baixo.
- `anti_saturation` reduz saturacao e melhora reward media, mas aumenta custo.
- `anti_saturation_warm_rbc_basic` foi a melhor reward media nesta janela e
  reduziu EV service penalty, mantendo custo perto de `current`.

## Conclusoes

1. A Fase 6F encontrou e corrigiu um problema real da reward EV.
2. Os resultados MADDPG anteriores ao fix nao devem ser usados para escolher
   receita final.
3. A melhor receita curta por agora e:
   - `maddpg_v2_compact`;
   - anti-saturation;
   - critic update `per_agent`;
   - actor update interval `2`;
   - target policy smoothing;
   - reward normalization ligada;
   - warm-start com `RBCBasicPolicy`.
4. Ainda nao ha evidencia suficiente para mexer em replay buffer avancado ou
   LSTM. Primeiro temos de repetir com janela com departures reais e/ou mais
   episodios.

## Proximo Passo

## Janela 15s Com Departures

Janela: 15s, steps `0-2879`, 2 episodios, seed `123`. Esta janela tem 3
departures EV exportaveis, dos quais 2 sao marcados como target/minimo
fisicamente infeasible pelo simulador `0.6.6`. Portanto o KPI principal e:

- `ev_departure_min_acceptable_feasible_rate`

| Variante | Custo | EV feasible min | EV raw min | EV deficit mean | EV shortfall tol | EV neg frac | Storage neg frac |
|---|---:|---:|---:|---:|---:|---:|---:|
| `RBCBasic` | 61.514 | 1.000 | 0.333 | 0.0687 | 0.0354 | 0.000 | 0.000 |
| `RBCSmart` | 60.402 | 1.000 | 0.333 | 0.0687 | 0.0354 | 0.000 | 0.000 |
| `current` | 18.749 | 0.000 | 0.000 | 0.5506 | 0.5006 | 0.781 | 0.775 |
| `anti_saturation_warm_rbc_basic` | 16.799 | 0.000 | 0.333 | 0.4287 | 0.3845 | 0.636 | 0.797 |
| `ev_priority_bc_warm_rbc_basic` | 39.862 | 0.000 | 0.000 | 0.4799 | 0.4299 | 0.131 | 0.290 |
| `ev_service_v2g_guard_warm_rbc_basic` | 72.287 | 1.000 | 0.333 | 0.1321 | 0.0988 | 0.006 | 0.285 |
| `ev_service_v2g_guard_prioritized_warm_rbc_basic` | 64.284 | 1.000 | 0.333 | 0.3079 | 0.2746 | 0.006 | 0.237 |

Leitura:

1. O MADDPG original aprende a reduzir custo, mas faz isso sacrificando EV
   service: descarrega EVs e baterias, falhando ate o departure factivel.
2. Anti-saturation e behavior cloning melhoram distribuicao de acoes, mas sem
   penalizar V2G contra servico ainda nao chegam para cumprir EV.
3. A reward nova `ev_v2g_service_penalty` muda o comportamento certo: a fração
   de EV discharge cai para perto de zero e o KPI EV factivel passa para `1.0`.
4. O custo ainda fica pior que RBC nesta run curta. Isto e aceitavel como marco
   tecnico: primeiro recuperamos servico EV, depois voltamos a otimizar custo.
5. O replay ponderado por reward tambem cumpre EV factivel e baixa o custo face
   ao guard uniforme (`64.284` vs `72.287`), mas aumenta deficit medio dos EVs
   infeasible. Pode estar a sobre-amostrar eventos muito penalizados do Building
   15; precisa de tuning de `priority_fraction`/`priority_alpha`.

## Alteracoes Implementadas

- Reward:
  - `ev_v2g_service_penalty` configuravel;
  - penaliza descarga EV (`last_charged_kwh < 0`) quando o EV esta abaixo do
    target de servico ou atrasado face ao schedule fisicamente necessario;
  - nao aplica esta penalizacao quando a departure e desconhecida/sentinel.
- MADDPG:
  - regularizacao opcional de behavior cloning sobre acoes do replay;
  - logs de loss/regularizacao de behavior cloning;
  - variante benchmark `ev_service_v2g_guard_warm_rbc_basic`.
- Replay:
  - `RewardWeightedMultiAgentReplayBuffer`, mantendo transicoes joint
    multi-agente alinhadas;
  - variante benchmark
    `ev_service_v2g_guard_prioritized_warm_rbc_basic`.

## Proximo Passo

Continuar a partir de `ev_service_v2g_guard_warm_rbc_basic` e da variante
priorizada:

- repetir com 3 a 5 episodios para ver se o custo desce mantendo EV feasible;
- testar replay ponderado com `priority_fraction` menor (`0.2-0.35`) para nao
  sobre-focar eventos infeasible;
- testar actor/critic menores e LayerNorm so depois de confirmar estabilidade;
- passar para 2022 apenas depois de manter EV feasible no 15s em mais de uma
  seed.

## Fase 6F.1 - Custo Sob Gate EV

Run:

- `runs/benchmarks/phase6f1_066_maddpg_15s_cost_service_gpu`

Ambiente:

- simulador `softcpsrecsimulator==0.6.6`;
- PyTorch com CUDA ativo numa RTX 4080 Laptop;
- dataset 15s, steps `0-2879`, 3 episodios, seed `123`;
- `maddpg_v2_compact`;
- actor `[128, 64]`, critic `[256, 128]`;
- warmup inicial com `RBCBasicPolicy`;
- noise ativo depois do warmup, com `sigma` a decair ate `0.03`.

Alteracoes testadas:

- `CostServiceGuardRewardV2`: perfil nomeado para proteger EV service;
- `CostServiceCostBalancedRewardV3`: perfil nomeado mais leve em EV service e
  com `battery_throughput_penalty`;
- `RewardWeightedMultiAgentReplayBuffer` com:
  - `priority_mode: negative_reward`;
  - `priority_fraction: 0.25`;
  - `priority_alpha: 0.7`;
  - `priority_max: 100.0`;
- decay configuravel de behavior cloning:
  - peso inicial `0.05` ou `0.04`;
  - peso minimo `0.01`;
  - decay depois do warmup.

Resultados:

| Variante | Custo | EV feasible min | EV raw min | EV success feasible | Deficit mean | Shortfall tol | EV neg frac | Storage neg frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `RBCBasic` | 61.514 | 1.000 | 0.333 | 0.000 | 0.0687 | 0.0354 | 0.000 | 0.000 |
| `RBCSmart` | 60.402 | 1.000 | 0.333 | 0.000 | 0.0687 | 0.0354 | 0.000 | 0.000 |
| `service_guard_v2_warm_rbc_basic` | 72.263 | 1.000 | 0.667 | 1.000 | 0.2123 | 0.1929 | 0.098 | 0.153 |
| `service_guard_v2_prioritized_low_bc_decay_warm_rbc_basic` | 69.737 | 1.000 | 0.333 | 1.000 | 0.2658 | 0.2325 | 0.080 | 0.169 |
| `cost_balanced_v3_prioritized_low_bc_decay_warm_rbc_basic` | 80.917 | 1.000 | 0.333 | 1.000 | 0.0857 | 0.0524 | 0.106 | 0.070 |

Leitura:

1. A 6F.1 confirmou que o MADDPG ja consegue preservar o gate principal
   `ev_departure_min_acceptable_feasible_rate = 1.0`.
2. O replay priorizado baixo com BC decay reduziu custo face ao V2 simples
   (`69.737` vs `72.263`), mas ainda fica longe do `RBCSmart` (`60.402`).
3. A V3 cost-balanced reduziu muito o deficit EV medio, mas ficou mais cara
   (`80.917`). Nao e boa candidata nesta forma.
4. O problema atual ja nao e falta de update, CUDA, replay basico ou KPI EV:
   e calibracao do tradeoff entre servico EV, uso de storage, excesso de carga
   e custo.
5. O `EV success feasible` fica `1.0` nas variantes MADDPG porque elas tendem a
   passar o target estrito no unico departure factivel; isto nao significa KPI
   global melhor, porque o custo e o comportamento de storage pioram.

Conclusao:

- melhor candidata atual: `service_guard_v2_prioritized_low_bc_decay_warm_rbc_basic`;
- rejeitar por agora: `cost_balanced_v3_prioritized_low_bc_decay_warm_rbc_basic`;
- proxima fase deve focar uma reward V4 ou config equivalente que mantenha o
  gate EV, mas desencoraje sobre-servico/armazenamento caro.

Testes:

- `pytest -q`: `173 passed`.

## Proximo Passo 6F.2

O caminho seguinte nao deve ser LSTM nem rede maior. Antes disso, testar:

- reward `CostServiceBandRewardV4` ou equivalente:
  - manter penalidade forte para deficit EV factivel;
  - reduzir incentivo a carregar muito acima do minimo aceitavel;
  - penalizar excesso/throughput de storage de forma um pouco mais visivel;
  - manter V2G guard so quando existe risco real de servico;
- variantes com `priority_fraction` ainda menor (`0.10-0.20`);
- BC decay mais cedo ou peso inicial mais baixo (`0.03 -> 0.005`);
- separar KPIs de custo por storage/EV para perceber onde nasce o custo extra;
- so depois repetir 2-3 seeds e escalar para 2022.

## Fase 6F.2 - Reward Comunitaria e Banda EV

Data: 2026-05-18.

Implementado para teste:

- `CostServiceCommunityBandRewardV4`;
- `local_cost_weight=0.0` e `community_settlement_cost_weight=1.0`, para treinar
  contra uma aproximacao do settlement comunitario em vez de duplicar custo
  local bruto;
- `community_local_price_ratio=0.8` e `community_grid_export_price=0.0`,
  alinhados com o dataset 15s atual;
- penalizacao suave de sobre-servico EV:
  `ev_over_service_penalty` sobre SOC acima de `required_soc + tolerance`;
- `battery_throughput_penalty=0.05`, para reduzir cycling caro de storage;
- variantes benchmark:
  - `community_band_v4_warm_rbc_basic`;
  - `community_band_v4_prioritized_tiny_bc_decay_warm_rbc_basic`.

Hipotese:

- o KPI EV primario continua a ser
  `ev_departure_min_acceptable_feasible_rate`;
- `ev_departure_within_tolerance_feasible_rate` passa a ser diagnostico
  importante para distinguir cumprir o utilizador de sobrecarregar EV;
- a V4 deve aproximar melhor o objetivo economico real quando ha energia local
  entre membros da comunidade;
- se a V4 falhar EV feasible, o peso de EV service ainda esta baixo;
- se cumprir EV mas ficar cara, olhar primeiro para storage throughput e
  EV/storage action fractions antes de mexer em redes.

Run:

- `runs/benchmarks/phase6f2_066_maddpg_15s_community_band_gpu`
- dataset 15s, steps `0-2879`, 3 episodios, seed `123`;
- actor `[128, 64]`, critic `[256, 128]`;
- warmup inicial com `RBCBasicPolicy`;
- CUDA ativo;
- `5/5` jobs completos, `0` falhas.

Resultados:

| Variante | Custo | EV feasible min | EV within feasible | EV success feasible | Deficit mean | Surplus mean | EV neg frac | Storage neg frac | Storage pos frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `RBCBasic` | 61.514 | 1.000 | 1.000 | 0.000 | 0.0687 | 0.0000 | 0.000 | 0.000 | 0.164 |
| `RBCSmart` | 60.402 | 1.000 | 1.000 | 0.000 | 0.0687 | 0.0000 | 0.000 | 0.000 | 0.000 |
| `service_guard_v2_prioritized_low_bc_decay_warm_rbc_basic` | 75.022 | 1.000 | 1.000 | 0.000 | 0.2969 | 0.0000 | 0.170 | 0.097 | 0.851 |
| `community_band_v4_warm_rbc_basic` | 86.571 | 1.000 | 0.000 | 1.000 | 0.0511 | 0.0500 | 0.041 | 0.133 | 0.812 |
| `community_band_v4_prioritized_tiny_bc_decay_warm_rbc_basic` | 69.794 | 1.000 | 0.000 | 1.000 | 0.4223 | 0.0500 | 0.187 | 0.106 | 0.842 |

Leitura:

1. A V4 nao resolveu ainda o objetivo final: continua pior que `RBCSmart`.
2. A melhor V4 foi a priorizada com BC decay leve: `69.794`, melhor que a V2
   equivalente nesta run (`75.022`), mas ainda acima de `RBCSmart` (`60.402`).
3. O gate principal continua preservado em todas as variantes MADDPG:
   `ev_departure_min_acceptable_feasible_rate = 1.0`.
4. A V4 mudou o comportamento EV: passa a cumprir target estrito factivel
   (`success_feasible=1.0`), mas falha `within_tolerance_feasible=0.0`, sinal de
   sobre/fora-da-banda no departure factivel.
5. O maior problema continua a ser uso de storage: MADDPG tem storage positivo
   acima de `0.8`, enquanto `RBCSmart` fica em `0.0` nesta janela.
6. A penalizacao de over-service aparece nos logs, mas ainda e pequena face ao
   custo/EV service; pode precisar de tuning ou de ser aplicada mais perto da
   departure para evitar incentivar V2G/oscilar.

Conclusao:

- manter `CostServiceCommunityBandRewardV4` como candidata, porque reduziu custo
  face a V2 nesta run quando combinada com replay/BC decay;
- rejeitar a variante V4 sem replay priorizado nesta forma (`86.571`);
- o proximo ajuste deve focar storage discipline e target-band EV, nao redes
  maiores;
- antes de escalar para 2022, testar uma V4.1 com:
  - `battery_throughput_penalty` mais forte ou dependente de energia;
  - penalizacao de EV over-service mais relevante perto da departure;
  - menor liberdade de V2G durante treino inicial, sem hard-code no wrapper;
  - repetir seed para confirmar se `69.794` e robusto.

## Fase 6F.2 - Warm Train, Phase-Out e EV Guard

Data: 2026-05-18.

Motivacao: a receita anterior aprendia durante pouco tempo com o RBC e depois o
actor degradava servico EV quando ficava sozinho. Foram testadas variantes com:

- treino durante a exploracao inicial com `RBCSmartPolicy`;
- phase-out do professor por mistura (`blend`) em vez de troca probabilistica;
- ruido negativo de EV reduzido/zero durante exploracao;
- regularizacao actor-side para V2G de EV e storage;
- BC decay mais lento;
- AMP desligado na variante phase-out por estabilidade, depois de erro NVIDIA
  `Xid`/CUDA que exige reboot para recuperar a GPU.

Runs guardadas:

- parcial: `runs/benchmarks/phase6f10_066_maddpg_15s_v42_blend_evguard_cpu`;
- completa: `runs/benchmarks/phase6f11_066_maddpg_15s_v42_blend_strong_evguard_cpu`;
- completa: `runs/benchmarks/phase6f12_066_maddpg_15s_v42_weighted_ev_bc_cpu`.

### Run Parcial 6F.10

Foi interrompida manualmente antes de terminar:

- episodio 4/6;
- step 1859/2880;
- global step 10496/17280;
- progresso 60.7407%;
- checkpoint guardado em
  `runs/benchmarks/phase6f10_066_maddpg_15s_v42_blend_evguard_cpu/jobs/phase6a_15s_maddpg_community_service_band_v42_prioritized_warmtrain_phaseout_rbc_smart_seed123/logs/checkpoints/latest_checkpoint.pth`.

Ultimo snapshot:

| EV neg frac | EV pos frac | Storage neg frac | Storage idle frac | BC efetivo | EV service penalty mean |
|---:|---:|---:|---:|---:|---:|
| 0.125 | 0.750 | 0.0588 | 0.9412 | 0.0236 | 126.606 |

Leitura: a receita estava a degradar EV service depois do phase-out. Nao usar
como comparacao final de KPIs, mas manter logs/checkpoint para diagnostico.

### Runs Completas

Referencia da janela 15s usada anteriormente:

- `RBCSmart`: custo `60.402`, `EV feasible min = 1.0`, deficit medio `0.0687`,
  EV negative fraction `0.0`.

| Run | Custo | EV feasible min | EV within feasible | EV success feasible | Deficit mean | Shortfall tol | EV surplus | EV neg frac | Storage neg frac | Storage idle frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `phase6f11` strong EV guard | 55.367 | 1.000 | 0.000 | 1.000 | 0.2007 | 0.1674 | 0.0500 | 0.0252 | 0.0088 | 0.9478 |
| `phase6f12` weighted EV BC | 48.957 | 0.000 | 0.000 | 0.000 | 0.2930 | 0.2430 | 0.0000 | 0.0271 | 0.0198 | 0.9561 |

Leitura:

1. `phase6f11` e o melhor marco ate agora: bate o `RBCSmart` em custo nesta
   janela (`55.367` vs `60.402`) e mantem `EV feasible min = 1.0`.
2. `phase6f11` ainda nao e uma vitoria limpa: o deficit EV medio piora bastante
   (`0.2007` vs `0.0687`) e aparece V2G/export EV residual.
3. `phase6f12` baixou mais o custo, mas falhou o KPI principal de EV factivel.
   A ponderacao de BC nos EVs nao resolveu a politica final; tornou a receita
   mais conservadora em alguns passos, mas ainda permitiu V2G/under-service.
4. O problema residual ja nao parece ser apenas ruido de exploracao. Na ultima
   avaliacao deterministica ainda existem acoes EV negativas, logo a politica
   aprendida esta a usar V2G/under-service como caminho de custo.
5. Proxima linha de trabalho deve focar reward/regularizacao informada por
   estado: penalizar V2G ou carga insuficiente quando SOC esta abaixo do target
   necessario/tempo ate departure, sem proibir V2G globalmente no wrapper.

Estado antes de reboot:

- nao ha runs ativas;
- resultados completos estao em `runs/benchmarks/phase6f11...` e
  `runs/benchmarks/phase6f12...`;
- run parcial `phase6f10...` tem logs, progress e checkpoint;
- GPU ficou em estado invalido apos erro NVIDIA/CUDA e deve recuperar com reboot.

## Fase 6F.3 - Teacher BC e Replay com Acoes do Professor

Data: 2026-05-18.

Motivacao: o MADDPG continuava a descobrir caminhos baratos mas perigosos
atraves de V2G/under-service EV. A ideia desta fase foi separar duas coisas que
antes estavam misturadas:

- a acao executada no replay, que pode ter ruido/exploracao;
- a acao do professor (`RBCSmartPolicy`), usada apenas como alvo de behavior
  cloning do actor.

Alteracoes implementadas:

- `MultiAgentReplayBuffer` e `RewardWeightedMultiAgentReplayBuffer` passam a
  poder guardar `behavior_actions` em paralelo com `actions`;
- MADDPG ganhou `actor_behavior_cloning_source`:
  - `replay_action`: comportamento antigo;
  - `warm_start_policy`: usa a policy de warm-start deterministica como alvo de
    BC, quando ha observacoes raw disponiveis;
- benchmark ganhou a variante
  `community_battery_value_v43_teacher_bc_stable_rbc_smart`;
- `RandomPolicy` passou tambem a respeitar limites observados de SOC de storage;
- `run_experiment.py` passou a marcar `result.json`/`progress.json` como
  `failed` se o treino rebentar, em vez de deixar artefactos em `pending`.

Testes focados:

- `86 passed` em:
  - `tests/test_replay_buffer.py`;
  - `tests/test_maddpg_exploration.py`;
  - `tests/test_baseline_policies.py`;
  - `tests/test_reward_functions.py`;
  - `tests/test_reward_function_registry.py`;
  - `tests/test_phase6a_benchmark.py`.

Run GPU longa parcial:

- `runs/benchmarks/phase6h_066_teacher_bc_gpu_longer_15s`;
- dataset 15s, seed `123`, alvo `12` episodios, `2880` steps por episodio;
- actor `[512, 256]`, critic `[1024, 512, 256]`;
- `RBCSmartPolicy` como warm-start/teacher;
- replay ponderado por reward;
- terminou por erro CUDA aos `33120/34560` global steps (`95.8333%`):
  `CUDA error: unspecified launch failure`.

O erro deixou o `torch.cuda.is_available()` a `False` em novos processos Python,
embora `nvidia-smi` ainda visse a RTX 4080. Portanto esta run fica como
diagnostico parcial, nao como benchmark final.

Comparacao parcial com a V43 anterior:

| Run | Rows treino | Ultimo step | EV neg frac mean | EV service mean | EV schedule deficit mean | Community settlement mean | Battery safety mean | Q std mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v43_prioritized_warmtrain_phaseout_rbc_smart` | 142 | 34320 | 0.0070 | 0.1254 | 0.0252 | 0.001156 | 0.0000086 | 8.525 |
| `v43_teacher_bc_stable_rbc_smart` parcial | 137 | 33120 | 0.0027 | 0.1037 | 0.0207 | 0.001248 | 0.0000156 | 9.429 |

Leitura:

1. Teacher BC contra `RBCSmartPolicy` melhora o sinal certo: reduz EV negative
   fraction e reduz EV service/schedule deficit medio face a V43 anterior.
2. Isto ainda nao chega para declarar vitoria. O agente 14 continua a dominar
   as perdas, com eventos fortes no Building 15.
3. A bateria esta a ser usada e nao ha violacao observada de SOC:
   `battery_soc_violation_penalty_amount_mean = 0.0`.
4. A critic continua instavel: `q_expected_std` cresce ate perto de `18.6`, e
   apareceu pelo menos um pico de critic loss `2.578`.
5. O custo comunitario nao melhorou nesta parcial; teacher BC melhora servico,
   mas ainda nao prova melhor KPI economico que RBC.

Conclusao:

- manter o suporte a teacher BC, porque e uma melhoria estrutural util e
  configuravel do MADDPG;
- nao promover esta variante a receita final;
- proximo trabalho deve combinar:
  - auditoria especifica do evento Building 15/agente 14;
  - replay/BC mais focado em janelas de servico EV, nao so reward total;
  - estabilizacao do critic antes de correr mais horas: `critic_lr` menor,
    batch maior se a memoria permitir, e possivelmente Huber/target clipping;
  - recuperar CUDA por reboot/driver antes de nova run longa.

## Fase 6F.4/6F.5 - V44/V45 e Capacidade EV Fase-Aware

Data: 2026-05-19.

Motivacao: a auditoria do Building 15 mostrou que a reward estava a calcular
pressao de carregamento EV com potencia nominal total do charger. Para chargers
limitados por fase, isso fazia o agente parecer mais atrasado do que estava
fisicamente: o Building 15 tem dois EVs ligados, mas a potencia efetiva fica
limitada pelas fases/servico eletrico, nao pela soma nominal dos chargers.

Alteracoes implementadas:

- `CostServiceCommunitySmoothServiceRewardV44` estabilizou critic com:
  - `critic_lr = 1e-4`;
  - Huber loss;
  - target clipping;
  - teacher BC persistente;
  - V2G EV negativo a zero na exploracao.
- `CostServiceCommunityFeasibleServiceRewardV45` mudou o termo schedule EV para
  usar potencia efetiva de carregamento quando ha fase/headroom disponivel.
- A V45 refresca metadata de fases durante `calculate()`, porque o simulador
  pode enriquecer metadata depois da inicializacao da reward.
- Quando o headroom observado e pos-acao, a V45 usa limites estaticos de fase
  vindos da metadata como capacidade efetiva, evitando capacidade falsa perto de
  zero enquanto o charger ja esta a carregar.

Diagnostico RBCSmart V45:

- run: `runs/benchmarks/phase6f5_rbcsmart_v45_static_phase_limit_per_agent_15s`;
- diagnostico:
  `runs/maddpg_diagnostics/phase6f5_rbcsmart_v45_static_phase_limit_per_agent_trajectory`;
- Building 15 passou a ver capacidade efetiva coerente:
  - dois EVs ligados: `12.0 kW`;
  - EV no L2: `5.0 kW`;
- agente 14: reward media `-0.2876`, minimo `-0.8206`, eventos criticos `3`.

Leitura:

1. O evento Building 15 nao era so erro do MADDPG. A reward V44/V45 precisava de
   conhecer capacidade efetiva por fase para nao exagerar o atraso EV.
2. Depois da correcao, o `RBCSmart` tambem tem eventos de pressao EV no Building
   15; logo estes eventos sao em parte reais e devem ser aprendidos, nao
   mascarados.
3. V45 passa a ser a reward experimental preferida para treino MADDPG nesta
   fase, porque informa atraso EV contra capacidade fisicamente disponivel.

## Fase 6F.6/6F.7 - Actor Pretraining

Motivacao: phase-out lento do professor melhora a acao executada durante treino,
mas a avaliacao deterministica chama apenas o actor. Se o actor nao aprendeu a
imitar o professor, a avaliacao final continua a falhar EV service mesmo que o
rollout de treino tenha usado boas acoes.

Alteracoes implementadas:

- MADDPG ganhou `actor_policy_loss_weight`;
- MADDPG ganhou schedule de policy loss:
  - `actor_policy_loss_warmup_weight`;
  - `actor_policy_loss_warmup_steps`;
  - `actor_policy_loss_warmup_start_step`;
- logs novos:
  - `MADDPG/actor_policy_loss_effective_weight`;
  - `MADDPG/actor_policy_loss_weighted_mean`;
- benchmark ganhou variante
  `community_feasible_service_v45_actor_pretrain_rbc_smart`.

Receita nova:

- reward `CostServiceCommunityFeasibleServiceRewardV45`;
- teacher `RBCSmartPolicy`;
- BC contra `warm_start_policy`;
- peso BC inicial `0.180`, minimo `0.120`;
- multiplicador BC EV `16.0`;
- multiplicador BC storage `0.50`;
- policy loss do actor comeca em `0.05` e sobe linearmente ate `1.0`;
- phase-out do professor mais lento.

Sanity:

- testes focados:
  `tests/test_maddpg_exploration.py tests/test_phase6a_benchmark.py
  tests/test_reward_functions.py tests/test_reward_function_registry.py`;
- resultado: `72 passed`.

Run curta completa:

- run: `runs/benchmarks/phase6f7_v45_actor_pretrain_cpu_1ep_15s`;
- diagnostico:
  `runs/maddpg_diagnostics/phase6f7_v45_actor_pretrain_cpu_1ep_15s_trajectory`;
- dataset 15s, full-window `5760` steps, 1 episodio, CPU;
- agente 14:
  - reward media `-0.1592`;
  - minimo `-0.8235`;
  - eventos abaixo de `-1.0`: `0`;
- EV negative fraction media/max: `0.0`;
- q expected std: media `1.83`, max `3.11`;
- BC loss media caiu de `0.1061` para `0.0240`;
- policy loss weight subiu de `0.057` para `0.377` durante o episodio.

Leitura:

1. Actor pretraining melhorou o evento Building 15 face a V45 estavel de 1
   episodio, que tinha agente 14 medio perto de `-0.3077`.
2. A critic continua a crescer, mas sem explosao imediata nesta run.
3. O resultado ainda nao prova aprendizagem final, porque esta run so mede um
   episodio de treino com professor ativo. A prova real e a avaliacao
   deterministica final.

Run em curso:

- `runs/benchmarks/phase6f7_v45_actor_pretrain_cpu_3train_1eval_15s`;
- objetivo: 3 episodios de treino + 1 episodio final deterministico;
- CPU apenas, porque PyTorch ainda falha inicializacao CUDA neste terminal
  apesar de `nvidia-smi` ver a RTX 4080.

Atualizacao parcial:

- a run foi parada manualmente no terceiro episodio de treino, depois de o
  actor voltar a falhar EV service;
- diagnostico:
  `runs/maddpg_diagnostics/phase6f7_v45_actor_pretrain_cpu_3train_1eval_15s_partial_trajectory`;
- no ponto de falha:
  - policy loss weight efetivo ja estava perto de `0.84`;
  - BC efetivo ja tinha descido para perto de `0.13`;
  - EV negative fraction atingiu `0.125`;
  - agente 14 teve minimo `-4.8385`;
  - agente 3 tambem degradou, com minimo `-1.8506`;
  - `q_expected_std` subiu ate `5.53`, sem explosao extrema mas com deriva.

Conclusao:

- actor pretraining ajuda no primeiro episodio, mas a rampa ainda deixa o
  gradiente RL dominar cedo demais;
- a proxima receita deve manter o professor por mais tempo e reduzir a policy
  loss no inicio, em vez de aumentar rede/LSTM;
- foi criada a variante
  `community_feasible_service_v45_actor_pretrain_slow_rbc_smart`:
  - `actor_policy_loss_weight = 0.60`;
  - `actor_policy_loss_warmup_weight = 0.02`;
  - warmup/decay `64 * random_exploration_steps`;
  - BC inicial `0.250`, minimo `0.200`;
  - multiplicador BC EV `24.0`;
  - multiplicador BC storage `1.0`;
  - penalizacao actor EV V2G `16.0`;
  - phase-out do professor `24 * random_exploration_steps`.

Sanity da variante slow:

- testes focados:
  `tests/test_maddpg_exploration.py tests/test_phase6a_benchmark.py
  tests/test_reward_functions.py tests/test_reward_function_registry.py`;
- resultado: `73 passed`;
- run: `runs/benchmarks/phase6f8_v45_actor_pretrain_slow_cpu_1ep_15s`;
- diagnostico:
  `runs/maddpg_diagnostics/phase6f8_v45_actor_pretrain_slow_cpu_1ep_15s_trajectory`;
- 1 episodio completo, CPU, full-window `5760` steps;
- agente 14:
  - reward media `-0.1542`;
  - minimo `-0.8217`;
  - eventos abaixo de `-1.0`: `0`;
- EV negative fraction media/max: `0.0`;
- `q_expected_std`: media `1.83`, max `2.87`;
- policy loss weight medio `0.0455`, max `0.0689`;
- BC efetivo medio `0.2485`;
- storage idle medio `0.893`.

Leitura:

- a variante slow mantem a melhoria do primeiro episodio e evita a degradacao
  rapida da variante anterior;
- ainda nao prova resultado final, porque falta o episodio deterministico;
- run 3 treino + 1 avaliacao deterministica iniciada em
  `runs/benchmarks/phase6f8_v45_actor_pretrain_slow_cpu_3train_1eval_15s`.

## Fase 6F.10-6F.14 - Teacher Clone EV Focus

Objetivo: antes de voltar a libertar policy loss RL, testar se o actor consegue
clonar uma politica de servico EV suficientemente bem para manter o gate EV e
baixar custo. Isto continua a ser treino do MADDPG/actor, nao uma regra no
wrapper.

Artefactos principais:

- comparacao:
  `runs/maddpg_diagnostics/phase6f10_f14_teacher_clone_comparison.csv`;
- run longa:
  `runs/benchmarks/phase6f14_ev_focus_cpu_3train_1eval_15s`;
- trajetoria:
  `runs/maddpg_diagnostics/phase6f14_ev_focus_cpu_3train_1eval_15s_trajectory`;
- eventos EV:
  `runs/maddpg_diagnostics/phase6f14_ev_focus_cpu_3train_1eval_15s_ev_departures`.

Resultados 15s:

| Run | Custo | EV feasible min | EV success feasible | EV within tol feasible | EV deficit | EV surplus | EV neg | Storage neg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `RBCBasic` | 99.495 | 1.000 | 0.000 | 1.000 | 0.0299 | 0.0000 | 0.000 | 0.043 |
| `RBCSmart` | 97.241 | 1.000 | 0.000 | 1.000 | 0.0299 | 0.0000 | 0.000 | 0.000 |
| `f10_ev_focus_1train` | 97.278 | 1.000 | 1.000 | 0.400 | 0.0304 | 0.0676 | 0.000 | 0.084 |
| `f11_ev_band_1train` | 87.543 | 0.800 | 0.600 | 0.600 | 0.0554 | 0.0215 | 0.000 | 0.083 |
| `f12_ev_balanced_1train` | 88.946 | 0.800 | 0.800 | 0.200 | 0.0509 | 0.0488 | 0.000 | 0.079 |
| `f13_ev_guarded_band_1train` | 90.541 | 0.800 | 0.800 | 0.400 | 0.0583 | 0.0344 | 0.013 | 0.083 |
| `f14_ev_focus_3train` | 87.460 | 1.000 | 0.800 | 0.600 | 0.0369 | 0.0426 | 0.000 | 0.062 |

Leitura:

1. `f14_ev_focus_3train` e o melhor marco MADDPG ate agora: baixa custo face
   ao `RBCSmart` e preserva `EV feasible min = 1.0`.
2. A diferenca ainda nao e uma vitoria final. O `RBCSmart` fica com
   `within_tolerance_feasible = 1.0`, enquanto `f14` fica em `0.6`; portanto o
   MADDPG ainda carrega alguns EVs acima/abaixo da banda desejada.
3. As variantes `f11`, `f12` e `f13` baixaram custo sacrificando eventos EV
   factiveis; ficam rejeitadas como receitas principais.
4. O treino longo de clonagem ajudou: face ao `f10`, o `f14` baixou custo
   (`97.278 -> 87.460`) e melhorou `within_tolerance_feasible` (`0.4 -> 0.6`)
   mantendo EV sem descarga.
5. O script bruto de eventos agora inclui tambem os KPIs oficiais exportados
   pelo simulador. No `f14`, os dois failures raw do Building 15 sao
   classificados pelo simulador como infeasible; o KPI oficial relevante e
   `official_min_acceptable_feasible_rate = 1.0`.

Conclusao operacional:

- `community_feasible_service_v45_teacher_clone_ev_focus_rbc_smart` passa a ser
  a base segura para continuar;
- proximo teste nao deve voltar a `ev_band`/`balanced`; deve partir de `f14` e
  introduzir RL muito devagar, mantendo BC EV forte e penalizacao anti-V2G;
- objetivo do proximo passo: preservar `EV feasible min = 1.0`, aproximar
  `within_tolerance_feasible` de `RBCSmart`, e testar se a policy loss consegue
  baixar custo sem reabrir undercharge.

## Fase 6F.15 - Slow Fine-Tune Rejeitado

Run:

- `runs/benchmarks/phase6f15_slow_finetune_cpu_1train_1eval_15s`

Variante:

- `community_feasible_service_v45_teacher_clone_ev_focus_slow_finetune_rbc_smart`

Objetivo:

- partir da receita `f14`;
- reintroduzir policy loss muito pequena e lenta;
- manter BC EV forte para tentar baixar custo sem perder o gate EV.

Resultado:

| Variante | Custo | EV feasible min | EV feasible dentro tolerancia | EV raw min | Deficit medio | EV V2G medio |
|---|---:|---:|---:|---:|---:|---:|
| `f14` teacher clone EV focus | 87.460 | 1.000 | 0.600 | 0.714 | 0.0369 | 0.0000 |
| `f15` slow fine-tune | 93.481 | 0.800 | 0.200 | 0.571 | 0.0655 | 0.0133 |

Conclusao:

- a reintroducao de policy loss, mesmo pequena, voltou a degradar EV service;
- nao vale a pena correr esta variante mais tempo sem uma protecao adicional;
- a proxima melhoria deve reforcar BC/replay em janelas EV criticas e so depois
  voltar a testar fine-tune RL.

## Fase 6F.16 - Event Replay EV Focus Rejeitado

Run:

- `runs/benchmarks/phase6f16_event_focus_cpu_1train_1eval_15s`

Variante:

- `community_feasible_service_v45_teacher_clone_ev_focus_event_rbc_smart`

Alteracao testada:

- policy loss continuou a `0.0`;
- BC EV mais forte;
- replay com `priority_fraction=0.35`;
- prioridade adicional por observacoes EV criticas:
  deficit/urgencia/potencia requerida ate departure.

Resultado:

| Variante | Custo | EV feasible min | EV feasible dentro tolerancia | EV raw min | Deficit medio | EV V2G medio |
|---|---:|---:|---:|---:|---:|---:|
| `f14` teacher clone EV focus | 87.460 | 1.000 | 0.600 | 0.714 | 0.0369 | 0.0000 |
| `f15` slow fine-tune | 93.481 | 0.800 | 0.200 | 0.571 | 0.0655 | 0.0133 |
| `f16` event replay EV focus | 90.405 | 0.800 | 0.400 | 0.571 | 0.0522 | 0.0053 |

Diagnosticos:

- trajetoria:
  `runs/maddpg_diagnostics/phase6f16_event_focus_cpu_1train_1eval_15s_trajectory`;
- eventos EV:
  `runs/maddpg_diagnostics/phase6f16_event_focus_cpu_1train_1eval_15s_ev_departures`.

Falhas principais no episodio de avaliacao:

- `Building 15 / charger_15_1`: SOC `0.67`, minimo aceitavel `0.75`;
- `Building 15 / charger_15_2`: SOC `0.64`, minimo aceitavel `0.77`;
- `Building 4 / charger_4_1`: SOC `0.70`, minimo aceitavel `0.80`.

Conclusao:

- event replay generico ajudou a reduzir EV V2G face ao f15, mas ainda quebrou
  o gate EV;
- aumentar prioridade de eventos sem mexer na qualidade/estrutura do professor
  nao chega;
- `f14` continua a ser a melhor base segura.

## Fase 6H.3 - Professor Explicito de Aprendizagem

Run:

- `runs/benchmarks/phase6h3_learning_teacher_gpu_3train_1eval_15s`

Variante:

- `community_feasible_service_v45_teacher_clone_ev_learning_teacher_rbc_smart`

Motivo:

- o `f14` tinha bons resultados, mas dependia de um professor configurado de
  forma implicita pelos defaults antigos;
- esta fase tornou esse professor explicito e mais adequado a aprendizagem:
  EV sem V2G no professor, carga EV menos agressiva, e BC forte em EV mas fraco
  em storage;
- a policy loss continuou desligada (`actor_policy_loss_effective_weight = 0.0`)
  para isolar imitacao/contrato antes de novo fine-tune RL.

Comparacao principal:

| Variante | Custo | EV feasible min | EV feasible dentro tolerancia | EV raw min | Deficit medio | Surplus medio | Erro abs medio | EV V2G medio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `RBCSmart` floor030 | 97.105 | 1.000 | 1.000 | 0.714 | 0.0299 | 0.0000 | 0.0299 | 0.0000 |
| `f14` professor antigo | 87.460 | 1.000 | 0.600 | 0.714 | 0.0369 | 0.0426 | 0.0795 | 0.0000 |
| `6H.3` professor explicito | 87.808 | 1.000 | 0.600 | 0.714 | 0.0298 | 0.0356 | 0.0654 | 0.0000 |

Diagnosticos:

- trajetoria:
  `runs/maddpg_diagnostics/phase6h3_learning_teacher_gpu_3train_1eval_15s_trajectory`;
- eventos EV:
  `runs/maddpg_diagnostics/phase6h3_learning_teacher_gpu_3train_1eval_15s_ev_departures`;
- storage:
  `runs/storage_audits/phase6h3_learning_teacher_gpu_3train_1eval_15s`;
- comparacao CSV:
  `runs/benchmarks/phase6h3_learning_teacher_gpu_3train_1eval_15s/comparison_key_metrics.csv`.

Falhas principais no episodio de avaliacao:

- `Building 15 / charger_15_1`: SOC `0.67`, minimo aceitavel `0.75`;
- `Building 15 / charger_15_2`: SOC `0.64`, minimo aceitavel `0.77`;
- `Building 1 / charger_1_1`: saiu acima da banda (`0.95` para target `0.87`).

Leitura:

- a variante recupera o custo baixo do `f14` e reduz o erro EV absoluto;
- ainda nao e vitoria sobre `RBCSmart`, porque a precisao EV dentro da banda
  continua pior;
- o pior agente continua a ser o Building 15/agente 14;
- o critic parece aprender o valor do clone, mas ainda nao ha otimizacao RL
  real da policy;
- o proximo passo deve melhorar os estados/eventos criticos de EV e storage
  antes de reabrir policy loss.

## Fase 6H.4 - Replay Event-Aware EV

Run:

- `runs/benchmarks/phase6h4_learning_teacher_event_gpu_3train_1eval_15s`

Variante:

- `community_feasible_service_v45_teacher_clone_ev_learning_teacher_event_rbc_smart`

Motivo:

- testar se dar mais peso a eventos EV no replay/BC melhorava os departures
  criticos;
- manter policy loss desligada e continuar a usar o professor `RBCSmart` de
  aprendizagem.

Resultado:

| Variante | Custo | EV feasible min | EV feasible dentro tolerancia | EV raw min | Deficit medio | Surplus medio | Erro abs medio | EV V2G medio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `6H.3` professor explicito | 87.808 | 1.000 | 0.600 | 0.714 | 0.0298 | 0.0356 | 0.0654 | 0.0000 |
| `6H.4` event-aware | 87.945 | 0.800 | 0.400 | 0.571 | 0.0570 | 0.0431 | 0.1002 | 0.0000 |

Diagnosticos:

- trajetoria:
  `runs/maddpg_diagnostics/phase6h4_learning_teacher_event_gpu_3train_1eval_15s_trajectory`;
- eventos EV:
  `runs/maddpg_diagnostics/phase6h4_learning_teacher_event_gpu_3train_1eval_15s_ev_departures`;
- storage:
  `runs/storage_audits/phase6h4_learning_teacher_event_gpu_3train_1eval_15s`;
- comparacao CSV:
  `runs/benchmarks/phase6h4_learning_teacher_event_gpu_3train_1eval_15s/comparison_key_metrics.csv`.

Leitura:

- a prioridade generica a eventos EV piorou o gate feasible;
- a falha nova relevante foi `Building 4 / charger_4_1` abaixo do minimo
  aceitavel;
- a hipotese mais provavel e que o replay event-aware estava a sobre-amostrar
  eventos infeasible do Building 15 e a degradar eventos feasible;
- conclusao: nao continuar esta linha sem distinguir eventos feasible,
  infeasible e over-service.

## Fase 6H.5 - Strong BC EV Sem Event Replay Generico

Run:

- `runs/benchmarks/phase6h5_learning_teacher_strong_bc_gpu_5train_1eval_15s`

Variante:

- `community_feasible_service_v45_teacher_clone_ev_learning_teacher_rbc_smart`

Alteracoes face a 6H.3:

- 5 episodios de treino + 1 avaliacao deterministica;
- BC EV mais forte;
- peso maior para acoes EV positivas do professor;
- peso maior para targets EV zero/idle;
- BC storage muito fraco;
- replay priorizado por acao EV do professor, mas sem prioridade generica por
  evento de observacao;
- policy loss continuou desligada.

Comparacao principal:

| Variante | Custo | EV feasible min | EV feasible dentro tolerancia | EV raw min | Deficit medio | Surplus medio | Erro abs medio | EV V2G medio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `RBCSmart` learning teacher | 89.010 | 1.000 | 1.000 | 0.714 | 0.0300 | 0.0000 | 0.0300 | 0.0000 |
| `6H.3` professor explicito | 87.808 | 1.000 | 0.600 | 0.714 | 0.0298 | 0.0356 | 0.0654 | 0.0000 |
| `6H.4` event-aware | 87.945 | 0.800 | 0.400 | 0.571 | 0.0570 | 0.0431 | 0.1002 | 0.0000 |
| `6H.5` strong BC | 85.172 | 1.000 | 0.800 | 0.714 | 0.0301 | 0.0182 | 0.0484 | 0.0000 |

Diagnosticos:

- trajetoria:
  `runs/maddpg_diagnostics/phase6h5_learning_teacher_strong_bc_gpu_5train_1eval_15s_trajectory`;
- eventos EV:
  `runs/maddpg_diagnostics/phase6h5_learning_teacher_strong_bc_gpu_5train_1eval_15s_ev_departures`;
- storage:
  `runs/storage_audits/phase6h5_learning_teacher_strong_bc_gpu_5train_1eval_15s`;
- comparacao CSV:
  `runs/benchmarks/phase6h5_learning_teacher_strong_bc_gpu_5train_1eval_15s/comparison_key_metrics.csv`.

Eventos finais:

- os dois failures min-acceptable crus sao `Building 15 / charger_15_1` e
  `Building 15 / charger_15_2`;
- ambos sao marcados como target-infeasible pelo KPI oficial, por isso
  `EV feasible min = 1.0`;
- o unico erro feasible fora da tolerancia e over-service:
  `Building 4 / charger_4_1`, SOC `0.97` para target `0.85`;
- `EV within_tolerance_feasible = 0.8`, melhor que 6H.3 mas ainda abaixo do
  professor (`1.0`).

Storage:

- episodio de avaliacao 6H.5: carga `63.153 kWh`, descarga `42.341 kWh`,
  throughput `105.493 kWh`;
- professor learning teacher: carga `95.711 kWh`, descarga `33.430 kWh`,
  throughput `129.141 kWh`;
- 6H.5 usa menos carga total que o professor, mas descarrega mais; o ganho de
  custo pode estar parcialmente ligado a maior autoconsumo/descarga, nao apenas
  a melhor EV policy.

Leitura:

- 6H.5 e o melhor marco MADDPG atual: bate o professor de aprendizagem em
  custo e aproxima a precisao EV;
- strong BC resolveu parte do over-service sem perder o gate EV feasible;
- event replay generico fica rejeitado por agora;
- o critic continua a saturar em valores negativos extremos perto/abaixo de
  `-35`, logo a proxima melhoria deve separar eventos infeasible do sinal de
  treino ou ajustar a escala/clipping do critic/reward;
- antes de reabrir policy loss, falta chegar a `EV within_tolerance_feasible =
  1.0` de forma consistente.

## Fase 6H.6 - V46 Feasible Precision

Run:

- `runs/benchmarks/phase6h6_precision_v46_strong_bc_gpu_5train_1eval_15s`

Variante:

- `community_feasible_precision_v46_teacher_clone_ev_learning_teacher_rbc_smart`

Alteracoes face a 6H.5:

- reward nova: `CostServiceCommunityFeasiblePrecisionRewardV46`;
- a V46 mantem componentes crus de deficit/surplus, mas limita o deficit usado
  em `ev_schedule_deficit_penalty` e `ev_departure_window_penalty`;
- tambem aumenta a penalizacao de over-service EV acima da banda;
- target clip do critic da variante baixa de `35` para `25`;
- policy loss continua desligada e o professor continua a ser o `RBCSmart`
  suave de aprendizagem;
- strong BC EV igual ao 6H.5.

Comparacao principal:

| Variante | Custo | EV feasible min | EV feasible dentro tolerancia | EV raw min | Deficit medio | Surplus medio | Erro abs medio | EV V2G medio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `RBCSmart` floor030 | 97.105 | 1.000 | 1.000 | 0.714 | 0.0299 | 0.0000 | 0.0299 | 0.0000 |
| `RBCSmart` learning teacher | 89.010 | 1.000 | 1.000 | 0.714 | 0.0300 | 0.0000 | 0.0300 | 0.0000 |
| `6H.5` V45 strong BC | 85.172 | 1.000 | 0.800 | 0.714 | 0.0301 | 0.0182 | 0.0484 | 0.0000 |
| `6H.6` V46 precision | 83.761 | 1.000 | 0.800 | 0.714 | 0.0304 | 0.0169 | 0.0473 | 0.0000 |

Diagnosticos:

- trajetoria:
  `runs/maddpg_diagnostics/phase6h6_precision_v46_strong_bc_gpu_5train_1eval_15s_trajectory`;
- eventos EV:
  `runs/maddpg_diagnostics/phase6h6_precision_v46_strong_bc_gpu_5train_1eval_15s_ev_departures`;
- storage:
  `runs/storage_audits/phase6h6_precision_v46_strong_bc_gpu_5train_1eval_15s`;
- comparacao CSV:
  `runs/benchmarks/phase6h6_precision_v46_strong_bc_gpu_5train_1eval_15s/comparison_key_metrics.csv`.

Eventos finais:

- os dois failures min-acceptable crus continuam em `Building 15 / charger_15_1`
  e `Building 15 / charger_15_2`;
- ambos continuam target-infeasible no KPI oficial, por isso
  `EV feasible min = 1.0`;
- o erro feasible fora da tolerancia passou a ser over-service no
  `Building 5 / charger_5_1`, SOC `0.97` para target `0.86`;
- `EV within_tolerance_feasible = 0.8`, ainda abaixo do professor
  (`1.0`), mas com surplus e erro absoluto ligeiramente melhores que 6H.5.

Storage:

- episodio de avaliacao 6H.6: carga `63.067 kWh`, descarga `40.981 kWh`,
  throughput `104.048 kWh`, SOC final medio `0.128`;
- episodio de avaliacao 6H.5: carga `63.153 kWh`, descarga `42.341 kWh`,
  throughput `105.493 kWh`, SOC final medio `0.115`;
- V46 reduziu um pouco descarga/throughput face a 6H.5, mas ainda usa storage
  de forma forte e precisa de auditoria antes de declarar a policy final.

Leitura:

- 6H.6 e o melhor marco MADDPG atual em custo (`83.761`) e mantem o gate EV
  feasible;
- a V46 validou a ideia de limitar pressao de deficits unrecoverable e apertar
  over-service sem perder custo;
- ainda nao resolveu totalmente precisao EV: falta chegar a
  `within_tolerance_feasible = 1.0`;
- o proximo passo deve ser uma melhoria incremental sobre V46, nao voltar ao
  event replay generico: classificar melhor eventos feasible/over-service e
  testar policy loss muito fraca ou fine-tune curto apenas depois de manter EV
  service estavel.
