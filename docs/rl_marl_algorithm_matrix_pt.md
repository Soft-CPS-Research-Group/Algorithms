# Matriz de Comparacao RL/MARL

Data: 2026-05-20.

Este documento prepara a fase seguinte: comparar algoritmos, nao apenas melhorar
MADDPG. O objetivo e encontrar o melhor controlador para a comunidade
energetica, com EVs, baterias, deferrables, limites fisicos, picos e uso de
renovaveis.

## Estado

Comparadores implementados e com smoke local ja validado:

- `MADDPG`;
- `MATD3`;
- `MASAC`;
- `IPPO`;
- `MAPPO`;
- `HAPPO`.

Baselines de referencia:

- `Random`;
- `NormalNoBattery`;
- `Normal`;
- `RBCBasic`;
- `RBCSmart`.

Configs pendentes, ainda nao submetidas:

- `configs/experiments/phase6_algorithm_matrix/remote_pending/`;
- manifesto: `configs/experiments/phase6_algorithm_matrix/remote_pending/pending_configs_manifest.csv`.

Estas configs exigem uma imagem Docker/SIF nova, construida depois dos commits
que adicionaram os comparadores e a matriz. A imagem remota `sha-969d417` nao
inclui ainda todos estes ficheiros.

## Matriz Preparada

Algoritmos:

- `MATD3`;
- `MASAC`;
- `IPPO`;
- `MAPPO`;
- `HAPPO`.

Datasets/variantes:

- `15s original`;
- `2022 original`;
- `15s no_v2g`;
- `2022 no_v2g`;
- `15s multi_charger`;
- `2022 multi_charger`.

Formato:

- seed `123`;
- track `short`;
- reward `CostServiceCommunityFeasiblePrecisionRewardV46`;
- encoding `maddpg_v2_compact`;
- interface `entity`;
- topologia `static`;
- MLflow desligado;
- export de KPIs ativo no fim.

O track `short` nao deve ser tratado como conclusao final. Serve para separar
rapidamente:

- algoritmo que nao aprende nada;
- algoritmo que quebra contrato/export;
- algoritmo que respeita EV/rede mas nao bate custo;
- algoritmo com sinal suficiente para passar a run longa/multi-seed.

## Smoke Local de Contrato

Data: 2026-05-21.

Foram corridos smokes locais no dataset 15s original, interface `entity`,
janela `0..63`, 1 episodio, reward
`CostServiceCommunityFeasiblePrecisionRewardV46` e encoding
`maddpg_v2_compact`.

Resultado:

| Algoritmo | Estado | Steps | ONNX por agente | Registos de treino |
|---|---:|---:|---:|---:|
| `MATD3` | passou | 64 | 17 | 15 |
| `MASAC` | passou | 64 | 17 | 15 |
| `IPPO` | passou | 64 | 17 | 15 |
| `MAPPO` | passou | 64 | 17 | 15 |
| `HAPPO` | passou | 64 | 17 | 15 |

Contrato validado:

- instanciam via registry;
- recebem observacoes encoded em `entity`;
- recebem bounds/nomes de acoes por agente;
- executam `predict`;
- executam pelo menos um ciclo de `update`;
- exportam ONNX por agente;
- geram `artifact_manifest.json`;
- nao rebentam com o dataset 15s original.

Resumo de topologia comum nos smokes:

- 17 agentes/buildings;
- 26 acoes totais;
- dimensoes de observacao encoded entre 71 e 131;
- agente 0 com storage, EV charger e deferrable;
- agente 14/Building 15 com storage e dois chargers.

Artefactos locais:

- `runs/comparator_smokes/20260521_212618/smoke_summary.md`;
- `runs/comparator_smokes/20260521_212618/smoke_summary.csv`;
- `runs/comparator_smokes/20260521_212618/contract_summary.csv`.

Estes smokes nao provam KPI, estabilidade ou qualidade de aprendizagem. Servem
apenas para dizer que os comparadores estao prontos para entrar na matriz curta
quando fizer sentido gastar compute.

## Comparabilidade da Reward

Para a primeira matriz, todos os algoritmos devem usar a mesma reward e os
mesmos KPIs. Isto torna a comparacao direta: o objetivo externo e igual, e o que
muda e a familia de algoritmo, exploracao, replay/on-policy e critic/value.

Leitura:

- `MATD3` e o comparador mais direto do MADDPG porque so altera critic/target
  policy smoothing/delayed actors;
- `MASAC` testa se a dificuldade principal e exploracao em continuous control;
- `IPPO` testa se uma policy independente por casa chega ou se MARL e mesmo
  necessario;
- `MAPPO` testa PPO multi-agent com value global, provavelmente o comparador
  on-policy mais serio;
- `HAPPO` testa update sequencial por agente, util quando a heterogeneidade dos
  buildings/importancia dos agentes for relevante.

Depois da primeira matriz, podem existir rewards ou escalas especificas por
familia de algoritmo, mas devem ser nomeadas como variantes explicitas. O
objetivo final nao deve mudar sem ser declarado no scorecard.

## Matriz Local Curta 2022 Original

Data: 2026-05-22.

Foi corrida uma matriz local curta no dataset 2022 original, janela `0..1999`,
com `RBCSmart`, `Random`, `MATD3`, `MASAC`, `IPPO`, `MAPPO` e `HAPPO`.

Detalhes:

- dataset: `citylearn_challenge_2022_phase_all_plus_evs`;
- interface: `entity`;
- encoding: `maddpg_v2_compact`;
- reward: `CostServiceCommunityFeasiblePrecisionRewardV46`;
- `MATD3`/`MASAC`: 3 episodios de 2000 steps;
- `IPPO`/`MAPPO`/`HAPPO`: 2 episodios de 2000 steps;
- `MATD3` e `MASAC` correram em CUDA local com a RTX 4080 Laptop GPU;
- sem MLflow, sem checkpoints, sem BAU export.

Resultado diagnostico:

| Policy | Cost EUR | Delta vs RBCSmart | EV min feasible | EV within tol feasible | Viol kWh | Peak daily ratio | Battery throughput | V2G export | Leitura |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `RBCSmart` | 3894.03 | 0.00 | 1.000 | 0.436 | 0.000 | 2.083 | 6847.00 | 18.78 | baseline forte desta janela |
| `MATD3` | 3894.48 | +0.45 | 1.000 | 0.596 | 0.000 | 1.446 | 12539.11 | 0.16 | melhor sinal tecnico; melhora precision/pico mas usa muita bateria |
| `MASAC` | 4696.15 | +802.12 | 1.000 | 0.051 | 0.000 | 2.227 | 1801.47 | 4.37 | cumpre minimo EV mas custo/precision maus; exploracao agressiva |
| `IPPO` | 4039.02 | +144.99 | 0.872 | 0.127 | 0.000 | 1.554 | 512.01 | 0.00 | falha EV service; baseline RL simples ainda cru |
| `MAPPO` | 3853.23 | -40.81 | 0.772 | 0.152 | 0.000 | 1.375 | 611.68 | 0.00 | custo baixo porque falha EV; precisa teacher/curriculum |
| `HAPPO` | 3851.17 | -42.86 | 0.772 | 0.152 | 0.000 | 1.376 | 619.83 | 0.00 | parecido com MAPPO; precisa guardrails |
| `Random` | 1369.51 | -2524.52 | 0.162 | 0.138 | 0.000 | 1.000 | 0.00 | 0.00 | custo baixo e irrelevante porque nao presta servico EV |

Leitura:

- `MATD3` e o unico comparador que passou o gate EV minimo nesta run curta e
  ficou praticamente empatado em custo com `RBCSmart`;
- `MATD3` tambem melhorou `EV within tolerance` e reduziu o ratio de pico, mas
  aumentou muito o throughput de bateria; antes de promover, auditar se esta a
  comprar melhoria com ciclos excessivos;
- `MASAC` precisa de tuning de entropia/exploracao e provavelmente teacher/BC
  mais restritivo antes de ser candidato serio;
- `IPPO`, `MAPPO` e `HAPPO` devem ser tratados como funcionais mas ainda crus:
  os custos baixos de `MAPPO/HAPPO` nao sao bons resultados porque falham EV
  service;
- nao tirar conclusoes finais desta matriz curta; serve para ordenar o proximo
  compute.

Artefactos:

- `runs/algorithm_matrix_local/20260522_095141/local_scorecard.md`;
- `runs/algorithm_matrix_local/20260522_095141/local_scorecard.csv`;
- `runs/algorithm_matrix_local/20260522_095141/configs/`.

Proxima decisao:

- promover `MATD3` para uma run 2022 maior/multi-seed depois de ler a matriz
  remota atual;
- testar uma variante `MATD3` com menor abuso de bateria;
- nao gastar Deucalion em `MAPPO/HAPPO/IPPO` longos sem primeiro adicionar
  teacher/curriculum/action priors para EV service;
- testar `MASAC` apenas depois de reduzir exploracao/temperatura ou adicionar
  warm-start mais conservador.

## Tuning Local Curto Dos Comparadores

Data: 2026-05-22.

Foram feitas alteracoes dentro dos agentes, sem mexer no wrapper:

- `MASAC`: passou a aceitar behavior cloning e regularizacao de acoes no actor,
  usando o mesmo replay/teacher contract dos off-policy agents;
- `IPPO`/`MAPPO`/`HAPPO`: passaram a aceitar warm-start policy, behavior
  cloning, extra BC updates e regularizacao de acoes no actor;
- todos mantem o mesmo contrato de observacoes/acoes/export.

Scorecard curto, mesma janela 2022 `0..1999`, comparado contra `RBCSmart`:

| Policy | Cost EUR | EV min | EV tol | EV success | Peak daily | Battery kWh | V2G kWh | Leitura |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `RBCSmart` | 3894.03 | 1.0000 | 0.4358 | 0.9777 | 2.0829 | 6847.00 | 18.78 | baseline forte |
| `MATD3StorageGuard` | 3817.43 | 0.9984 | 0.6561 | 0.9777 | 1.3866 | 10882.09 | 0.62 | melhor custo/precision/pico, mas ainda usa bateria acima do RBC |
| `MATD3ServiceStorageGuard` | 3839.35 | 1.0000 | 0.6149 | 0.9650 | 1.3805 | 10299.65 | 0.28 | melhor variante curta: EV minimo perfeito, custo melhor que RBCSmart, menos bateria que MATD3 antigo |
| `MASACConservativeBC` | 4311.21 | 0.9525 | 0.1347 | 0.9204 | 1.6070 | 14798.84 | 0.00 | melhorou custo face a MASAC cru, mas falha EV e abusa bateria |
| `IPPOBCExtra` | 4115.52 | 0.9968 | 0.4216 | 0.9761 | 1.7627 | 21687.88 | 178.09 | extra BC salva EV, mas bateria/custo maus |
| `MAPPOBCExtraGuard` | 4043.86 | 0.9952 | 0.4596 | 0.9682 | 1.7705 | 22420.79 | 208.36 | melhor PPO em EV tol, ainda nao competitivo em custo/bateria |
| `HAPPOBCExtra` | 4061.02 | 0.9937 | 0.4580 | 0.9538 | 1.7744 | 21912.83 | 249.39 | passa EV minimo, mas nao custo/bateria |

Conclusoes:

- `MATD3ServiceStorageGuard` e o unico novo comparador que, nesta janela curta,
  bate `RBCSmart` em custo mantendo EV minimo perfeito e zero violacoes;
- `MATD3StorageGuard` tem melhor precision EV e melhor pico, mas falha ligeira
  no EV minimo feasible rate (`0.9984`) e ainda usa muita bateria;
- `MASAC` ainda nao merece run longa: a entropia/BC ajudou no custo, mas piorou
  service e bateria;
- PPO-like agents agora ja conseguem aprender o servico EV com extra BC, mas
  usam demasiado storage/V2G; para os promover sera preciso uma solucao melhor
  de regularizacao/constraint de storage ou um treino mais longo/curriculum.

Artefactos:

- `runs/algorithm_matrix_local/20260522_135240/local_tuned_scorecard.md`;
- `runs/algorithm_matrix_local/20260522_135240/local_tuned_scorecard.csv`;
- `runs/algorithm_matrix_local/20260522_135240/configs/`.

Decisao atual:

- promover `MATD3ServiceStorageGuard` para a proxima matriz longa 2022/multi-seed;
- manter `MATD3StorageGuard` como alternativa se quisermos maximizar EV precision
  e reduzir picos;
- nao promover `MASAC`, `IPPO`, `MAPPO` ou `HAPPO` para long runs ate resolver
  custo/bateria.

## Ordem Recomendada

1. Esperar pelas runs ja submetidas de `MADDPG V48`, baselines e variants.
2. Gerar scorecard unico com `scripts/build_phase6_remote_scorecard.py`.
3. So depois submeter a matriz `short` dos novos algoritmos.
4. Priorizar no remoto:
   - `MATD3` e `MASAC` em GPU para original `15s` e `2022`;
   - `IPPO`, `MAPPO`, `HAPPO` em CPU/server se GPU estiver ocupada;
   - variants `no_v2g` e `multi_charger` depois dos originais.
5. Promover para long run apenas candidatos que passem EV, rede e custo
   competitivo contra `RBCSmart`.

## Scorecard

O scorecard deve comparar todos os learning controllers, nao so MADDPG:

- `MADDPG`;
- `MADDPG_v48`;
- `MATD3`;
- `MASAC`;
- `IPPO`;
- `MAPPO`;
- `HAPPO`.

Gates principais:

- EV minimo aceitavel em departures feasible;
- EV dentro da tolerancia do target;
- ausencia de violacoes de rede/fase;
- custo contra `RBCSmart`;
- custo e ratio contra BAU;
- throughput de bateria;
- V2G export.

Sinais comunitarios que passam a estar destacados quando existirem nos KPIs:

- importacao comunitaria;
- exportacao comunitaria;
- net exchange comunitario;
- picos diarios/all-time contra BAU;
- geracao solar comunitaria;
- export solar;
- self-consumption solar;
- community-market import share.

Isto e importante porque o objetivo nao e so "cada casa cumprir o seu EV". O
controlador deve tambem melhorar a comunidade como sistema: menos pico, mais
uso local/comunitario de renovaveis e menos energia comprada fora.

## Decisoes Esperadas

Se `MADDPG V48` estiver forte:

- manter como referencia;
- correr multi-seed;
- usar `MATD3` e `MASAC` como sanity check de melhoria marginal.

Se `MADDPG V48` tiver critic instavel ou Q-values suspeitos:

- promover `MATD3`;
- manter reward/encoding iguais;
- comparar twin critics e target smoothing.

Se o problema for exploracao:

- promover `MASAC`;
- observar alpha/entropia;
- comparar com e sem warm-start.

Se off-policy for instavel:

- testar `MAPPO`;
- usar `IPPO` como baseline RL simples;
- usar `HAPPO` como comparador on-policy sequencial.

Se `multi_charger` falhar:

- nao concluir logo que o algoritmo e mau;
- auditar dimensao de acoes, fases, headroom e escala de outputs;
- considerar heads por tipo de ativo antes de saltar para GNN/attention.

## Trabalho Futuro

Depois da matriz curta, candidatos de segunda vaga:

- heads separados por tipo de ativo;
- critic com attention por agentes/ativos;
- GNN para generalizacao de topologia;
- curriculum V2G;
- reward com termo comunitario mais explicito;
- ablation de storage para perceber valor real de bateria;
- scorecard por building, com destaque para Building 15.
