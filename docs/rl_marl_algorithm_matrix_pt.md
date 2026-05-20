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

