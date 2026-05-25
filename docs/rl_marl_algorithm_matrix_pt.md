# Matriz de Comparacao RL/MARL

Snapshot: 2026-05-25

Este documento mantem so a matriz conceptual. As configs antigas foram
removidas para evitar relancar uma ronda obsoleta. Quando houver nova imagem/SIF,
gerar configs novos a partir de `configs/templates/`.

## Objetivo

Comparar algoritmos, nao apenas melhorar MADDPG, para o caso:

- comunidade energetica;
- EVs, baterias, deferrables e possivel V2G;
- limites fisicos, fases/headroom e bounds;
- custo local/comunitario, autoconsumo, picos e requisitos de servico;
- export/inference via bundle.

## Baselines Obrigatorios

- `RandomPolicy`: lower bound/sanidade.
- `NormalNoBatteryPolicy`: BAU sem bateria.
- `NormalPolicy`: BAU com bateria simples.
- `RBCBasicPolicy`: heuristica minima.
- `RBCSmartPolicy`: baseline local forte.
- `RBCCommunityPolicy`: baseline comunitario forte.

## Algoritmos Candidatos

- `MADDPG`: candidato neural principal ja otimizado para `maddpg_v3`.
- `MATD3`: comparador direto de MADDPG com twin critics / TD3-style.
- `MASAC`: testa exploracao entropy-based em continuous control.
- `IPPO`: baseline PPO independente por building.
- `MAPPO`: PPO multi-agent com value/critic centralizado.
- `HAPPO`: variante PPO multi-agent com update sequencial por agente.

## Datasets E Perfis

Primeira matriz limpa:

- dataset: `citylearn_challenge_2022_phase_all_plus_evs`;
- interface: `entity`;
- topology: `static` para MADDPG/fixed-vector;
- encoding: `maddpg_v3_operational`;
- reward inicial: `CostServiceCommunityFeasiblePrecisionRewardV46`;
- export: KPI/series so no necessario para scorecard.

Comparacao real-world-safe posterior:

- encoding: `maddpg_v3_realtime`;
- evitar features de forecast derivado que dependam de futuro perfeito.

## Ordem Recomendada

1. Validar imagem/SIF com Gate 0: `RBCSmartPolicy`, `RBCCommunityPolicy` e
   `MADDPG` curto.
2. Correr scorecard anual dos baselines fortes.
3. Correr MADDPG multi-seed com o mesmo reward/encoding.
4. So depois promover `MATD3`, `MASAC`, `IPPO`, `MAPPO` e `HAPPO`.
5. Promover para run longa apenas candidatos que passem os gates em
   `community_optimization_success_scorecard_pt.md`.

## Regra De Fairness

Todos os candidatos de uma matriz devem partilhar:

- mesmo dataset/janela;
- mesmo perfil de observacao;
- mesmo reward externo;
- mesma politica de export/KPIs;
- seeds declaradas;
- comparacao contra `RBCSmartPolicy` e `RBCCommunityPolicy`.

Se uma familia precisar de teacher, BC, action priors ou reward scaling proprio,
isso deve aparecer como variante nomeada, nao como substituicao silenciosa da
matriz principal.
