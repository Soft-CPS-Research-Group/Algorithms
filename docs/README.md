# Documentacao Ativa

Snapshot: 2026-05-26

Este diretorio fica reduzido aos documentos que ainda ajudam trabalho futuro.
Historico de fases antigas, configs remotos ja executados e relatorios
intermedios foram removidos do repo. Resultados brutos devem ficar em
`runs/`, que e ignorado pelo git.

## Leitura Principal

- `team_update_baselines_maddpg_performance_pt.md` - estado atual para a equipa:
  baselines, MADDPG, Simulator 1.1.0, performance e proximos passos.
- `maddpg_current_roadmap_pt.md` - roadmap operacional curto para a proxima
  ronda de smokes/runs.
- `community_optimization_success_scorecard_pt.md` - gates e KPIs para decidir
  se um controlador e candidato serio.

## Contratos E Plataforma

- `platform_guide.md` - fluxo da plataforma, runner, wrapper, artefactos e
  manifests.
- `inference_bundle.md` - contrato de export/inference.
- `entity_interface_playbook_pt.md` - contrato entity do Simulator.
- `entity_encoding_profiles_v1_pt.md` - perfis de encoding ativos para
  Simulator 1.1.0.
- `simulator_limits.md` - limites e cuidados conhecidos do simulador.

## Algoritmos E Comparadores

- `marl_algorithm_comparators_pt.md` - alternativas RL/MARL implementadas ou
  relevantes.
- `rl_marl_algorithm_matrix_pt.md` - matriz conceptual de comparacao. As configs
  concretas devem ser geradas de novo a partir de `configs/templates/` quando
  houver SIF/imagem nova.

## Investigacao

- `phd_framework_roadmap.md` - enquadramento mais largo da tese/projeto.

## Regra De Organizacao

- `docs/` guarda estado atual, contratos e decisoes.
- `configs/templates/` guarda templates reutilizaveis.
- `configs/experiments/` so deve receber configs ativos de uma ronda nova.
- `runs/` guarda outputs locais/remotos gerados e pode ser limpo sem afetar o
  codigo versionado.
