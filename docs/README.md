# Documentacao Ativa

Snapshot: 2026-06-04

Este diretorio fica reduzido aos documentos que ainda ajudam trabalho futuro.
Historico de fases antigas, configs remotos ja executados e relatorios
intermedios foram removidos do repo. Resultados brutos devem ficar em
`runs/`, que e ignorado pelo git.

## Leitura Principal

- `team_update_baselines_maddpg_performance_pt.md` - estado atual para a equipa:
  baselines, MADDPG/MATD3, Simulator 1.5.3, dynamic datasets e proximos passos.
- `community_optimization_success_scorecard_pt.md` - gates e KPIs para decidir
  se um controlador e candidato serio.
- `phase10_w6_guided_training_plan_pt.md` - plano atual de treino guiado para
  MADDPG/MATD3 contra os RBCs fortes.

## Contratos E Plataforma

- `platform_guide.md` - fluxo da plataforma, runner, wrapper, artefactos e
  manifests.
- `inference_bundle.md` - contrato de export/inference.
- `entity_interface_playbook_pt.md` - contrato entity do Simulator.
- `entity_encoding_profiles_v1_pt.md` - perfis de encoding ativos para
  Simulator 1.5.3.
- `simulator_limits.md` - limites e cuidados conhecidos do simulador.

## Algoritmos E Comparadores

- `rl_marl_algorithm_matrix_pt.md` - matriz conceptual de comparacao. As configs
  concretas devem ser geradas de novo a partir de `configs/templates/` quando
  houver SIF/imagem nova.

## Regra De Organizacao

- `docs/` guarda estado atual, contratos e decisoes.
- `configs/templates/` guarda templates reutilizaveis.
- `configs/experiments/` so deve receber configs ativos de uma ronda nova.
- `runs/` guarda outputs locais/remotos gerados e pode ser limpo sem afetar o
  codigo versionado.
