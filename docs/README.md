# Documentacao Ativa

Snapshot: 2026-06-04

Este diretorio fica reduzido aos documentos que ainda ajudam trabalho futuro.
Historico de fases antigas, configs remotos ja executados e relatorios
intermedios foram removidos do repo. Resultados brutos devem ficar em
`runs/`, que e ignorado pelo git.

## Leitura Principal

- `community_optimization_success_scorecard_pt.md` - gates e KPIs para decidir
  se um controlador e candidato serio.

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
- Receitas de campanha, demonstrações e evidência experimental ficam locais e
  são ignoradas pelo Git, tal como os outputs em `runs/`.

O ciclo remoto pode ser preparado, submetido, acompanhado e arquivado com
`scripts/manage_remote_experiment.py`. A submissao exige sempre
`--confirm-submit`; o CLI nao inclui comandos de stop/delete.
