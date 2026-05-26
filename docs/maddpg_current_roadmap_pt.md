# Roadmap Atual RL/MARL

Snapshot: 2026-05-26

Este e o documento operacional curto para a proxima ronda. A historia antiga de
Phase 6 foi removida do repo; o que interessa agora e correr uma matriz limpa em
cima do Simulator 1.1.0, dos encodings atuais e dos RBCs atualizados.

## Estado Atual

- Simulador requerido: `softcpsrecsimulator==1.1.0`.
- Interface principal: `entity`.
- Topologia MADDPG atual: `static`.
- Perfil RL default: `maddpg_v3_operational`.
- Perfil real-world-safe: `maddpg_v3_realtime`.
- Dataset principal: `citylearn_challenge_2022_phase_all_plus_evs`.
- Dataset 15s/dynamic continua util para smokes e diagnosticos, mas nao deve
  substituir scorecards anuais.

## Baselines Ativos

- `RandomPolicy`: lower bound/sanidade.
- `NormalNoBatteryPolicy`: BAU sem bateria.
- `NormalPolicy`: BAU com bateria simples.
- `RBCBasicPolicy`: heuristica minima.
- `RBCSmartPolicy`: baseline local forte; usa preco, PV/load, headroom, bounds,
  EV deadlines, feasibility e capacidades instantaneas.
- `RBCCommunityPolicy`: baseline comunitario forte; usa preco local
  comunitario, import/export comunitario, surplus, EVs, deferrables, storage e
  reducao de picos dentro de regras conservadoras.

## Algoritmos Ativos

- `MADDPG`: candidato neural principal existente.
- `MATD3`, `MASAC`, `IPPO`, `MAPPO`, `HAPPO`: comparadores implementados no
  registry e mantidos para a fase posterior.

## Configs

Templates reutilizaveis vivem em:

```text
configs/templates/baselines/
configs/templates/maddpg/
configs/templates/rl/
configs/templates/dynamic/
```

`configs/experiments/` deve ficar vazio ou conter apenas configs da ronda ativa.
Nao versionar dumps remotos antigos nem manifests de submissao ja executados.

## Proxima Ronda

Depois do push e da nova imagem/SIF:

1. Gate 0: smoke curto em CPU Deucalion com `RBCSmartPolicy`,
   `RBCCommunityPolicy` e `MADDPG` em `maddpg_v3_operational`.
2. Confirmar que a imagem publicada inclui este commit e
   `softcpsrecsimulator==1.1.0`.
3. Repor apenas os datasets/configs necessarios no orchestrator.
4. Correr scorecard anual dos baselines:
   `Random`, `NormalNoBattery`, `Normal`, `RBCBasic`, `RBCSmart`,
   `RBCCommunity`.
5. So depois correr MADDPG/matriz RL, para nao misturar bugs de infraestrutura
   com comparacao cientifica.

## Criterios De Decisao

Usar `community_optimization_success_scorecard_pt.md` como gate:

- EV service minimo quase perfeito.
- Sem violacoes fisicas relevantes.
- Deferrables servidos dentro da janela.
- Custo comunitario e picos abaixo dos baselines fortes.
- Battery/V2G sem throughput suspeito.

Um candidato neural so conta se bater baselines fortes sob a mesma janela,
dataset, reward, encoding e export settings.

## Performance Atual

Leitura local recente para Simulator 1.1.0, 2022 all-plus-EVs, 17 agentes:

```text
RandomPolicy, all bundles, 700 steps:
  wall time:                  ~13.6 ms/step
  RSS delta:                  +13.7 MB

RBCSmartPolicy, all bundles, 700 steps:
  wall time:                  ~14.8 ms/step
  RSS delta:                  +2.0 MB

MADDPG v3 operational, 96-step smoke:
  wall time:                  ~43.8 ms/step
  RSS delta:                  +588.5 MB
```

O ponto principal desta ronda e que o leak de `entity_action_feedback` deixou de
aparecer nos smokes locais. O MADDPG continua caro em RAM por causa das redes e
do replay buffer, nao por crescimento por step do cache do simulador.

## Regra De Trabalho

Nao adicionar configs remotas versionadas sem necessidade. Gerar configs de uma
ronda em `runs/remote_configs/...`, submeter, recolher, promover so o resumo que
for realmente necessario para `docs/`.
