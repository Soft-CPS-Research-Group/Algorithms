# Roadmap Atual RL/MARL

Snapshot: 2026-05-25

Este e o documento operacional curto para a proxima ronda. A historia antiga de
Phase 6 foi removida do repo; o que interessa agora e correr uma matriz limpa em
cima do Simulator 1.0.2, dos encodings atuais e dos RBCs atualizados.

## Estado Atual

- Simulador requerido: `softcpsrecsimulator==1.0.2`.
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
   `softcpsrecsimulator==1.0.2`.
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

Leitura local recente para Simulator 1.0.2, 2022 all-plus-EVs, 17 agentes:

```text
MADDPG v3 operational:
  mediana step perfilado:     ~20.7 ms
  env.step:                   ~12.4 ms
  entity->maddpg_v3 direto:   ~7.2 ms
  update neural real:         ~0.6 ms
```

Isto ja removeu a aberração anterior perto de nove dias estimados para uma run
longa, mas ainda e caro. O proximo corte deve vir do simulador ou de reducao de
features/exports na ronda concreta, nao de voltar a duplicar encodings no
wrapper.

## Regra De Trabalho

Nao adicionar configs remotas versionadas sem necessidade. Gerar configs de uma
ronda em `runs/remote_configs/...`, submeter, recolher, promover so o resumo que
for realmente necessario para `docs/`.
