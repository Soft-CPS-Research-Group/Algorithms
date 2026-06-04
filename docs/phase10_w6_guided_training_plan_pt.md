# Phase 10 W6 - Treino Guiado Contra RBCs Fortes

Snapshot: 2026-06-04.

Objetivo: treinar `MADDPG` e `MATD3` sem voltar a sweeps cegos. A comparacao
principal passa por bater `RBCSmartPolicy` e `RBCCommunityPolicy` na mesma
janela, dataset, versao do simulador e politica de export.

## Alvos

Um candidato RL/MARL so e promovido se:

- `ev_min_acceptable_feasible_rate >= 0.99`;
- `electrical_violation_kwh == 0`;
- `ev_within_tolerance_feasible_rate >= 0.40` como alvo inicial;
- custo oficial do simulador menor ou igual ao melhor RBC forte da mesma janela;
- bateria/V2G sem abuso evidente;
- import/export/self-consumption/picos nao degradam sem ganho claro de custo.

O custo oficial vem do simulador:

1. `district_cost_community_market_settled_total_eur`, quando existe;
2. `district_cost_total_control_eur`, como fallback.

Nao usar custo recalculado do lado dos algoritmos para scorecard.

## Geração De Configs

```bash
.venv/bin/python scripts/generate_phase10_w6_configs.py \
  --stage all \
  --output-dir runs/generated_configs/phase10_w6
```

As configs geradas ficam em `runs/`, nao em `docs/`.

Stages relevantes:

| Stage | Conteudo | Uso |
|---|---|---|
| `w6-smoke-local` | MADDPG/MATD3 curto | validar wiring local |
| `w6a-local` | janelas locais + RBCs | escolher recipes |
| `w6b-remote-smoke` | smoke remoto curto | validar imagem/SIF/CUDA/export |
| `w6c-full-year` | full-year static | comparar contra RBCs fortes |

Janelas locais:

- `0:2048`
- `2048:4096`
- `4096:6144`
- `6144:8192`

## Recipes Prioritárias

Para continuar trabalho novo, priorizar:

- `w6_flex_ev_gate_repair_mid_bc`
- `w6_flex_ev_gate_repair_cost_push`
- `w6_flex_ev_gate_repair_policy_open`
- `w6_flex_v2g_open_value`, apenas como diagnostico de V2G/value

Todas devem manter:

- teacher/warm-start com `RBCSmartPolicy`;
- `RewardWeightedMultiAgentReplayBuffer`;
- prioridade de eventos EV;
- reward comunitaria com `community_settlement_cost_weight > 0`;
- export final-only em runs multi-episodio/longas.

## Algoritmos

Primarios:

- `MADDPG`
- `MATD3`

Adiar `MASAC`, `IPPO`, `MAPPO` e `HAPPO` ate haver uma recipe estavel nos dois
primarios. Esses algoritmos ja serviram para smoke infra, mas ainda nao ha sinal
forte para gastar full-year neles.

## Dynamic Topology

Dynamic topology fica reservado para baselines/RBCs por agora.

`MADDPG` e `MATD3` assumem layout fixo; `entity + dynamic` deve continuar a
falhar cedo no schema para estes algoritmos.

## Test Plan Antes De Remote

1. Validar configs com schema.
2. Smoke local curto para `MADDPG` e `MATD3`.
3. Smoke dynamic local dos RBCs quando o dataset dynamic mudar.
4. Testes:

```bash
.venv/bin/python -m pytest -q \
  tests/test_dataset_config.py \
  tests/test_config_validation.py \
  tests/test_baseline_policies.py \
  tests/test_phase10_w6_configs.py \
  tests/test_reward_functions.py
```

## Recursos Remotos

- Smokes CPU/server: usar margem suficiente, mas sem GPU se nao for necessario.
- Smokes GPU: A100, 64 GB RAM, 4 CPUs, 4h.
- Full-year RL: A100, 96 GB RAM, 4 CPUs, 12h.
- Nenhum job deve pedir mais de 48h.

## Decisão

Promover para full-year apenas se:

- baselines full-year novos ja estiverem fechados com Simulator 1.5.1;
- smoke remoto confirmar imagem/SIF e export;
- recipe local/remota curta passar gates de EV/rede sem throughput absurdo.
