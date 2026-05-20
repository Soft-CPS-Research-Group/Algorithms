# Fase 6E.1 - Baselines Calibrados com Simulador 0.6.6

Data: 2026-05-17.

## Contexto

Objetivo: corrigir/calibrar os baselines heuristicos depois da Fase 6E, usando
o simulador local `0.6.6`.

Ambiente:

- simulador local editable: `/home/tiago/dev/Simulator`;
- `citylearn.__version__`: `0.6.6`;
- import efetivo: `/home/tiago/dev/Simulator/citylearn/__init__.py`;
- `requirements.txt` deste repo ainda aponta para `softcpsrecsimulator==0.6.4`;
- testes focados: `31 passed`;
- matriz final: `10/10` jobs completos, `0` falhas.

Output final:

- `runs/benchmarks/phase6e1_066_baselines_calibrated_v2`;
- resumo: `runs/benchmarks/phase6e1_066_baselines_calibrated_v2/benchmark_summary.csv`.

Comando:

```bash
.venv/bin/python scripts/run_phase6a_benchmark.py \
  --output-dir runs/benchmarks/phase6e1_066_baselines_calibrated_v2 \
  --dataset 15s \
  --dataset 2022 \
  --agent random \
  --agent normal_no_battery \
  --agent normal \
  --agent rbc_basic \
  --agent rbc_smart \
  --seed 123 \
  --episodes 1 \
  --full-window \
  --metric-interval 240
```

## Alteracoes

- Simulador instalado localmente como `0.6.6`.
- `RBCBasic`:
  - EV deixa de forcar target absoluto `1.0`; garante o minimo requerido;
  - arbitragem de bateria reduzida para `0.15`;
  - banda de SOC mais conservadora;
  - deferrables usam ultimo arranque seguro.
- `RBCSmart`:
  - EV tambem garante minimo requerido;
  - bateria fica conservadora: PV sem stress e descarga so em contexto caro/stress;
  - config final evita arbitragem de preco agressiva;
  - deferrables usam ultimo arranque seguro.
- `Normal`:
  - continua a representar dia normal com bateria simples;
  - discharge de bateria ficou menos sensivel a importacoes muito pequenas.

## Resultados 15s

| policy | cost EUR | EV min feasible | EV min raw | EV success feasible | battery penalty | deferrable penalty |
|---|---:|---:|---:|---:|---:|---:|
| Random | 61.22 | 0.000 | 0.000 | 0.000 | 0.0020 | 0.0000 |
| NormalNoBattery | 112.51 | 1.000 | 0.714 | 1.000 | 0.0000 | 0.0000 |
| Normal | 113.88 | 1.000 | 0.714 | 1.000 | 0.0003 | 0.0000 |
| RBCBasic | 99.49 | 1.000 | 0.714 | 0.000 | 0.0002 | 0.0000 |
| RBCSmart | 97.24 | 1.000 | 0.714 | 0.000 | 0.0000 | 0.0000 |

Leitura:

- O fix do simulador `0.6.6` torna o KPI feasible coerente: deterministicos
  ficam a `1.000` em `ev_departure_min_acceptable_feasible_rate`.
- O raw continua `0.714` porque ha eventos fisicamente infeasible no cenario.
- `RBCBasic` e `RBCSmart` reduzem custo porque deixam de carregar EVs ate 100%;
  garantem o minimo aceitavel.
- `RBCSmart` ficou melhor que `RBCBasic` no 15s: menor custo e sem penalty de
  bateria.
- O strict success dos RBCs fica `0.000` porque estes baselines miram minimo
  aceitavel, nao target exato. Isto e aceitavel para comparacao principal, mas
  deve ficar visivel.

## Resultados 2022

| policy | cost EUR | EV min feasible | EV min raw | EV success feasible | battery penalty | deferrable penalty |
|---|---:|---:|---:|---:|---:|---:|
| Random | 2788.85 | 0.152 | 0.151 | 0.124 | 0.2728 | 0.0000 |
| NormalNoBattery | 4729.07 | 1.000 | 0.995 | 0.986 | 0.0000 | 0.0000 |
| Normal | 4847.26 | 1.000 | 0.995 | 0.986 | 0.0310 | 0.0000 |
| RBCBasic | 3805.41 | 1.000 | 0.995 | 0.978 | 0.0750 | 0.0000 |
| RBCSmart | 3967.06 | 1.000 | 0.995 | 0.978 | 0.0000 | 0.0000 |

Leitura:

- EV service feasible mantem `1.000` nos deterministicos.
- `RBCBasic` tem menor custo, mas usa bateria e paga throughput penalty.
- `RBCSmart` e o tradeoff conservador: custo maior que `RBCBasic`, mas sem
  penalty de bateria e ainda muito melhor que `NormalNoBattery`.
- `Normal` com bateria continua a piorar custo face a `NormalNoBattery`, o que
  e aceitavel como comportamento simples de dia normal.

## Decisao

Fase 6E.1 passa o gate minimo.

Baselines a usar a partir daqui:

- `Random`: sanity/lower bound, nunca baseline de custo isolado;
- `NormalNoBattery`: dia normal sem bateria;
- `Normal`: dia normal com bateria simples;
- `RBCBasic`: heuristica de menor custo com bateria moderada;
- `RBCSmart`: heuristica conservadora que preserva EV/deferrables e evita
  bateria agressiva.

Comparacao MADDPG deve usar primeiro:

- `ev_departure_min_acceptable_feasible_rate`;
- custo condicionado a EV/deferrables cumpridos;
- battery safety/throughput;
- grid violations.

## Pendencias

- Quando `0.6.6` estiver no PyPI, instalar por PyPI e atualizar
  `requirements.txt`.
- A diferenca entre strict EV success e min acceptable deve continuar explicita:
  os RBCs otimizados miram servico aceitavel, nao target exato.
- Proxima fase: repetir diagnostico/tuning MADDPG contra estes baselines.
