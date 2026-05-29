# Phase 10 W6 - Treino Guiado Contra RBCSmart

Objetivo: deixar de fazer sweeps full-year cegos e atacar o modo de falha real:
EV service melhora com teacher/BC, mas ainda ha excesso de bateria/V2G e custo
acima de `RBCSmartPolicy`.

## Alvo Oficial

`RBCSmartPolicy` e o alvo W6. Uma run so e promovida se passar:

- `ev_min_acceptable_feasible_rate >= 0.99`
- `electrical_violation_kwh == 0`
- `ev_within_tolerance_rate >= 0.40`
- custo full-year `<= 17884.3`
- bateria preferencialmente `<= 49000 kWh`
- metricas comunitarias dentro de ~3% de RBCSmart, salvo melhoria clara de custo

`cost_ratio_to_bau` continua fora do gate porque BAU esta desligado de proposito.

## Implementacao

Gerador de configs:

```bash
.venv/bin/python scripts/generate_phase10_w6_configs.py --stage all --output-dir runs/generated_configs/phase10_w6
```

Saidas geradas:

- `runs/generated_configs/phase10_w6/run_matrix.csv`
- `runs/generated_configs/phase10_w6/run_matrix.json`
- `runs/generated_configs/phase10_w6/w6a-local/*.yaml`
- `runs/generated_configs/phase10_w6/w6b-remote-smoke/*.yaml`
- `runs/generated_configs/phase10_w6/w6c-full-year/*.yaml`

Scorecard contra RBCSmart:

```bash
.venv/bin/python scripts/build_phase10_candidate_scorecard.py \
  --summary-csv runs/remote_results/phase10_wave5_completed_20260528/summary.csv \
  --output-csv docs/phase10_wave5_candidate_scorecard_clean.csv \
  --output-md docs/phase10_wave5_candidate_scorecard_pt.md
```

## Matriz W6

| Stage | Conteudo | Recursos |
|---|---|---|
| `w6-smoke-local` | 4 recipes MADDPG, 2 seeds, 256 steps | local |
| `w6a-local` | RBCSmart/RBCCommunity por janela + 4 recipes MADDPG, 2 seeds, 4 janelas, 8 episodios | local |
| `w6b-remote-smoke` | Top 2 recipes MADDPG, 2 seeds, 4096 steps | A100, 64 GB, 4 CPUs, 4h |
| `w6c-full-year` | Top 2 recipes, MADDPG + MATD3, 2 seeds, 2 episodios full-year | A100, 96 GB, 4 CPUs, 12h |

Janelas locais:

- `0:2048`
- `2048:4096`
- `4096:6144`
- `6144:8192`

## Recipes

| Recipe | BC | Teacher | Storage | Regularizacao |
|---|---:|---|---|---|
| `w6_ev_only_bc_primary` | `0.06 -> 0.006` | `RBCSmartPolicy`, blend `4096` | BC `0.0` | storage L2 `0.004`, EV V2G L2 `0.05` |
| `w6_balanced_bc_storage_light` | `0.04 -> 0.004` | `RBCSmartPolicy`, blend `4096` | BC `0.15` | storage L2 `0.008`, EV V2G L2 `0.05` |
| `w6_fast_decay_less_teacher` | `0.03 -> 0.0` | `RBCSmartPolicy`, blend `2048` | BC `0.0` | storage L2 `0.004`, EV V2G L2 `0.05` |
| `w6_clone_diagnostic` | `0.50 -> 0.50` | `RBCSmartPolicy`, policy loss `0` | BC `0.25` | diagnostico de mapping/clonagem |

Todos usam `CostServiceCommunityFeasiblePrecisionRewardV46`,
`RewardWeightedMultiAgentReplayBuffer`, `priority_fraction=0.35`,
`priority_mode=negative_reward` e prioridade de eventos EV em partidas.

## Watchdog

O watchdog de stall fica desligado em local/smoke e ligado nos full-year. Para
nao pesar em I/O, o contexto JSON so e reescrito a cada
`stall_watchdog_context_interval_steps=64` nos `step_start`.

## Decisao Operacional

Nao lancar `w6c-full-year` ate:

1. `w6a-local` confirmar que pelo menos uma recipe bate RBCSmart na propria janela;
2. `w6b-remote-smoke` validar CUDA/update/export/RAM;
3. scorecard W6 mostrar EV gate e custo de janela aceitaveis.

## Validacao Local Inicial

Executado em 2026-05-29:

- `w6-smoke-local`: 8/8 completed, `exported_kpis.csv` presente em todas.
- `w6a-local` baselines: 8/8 completed, `exported_kpis.csv` presente em todas.

O smoke de 256 steps valida schema, runner, replay, teacher/warm-start,
export final-only e manifest flow. Nao distingue performance entre recipes
porque o episodio e curto e ainda dominado pelo teacher.

Targets RBCSmart por janela:

| Window | Cost EUR | EV min | EV tol | Battery kWh | V2G kWh | Import kWh | Export kWh | Solar self-cons. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `0:2048` | 4018.1 | 1.000 | 0.406 | 5886 | 0.74 | 37252 | 11127 | 0.514 |
| `2048:4096` | 4671.2 | 1.000 | 0.479 | 5417 | 0.00 | 42291 | 6925 | 0.509 |
| `4096:6144` | 5535.3 | 1.000 | 0.417 | 5474 | 0.00 | 37162 | 13456 | 0.422 |
| `6144:8192` | 2727.7 | 1.000 | 0.434 | 5916 | 0.00 | 32563 | 17761 | 0.458 |

Targets RBCCommunity por janela ficam como referencia comunitaria secundaria,
mas o gate W6A continua a ser bater a RBCSmart da propria janela.
