# Pipeline de Analise Remota Phase 6

Data: 2026-05-20.

Este ficheiro descreve o pipeline para transformar jobs remotos em evidencia
acionavel. Serve para as runs no server/Deucalion, mas tambem para qualquer
colecao futura de jobs do orchestrator.

## Comando Unico

Quando as runs terminarem:

```bash
.venv/bin/python scripts/run_phase6_remote_analysis.py \
  --jobs-file runs/remote_configs/phase6_remote_2026_05_20/submitted_jobs_2026_05_20_sha969d417.csv \
  --jobs-file runs/remote_configs/phase6_remote_2026_05_20/submitted_cpu_jobs_2026_05_20_sha969d417.csv \
  --jobs-file runs/remote_configs/phase6_remote_2026_05_20/submitted_server_variant_full_jobs_2026_05_20_sha969d417.csv \
  --output-dir runs/remote_results/phase6j_sha969d417
```

Isto faz tres coisas:

1. recolhe status, logs, configs resolvidas e KPIs;
2. gera scorecard agregado por job;
3. gera scorecard por building.

Se os resultados ja estiverem recolhidos e so quiseres refazer os relatorios:

```bash
.venv/bin/python scripts/run_phase6_remote_analysis.py \
  --skip-collect \
  --output-dir runs/remote_results/phase6j_sha969d417
```

## Outputs

O diretorio de output fica com:

- `summary.csv`: tabela compacta recolhida do orchestrator;
- `summary.json`: igual em JSON;
- `jobs/<job_id>/`: status, job info, logs tail, config resolvida e
  `exported_kpis.csv`;
- `scorecard.csv`: comparacao agregada por job;
- `scorecard.md`: leitura humana do scorecard agregado;
- `building_scorecard.csv`: metricas por building;
- `building_scorecard.md`: leitura humana dos buildings problemáticos.

## Scorecard Agregado

O scorecard agregado classifica cada run com:

- `decision`: resultado tecnico da run;
- `decision_bucket`: `promote`, `iterate`, `reject`, `wait` ou `reference`;
- `risk_flags`: pontos que merecem inspecao;
- `next_action`: acao recomendada.

Gates default:

- `ev_min_acceptable_feasible_rate >= 0.999`;
- `ev_within_tolerance_feasible_rate >= 0.80`;
- `electrical_violation_kwh <= 1e-6`;
- candidato forte precisa bater `RBCSmart` em custo;
- custo ate `5%` acima de `RBCSmart` pode ficar em iteracao, mas nao promove.

Flags relevantes:

- `ev_service_below_gate`;
- `ev_precision_below_gate`;
- `grid_violation`;
- `cost_above_rbcsmart`;
- `battery_throughput_high`;
- `peak_worse_than_bau`;
- `v2g_used`.

## Scorecard Por Building

O scorecard por building responde onde a run falhou ou ganhou.

Metricas extraidas por building:

- custo e delta para BAU;
- import/export/net exchange;
- EV departure feasibility e precision;
- deficits/surplus EV;
- V2G export;
- violacoes de fase/headroom;
- throughput de bateria;
- solar self-consumption;
- local share comunitario;
- deferrable service.

Flags relevantes:

- `building_15`;
- `cost_worse_than_bau`;
- `ev_service_below_gate`;
- `ev_precision_below_gate`;
- `ev_infeasible_departures`;
- `grid_violation`;
- `battery_throughput_high`;
- `solar_self_consumption_low`;
- `v2g_used`;
- `deferrable_service_gap`.

## Como Decidir Depois

Ordem de leitura:

1. abrir `scorecard.md`;
2. filtrar `decision_bucket=promote` e `decision_bucket=iterate`;
3. abrir `building_scorecard.md` para esses candidatos;
4. confirmar se as falhas aparecem concentradas num building, especialmente
   `Building_15`;
5. comparar custo, EV service, EV precision, picos, net exchange e bateria;
6. so depois decidir se vale continuar MADDPG, testar MATD3/MASAC/MAPPO ou
   mexer em reward/exploracao.

## Notas

- `risk_flags` nao sao prova automatica de bug. Sao sinais para guiar a
  auditoria.
- `v2g_used` e `battery_throughput_high` podem ser bons ou maus. So ficam maus
  se nao houver poupanca, renovavel aproveitada ou reducao de picos.
- Custo pior num building pode ser aceitavel se o custo comunitario melhorar,
  mas se repetir nos mesmos buildings temos de investigar credit assignment.
