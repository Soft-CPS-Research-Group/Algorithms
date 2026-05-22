# Scorecard de Sucesso para Otimizacao Comunitaria

Este scorecard fica congelado como criterio pratico inicial para comparar baselines, MADDPG e outros algoritmos RL/MARL. Pode ser expandido mais tarde com objetivos comunitarios adicionais, mas por agora evita comparacoes soltas por reward interna.

## Gates Obrigatorios

Um controlador so e candidato serio se passar estes gates:

| Area | KPI | Criterio inicial |
|---|---|---:|
| EV service minimo | `ev_min_acceptable_feasible_rate` | `>= 0.99`, idealmente `1.0` |
| Rede/fases | `electrical_violation_kwh` | `0.0` |
| Rede/fases | `electrical_violation_events` | `0`, ou explicar tolerancias numericas residuais |
| Deferrables | deadlines/falhas de servico | sem falhas relevantes |
| Storage | SOC min/max violation | `0`, respeitando limites observados quando existem |

Se um algoritmo falhar estes gates, o custo mais baixo nao deve ser lido como melhoria real.

## KPIs de Otimizacao

Depois de passar os gates, a ordenacao deve olhar para:

| Prioridade | KPI | Leitura |
|---:|---|---|
| 1 | `community_cost_eur` | menor e melhor |
| 2 | `cost_delta_to_bau_eur` / `cost_ratio_to_bau` | negativo ou ratio `< 1` e melhor, quando BAU estiver disponivel |
| 3 | `ev_within_tolerance_feasible_rate` | maior e melhor; mede precisao face ao SOC escolhido |
| 4 | `peak_daily_ratio_to_bau` / `peak_all_time_ratio_to_bau` | menor e melhor, quando BAU estiver disponivel |
| 5 | `community_solar_self_consumption_rate` | maior e melhor |
| 6 | `community_market_import_share_rate` / importacao externa | menor tende a ser melhor |
| 7 | `battery_throughput_kwh` | diagnostico, nao objetivo isolado; abuso de ciclos e suspeito |
| 8 | `v2g_export_kwh` | util apenas se nao sacrificar EV service/rede/custo |

## Regras de Decisao

- `Reject`: falha EV minimo, rede, deferrables ou SOC de storage.
- `Inspect`: passa gates e reduz custo, mas com throughput/V2G/picos anormais.
- `Candidate`: passa gates e melhora custo face a `RBCSmartPolicy`.
- `Strong candidate`: passa gates, melhora custo face a `RBCSmartPolicy`, melhora ou nao degrada muito `ev_within_tolerance_feasible_rate`, e nao cria picos/throughput suspeitos.

## Baselines de Referencia

A comparacao principal deve incluir:

- `RandomPolicy`, apenas sanidade/lower bound;
- `NormalNoBatteryPolicy`;
- `NormalPolicy`;
- `RBCBasicPolicy`;
- `RBCSmartPolicy`, baseline heuristico forte;
- candidato RL/MARL em avaliacao.

Para resultados finais, reportar sempre a janela temporal usada. Resultados de janela curta nao devem ser apresentados como resultado anual.
