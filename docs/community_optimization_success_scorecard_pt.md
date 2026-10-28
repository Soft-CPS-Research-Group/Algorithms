# Scorecard de Sucesso para Otimizacao Comunitaria

Para os datasets Portugal, consultar o [protocolo do motor original](portugal_evaluation_protocol_v2.md). A recuperação fixa `978be68b` sem extensões. A convenção financeira está definida: tarifa comum e fator de 80% no schema da experiência, usando a liquidação original. A tarefa corrente limita-se à construção dos dados, com avaliações suspensas. As regras executáveis Portugal anteriores não estão certificadas para este contrato; a integração de métricas será revista quando retomarmos as experiências. Os critérios gerais abaixo são orientação e os resultados históricos conservam as suas regras.

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
| 2 | `cost_delta_to_bau_eur` / `cost_ratio_to_bau` | diferença em euros; percentagem com denominador absoluto não nulo; não ordenar custos negativos por um quociente simples |
| 3 | `ev_within_tolerance_feasible_rate` | maior e melhor; mede precisao face ao SOC escolhido |
| 4 | `peak_daily_ratio_to_bau` / `peak_all_time_ratio_to_bau` | menor e melhor, quando BAU estiver disponivel |
| 5 | `community_solar_self_consumption_rate` | maior e melhor |
| 6 | `community_market_import_share_rate` / importacao externa | menor tende a ser melhor |
| 7 | `battery_throughput_kwh` | diagnostico, nao objetivo isolado; abuso de ciclos e suspeito |
| 8 | `v2g_export_kwh` | util apenas se nao sacrificar EV service/rede/custo |

Nota sobre custo: desde `softcpsrecsimulator==1.5.1`, o custo oficial vem do
simulador e ja recorta exportacao a zero (`max(net_kWh, 0) * price`). A versao
operacional atualmente fixada e `softcpsrecsimulator==1.9.1`. Resultados
historicos obtidos com `1.5.6` mantem essa versao na sua proveniencia e exigem
replay emparelhado antes de serem comparados causalmente com novas runs. Portanto
`community_cost_eur` deve mapear diretamente para o KPI de custo do simulador:
quando o mercado comunitario esta ativo, primeiro
`district_cost_community_market_settled_total_eur`; caso contrario,
`district_cost_total_control_eur`. Nao usamos custo alternativo recalculado
deste lado. Import/export/self-consumption ficam apenas como KPIs de diagnostico
e qualidade da politica.

## Referência comercial de DR

O motor original usa a média da hora anterior e DR apenas no Premium All-in. Não tem o contrato calibrado do motor alterado. As métricas abaixo descrevem a revisão histórica e objetivos de análise; não são garantidas pelos datasets correntes nem podem ser preenchidas com zero para aprovar uma execução:

- `dr_reference_valid_fraction`: fração dos pedidos/passos com referência qualificada, sem omitir recusas;
- `dr_reference_member_coverage`, `dr_reference_bess_power_coverage` e `dr_reference_charger_coverage`: população e equipamentos efetivamente representados;
- `dr_baseline_mae_kw`, `dr_reference_uncertainty_kw` e `dr_reference_interval_coverage`: erro observado, margem e cobertura empírica;
- `dr_credit_without_response_kwh`: crédito residual no par que ignora o pedido;
- adicionalidade, cumprimento físico por quarto de hora e rebound, já presentes no diagnóstico emparelhado.

Coberturas usam frações entre 0 e 1. O resumo conserva o mínimo de cobertura e o máximo de incerteza; esses extremos não substituem a leitura das janelas individuais. A calibração não altera o custo oficial nem transforma um pedido sem referência válida em resposta cumprida.

## Regras de Decisao

- `Reject`: falha EV minimo, rede, deferrables ou SOC de storage.
- `Inspect`: passa gates e reduz custo, mas com throughput/V2G/picos anormais.
- `Candidate`: passa gates e melhora custo face a `RBCSmartPolicy` e
  `RBCCommunityPolicy`, quando ambos forem aplicaveis.
- `Strong candidate`: passa gates, melhora custo face aos RBCs fortes, melhora
  ou nao degrada muito `ev_within_tolerance_feasible_rate`, e nao cria
  picos/throughput suspeitos.

## Baselines de Referencia

A comparacao principal deve incluir:

- `RandomPolicy`, apenas sanidade/lower bound;
- `NormalNoBatteryPolicy`;
- `NormalPolicy`;
- `RBCBasicPolicy`;
- `RBCSmartPolicy`, baseline heuristico local forte;
- `RBCCommunityPolicy`, baseline heuristico comunitario forte;
- candidato RL/MARL em avaliacao.

Para resultados finais, reportar sempre a janela temporal usada. Resultados de janela curta nao devem ser apresentados como resultado anual.
