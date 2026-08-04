# PPO local anual: validação em três seeds e freeze para o CC

- Campanha: `ppo_residual_battery_annual_3seed_freeze_20260804`
- Arquivo: `2026-08-04T05:09:10+01:00`
- Dataset: `citylearn_three_phase_electrical_service_demo_15min_parquet`
- Janela: `0:35039`, 35 040 amostras de 15 minutos, 35 039 transições
- Código final: `876293a`
- Mercado comunitário: desligado
- Perfil local: `building_local_v1`
- Seeds de rede: `123`, `456`, `789`

## Decisão

O PPO local anual fica `ACCEPTED` no perfil
`building_local_phase10_w6_safety_projection_v1` e pode ser congelado para a
integração com o Community Coordinator. A seed `123` foi a seed de
desenvolvimento; a receita foi congelada antes de treinar e avaliar `456` e
`789`, que funcionaram como validação sem afinação por seed.

O leaf entregue não é um ONNX isolado. É o composto:

`RBCSmartLocalPolicy + 17 atores PPO residual-battery + safety projector + price adapter`

EV e deferrables continuam sob serviço local RBCSmart; o PPO decide apenas o
residual da bateria estacionária. Nenhum ator recebe observações comunitárias.
O único contexto futuro vindo do CC é o multiplicador de preço efetivo, sem
alterar o preço real de settlement.

## Comparadores

| Referência | Custo local anual | Semântica |
|---|---:|---|
| RBC Smart local | EUR 24 569,0692 | baseline local contemporânea, 17/17 gates |
| MILP individual replay | EUR 22 508,0351 | battery-only, fixed-service, referência feasible condicional |

O MILP continua a ser uma referência condicional, não um ótimo total com
EV/V2G, deferrables e rede como decisões conjuntas.

## Resultados congelados

| Seed | Custo local | Poupança vs. RBC | Gap fechado até MILP | Gates estritas | Gates tolerantes | Casas abaixo do RBC |
|---:|---:|---:|---:|---:|---:|---:|
| 123 | EUR 23 917,1205 | EUR 651,9486 (2,6535%) | 31,6321% | 17/17 | 17/17 | 17/17 |
| 456 | EUR 23 861,5365 | EUR 707,5327 (2,8798%) | 34,3290% | 16/17 | 17/17 | 17/17 |
| 789 | EUR 23 826,7980 | EUR 742,2711 (3,0212%) | 36,0145% | 17/17 | 17/17 | 17/17 |

Média: EUR 23 868,4850, menos EUR 700,5842 que o RBC (2,8515%). O desvio
amostral entre seeds foi EUR 45,5604. Todas as 17 casas ficaram abaixo do RBC
em todas as três seeds; o pior delta casa-seed ainda foi uma melhoria de
EUR 2,5345 no `Building_12`.

## Ressalva de segurança

A seed `456` registou no `Building_15` 0,076095 kWh em dois eventos de pedido
pré-projeção. A execução foi certificada dentro dos limites, a energia ficou
abaixo da tolerância explícita de 1 kWh/ano e a taxa de eventos ficou abaixo
de 0,1%. Por isso passa o perfil tolerante, mas não é apresentada como 17/17
estrita. As seeds `123` e `789` passaram 17/17 estritas.

## Freeze e handoff

A seed `789` foi escolhida para o primeiro handoff: é uma seed de validação,
passou 17/17 gates estritas e obteve o menor custo anual. O pacote local está
em:

`runs/handoffs/pedro_cc/ppo_local_residual_annual_seed789_20260804`

Inclui 17 ONNX, manifest, configuração resolvida, scorecards, contrato de
integração e checksums. O bundle valida, mas declara corretamente
`deployable: false` para o ONNX isolado porque requer a política base, safety e
adaptador de preço do runtime.

## Próximo gate

Antes de treinar o CC:

1. montar `CCLevel1 -> ensemble PPO congelado`;
2. carregar apenas o checkpoint do leaf, deixando o CC inicializar de raiz;
3. executar um smoke com multiplicador fixo `1.0`;
4. exigir paridade com a seed `789` standalone;
5. só depois treinar o CC com preço não neutro e comparar com
   `RBCCommunityPolicy` e o MILP comunitário na mesma superfície económica.

## Evidência local ignorada pelo Git

- Resumo de três seeds:
  `runs/analysis/ppo_residual_battery_dagger_frozen_annual_3seed_summary_20260804`
- Auditorias individuais:
  `runs/analysis/ppo_residual_battery_dagger_frozen_annual_s{123,456,789}_b12_deadband003_audit_20260804`
- Treinos DAgger:
  `runs/jobs/ppo-residual-battery-annual-dagger-s123-23febf8` e
  `runs/jobs/ppo-residual-battery-annual-dagger-s{456,789}-876293a`
- Avaliações congeladas:
  `runs/jobs/ppo-residual-battery-dagger-frozen-annual-eval-s{123,456,789}-*`
