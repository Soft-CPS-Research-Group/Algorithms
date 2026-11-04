# Comparação atual dos baselines Core30

Decisão de Tiago em 03/11/2026: **BAU = `NormalTargetPolicy`**, com carga EV
imediata limitada ao SOC pedido, arranque imediato dos eletrodomésticos e bateria
doméstica em autoconsumo. **BAU 100% = `NormalPolicy`**, preservado como variante.
`NormalNoBatteryPolicy` continua disponível como variante auxiliar a 100%, sem
controlo da bateria doméstica; não é o denominador das comparações.

Os templates `bau.yaml` e `bau_target.yaml` selecionam a referência principal;
`bau_full_charge.yaml` seleciona a variante a 100%. Os templates Core30 desligam
o contrafactual nativo do Simulator, que tem outra definição. Se alguém o ligar,
os seus rácios continuam a ser diagnósticos nativos e não poupanças face ao BAU
principal. `results/kpis.json` declara esta distinção e exige um job de referência
correspondente; não inventa uma execução de BAU nem recalcula custos oficiais.

## Resultados comparáveis

Core30 v3.3, Simulator 2.0.0, 30 membros, seed de avaliação 6763, 14 dias:
1–7 janeiro e 1–7 julho de 2024 UTC. Janelas `[35040,35712)` e `[52512,53184)`,
15 minutos por intervalo, com resets separados. Dataset, hashes dos inputs,
preços e mercado coincidentes. Custos somados, pico máximo dos 1344 intervalos
e média dos 14 máximos diários. Não são resultados anuais.

| Controlador | Custo (€) | Redução do custo | Pico máximo (kW) | Redução do pico | Redução da média dos picos diários |
|---|---:|---:|---:|---:|---:|
| **BAU** | **270,08** | **0%** | **59,59** | **0%** | **0%** |
| Basic | 261,30 | 3,25% | 58,50 | 1,81% | 10,48% |
| Smart | 255,11 | 5,54% | 55,37 | 7,07% | 11,70% |
| Community | 250,99 | 7,07% | 55,37 | 7,07% | 12,49% |
| BAU 100% — variante | 427,97 | −58,46% | 81,02 | −35,98% | −39,20% |

Todas as percentagens usam `100 × (BAU − controlador) / BAU`. Um valor
positivo significa redução; um negativo significa aumento. A referência anterior
a 100% dava 41,35% de redução de custo ao Community; face ao BAU principal,
o mesmo resultado é 7,07%. Não houve alteração do resultado Community.

O Basic mantém o preço atual e a urgência para decidir EVs e eletrodomésticos,
com preço barato até 0,0765 €/kWh. A bateria passou a usar o mesmo autoconsumo
reativo do BAU: guardar excedente local e servir consumo local, sem arbitragem
de rede nem previsões. Esta alteração reduziu o custo do Basic anterior de
270,11 € para 261,30 € (3,26%) e mantém-no entre o BAU e o Smart neste agregado.
Smart e Community mantêm as políticas auditadas; o Community usa o gatilho
de apoio de 15 kW.

A escolha foi feita em janeiro/julho de 2023: menos 1,72% de custo do Basic,
sem aumento do pico máximo e com todos os critérios de serviço cumpridos.
Os parâmetros foram congelados antes das novas execuções de 2024. Uma primeira
tentativa de alterar apenas o limiar caro foi rejeitada por piorar o custo em
2023. As janelas de 2024 já tinham sido observadas anteriormente; esta é uma
validação temporal de regressão, não um conjunto de teste intocado. A ordenação
agregada não garante dominância em todas as semanas ou métricas.

Custos oficiais em `district_cost_community_market_settled_total_eur`.
Os números completos e a identidade do denominador estão em
[`validation/core30_baseline_comparison.json`](validation/core30_baseline_comparison.json).

## Serviço, pressupostos e limitações

Serviço mínimo viável 100%, 170 saídas EV e 153 ciclos concluídos por controlador,
zero violações elétricas e zero ciclos perdidos. Critérios: mínimo ≥99,9%,
precisão ≥80%, coberturas 100%, energia de violação ≤1e−6 kWh e zero eventos,
SOC físico em [0,1] ±1e−6 e verificação de exportação passada.

O BAU com limite não provocou ultrapassagens do alvo. O défice positivo de SOC
na saída foi 0,244 pontos percentuais em média, máximo 2,062, devido ao controlo
discreto e à potência mínima. Os RBCs podem ultrapassar ligeiramente o alvo;
os pedidos de SOC são iguais, mas a energia entregue não é exatamente igual.
Alguns EVs já chegam acima do pedido, e o BAU não os descarrega.

Compra à rede ao preço completo do dataset; transações internas a 80%; venda
externa a zero; PV próprio sem custo operacional de energia. As transações
internas cancelam no custo agregado. O SOC de chegada de cada sessão vem do
dataset. Não se impõe igualdade de SOC terminal das baterias nem preço do desgaste.

Evidência: [promoção do BAU e revisão do Basic](experiment_history/20261103_bau-main-basic-20261103.md),
[ensaio do BAU com limite](experiment_history/20261103_bau-user-target-20261103.md),
`runs/bau-main-basic-20261103/`,
`runs/bau-user-target-20261103/`, `runs/bau-reference-comparison-20261103/`,
`runs/baseline-reference-close-20261103/` e
`runs/community-peak-core30-20261103/validation/remote-results/`.

As referências offline com conhecimento do futuro são comparadas separadamente
na [auditoria MILP](milp_core30_audit_2026_11_03.md), com os mesmos baselines e
limites terminais explícitos. Não alteram a definição dos controladores desta tabela.

## MILP com decisão em cada passo

A [referência física atual](milp_physics_reference_core30_2026_11_03.md) substitui os valores MILP do primeiro ensaio abaixo, com física alinhada, validação das trajetórias e intervalo residual de ótimo. Os baselines desta página mantêm-se inalterados.

A [comparação de 24 horas](milp_mpc_core30_2026_11_03.md) repete estes quatro baselines e acrescenta MILP a cada passo: 222,29 € (−17,70% face ao BAU), pico de 61,79 kW (+3,69%) e serviço mínimo viável de 100%. O semanal fica separado como oráculo offline: 222,97 € (−17,44%), com reservas finais por ativo não inferiores ao MPC. Esta diferença pequena reforça que o certificado do modelo linear não é um certificado de ótimo global do Simulator. As políticas desta página permanecem inalteradas.
