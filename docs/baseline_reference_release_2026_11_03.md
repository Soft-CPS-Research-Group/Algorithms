# Fecho das referências BAU, Basic, Smart e Community

Decisão de Tiago em 03/11/2026, após a calibração e o diagnóstico dos picos.

| Nome público | Identificador do job | Decisão |
|---|---|---|
| BAU com bateria | `NormalPolicy` | Mantém o comportamento auditado. |
| BAU sem controlo da bateria | `NormalNoBatteryPolicy` | Mantém o comportamento auditado. |
| Basic | `RBCBasicPolicy` | Preço atual e limiares fixos, mantendo as proteções de serviço. |
| Smart | `Smart` | Mesmo controlador local validado anteriormente, com novo nome público. |
| Community | `RBCCommunityPolicy` | Adota o gatilho agregado de 15 kW como referência oficial. |

## Community e Smart

O Community oficial é a configuração que a campanha chamou «Community 15».
O parâmetro continua explícito e exportado; 15 kW é um gatilho de apoio das
baterias em períodos caros, não um teto físico. Não se implementou a coordenação
adicional de picos sugerida no diagnóstico. A decisão de Tiago sucede à recomendação
inicial de preservar 210 kW; os relatórios históricos conservam os resultados e
a decisão que existiam na altura. A adoção não implica uma nova alegação de
redução do pico: na validação, o custo era 410,511 €, o máximo 55,373 kW e a
média dos máximos diários 29,911 kW, nas quatro semanas de 2024.

`Smart` é um nome registado que aponta exatamente para a implementação
`RBCSmartLocalPolicy`. Os novos jobs e bundles usam `Smart`. O identificador
antigo permanece disponível e o template `smart_local.yaml` é um caminho de
compatibilidade para `smart.yaml`. O histórico `RBCSmartPolicy`, usado por
professores/residuais, mantém o seu comportamento; não se trocou essa implementação
pela política local. O catálogo tem 38 nomes, incluindo o novo alias.

## Simplificação do Basic

O Basic deixou de consultar as previsões para decidir se o preço é favorável.
Usa o preço atual em €/kWh, com:

- barato: preço ≤ **0,0765**;
- caro: preço ≥ **0,1078**;
- neutro: entre os limiares, ou se não houver preço disponível.

Estes valores são os tercis 1/3 e 2/3 das **35 040 observações de 2023** de
`shared/pricing_omie.parquet` do Core30 v3.3, arredondados a quatro casas decimais.
Os valores sem arredondamento são 0,0764933327833811 e 0,107843334476153.
SHA-256 do ficheiro de origem:
`9877634adf48afb13db9464a667c9092d4c62f8b237d5beccd7ac01771c8ee61`.
Só as primeiras 35 040 linhas foram usadas; os dados de 2024 e os resultados dos
controladores não entraram nesta escolha. Não houve procura de limiares para
forçar uma ordenação de custos.

Continuam iguais as taxas de carga/descarga, os objetivos SOC, a proteção de
prazos EV/eletrodomésticos, a potência mínima dos carregadores e os limites
elétricos. O Basic não usa V2G. Uma tarifa constante abaixo do limiar continua
a ser barata para esta regra; não existe fallback horário no Basic atual.

Smart, Community e os professores históricos conservam o classificador relativo
às previsões. A herança da classe Basic foi explicitamente isolada neste ponto.
`basic_price_policy: forecast_relative` reproduz a classificação anterior em
novas experiências. Bundles executáveis antigos sem esse campo conservam a
semântica anterior ao serem carregados; os novos serializam o modo e os limiares.

Os templates oficiais estão em `configs/templates/baselines/core30/`: `bau.yaml`,
`bau_no_battery.yaml`, `basic.yaml`, `smart.yaml`, `community.yaml`.

## Validação

A validação verifica invariância do Basic a previsões, decisões de serviço urgente,
respeito pelos limites, preservação do classificador Smart/Community, igualdade
dos dois nomes Smart, configurações e roundtrip dos bundles novos e históricos.

No checkout limpo do commit de implementação
`258fc6009e7df83dea08270a50cadd89fbda31e9`, com Simulator 2.0.0 publicado:
**1 971 testes gerais passaram, 23 foram ignorados com motivos explícitos, e
69 testes de jobs completos/exportação passaram**. Ruff, formatação e mypy
passaram. Foi usada a configuração CPU da CI. Os ficheiros locais de experiências
ignorados pelo Git ficaram preservados e fora do checkout de aceitação.

Repetiram-se apenas as quatro semanas de 2024 do Basic pelo runner comum,
em dois processos CPU locais, com zero treino, seed de avaliação 6763,
672 intervalos por semana e verificação obrigatória de 32 decisões por bundle.
O dataset e os 87 ficheiros referenciados nos manifests, os preços exportados,
as janelas, o mercado e o Simulator coincidiram com a campanha anterior.
O caminho local do dataset e o nome dos jobs mudaram; as hipóteses económicas
e de serviço permaneceram iguais. Native BAU desativado, como na comparação anterior.

| Semana de 2024 | Basic atual (€) | Basic anterior (€) | Máximo atual (kW) | Média dos máximos diários (kW) |
|---|---:|---:|---:|---:|
| 1–7 janeiro | 192,792932 | 187,310688 | 58,700710 | 40,192889 |
| 1–7 abril | 15,066227 | 13,222251 | 58,909031 | 37,443913 |
| 1–7 julho | 77,316247 | 68,686144 | 28,464643 | 21,804848 |
| 1–7 outubro | 158,424730 | 152,421481 | 51,828156 | 38,923922 |

O Basic atual totalizou **443,600136 €**, mais **5,208%** que os 421,640563 €
anteriores. O máximo foi 58,909031 kW e a média diária 34,591393 kW. Foram
observadas 352 saídas EV e 302 ciclos concluídos, sem ciclos perdidos ou
violações elétricas. Serviço mínimo viável 100%, precisão de alvo 341/352 =
96,875%, cobertura completa e zero desconhecidos. SOC das três baterias ativas
entre 30,474% e 90%, throughput 133,942 kWh. Os exports de baterias vazias de
edifícios sem armazenamento foram excluídos da estatística SOC. Todos os bundles
passaram os 32 exemplos com erro máximo zero.

Critérios mantidos: serviço mínimo viável ≥99,9%, precisão ≥80% em cada semana,
coberturas 100%, zero ciclos perdidos, energia de violação ≤1e−6 kWh e zero
eventos, SOC físico em [0,1] com tolerância 1e−6. Os quatro jobs passaram.

| Referência atual | Custo dos 28 dias (€) | Máximo (kW) | Média dos máximos diários (kW) |
|---|---:|---:|---:|
| Basic | 443,600136 | 58,909031 | 34,591393 |
| Smart | 420,035168 | 55,372711 | 30,571767 |
| Community | 410,510912 | 55,372711 | 29,911209 |

Os resultados Smart e Community são os da campanha anterior, cujas políticas
foram preservadas; não foram simulados de novo neste fecho. Smart custa 5,312%
menos que o novo Basic neste conjunto. A simplificação aumentou significativamente
o pico de abril: 34,494 → 58,909 kW, embora o maior pico das quatro semanas
tenha mudado pouco. Não se promete uma hierarquia universal ou desempenho anual.
Não se ajustaram os limiares depois de observar estes resultados.

Evidência local: `runs/baseline-reference-close-20261103/`, incluindo `protocol.json`,
`execution.json`, `summary.json/csv`, configurações, logs, outputs completos dos
quatro jobs e os scripts `analysis/run_basic.py` e `analysis/score_basic.py`.
Os jobs são `closed-2024-01-basic`, `closed-2024-04-basic`, `closed-2024-07-basic`
e `closed-2024-10-basic`. Os testes de aceitação encontram-se também registados
no PR deste fecho. Nenhum job remoto anterior foi alterado.

As avaliações anteriores do Basic descrevem a versão com previsões. Não devem
ser apresentadas como resultados desta política simplificada. O diagnóstico
dos picos fica em [documento próprio](community_peak_diagnosis_2026_11_03.md).
As restantes famílias, integração especializada e etapa 5 continuam adiadas.
