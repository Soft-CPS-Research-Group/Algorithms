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
Os resultados finais dos testes e ensaios são registados após a sua conclusão.

As avaliações anteriores do Basic descrevem a versão com previsões. Não devem
ser apresentadas como resultados desta política simplificada. O diagnóstico
dos picos fica em [documento próprio](community_peak_diagnosis_2026_11_03.md).
As restantes famílias, integração especializada e etapa 5 continuam adiadas.
