# Community: diagnóstico dos picos nas quatro semanas de 2024

Análise dos resultados já concluídos da [calibração](community_peak_calibration_2026_11_03.md),
a pedido de Tiago. Não foram lançadas novas simulações nem alterados os controladores.
Mantêm-se Simulator 2.0.0, Core30 Normal v3.3, preços, mercado e serviço da campanha.
São quatro episódios independentes de sete dias, não um mês contínuo ou um ano.

## O que significa o limiar

`community_import_threshold_kw: 15` é um **gatilho agregado de apoio das baterias**,
não um limite físico de importação nem 15 kW por casa. Em
`RBCCommunityPolicy._storage_demand`, o orçamento comunitário de importação tem
de atingir o limiar **e** o preço tem de ser classificado como caro. A regra
acrescenta procura para descarga até ao excesso sobre o limiar, sujeita à potência,
energia, reservas e limites elétricos. As regras locais continuam a existir.

Por exemplo, observar 30 kW com um limiar de 15 kW não garante reduzir a rede a
15 kW. O pedido adicional só surge se o preço for caro; as três baterias somam
12,68 kW nominais, e a taxa estratégica de 65% pede no máximo 8,242 kW antes dos
restantes limites. Não se suspendem automaticamente os carros ou as lavagens.

O original calcula 7 kW locais × 30 membros = 210 kW. É muito superior aos picos
observados. Preservá-lo como controlo reprodutível não significa que seja um
limiar adequado para limitar os picos desta comunidade.

## Método e verificações

Recolheram-se os 720 exports Parquet de edifícios/carregadores dos 16 jobs de
validação. Estes complementam as séries comunitárias, preços, baterias e auditorias
de serviço já guardadas. Para cada um dos **10 752 intervalos**:

`rede = consumo não controlável + eletrodomésticos + carros + bateria − solar`.

A energia dos eletrodomésticos foi reconstruída pelos inícios efetivos da auditoria
de serviço e pelos perfis originais de cada ciclo. Foram respeitados os sinais
negativos de produção solar e excluídos os sentinelas −1 dos carregadores. O erro
máximo de fecho foi **0,000002679 kW**, compatível com o arredondamento dos exports.
Os máximos calculados coincidem com os já reportados.

A classificação de preço foi reconstruída com as séries da campanha, os horizontes
de publicação do Simulator 2.0.0 e a função original `_get_price_context` do RBC.
Inclui as três previsões existentes e as cinco previsões derivadas. «Barato» é
relativo às previsões recebidas, não a uma hora fixa ou a um preço absoluto.
Esta reconstrução de sinais exógenos não é um registo completo dos ramos internos
executados pelo controlador. Os exports de SOC dos EVs são arredondados e posteriores
à ação; não bastam para certificar a flexibilidade de cada carga.

## Os quatro máximos do Community 15

Horas de Portugal continental. Valores em kW médios de 15 minutos. Bateria
positiva significa carga; negativa significa descarga. Solar é produção subtraída.

| Instante | Rede | Não controlável | Eletrodomésticos | Carros | Bateria | Solar |
|---|---:|---:|---:|---:|---:|---:|
| 3 janeiro, 14:00 | 55,373 | 30,214 | 2,816 | 31,464 | −3,328 | 5,794 |
| 1 abril, 19:15 | 34,494 | 33,314 | 0,952 | 5,152 | 0,000 | 4,924 |
| 4 julho, 01:00 | 25,655 | 5,663 | 0,000 | 15,456 | +4,536 | 0,000 |
| 3 outubro, 02:00 | 38,354 | 4,986 | 0,000 | 25,760 | +7,608 | 0,000 |

**Os quatro máximos ocorreram em períodos classificados como baratos.** A condição
de preço caro do apoio comunitário não se verifica nesses instantes, qualquer que
seja o limiar. Dos 28 máximos diários do Community 15, 18 foram baratos, sete neutros
e apenas três caros. Entre os 80 intervalos formados pelos 20 maiores de cada
semana, 64 foram baratos, dez neutros e seis caros. A condição de preço exclui
assim a maioria dos momentos que se pretende aliviar.

- **Janeiro:** cinco carregadores, concentrados nos membros 025 e 028, consumiam
  31,464 kW. Esses dois membros representavam 32,448 kW líquidos dos 55,373 kW
  comunitários. Apenas a bateria do membro 025 descarregava, a 3,328 kW; as outras
  duas estavam paradas. Antes do intervalo, os SOC das três eram aproximadamente
  41,0%, 48,9% e 36,1%, acima da reserva estratégica de 20%. Existia energia
  armazenada; a ausência de apoio adicional não se explica por baterias vazias.
- **Abril:** predominava o consumo não controlável; os carros contribuíam apenas
  5,152 kW. As três baterias estavam perto de 90% e não descarregaram. Uma política
  centrada só no preço conserva-as nesse momento, embora o objetivo de pico possa
  justificar outra decisão. A potência efetivamente disponibilizável teria de
  respeitar as ligações e fases.
- **Julho:** a importação sobe de 8,525 para 25,655 kW em 15 minutos, quando o preço
  passa de 0,0800 para 0,0725 €/kWh. Cinco carregadores pedem 15,456 kW enquanto as
  baterias carregam a 4,536 kW. Há sincronização de cargas na oportunidade barata.
- **Outubro:** a importação sobe de 9,240 para 38,354 kW quando o preço passa de
  0,0400 para 0,0178 €/kWh. Oito carregadores consomem 25,760 kW e as baterias
  7,608 kW. No Community 15, as baterias estavam perto da reserva de 20%; aqui
  o primeiro recurso de pico é evitar ou distribuir carga adicional, não pressupor
  que se pode descarregar. O original tinha mais energia armazenada, mas carregou
  as baterias à mesma potência e teve o mesmo pico.

Uma sensibilidade contabilística ilustra a escala: retirar apenas a carga das
baterias desses intervalos daria **21,119 kW em julho** e **30,746 kW em outubro**.
Não são resultados de uma política validada: a energia teria de ser reposta noutro
momento e poderia surgir outro máximo. Mostram, porém, uma parcela diretamente
associada à decisão de carregar baterias durante o pico.

## Limitação da política, além do parâmetro

O Community operacional partilha excedentes e acrescenta apoio das baterias,
mas o caminho económico dos EVs ainda termina na decisão de preço/serviço do
Basic quando não há excedente. Não há um orçamento agregado de importação que
distribua globalmente a carga flexível dos carros. Cada casa pode respeitar o seu
limite e, mesmo assim, todas carregarem simultaneamente.

Também não existe uma passagem prévia que reúna todos os pedidos antes de decidir
as baterias: o despacho percorre os membros e, dentro de cada membro, EVs,
eletrodomésticos e bateria. Os pedidos de membros posteriores ainda não foram
incorporados quando uma bateria anterior decide. Esta é uma limitação estrutural;
os exports disponíveis não quantificam isoladamente o seu efeito causal.

Há ainda limites de potência/energia e medições endógenas do último intervalo
concluído. Logo, remover a condição de preço caro ou reduzir mais o limiar não
garante, por si, um pico inferior nem um melhor custo ao longo do episódio.

## Custo e comparação com o Basic

Mantém-se o KPI oficial de custo liquidado, sem recalcular outra contabilidade.

| Política | Custo dos 28 dias (€) | Máximo (kW) | Média dos máximos diários (kW) |
|---|---:|---:|---:|
| Basic | 421,641 | 58,701 | 31,422 |
| Smart Local | 420,035 | 55,373 | 30,572 |
| Community original | 419,405 | 55,373 | 29,928 |
| Community 15 | 410,511 | 55,373 | 29,911 |

Community 15 poupou **8,894 € (2,121%)** relativamente ao original no conjunto
das 30 casas. Houve redução nas quatro semanas; não foi demonstrada significância
estatística ou generalização anual. A energia movimentada nas baterias cresceu
57,53%, e a energia armazenada no último estado foi, em média, 4,114 kWh inferior
por episódio. Não se igualaram reservas finais nem se monetizou desgaste. A
melhoria é do custo de energia observado, não do custo total de propriedade.

Smart Local já foi ligeiramente melhor no agregado: **1,605 € (0,381%)** de
poupança face ao Basic, máximo 5,669% inferior e média diária aproximadamente
2,71% inferior. Não dominou todas as semanas; em abril o custo foi 0,015 € superior.

Todos os controladores conservaram os critérios da campanha: serviço mínimo
viável EV ≥99,9% (observado 100%), precisão de alvo ≥80% (observado 96,875%),
cobertura completa, zero ciclos perdidos, energia de violação elétrica ≤1e−6 kWh
(observado zero), zero eventos e paridade dos 32 exemplos exportados. Isto abrange
352 saídas EV e 302 ciclos por controlador; cumprir o mínimo e atingir o alvo
com precisão são indicadores distintos.

## Proposta a discutir antes de implementar

1. Dar ao Community uma regra explícita para evitar carga de baterias que agrave
   o pico agregado, incluindo períodos baratos. Recolher primeiro a procura dos
   membros para decidir com uma visão conjunta.
2. Distribuir o carregamento EV que tenha folga entre intervalos, protegendo
   serviço, prazos, potência mínima e limites físicos. Examinar depois os arranques
   de eletrodomésticos; não interromper ciclos em curso.
3. Separar a decisão de alívio de pico da classificação de preço, com reservas
   de bateria e tolerância de custo explícitas. Calibrar em 2023 e medir efeitos
   nos mesmos 28 dias; comparar também desgaste aproximado por throughput e
   reservas finais. Como 2024 já informou este diagnóstico, reservar outras
   janelas para uma nova validação independente se a política for desenvolvida.

Para o **Basic**, uma simplificação defensável seria usar preço atual e limiares
fixos de barato/caro definidos em 2023, sem consultar os vários horizontes de
previsão. Manteria a proteção comum de serviço e a decisão simples por preço e
urgência. Smart Local conservaria previsões, solar e gestão local; Community
acrescentaria coordenação agregada. A proposta define uma diferença de capacidade,
não garante que o Basic fique sempre mais caro. Preservar o resultado e a versão
atuais como referência histórica; decidir a especificação antes de comparar a nova.
Nenhuma destas alterações foi aplicada.

Os BAUs mantêm as definições operacionais e a evidência da auditoria anterior.
Esta análise não os voltou a simular. As restantes famílias e a etapa 5 continuam
fora deste trabalho.

## Evidência reproduzível

- Campanha: `runs/community-peak-core30-20261103/`.
- Análise: `analysis/inspect_2024_peaks.py`, `analysis/plot_2024_peak_attribution.py`.
- Resultados: `validation/remote-results/peak-attribution/`, com `weekly.json/csv`,
  `daily.json/csv`, `statistics.json/csv`, `intervals.parquet` e `checks.json`.
- Figura: `peak_attribution_2024.png` e PDF no mesmo diretório.
- Código auditado: `algorithms/agents/baseline_policies.py`, especialmente
  `predict`, `_get_price_context`, `RBCBasicPolicy`, `_EconomicSmartPolicy` e
  `RBCCommunityPolicy._storage_demand`.

Os dados brutos e scripts de diagnóstico permanecem na pasta ignorada de runs;
este documento é o registo conciso no repositório. A recolha foi apenas leitura
dos jobs já terminados, sem alterar pastas ou estado remoto.
