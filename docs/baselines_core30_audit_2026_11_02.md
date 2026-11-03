# Auditoria dos baselines no Core30 e Simulator 2.0.0

## Âmbito e critério de fecho

Esta revisão cobre `NormalPolicy`, `NormalNoBatteryPolicy`, `RBCBasicPolicy`,
`RBCSmartLocalPolicy`, `RBCCommunityPolicy` e as referências MILP. O objetivo é
ter implementações coerentes com a sua política, execução no ciclo comum,
KPIs oficiais e exportação verificável. Não se altera o Basic para fabricar uma
hierarquia de resultados. A etapa 5 e a integração especializada do TIMARL
continuam adiadas.

O contrato comportamental está em [baseline_controllers.md](baseline_controllers.md).
As regras são políticas online determinísticas com estado/configuração explícitos;
os MILPs são referências offline com conhecimento do futuro. Os primeiros exportam
bundles de regras reproduzíveis; os planos MILP são artefactos de investigação
para a janela calculada, sem promessa de inferência em cenários novos.

## Ambiente e dados

- Pacote oficial PyPI `softcpsrecsimulator==2.0.0`, publicado em 02/11/2026.
  Wheel SHA-256 `8a9d2fb44a9e8e708364e99c95a31d79e52b6da4481cd0bebbe2246d073d2b94`.
- Fonte/fixtures do Simulator: `a0adc12cb6ea626f401484d84ae0a10fad8125c9`,
  release `softcpsrecsimulator-v2.0.0`. O tag upstream `v2.0.0` não é esta release.
- Dataset `rec_pt_2023_core_30_v2`, `CURRENT_EVALUATION.json`
  `portugal-native-2023-2024-v3.3`. O nome histórico não significa apenas 2023:
  contém 70 176 amostras de 15 minutos, cobrindo 2023 e 2024.
- Core30 Normal: 30 membros, 15 carregadores e 3 baterias estacionárias físicas.
  O schema contém também equipamentos de capacidade nula, que não são baterias úteis.
- Mercado ativo: preço local de 0,8 do preço de compra à rede; venda externa a zero.
  As quatro séries de preço foram verificadas no período completo: mínimo zero,
  máximo aproximadamente 0,22 €/kWh, sem preços negativos.
- Normal schema SHA-256 `990ece4b767f3b1d59ecabd82845f355ba6051f2fc79d4025b62e08f052c9eab`.
  A cópia remota conserva os 618 ficheiros de origem, verificados por SHA-256;
  os dados originais não foram alterados.
- Todos os jobs de comparação usam seed de configuração 22, seed de avaliação
  derivada 6763, interface entity, uma instância comunitária do controlador,
  zero episódios de treino e uma avaliação determinística. Exportação apenas
  final, MLflow e métricas de sistema desligados.
- Cada janela começa com um reset. Estes resultados não equivalem a recortes
  de uma simulação anual contínua, nem a uma validação anual.

## Políticas e diferenças esperadas

| Referência | Serviço EV e eletrodomésticos | Bateria e informação |
|---|---|---|
| BAU sem controlo de BESS | Carrega imediatamente até 100%; inicia o ciclo no primeiro instante permitido. | Ação de bateria zero; perdas passivas podem continuar. |
| BAU com BESS | O mesmo serviço imediato. | Autoconsumo local, sem arbitragem, com limiares estratégicos de SOC 20–90%. |
| Basic | Usa preço relativo e urgência para atingir o SOC pedido; lava por preço ou prazo. | Arbitragem simples de preço, descarga para consumo local útil. |
| Smart Local | Mantém o serviço e acrescenta oportunidades solares locais. | Reserva para preços altos, captura solar e alívio de pico local; não consulta sinais comunitários. |
| Community | Acrescenta acesso ao excedente partilhável da comunidade. | Orçamento comum sem duplicação de excedente; apoio a pico comunitário em períodos caros. Sem mercado, reproduz a política local. |

Não existe uma restrição fixa de carregar só de dia ou só de noite. Os preços
publicados e o serviço determinam as decisões. O horário só é fallback quando
falta preço. V2G está desligado por omissão nos três RBCs.

A hierarquia é de informação e capacidades, não uma garantia de dominância.
O Community tem uma ordem de atribuição determinística e não otimiza globalmente
a repartição entre membros. Por omissão, o seu limiar de pico agregado é
7 kW vezes o número de membros: 210 kW para 30 membros. Assim, um pico inferior
não ativa essa regra comunitária, embora as regras locais continuem ativas.
Calibrar esse limiar é uma decisão de protocolo, a fazer nos dados de desenvolvimento
de 2023 antes de avaliar 2024, sem escolher parâmetros pelos resultados de teste.

O BAU presta mais energia aos EVs do que os controladores que seguem o pedido de
saída. A diferença de custo inclui essa diferença de serviço. A precisão face ao
SOC pedido não é critério de aprovação do BAU: o seu alvo declarado é 100%.

## Problemas corrigidos

1. Referências de instalação e CI atualizadas de Simulator 1.9.x para 2.0.0.
   As fixtures usam o commit exato da release do fork.
2. Os BAU deixaram de reservar sequencialmente a margem elétrica para o primeiro
   carregador. Pedem potência máxima simultaneamente; o Simulator reparte a
   capacidade. Um ensaio físico com dois EVs e limite de 4 kW entregou 2 kW a cada um.
3. O BAU deixou de reduzir o comando por uma estimativa de energia remanescente
   perto de 100%, evitando a zona morta da potência mínima do carregador.
   A estimativa de potência aceite continua a limitar o consumo antecipado
   que a bateria da casa tenta servir; comando máximo não significa consumo máximo.
4. O Basic passou a limitar a energia de carga ao espaço até ao teto de SOC.
   Antes podia passar de menos de 90% para 96,9% num único passo no Core30.
5. O MILP preserva fases não ligadas com limite zero e resolve equipamentos sem
   fase explícita pela fase predefinida da ligação. A distribuição equilibrada
   anterior tornava planos de casas monofásicas artificialmente inviáveis.
6. Planos MILP incluem equipamentos sem pedidos na janela com ações zero.
   A ausência dessas séries fazia o replay estrito rejeitar planos válidos.
7. O MILP de serviço fixado lê CSV, Parquet direto e partes Parquet ordenadas,
   incluindo os nomes `Member_001`. Rejeita partes em falta e representações ambíguas.
8. Adaptadores estáticos rejeitam topologia dinâmica, perturbações, outages e
   equipamentos com ações desativadas, em vez de certificar um problema que omite esses mecanismos.
9. O coletor remoto usa os KPIs finais oficiais do runtime quando o orquestrador
   só lista CSV. Preserva o JSON e cria uma vista CSV para os relatórios existentes;
   não recalcula custos nem transforma métricas inválidas em zeros.
10. Baterias instaladas ou substituídas durante o episódio usam potência,
    capacidade e fase observadas. O schema inicial já não determina a escala
    de um equipamento que entretanto mudou.
11. O runtime reconhece a preparação isolada `attempt_artifacts_v1` do worker
    0.7.0. O config pré-criado já não provoca uma segunda pasta de tentativa que
    escondia os resultados da API. A adoção exige o layout e ligação corrente
    esperados, saídas ainda vazias e obtenção da marca exclusiva de execução.
    Repetições locais e artefactos já existentes continuam preservados.
12. Há cinco templates Core30 atuais, com os defaults desta auditoria. Os
    exemplos históricos mantêm a sua identificação e os seus parâmetros;
    já não é preciso escolher um desses exemplos para iniciar uma referência atual.

### Verificação do código

Passaram 1 949 testes gerais, com 23 omissões por condições explicitadas, e
68 testes de execução completa/exportação. Ruff, formatação e o conjunto
configurado de verificações mypy passaram. A instalação com o ficheiro de
constraints atualizado foi resolvida sem conflito. Os cinco templates Core30
passaram a validação de configuração; o template BAU inicializou o ambiente
em modo de inspeção, sem simular passos. Os testes e ensaios usam o pacote
instalado do Simulator, não uma importação do checkout de desenvolvimento.


## Evidência local

Os números abaixo são KPIs do Simulator depois de executar os controladores ou
os planos. O custo é `district_cost_community_market_settled_total_eur`.
O pico absoluto é o máximo da energia líquida exportada na série comunitária,
dividido por 0,25 h; não é a soma de máximos das três fases.

Janela Normal: transições `[0,192)`, 1–2 de janeiro de 2023. O estado terminal é 192.

| Controlador | Custo (€) | Pico de importação (kW) | Serviço mínimo EV | Lavagens concluídas/perdidas | Violações elétricas (kWh) |
|---|---:|---:|---:|---:|---:|
| BAU sem BESS | 83,1372 | 61,1766 | 100% | 15 / 0 | 0 |
| BAU com BESS | 81,2115 | 60,2726 | 100% | 15 / 0 | 0 |
| Basic corrigido | 48,3009 | 43,6475 | 100% | 15 / 0 | 0 |
| Smart Local | 48,4250 | 37,1028 | 100% | 15 / 0 | 0 |
| Community | 48,2807 | 37,1028 | 100% | 15 / 0 | 0 |
| MILP individual, replay conjunto | 47,9207 | 44,4666 | 100% | 15 / 0 | 0 |
| MILP comunitário, replay conjunto | 43,1619 | 39,2166 | 100% | 15 / 0 | 0 |
| MILP de bateria, serviço Smart Local | 45,8049 | 38,2745 | 100% | 15 / 0 | 0 |
| MILP de bateria individual | 49,8931 | 35,7715 | 100% | 15 / 0 | 0 |
| MILP de bateria para picos/rampas | 47,6050 | 31,3721 | 100% | 15 / 0 | 0 |

Há 20 partidas EV. Nos dois BAU, as 20 partiram com SOC de 100%, confirmado nas
séries exportadas de cada EV. Nos RBCs, 80% ficaram dentro da tolerância de precisão do
Simulator. Os sete ligeiros défices em relação ao alvo exato ficaram abaixo
de 0,1 ponto percentual, a tolerância de decisão configurada. Há também EVs
que já chegam acima do SOC solicitado; sem V2G, o controlador não os deve
descarregar apenas para melhorar esse indicador. Não confundir precisão com
incumprimento de serviço mínimo.

As baterias dos RBCs ficaram dentro dos limites físicos. Depois da correção,
o Basic também respeitou o teto estratégico de 90%. O BAU pode terminar um passo
ligeiramente abaixo do limiar estratégico de 20% devido a perdas; esse limiar
não é o mínimo físico do dataset (10%). A observação de consumo/geração é do
último intervalo concluído; autoconsumo não garante exportação nula por passo
face a variações, estimativas de energia aceite e perdas.

`TotalHomeOracleReplayPolicy` também executou `Member_001`, `[0,160)`, com dois
serviços EV fechados: custo 0,0381144 €, serviço mínimo 100% e zero violações.
Esta casa isolada não é comparável à tabela de 30 membros.

Os cinco controladores passaram ainda um ensaio de mudança de topologia em
`[39116,39124)`, atravessando a entrada de `Member_031` no passo 39120.
Os bundles dos cinco reproduziram as decisões através do contrato comum.
Os cinco passaram também `[47848,47860)`, atravessando a instalação da bateria
de `Member_010` no passo 47852. A correção de metadados reproduziu sem diferença
numérica as 32 referências estáticas guardadas de cada um dos cinco controladores.

### Limites das referências MILP

O modelo individual obteve limite inferior 60,8777 € e plano linear conservador
60,9095 €; esse objetivo individual é anterior à liquidação comunitária.
O modelo comunitário obteve 43,1745 €. A execução produziu 43,1619 € porque a
aproximação linear e a física do Simulator não são idênticas. Os certificados
são do modelo declarado, não provas de ótimo do Simulator não linear.

Os oito serviços EV ainda abertos na fronteira direita não receberam um alvo
artificial de saída. `boundary_service_exact` é falso. As baterias do MILP têm
restrição terminal; os RBCs não recebem a mesma restrição. Por isso, estes
valores validam execução e contabilização, mas não demonstram superioridade
económica em condições terminais equivalentes. Uma campanha de distância ao
ótimo exige fixar essas condições e a política de serviços que atravessam a janela.

O MILP de bateria mantém o serviço fixo no problema matemático. No replay,
o RBC volta a decidir serviço e as proteções físicas podem intervir. Não se
declara equivalência integral das trajetórias apenas por o custo reconstruído
de origem coincidir com o export oficial.

A variante individual de bateria minimiza compras isoladas, com venda a zero;
o seu custo comunitário pode ser pior do que o Smart Local. A variante de
picos/rampas foi resolvida com teto linear de custo igual aos 48,4250 € do
Smart Local e pesos de rampa/pico diário/pico global de 1/1/0,25. Produziu no
Simulator um pico menor (31,3721 kW), mantendo serviço mínimo e sem violações.
O teto pertence ao problema linear; a execução é sempre verificada separadamente.

## Validação remota e decisão

Campanha executada na pasta pessoal **Baselines Core30**, de Tiago
Fonseca, no worker CPU `server`: cinco controladores nas janelas Normal de
inverno `[35040,35712)` e verão `[52512,53184)`, sete dias cada, e Combined
`[39072,39360)`, três dias. Esta última cruza admissão de membro e perda de
observações; não representa cobertura de todas as perturbações possíveis.

### Semanas sazonais, 2024

| Referência | Inverno: custo (€) | Inverno: pico (kW) | Verão: custo (€) | Verão: pico (kW) |
|---|---:|---:|---:|---:|
| BAU sem BESS | 302,0420 | 81,0246 | 133,7387 | 46,7757 |
| BAU com BESS | 299,5291 | 81,0246 | 128,4457 | 42,3957 |
| Basic | 187,3107 | 58,7007 | 68,6861 | 28,1974 |
| Smart Local | 186,5761 | 55,3727 | 68,5375 | 25,6552 |
| Community | 186,5338 | 55,3727 | 68,0941 | 25,6552 |

Nestas duas semanas, Smart Local teve menor custo e pico do que Basic.
Community teve custo ligeiramente menor do que Smart Local e o mesmo pico.
Isto descreve estes ensaios, sem impor uma ordenação para outros períodos.

### Janela dinâmica Combined, 2024

| Referência | Custo (€) | Pico de importação (kW) | Serviço mínimo EV | Precisão EV |
|---|---:|---:|---:|---:|
| BAU sem BESS | 118,7665 | 60,5743 | 100,0% | não se aplica ao alvo BAU |
| BAU com BESS | 118,1520 | 60,6408 | 100,0% | não se aplica ao alvo BAU |
| Basic | 72,4890 | 48,7443 | 100,0% | 100,0% |
| Smart Local | 72,5392 | 48,7358 | 100,0% | 100,0% |
| Community | 72,4753 | 48,7358 | 100,0% | 100,0% |

As 15 referências finais passaram os critérios declarados abaixo: serviço mínimo EV 100%,
zero violações elétricas, zero lavagens perdidas, cobertura de auditoria completa e
zero partidas de viabilidade desconhecida. A precisão dos RBCs variou entre 88,5%
e 100%. Os 15 bundles passaram 32 amostras cada, com erro máximo de reprodução zero.

Na janela dinâmica, o Smart Local ficou ligeiramente mais caro do que o Basic.
A ordenação observada nas duas semanas sazonais não é, portanto, universal.
Os SOCs das baterias controladas respeitaram o teto estratégico de 90%.
A ligeira descida abaixo de 10% nas baterias inicializadas nesse valor ocorreu
por perdas passivas; nas amostras abaixo de 10% revistas não houve descarga.

Foram executados 19 jobs remotos: 15 referências finais e quatro diagnósticos
dinâmicos anteriores à correção dos metadados de bateria. Estes quatro estão
excluídos da comparação. As primeiras imagens exigiram recuperar a ligação da
API aos resultados originais, sem alterar os valores; essa intervenção ficou
registada. Os cinco jobs da imagem final `a0eeb04` publicaram resultados
diretamente, sem essa recuperação. Todos os 19 terminaram com sucesso.

A execução usou o worker CPU `server`, versão 0.7.0, capacidade de um job,
na pasta **Baselines Core30**, de **Tiago Fonseca**. As imagens de auditoria
foram publicadas pela CI e identificadas pelos commits `2d203f5`, `80fc555`,
`987e72e` e `a0eeb04`; tags e digests estão guardados com os payloads.
Os resultados estáticos usam a versão corrigida da respetiva política. A
correção posterior de metadados dinâmicos reproduziu sem diferenças as
referências estáticas guardadas dos cinco controladores.

O [histórico versionado](experiment_history/20261103_baselines-core30-sim200.md)
e o seu ledger guardam versões, janelas, decisões e IDs dos jobs. Os ficheiros
`remote-results/reference-scorecard.csv` e `reference-evidence.json` dentro da
campanha contêm a comparação explícita por schema, janela e seed. Os relatórios
legados Phase 6 foram produzidos em `remote-results/legacy-phase6/`, mas as suas
heurísticas históricas de nomes e precisão BAU não são a decisão desta auditoria.

Evidência bruta em `runs/baselines-core30-20261102/`; os resultados locais
anteriores às correções permanecem separados dos ensaios corrigidos.

O critério desta revisão é serviço mínimo EV ≥ 99,9% nas partidas classificadas
como viáveis, energia de violação elétrica ≤ 10⁻⁶ kWh, zero eventos de violação,
zero lavagens perdidas e SOC dentro de [0,1], com inspeção dos limites de descarga
e das perdas passivas. O mínimo de descarga não é uma barreira à autodescarga: uma
bateria inicializada a 10% e sem controlo pode descer ligeiramente abaixo disso.
Precisão EV ≥ 80% é um indicador
de revisão para os RBCs, acompanhado dos casos que chegam acima do alvo;
não se aplica ao BAU. Valores ausentes não são aprovação. Só depois destes
critérios se comparam custos e picos em janelas e condições iguais.


### Interpretação e trabalho seguinte

A referência BAU deve ser descrita como «pedido imediato de carga máxima até 100%,
sujeito aos limites do equipamento, ligação e tempo disponível». Não deve ser
apresentada como garantia de que todas as partidas atingem 100%. Nos ensaios
sazonais sem BESS, 84/87 partidas de inverno e 80/83 de verão atingiram 100%.
As seis exceções foram revistas nas séries de EV e carregador: houve carga em
todos os intervalos de ligação, incluindo o último; o comando não parou no
SOC pedido. Os quatro intervalos de aproximação ao carregador são estado
«incoming», e não tempo de ligação disponível. Os casos de inverno em
`EV-M004-01` receberam o máximo nominal em todos os 25 intervalos ligados.

Não se ajustaram parâmetros para impor Basic > Smart Local > Community em custo.
A reserva, o escalonamento solar e as regras de pico são decisões heurísticas,
não uma prova de ótimo económico. A diferença de 0,1241 € entre Basic e Smart
Local nas 48 h locais é pequena, e o Smart Local reduz o pico em 6,5447 kW nessa
mesma janela. Os SOCs terminais das três baterias são praticamente iguais entre
os três RBCs nessa janela; o sinal da diferença não se explica por uma grande
reserva terminal extra.

O fecho desta revisão refere-se à implementação e à evidência destas janelas.
Antes de usar uma ordenação como conclusão científica, o passo seguinte é fixar
um protocolo: serviço equivalente quando se pretende isolar apenas o preço,
condições terminais dos MILPs, limiar de pico comunitário e avaliação anual ou
janelas mais abrangentes. Calibração em 2023 e avaliação congelada em 2024.
Isto é trabalho de avaliação posterior; não reabre a etapa 5 nem o TIMARL.
