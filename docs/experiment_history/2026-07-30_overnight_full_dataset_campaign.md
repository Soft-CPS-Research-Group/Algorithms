# Campanha noturna: benchmark local e oráculos MILP

- Dataset: `citylearn_three_phase_electrical_service_demo_15min_parquet`
- Cobertura total: 35 040 passos de 15 minutos, 365 dias, 17 edifícios
- Janela de desenvolvimento: `0:1023` (10,67 dias)
- Janela de promoção: `0:35039` (ano completo)
- Mercado no track local: desligado
- Perfil de observações RL: `building_local_v1`
- Reward local: `LocalScorecardGuardRewardV2`

## Taxonomia vinculativa

| Track | Controlador | Unidade de decisão | Informação comunitária | Coordenação | Comparador correto |
|---|---|---|---|---|---|
| Edifício | `RBCSmartLocalPolicy` | um edifício | não | não | referência local |
| Edifício | PPO `count: 17` | 17 learners independentes, cada um com um edifício | não | não | RBC local e MILP individual |
| Edifício | TD3 `count: 17` | 17 learners independentes, cada um com um edifício | não | não | RBC local e MILP individual |
| Edifício | MILP individual | 17 problemas isolados | não | não | limite condicional por edifício |
| Comunidade | RBC Community / controladores MARL | conjunto dos 17 edifícios | sim | sim | baselines comunitárias |
| Comunidade | MILP comunitário | problema conjunto/agregado | sim | sim | limite condicional comunitário |

PPO e TD3 neste benchmark **não são controladores comunitários**. `count: 17`
significa 17 instâncias single-agent independentes; não existe critic partilhado,
reward comunitário, mercado, broadcast nem troca de observações entre casas.

## Força atual dos MILPs

Existem as duas topologias pedidas:

1. `individual_building_milp.py` resolve cada edifício isoladamente e soma os
   17 certificados;
2. `perfect_foresight_milp.py` resolve a bateria equivalente no problema
   comunitário com netting conjunto.

Esses dois solvers continuam a ser referências históricas **fixed-service,
battery-only**. Em paralelo foi implementada a formulação `total-energy`, que
decide bateria, EV/V2G, ciclos deferrable e envelopes elétricos total/L1/L2/L3,
com duas superfícies económicas: import positivo por casa (`individual`) e
netting distrital (`community`). O replay CityLearn continua obrigatório: um
ótimo do modelo linear não é automaticamente um ótimo do simulador não linear.

O MILP total individual foi inicialmente resolvido por decomposição estrutural
em 17 problemas e reproduzido em CityLearn. Uma reconciliação posterior
detetou, porém, que o extrator usava `1 - depth_of_discharge` quando o schema
não declarava `initial_soc`, enquanto o CityLearn inicializa esses EVs através
de um gerador determinístico dependente da seed e do identificador. Por isso,
os replays continuam a ser evidência empírica feasible, mas os certificados
lineares individual e comunitário produzidos antes da correção ficam
superseded como referências ótimas do dataset. A implementação passou a usar
exatamente o mesmo fallback determinístico do simulador; os benchmarks só são
readmitidos depois de novo solve, replay e scorecard. Essa repetição está
fechada mais abaixo: os MILPs total-energy individual e comunitário corrigidos
foram readmitidos no âmbito exato dos respetivos modelos lineares e replays.

## Correção de proveniência local

O `RBCSmartPolicy` histórico podia recorrer a forecast PV/headroom comunitário
quando faltava o equivalente local. Essa política continua disponível por
compatibilidade, mas os resultados antigos deixam de contar como prova
strict-local.

A campanha nova usa `RBCSmartLocalPolicy`, que:

- remove do mapa raw todas as observações cujo nome contém `community`,
  preservando os índices originais;
- neutraliza explicitamente todos os accessors comunitários herdados;
- é também o teacher de EV/deferrables de
  `FixedServiceOracleReplayPolicy`.

## Evidência curta regenerada

No horizonte `0:1023`:

| Candidato | Custo local | Gates locais | Edifícios melhores que RBC | Estado |
|---|---:|---:|---:|---|
| RBC Smart strict-local | EUR 673,2704 | 17/17 | referência | válido |
| MILP individual fixed-service, replay | EUR 629,1809 | 17/17 | 17/17 | válido, condicional |
| PPO assisted, teacher strict-local | EUR 645,8921 | 17/17 | 17/17 | válido, assisted |
| TD3 assisted, teacher strict-local | EUR 635,4363 | 17/17 | 17/17 | válido, assisted |

O PPO regenerado reduziu o custo local em 4,07% face ao RBC, fechou 62,10%
da oportunidade observada até ao replay MILP e melhorou todos os edifícios.
Continua a ser um controlador de storage assistido em runtime pelo teacher
strict-local para EV/deferrables; não é uma claim de PPO autónomo.

O TD3 regenerado reduziu o custo local em 5,62%, fechou 85,81% da oportunidade
até ao replay MILP e também melhorou todos os 17 edifícios. Tal como o PPO,
continua a ser uma variante assisted para os serviços EV/deferrable.

O MILP comunitário fixed-service derivado da mesma fonte strict-local produziu
um replay de EUR 521,2672 no custo comunitário e passou os hard gates. Este
valor pertence exclusivamente ao track comunitário.

## Campanha de maior horizonte

Fila local iniciada em 2026-07-30:

- RBC Smart strict-local: ano completo;
- PPO local com teacher strict-local: treino piloto regenerado e avaliação no
  ano completo;
- TD3 local com teacher strict-local: treino piloto regenerado e avaliação no
  ano completo;
- MILP individual e comunitário: regeneração anual a partir do rollout RBC
  strict-local anual, com gap e timeout reportados sem promover solves
  truncados a ótimos.

O RBC Smart strict-local anual já terminou com EUR 24 569,0692, 17/17 gates
locais e zero custo de settlement comunitário. O restante da fila continua em
execução.

A primeira avaliação anual do PPO treinado apenas em `0:1023` foi rejeitada
para promoção económica: manteve 17/17 gates, mas custou EUR 26 256,3357
(+6,87% face ao RBC) e melhorou apenas 1/17 edifícios. A campanha foi então
expandida para treino matching de 8 semanas (`0:5375`, cinco passagens), sem
repetição modular da demonstração curta.

A avaliação anual TD3 treinada na mesma janela curta teve a mesma conclusão:
17/17 gates, mas EUR 26 349,0846 (+7,24% face ao RBC) e apenas 1/17 edifícios
melhor. Também ficou `REJECT` para promoção anual e entrou na ronda matching de
8 semanas.

A referência strict-local matching de 8 semanas terminou com EUR 3 786,4090,
17/17 gates locais e zero custo de settlement comunitário. Esta é a baseline
que será usada, sem mistura de horizontes, para decidir os novos PPO e TD3 de
8 semanas.

O MILP individual matching resolveu 17/17 formulações com certificado válido.
Somou um lower bound de modelo de EUR 3 114,7902 e uma solução factível de
modelo de EUR 3 455,1413 (gap agregado 9,85%). O replay no simulador custou
EUR 3 446,5064: menos EUR 339,9026 (-8,98%) do que o RBC, com 17/17 edifícios
melhores e 17/17 gates locais. Houve 431 partidas EV e 56 ciclos deferrable sem
falhas. O replay é a referência empírica feasible; os bounds do modelo mantêm
semântica separada e continuam limitados a storage estacionário com os outros
serviços fixados pelo RBC strict-local.

O PPO matching treinado em cinco passagens sobre a janela de 8 semanas custou
EUR 3 585,7133 na última época: menos EUR 200,6957 (-5,30%) do que o RBC,
17/17 edifícios melhores e 17/17 gates locais. Fechou 59,05% da oportunidade
empírica entre o RBC e o replay MILP. Fica `ACCEPTED` dentro da janela matching,
mantendo a classificação `assisted`.

Na avaliação anual out-of-training-window, o mesmo PPO custou EUR 25 559,7911:
mais EUR 990,7220 (+4,03%) do que o RBC anual. Manteve 17/17 gates, mas só
1/17 edifícios ficou mais barato. A expansão de treino reduziu o excesso anual
do PPO curto de EUR 1 687,2665 para EUR 990,7220, sem inverter o ranking; por
isso fica `REJECT` para promoção anual.

O TD3 matching treinado nas mesmas cinco passagens custou EUR 3 543,1881:
menos EUR 243,2209 (-6,42%) do que o RBC, com 17/17 edifícios melhores e
17/17 gates. Fechou 71,56% da oportunidade empírica até ao replay MILP e fica
`ACCEPTED` na janela matching, também como controlador `assisted`.

Na avaliação anual, o TD3 custou EUR 25 534,9633: mais EUR 965,8941 (+3,93%)
do que o RBC anual. Passou 17/17 gates e melhorou 2/17 edifícios. A janela
maior reduziu o excesso do TD3 curto de EUR 1 780,0155 para EUR 965,8941, mas
o ranking anual continua desfavorável; fica `REJECT` para promoção anual.

O MILP comunitário anual conditional fixed-service produziu lower bound de
EUR 16 544,2389 e schedule conservador de modelo de EUR 18 562,5595, com
band estrutural de 10,87%; o MIP conservador ficou dentro da tolerância do
solver. O replay custou EUR 18 578,9889 na superfície comunitária settled,
apenas EUR 16,4293 acima do schedule de modelo. Contudo, registou três pedidos
com violação elétrica pré-projeção no `Building_15`, totalizando 0,0201 kWh, e
fica `REJECT_electrical_energy+electrical_events`. Os comandos executados foram
projetados para dentro dos limites, mas o gate rejeita corretamente os pedidos
inseguros.

O custo oficial do replay comunitário não é diretamente comparável com os
EUR 24 569,0692 do RBC strict-local anual: o primeiro usa settlement de mercado
comunitário e o segundo a soma local sem mercado. Na superfície reconstruída de
district import/zero export credit, a fonte RBC vale EUR 21 967,2764 contra
EUR 18 578,9889 do replay; isto é uma comparação model-aligned, não uma claim
do scorecard oficial. O resultado continua battery-only, fixed-service e sem
claim de ótimo global.

O processo do MILP individual anual terminou sem produzir certificado,
schedule ou logs finais no diretório de resultado. A causa de terminação não
ficou registada e o resultado é classificado como `NO_RESULT`, não como solve
pendente nem como ótimo. Uma repetição anual só deve ser feita com checkpoints
de progresso e limite de wall time externo auditável.

## Ronda autónoma full-action

Foi removida a dependência runtime do `RBCSmartLocalPolicy` para EV e
deferrables. PPO e TD3 usam agora BC full-action do MILP total individual,
mantêm apenas observações `building_local_v1` e declaram explicitamente
`runtime_service_teacher: false`.

O primeiro smoke `Building_1`, janela fechada `0:320`, revelou dois erros de
contrato antes de produzir resultados: o adaptador de preço estava a validar o
catálogo raw global em vez da vista encoded local, e o replay teacher não
aceitava a assinatura `predict_at_step(..., deterministic=...)`. Ambos foram
corrigidos e cobertos por testes.

O smoke revelou também que o safety shield anterior não era economicamente
neutro: forçava desde a chegada a potência média necessária de cada EV e não
impedia carga para lá do target. Foi adicionada uma modalidade
`deadline_feasible`, que só impõe a energia que já não pode ser diferida para
passos futuros, e um cap físico até ao SOC de serviço. Isto continua a ser uma
camada de segurança auditável, não um teacher de serviço.

Resultados matching em `Building_1`:

| Janela | Candidato | Custo local | vs. RBC Smart | Gates | Gap fechado até MILP |
|---|---|---:|---:|---:|---:|
| `0:320` (80 h) | RBC Smart | EUR 28,1402 | referência | 1/1 | 0,00% |
| `0:320` (80 h) | MILP total replay | EUR 22,3814 | -20,46% | 1/1 | 100,00% |
| `0:320` (80 h) | PPO autónomo | EUR 25,5207 | -9,31% | 1/1 | 45,49% |
| `0:320` (80 h) | TD3 autónomo | EUR 25,4498 | -9,56% | 1/1 | 46,72% |
| `0:728` (182 h) | RBC Smart | EUR 50,3658 | referência | 1/1 | 0,00% |
| `0:728` (182 h) | MILP total replay | EUR 39,9384 | -20,70% | 1/1 | 100,00% |
| `0:728` (182 h) | PPO autónomo | EUR 45,7946 | -9,08% | 1/1 | 43,84% |
| `0:728` (182 h) | TD3 autónomo | EUR 48,3419 | -4,02% | 1/1 | 19,41% |

Na janela de 182 h, PPO e TD3 obtiveram EV mínimo e precisão a 100%, zero
ciclos deferrable falhados e zero violações elétricas. Estes resultados são
matching/in-sample e demonstram autonomia mecânica e vantagem sobre RBC; ainda
não demonstram generalização sazonal nem promoção anual.

Foi depois executada uma avaliação congelada numa janela posterior e disjunta
do treino, `728:1492` (191 h). Os checkpoints de `0:728` não fizeram updates,
não chamaram o teacher em runtime e mantiveram a dimensão local original:

| Candidato held-out | Custo local | vs. RBC Smart | Gates | Gap fechado até MILP |
|---|---:|---:|---:|---:|
| RBC Smart | EUR 48,4585 | referência | 1/1 | 0,00% |
| MILP total replay | EUR 44,0756 | -9,04% | 1/1 | 100,00% |
| PPO autónomo congelado | EUR 45,5967 | -5,91% | 1/1 | 65,29% |
| TD3 autónomo congelado | EUR 46,3071 | -4,44% | 1/1 | 49,09% |

PPO e TD3 atingiram 100% de serviço mínimo e precisão EV, nenhum ciclo
deferrable falhado e zero violações elétricas. Esta é evidência de
generalização temporal dentro do mesmo edifício, embora ainda não seja uma
validação sazonal ou multi-edifício.

O MILP total de `Building_1` foi ainda alargado para `0:2736` (quatro semanas
fechadas), com 29 sessões EV e 29 ciclos deferrable. O solver terminou com
status ótimo, objetivo EUR 167,6804, gap 0,0092%, zero shortfall EV e zero
violações do modelo. O primeiro replay com o modo legado `average` do safety
shield custou EUR 207,8724 e falhou apenas a gate de precisão EV (37,93%),
revelando intervenção excessiva da camada de replay. O replay sem essa
projeção custou EUR 169,0779, menos 21,14% que o RBC matching de EUR 214,4002,
e passou todas as gates com precisão EV de 100%. Assim, a schedule foi
confirmada como CityLearn-feasible; a variante com shield legado fica apenas
como diagnóstico e não como valor do oráculo.

Os checkpoints promovidos treinados em `0:728` foram avaliados congelados nas
quatro semanas completas:

| Candidato em `0:2736` | Custo local | vs. RBC Smart | Gates | Gap fechado até MILP |
|---|---:|---:|---:|---:|
| RBC Smart | EUR 214,4002 | referência | 1/1 | 0,00% |
| MILP total replay zero-intervenção | EUR 169,0779 | -21,14% | 1/1 | 100,00% |
| PPO checkpoint de `0:728` | EUR 201,9560 | -5,80% | 1/1 | 27,46% |
| TD3 checkpoint de `0:728` | EUR 212,5132 | -0,88% | 1/1 | 4,16% |

Foram testadas três continuações longas a partir dos mesmos checkpoints:
fine-tuning RL+BC com BC decrescente, BC conservadora forte e uma primeira
passagem teacher-forced para recolher estados da trajetória do oráculo. Todas
mantiveram 100% de serviço e gates, mas ficaram piores que o RBC (PPO entre
+2,02% e +3,51%; TD3 entre +0,82% e +1,56%). Ficam `REJECT` e não substituem
os checkpoints anteriores. A investigação mostrou depois que estas
continuações não eram uma experiência limpa de “mais treino”: o teacher usava
o `exploration_step` restaurado (`5096`) e começava a nova janela no label
`2360`, embora o ambiente tivesse reiniciado no passo `0`; adicionalmente,
`fine_tune` misturava optimizers novos com exploração e normalização antigas e
o replay expert era descartado. Logo, estes runs provam apenas que aquela
continuação era inválida, não que mais dados ou épocas prejudiquem TD3/PPO.

## MILP total individual, 17 edifícios

O problema `0:672` com settlement individual foi decomposto por edifício. Os
17 solves da primeira extração terminaram `optimal`, com custo linear agregado
EUR 436,8081; o solve bounded registou lower bound EUR 436,1863 e upper bound
EUR 436,8080. Estes números ficam arquivados apenas como diagnóstico: a
extração antecedeu a correção do SOC inicial determinístico de EVs e, por isso,
já não certifica o modelo alinhado com o dataset CityLearn.

O replay CityLearn matching custou EUR 428,3052 contra EUR 477,0224 do RBC
Smart local: melhoria de 10,21%, 17/17 gates estritas, 16/17 edifícios mais
baratos, zero ciclos deferrable falhados e zero violações elétricas. Apenas
`Building_12` ficou EUR 0,8015 acima do RBC. O replay continua válido como
schedule simulator-feasible, mas deixa de ser aceite como ótimo do modelo
individual alinhado até à repetição com a extração corrigida.

Como candidato comunitário intermédio, a mesma schedule individual foi
reproduzida com settlement comunitário. Custou EUR 387,7530 contra
EUR 440,0610 do `RBCCommunityPolicy` matching (-11,89%) e passou 17/17 gates.
Este run está explicitamente marcado `community_optimum_claim: false`: prova
uma solução total comunitária CityLearn-feasible, mas não que o netting
conjunto ou o modelo corrigido tenham sido otimizados.

Foi também integrado o contrato de preço efetivo para futuro CC. Um
multiplicador `None`/`1.0` é bitwise no-op; valores não neutros alteram apenas
`district__electricity_pricing` na cópia encoded vista pelo ator. Reward,
settlement e scorecard continuam a usar o preço real, e nenhuma feature
comunitária entra nos atores locais. Treino online com multiplicador não neutro
falha deliberadamente; o uso previsto sob CC é inference-only com leaves
congelados.

Validação de regressão após as alterações: `758 passed`, `28` warnings
conhecidos.

## Reconciliação de SOC e repetição dos MILPs totais

O primeiro solve conjunto com caps por edifício terminou `optimal` e o replay
passou 17/17 gates, mas uma reconciliação modelo/simulador mostrou que não
podia ser promovido. Nos EVs sem `initial_soc` explícito, o extrator usava
`1 - depth_of_discharge`; o CityLearn usa uma inicialização determinística
baseada em MD5 da seed, tipo e identificador do EV. Por exemplo, no primeiro
EV de `Building_10` o modelo assumia 20%/15 kWh, enquanto o simulador começava
em 71,451492%/53,5886 kWh; no `Building_12`, 10%/4,5 kWh contra
70,718413%/31,8233 kWh. Parte da carga comandada saturava por isso no replay.

O fallback foi centralizado e passou a reproduzir exatamente o algoritmo do
CityLearn. O replay também ganhou uma tolerância de `1e-6` apenas para resíduos
float32 nos limites de ação e um nudge de `1e-6 kW` para comandos exatamente no
deadband mínimo do charger. Há testes dedicados aos SOCs de `Building_10` e
`Building_12`, ao override explícito, ao deadband e à rejeição de violações
materiais. O solve/replay anterior de EUR 355,4025 fica
`DIAGNOSTIC_SUPERSEDED`.

Com a extração corrigida, o MILP total individual `0:672` voltou a resolver os
17 problemas locais com `status=optimal` e custo linear agregado de
EUR 424,3984. O shortfall mínimo/realizado é 28,9027 kWh, exclusivamente no
`Building_15`. O replay CityLearn custou EUR 424,0119 contra EUR 477,0224 do
RBC Smart local (-11,11%), passou 17/17 gates e, desta vez, os 17 edifícios
ficaram abaixo do RBC. Esta schedule é o novo teacher local corrigido; a
schedule pré-correção fica apenas como evidência feasible histórica.

## MILP total comunitário ótimo e replay

O problema comunitário conjunto corrigido tem 17 edifícios, 672 passos e a
formulação completa de bateria, EV/V2G, deferrables e envelopes elétricos
total/L1/L2/L3. A fase de serviço terminou `optimal`: shortfall mínimo de
28,902728 kWh, todo em `Building_15`. A fase económica realizou
28,903728 kWh, usando exatamente a tolerância lexicográfica máxima de 1 Wh no
único edifício com shortfall positivo; todos os restantes ficaram fixados a
zero.

O solve económico terminou `optimal` com:

- objetivo do modelo: EUR 347,7026176791;
- dual bound: EUR 347,6787858945;
- MIP gap: 0,0068547%;
- replay CityLearn: EUR 347,7026162444;
- diferença replay menos modelo: -EUR 0,0000014346.

Scorecard matching na superfície comunitária:

| Candidato | Custo | vs. RBCCommunity | Gates | Edifícios abaixo do RBC |
|---|---:|---:|---:|---:|
| RBCCommunity | EUR 440,0610 | referência | 17/17 | referência |
| Schedule individual pré-correção, upper candidate | EUR 387,7530 | -11,89% | 17/17 | 16/17 |
| MILP comunitário total corrigido, replay | EUR 347,7026 | -20,99% | 17/17 | 17/17 |

O novo replay poupa EUR 92,3584 face ao RBC e EUR 40,0504 face ao candidate
anterior, sem violações elétricas, ciclos deferrable falhados ou gates SOC/EV
rejeitadas. Fica `ACCEPTED_OPTIMAL_SUPPLIED_LINEAR_MODEL` e
`ACCEPTED_CITYLEARN_FEASIBLE_REPLAY`. Isto não é uma claim de ótimo global do
simulador não linear. A janela `[0,672)` também está truncada à direita em oito
sessões EV; serve como benchmark week-one matching, não como evidência sazonal.
Os artefactos brutos estão em
`runs/analysis/total_energy_community_week1_corrected_soc_solve_20260731` e o
scorecard em
`runs/analysis/total_energy_community_week1_corrected_soc_optimal_replay_audit_20260731`.

## TD3: início da cobertura dos 17 edifícios

Antes de repetir treino longo, o relógio do teacher passou a ser local ao
episódio, o resume foi separado em flags independentes para optimizers, replay,
exploração e normalização, e a reward `LocalEconomicSafetyRewardV3` deixou de
penalizar densamente a decisão economicamente válida de adiar carga EV. O ator
usa agora heads semânticos separados e a demonstração individual corrigida foi
empacotada de forma portátil com manifesto e hashes em
`configs/demonstrations/local_total_energy_v1/week1_corrected_soc_diagnostic`.

Dois smokes autónomos seed 123 correram 17 TD3 independentes, sem mercado,
features comunitárias ou service teacher em runtime:

| Receita | Custo | vs. RBC local | Gates | Casas abaixo do RBC | Gap fechado até MILP |
|---|---:|---:|---:|---:|---:|
| TD3+BC, 3 episódios | EUR 448,4700 | -5,99% | 16/17 | 12/17 | 53,86% |
| TD3 BC-only diagnóstico, 5 episódios | EUR 441,3098 | -7,49% | 16/17 | 12/17 | 67,37% |

Ambos passaram todas as gates em 16 edifícios. A única rejeição foi
`Building_15`: com dois EVs e limites trifásicos partilhados, EV minimum ficou
em 0,6364 e precisão em 0,5455. O BC-only melhorou custo e regret mas não
removeu esta falha, indicando que o próximo diagnóstico deve atacar a cobertura
dos eventos positivos/coordenação dos dois heads EV, e não simplesmente subir
o peso global de BC. Estes smokes são `REJECT_B15_EV_SERVICE`; demonstram que a
cobertura multi-building começou, não que os agentes estejam prontos para CC.

## Estado de promoção e execução remota

A implementação está isolada na branch
`codex/local-rl-milp-validation-20260730`. Os snapshots one-off foram retirados
de `configs/experiments` e preservados sob `runs/config_snapshots`; os templates
permanentes já não contêm caminhos para `/home` ou `runs`. A suite completa
terminou com `758 passed` e 28 warnings conhecidos.

O preflight read-only do Job Orchestrator encontrou a fila vazia, CPU local e
Deucalion CPU disponíveis e o GPU local disponível. O GPU Deucalion estava
ocupado. O worker INESC TEC/Union estava online e com imagem pronta, mas a
autenticação tinha excedido a janela estrita de 24 horas. Não foram submetidos,
cancelados ou reautenticados jobs. A imagem publicada ainda corresponde ao
commit anterior; a nova demonstração portátil elimina a dependência de
artefactos ignorados, mas é necessário publicar uma imagem do novo commit e
repetir o preflight antes da campanha sazonal/multi-seed.

O próximo gate de promoção é: janelas sazonais boundary-exact, seeds
123/456/789, 17/17 hard gates, custo agregado abaixo do RBC e cada edifício a
bater o RBC em pelo menos duas seeds. Só depois se congelam os agentes locais.
O CC permanece fora dos atores: quando chegar essa fase, verá a comunidade e
atuará apenas através do multiplicador de preço efetivo; os leaves continuam
sem observações comunitárias.
