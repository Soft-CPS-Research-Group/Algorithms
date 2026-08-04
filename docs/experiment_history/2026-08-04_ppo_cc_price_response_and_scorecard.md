# Resposta ao preço e scorecard do CC sobre o PPO congelado

- Campanha: `ppo_cc_price_response_scorecard_20260804`
- Dataset: `citylearn_three_phase_electrical_service_demo_15min_parquet`
- Leaf: PPO residual-battery seed `789`, anual, congelado
- Mercado comunitário: desligado para isolar o efeito do sinal
- Horizonte de promoção: `0:35039`, 35 040 amostras
- Scorecard: `cc_frozen_leaf_scorecard_v1`

## Decisão

O PPO local entregue ao CC mantém-se inalterado, congelado e sem observações
comunitárias. A paridade do pipeline neutro `FixedPriceSignal(1.0) -> PPO` com
a avaliação anual standalone aceite foi exata: os 53 ficheiros exportados são
iguais byte a byte e o `exported_kpis.csv` tem SHA-256
`63ddbd04eb1c739cd3aae45004d6101413147007305eddd6ff680c92b05a3034`.

O multiplicador anual fixo `1.025` foi o primeiro sinal do CC a passar o novo
scorecard completo. A extensão da superfície mostrou um patamar melhor entre
`1.05` e `1.075`. O menor custo medido foi EUR 23 804,8356 em `1.075`, mas é
apenas EUR 0,0901 inferior a `1.05`, tem uma projeção de segurança tolerante,
melhora menos edifícios (10/17 contra 12/17) e apresenta pior fairness. Por
isso, `1.05` é o incumbente robusto: reduz o custo de EUR 23 826,7980 para
EUR 23 804,9257 (`-EUR 21,8723`), passa os hard gates estritos e não tem
trade-offs secundários materiais. O `0.90`, que parecia melhor numa janela
curta, é rejeitado no ano.

O primeiro checkpoint adaptativo, centrado em `1.025`, passou o scorecard mas
não superou o fixo `1.05` e não foi promovido. Depois de tornar as 17 folhas
PPO verdadeiramente determinísticas durante o treino do manager, o segundo
`CCLevel1` atingiu EUR 23 804,4761: `-EUR 22,3220` contra PPO neutro, `-EUR
0,4496` contra o fixo robusto `1.05` e `-EUR 0,3596` contra o menor custo fixo
`1.075`. Passa 17/17 hard gates estritos, não apresenta regressões secundárias
materiais e melhora 13/17 edifícios. É promovido como **candidato anual
CC+PPO**, mas ainda não como prova de generalização, porque treino e avaliação
percorrem o mesmo ano do mesmo dataset.

O replay anual contínuo do schedule sazonal robusto terminou em EUR
23 804,4893. O `CCLevel1` aprendido foi ainda EUR 0,0132 melhor, uma diferença
economicamente indistinguível mas importante como controlo: o candidato
aprendido atingiu, sem receber o calendário oracle, praticamente todo o ganho
disponível no schedule conservador construído a posteriori.

O schedule de teto retrospetivo atingiu EUR 23 800,5018 (`-EUR 26,2962`
contra PPO), deixando apenas EUR 3,9743 entre o candidato aprendido e o melhor
calendário mensal discreto construído com conhecimento do próprio ano. Esse
teto usa `1.075`, requer projeção de segurança e melhora menos casas (11/17),
por isso serve para orientar o próximo treino, não para promoção ou entrega.

## Porque é que baixar o preço percebido podia baixar o custo real?

O CC não altera a tarifa de settlement. Só altera a coordenada de preço atual
que o PPO vê. A configuração local usa
`local_price_forecast_mode: real_unmodified`, pelo que os forecasts continuam
com o preço real. Assim, `0.90` faz o instante atual parecer relativamente
mais barato que o futuro e pode atravessar limiares da rede, do residual RBC,
do deadband e da projeção de segurança. O efeito observado nas quatro semanas
foi carregar mais em instantes baratos e descarregar mais em instantes caros:

- custo: EUR 2 007,9620 -> EUR 2 000,8431;
- carga da bateria: `+116,499 kWh`;
- descarga: `+80,930 kWh`;
- preço médio ponderado de carga: `0,13537 -> 0,13489 EUR/kWh`;
- preço médio ponderado de descarga: `0,18087 -> 0,18298 EUR/kWh`;
- spread realizado: `4,55 -> 4,81 cêntimos/kWh`.

Um controlo causal em que preço atual **e os forecasts** foram todos
multiplicados por `0.90` poupou apenas EUR 2,3012 nessa janela, contra EUR
7,1188 quando só o preço atual foi alterado. A cunha atual-versus-forecast
explica, portanto, a maior parte do ganho curto; o remanescente é compatível
com a não linearidade e a ausência de invariância de escala da política.

## Artefacto descoberto nas janelas curtas

As janelas `0:2735`, `8760:11495` e `17520:20255` recalculavam os limites
`minmax_space` apenas sobre a respetiva janela. O PPO tinha sido treinado com
limites anuais. Os resets brutos curto e anual eram idênticos nos 17
edifícios, mas o encoding e a ação já divergiam no primeiro passo.

| Variável no passo 0 | Bruto | Bounds `0:2735` -> encoded | Bounds anuais -> encoded |
|---|---:|---:|---:|
| Preço atual | 0,16464 | 0,02509--0,250 -> 0,62047 | 0,00003--0,651 -> 0,25287 |
| Temperatura | 20,0 | 16,7--25,6 -> 0,37079 | 5,6--32,2 -> 0,54135 |
| Carbono | 0,17072 | 0,10276--0,25930 -> 0,43417 | 0,07038--0,28180 -> 0,47462 |

Com o mesmo checkpoint determinístico, por exemplo, a ação de storage do
Building 2 foi `0` no curto e `0,0204565` no anual; no Building 8 foi `0` e
`0,0195420`. Isto explica por que o prefixo curto não reproduzia o prefixo do
ano.

As comparações neutral-candidato continuam emparelhadas dentro de cada janela,
mas deixam de ser evidência de deployment. A promoção passa a exigir ano
exato. Para desenvolvimento curto, o ambiente deve manter
`simulation_start_time_step: 0` e `simulation_end_time_step: 35039` e selecionar
a subjanela através de `episode_time_steps`, preservando os bounds anuais. O
handoff futuro deve persistir/hashar os bounds e falhar se dataset, perfil,
nomes ou dimensões não coincidirem.

## Superfície anual comparável

| Multiplicador | Custo | Delta vs. `1.0` | Casas melhores | Gates | Decisão |
|---:|---:|---:|---:|---:|---|
| 0,900 | EUR 23 957,3636 | EUR +130,5656 | 1/17 | 17/17 tolerantes | REJECT_COST |
| 0,975 | EUR 23 845,6148 | EUR +18,8168 | 3/17 | 17/17 | REJECT_COST |
| 1,000 | EUR 23 826,7980 | referência | referência | 17/17 | REFERENCE |
| 1,025 | EUR 23 812,1599 | EUR -14,6381 | 14/17 | 17/17 | PASS_CC_SCORECARD |
| 1,050 | **EUR 23 804,9257** | **EUR -21,8723** | **12/17** | **17/17 estritos** | **PASS_CC_SCORECARD / incumbente robusto** |
| 1,075 | EUR 23 804,8356 | EUR -21,9624 | 10/17 | 17/17 tolerantes | PASS_CC_SCORECARD / mínimo medido |
| 1,100 | EUR 23 806,0504 | EUR -20,7477 | 10/17 | 17/17 tolerantes | PASS_COST_WITH_TRADEOFFS |
| 1,125 | EUR 23 811,7546 | EUR -15,0435 | 8/17 | 17/17 tolerantes | PASS_COST_WITH_TRADEOFFS |
| 1,150 | EUR 23 821,4695 | EUR -5,3286 | 8/17 | 17/17 tolerantes | PASS_COST_WITH_TRADEOFFS |

### Políticas aprendidas contra a superfície fixa

| Política | Custo | Delta vs. PPO | Casas melhores | Gates | Decisão |
|---|---:|---:|---:|---:|---|
| CC adaptativo v1 | EUR 23 812,1054 | EUR -14,6927 | 14/17 | 17/17 estritos | PASS, não promover |
| **CC adaptativo v2** | **EUR 23 804,4761** | **EUR -22,3220** | **13/17** | **17/17 estritos** | **PASS, candidato anual** |

### Scorecard do incumbente robusto `1.05` contra PPO neutro

| Métrica | PPO neutro | CC fixo `1.05` | Delta | Leitura |
|---|---:|---:|---:|---|
| Custo retail | EUR 23 826,7980 | EUR 23 804,9257 | EUR -21,8723 | melhora |
| Importação | 129 871,287 kWh | 129 600,673 kWh | -270,614 kWh | melhora |
| Pico diário / BAU | 1,080662 | 1,078828 | -0,001834 | melhora |
| Pico máximo / BAU | 1,148736 | 1,148668 | -0,000068 | melhora |
| Ramping / BAU | 1,361981 | 1,367929 | +0,005948 | piora imaterial |
| Penalização load factor / BAU | 1,103383 | 1,102662 | -0,000720 | melhora |
| Emissões | 22 423,065 kgCO2 | 22 380,562 kgCO2 | -42,503 kgCO2 | melhora |
| Autoconsumo solar | 0,726217 | 0,727874 | +0,001657 | melhora |
| Exportação | 39 818,472 kWh | 39 579,116 kWh | -239,357 kWh | monitor, melhora |
| Throughput bateria | 49 928,812 kWh | 49 729,503 kWh | -199,309 kWh | monitor |

O `0.90` anual, pelo contrário, aumentou a importação em `870,050 kWh`, as
emissões em `134,819 kgCO2` e reduziu a autoconsumo solar em `0,005962`, apesar
de pequenas melhorias no pico máximo e no ramping. Esta é a razão para não
avaliar o CC apenas pelo custo.

O ganho do `1.05` também foi temporalmente consistente: venceu o neutro em
11/12 meses. A única exceção foi maio, com uma regressão de apenas EUR 0,0927;
o melhor mês foi setembro, com EUR 3,8913 de poupança. O `1.075` venceu em
10/12 meses e regrediu EUR 0,18597 em junho e EUR 0,66088 em maio. Isto reforça
a escolha conservadora de `1.05` apesar da diferença anual de nove cêntimos.

## Comparação anual com o RBC Community

Para não misturar o dataset atual com a campanha antiga de custos próximos de
EUR 32 mil, foi executado um novo `RBCCommunityPolicy` no mesmo dataset e no
mesmo horizonte anual. O resultado expõe duas contabilidades diferentes:

| Controlador | Mercado | Custo retail/contrafactual | Custo comunitário liquidado |
|---|---|---:|---:|
| RBC Smart local | desligado | EUR 24 569,0692 | n/a |
| RBC Community | ligado | EUR 26 278,3000 | EUR 22 839,3124 |
| PPO seed 789 | desligado | EUR 23 826,7980 | n/a |
| CCLevel1 v2 + PPO | desligado | EUR 23 804,4761 | n/a |

O valor liquidado do RBC Community incorpora EUR 3 438,9876 de poupança de
mercado/netting relativamente ao contrafactual retail dos membros. Não deve ser
comparado diretamente com PPO ou CC+PPO enquanto estes forem avaliados com
`community_market: false`. Na base retail comum, o RBC Community é EUR
1 709,2308 mais caro que o RBC Smart local; o PPO é EUR 742,2711 mais barato e
o CC+PPO é EUR 764,5931 mais barato.

O scorecard físico confirma que o custo liquidado, sozinho, esconderia uma
troca desfavorável:

| Métrica anual | RBC Smart local | RBC Community | PPO | CC+PPO |
|---|---:|---:|---:|---:|
| Importação (kWh) | 132 812,128 | 136 608,339 | 129 871,287 | 129 595,295 |
| Pico diário / BAU | 1,070850 | 1,180207 | 1,080662 | 1,078961 |
| Pico máximo / BAU | 1,134396 | 1,321582 | 1,148736 | 1,148758 |
| Ramping / BAU | 2,403322 | 2,910526 | 1,361981 | 1,367858 |
| Emissões (kgCO2) | 22 425,307 | 24 092,282 | 22 423,065 | 22 379,792 |
| Autoconsumo solar | 0,697469 | 0,668097 | 0,726217 | 0,727927 |
| Hard gates | 17/17 estritos | 17/17 estritos | 17/17 estritos | 17/17 estritos |

O próximo confronto justo é executar PPO e CC+PPO congelados com exatamente o
mesmo mercado/settlement do RBC Community, reportando em paralelo custo
liquidado, contrafactual retail e todo o scorecard físico.

## Contrato de scorecard daqui para a frente

1. Hard gates primeiro: serviço EV/deferrables, segurança elétrica, SoC e
   outage.
2. Custo retail anual como objetivo primário.
3. Importação, pico diário, pico máximo, ramping, load factor, emissões e
   autoconsumo solar como métricas secundárias explícitas.
4. Exportação, net exchange, throughput da bateria e V2G como métricas de
   monitorização.
5. Fairness por edifício: número de casas melhores, pior delta local e índice
   de Jain sobre poupanças não negativas.

Uma regressão secundária relativa superior a 1%, ou perda absoluta de
autoconsumo superior a 0,005, fica visível como trade-off mesmo quando o custo
melhora. O custo nunca pode compensar a falha de um hard gate.

## Alterações de implementação desta campanha

- `4d24f00`: sinais fixos vetoriais por edifício;
- `66f7787` e `53df50a`: scorecard completo e evidência dos hard gates;
- `661083a` e `11bc2f9`: correções e blending conservador do CCLevel2;
- `a99ef61` e `28a622a`: reward `member_retail` alinhado com o KPI oficial;
- `da9810d`: largura do contexto CCLevel1 corrigida de 16 para 17;
- `e10af2b`: contrato residual seguro no CCLevel1;
- `8c0b87d`: templates corrigidos para as 35 040 amostras anuais;
- `e73d924` e `f4a3e8f`: campanha anual escalar reprodutível e inicialização
  exata na referência `1.025`;
- `2703e3e`: teste que fixa as 35 040 amostras nos templates anuais;
- `ec96fb7`: avaliação anual congelada e determinística do CC escalar;
- `fb49301`: estágios congelados do pipeline passam a agir deterministicamente
  enquanto apenas o estágio treinável explora;
- `68516b6`: segunda campanha anual em torno do incumbente robusto `1.05`,
  com PPOs locais determinísticos durante o treino do CC;
- `ee0cbc9`: sinal de preço com schedule por `start_step`, validado e exportado
  como baseline auditável;
- `289bb73` e `206e032`: oracles sazonais conservadora e de teto para medir a
  margem de adaptação lenta sem a confundir com generalização.

Suite completa final: `861 passed`, 29 warnings esperados. O alvo direcionado
do pipeline/CC e do sinal sazonal passou antes com `104 passed`.

## CC adaptativo anual

O primeiro treino anual terminou no job
`cclevel1-safe-residual-frozen-ppo-seed789-annual-train-f4a3e8f`. A avaliação
anual congelada e determinística corre no job
`cclevel1-safe-residual-frozen-ppo-seed789-annual-eval-ec96fb7`.

Contrato:

- um único multiplicador comunitário;
- referência `1.025`;
- output possível limitado por `policy_residual_scale: 0.05` dentro da gama
  base `0.85--1.15`;
- recompensa apenas `member_retail` para alinhar o gradiente com o custo;
- decisões horárias;
- PPOs locais seed `789` congelados e community-blind;
- treino anual e avaliação anual determinística em jobs separados;
- promoção apenas se superar simultaneamente o PPO neutro, a sua referência
  `1.025` e o incumbente robusto `1.05`, sem falhar o scorecard.

### Diagnóstico do primeiro treino adaptativo

O atributo `frozen` impedia updates nos 17 PPOs, mas o pipeline continuava a
passar-lhes `deterministic=False` durante o treino do CC. Assim, os PPOs
congelados amostravam a sua distribuição de ações e o CC aprendia contra folhas
diferentes das usadas em avaliação/deployment. O rollout de treino estocástico
custou EUR 25 346,1592 (`+EUR 1 519,3612` contra o PPO neutro), aumentou a
importação em 1 511,496 kWh e falhou o scorecard, embora o multiplicador do CC
tivesse média `1,024959` e desvio padrão de apenas `0,000367`. O ruído vinha,
portanto, sobretudo das folhas e não de uma exploração economicamente útil do
coordenador.

O pipeline foi corrigido para que um estágio congelado seja também
determinístico. O segundo treino usa:

- referência `1.05` e output limitado a `1.00--1.075`;
- PPOs seed `789` congelados, determinísticos e sem contexto comunitário;
- apenas o CC estocástico, com `initial_log_std: -1.5`;
- rollouts de 336 decisões horárias e duas passagens anuais;
- reward primário `member_retail`; restantes KPIs aplicados como scorecard de
  promoção, sem esconder trade-offs numa soma de pesos;
- job `cclevel1-deterministic-leaf-frozen-ppo-seed789-annual-train-v2-fb49301`.

A avaliação determinística do primeiro checkpoint terminou com EUR
23 812,1054 (`-EUR 14,6927` contra PPO neutro), 17/17 hard gates estritos,
14/17 edifícios melhores e sem regressões secundárias materiais. Passou o
scorecard, mas não foi promovida: melhora apenas EUR 0,0546 face ao fixo
`1.025` e perde EUR 7,1797 para o incumbente robusto `1.05`. O trace confirma
que era praticamente constante (`1,024944 ± 0,000125`, min `1,024709`, max
`1,025183`). Este resultado valida o pipeline de avaliação, não uma vantagem
adaptativa relevante.

O segundo treino concluiu duas passagens anuais. Durante a segunda, a política
estocástica explorou `1,017940--1,068338` com média `1,048331` e desvio padrão
`0,007749`, mantendo as folhas determinísticas. A avaliação anual separada do
checkpoint final terminou no job
`cclevel1-deterministic-leaf-frozen-ppo-seed789-annual-eval-v2-68516b6`.

O checkpoint determinístico emitiu `1,043978--1,053768`, com média `1,048771`
e desvio padrão `0,002538`. O scorecard anual foi:

- custo EUR 23 804,4761 (`-EUR 22,3220` contra PPO; `-EUR 0,4496` contra
  `1.05`);
- importação `-275,992 kWh` e emissões `-43,273 kgCO2`;
- pico diário `-0,001701`, pico máximo `+0,000022` imaterial e ramping
  `+0,005877` imaterial;
- autoconsumo solar `+0,001710`, exportação `-246,886 kWh` e throughput de
  bateria `-177,024 kWh`;
- 17/17 gates estritos, 13/17 casas melhores, pior casa `+EUR 2,5669` e Jain
  `0,4341`.

Decisão: `PASS_CC_SCORECARD` e promoção como candidato anual. A melhoria
sobre o fixo é pequena, logo a política deve conservar `1.05` como fallback e
precisa de evidência fora da trajetória anual de treino antes de deployment.

### Oracles sazonais de diagnóstico

Os custos mensais dos sinais fixos permitem construir dois controlos
determinísticos, sempre com o mesmo PPO congelado:

- `robust_seasonal`: escolhe apenas entre `1.0`, `1.025` e `1.05`, evitando o
  ponto `1.075` que precisou de projeção de segurança;
- `upper_bound_seasonal`: escolhe também `1.075` e representa o melhor valor
  mensal observado na superfície discreta.

Antes do replay sequencial, a soma dos meses independentes estimava EUR
23 804,1786 para o oracle robusto (`-EUR 0,7471` contra o fixo `1.05`) e EUR
23 800,5276 para o oracle de teto (`-EUR 4,3982`). Como o estado da bateria
atravessa fronteiras mensais, essas somas não foram aceites como resultado.
Ambos os schedules foram reexecutados no ano contínuo e submetidos ao
scorecard. Continuam a ser upper bounds in-sample e não checkpoints para
deployment.

O replay contínuo `robust_seasonal` produziu EUR 23 804,4893 (`-EUR 22,3087`
contra PPO e `-EUR 0,4364` contra o fixo `1.05`), 17/17 hard gates estritos,
13/17 casas melhores e nenhuma regressão secundária material. A diferença para
a soma mensal otimista foi EUR 0,3107, confirmando que o estado transportado
entre meses não pode ser ignorado. O candidato `CCLevel1` v2 ficou EUR 0,0132
melhor que este controlo.

O `upper_bound_seasonal` contínuo terminou em EUR 23 800,5018 (`-EUR 26,2962`
contra PPO e `-EUR 3,9743` contra o `CCLevel1` v2), com 11/17 casas melhores e
sem regressões secundárias materiais. Passou apenas com projeção de segurança:
o pedido excedente total foi `0,114567 kWh` num evento. Ao contrário da soma
mensal preliminar, o estado contínuo melhorou este teto em EUR 0,0258. Como o
schedule foi escolhido ex post no mesmo ano, não é uma política generalizável;
quantifica somente a margem in-sample ainda disponível na família discreta.

## Evidência local ignorada pelo Git

- Scorecard anual completo:
  `runs/analysis/ppo_cc_scorecard_v1_annual_price_surface_complete_20260804`
- Scorecard anual final (fixos, adaptativos e schedules contínuos):
  `runs/analysis/ppo_cc_scorecard_v1_annual_cc_final_20260804`
- Auditoria anual matching do RBC Community:
  `runs/analysis/rbccommunity_matching_annual_20260804`
- Scorecard do schedule robusto contínuo:
  `runs/analysis/ppo_cc_scorecard_v1_annual_robust_seasonal_20260804`
- Scorecard de escalas CCLevel2:
  `runs/analysis/ppo_cc_scorecard_v1_cclevel2_dev_scales_20260804`
- CCLevel2 alinhado com retail:
  `runs/analysis/ppo_cc_scorecard_v1_cclevel2_retail_aligned_dev_scales_20260804`
- Vetor in-window:
  `runs/analysis/ppo_cc_scorecard_v1_vector_oracle_20260804`
- Vetor heldout abril:
  `runs/analysis/ppo_cc_scorecard_v1_vector_heldout_8760_11495_20260804`
- Vetor heldout julho:
  `runs/analysis/ppo_cc_scorecard_v1_robust_vector_heldout_17520_20255_20260804`
