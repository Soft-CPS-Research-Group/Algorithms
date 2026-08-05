# Protocolo congelado: SMART, PPO e Community Coordinator com settlement

- Protocolo: `ppo_cc_settlement_annual_v1`
- Estado: condições congeladas; nenhum job submetido por esta auditoria
- Commit auditado: `6f490df9684d7f5e00e811ad936df0cbe2826589`
- Dataset: `citylearn_three_phase_electrical_service_demo_15min_parquet`
- CityLearn/SoftCPS REC Simulator local: `1.5.6`
- Evidência bruta local: `runs/local_audits/ppo_cc_settlement_protocol_20260804/contract_audit.csv`

## Decisão

Todo o trabalho novo desta linha usa `community_market.enabled: true`. Os
resultados históricos sem settlement são preservados numa tabela separada,
mas não são usados como baseline económico para promover resultados novos.

A superfície canónica é o ano completo, `0:35039`, com 35 040 amostras de 15
minutos. Isto preserva os PPO congelados e o `CC-TD3` anual do Pedro. O
`CC-SMART` histórico com 34 944 passos não é descartado, mas fica marcado como
evidência histórica sem settlement e fora da nova superfície comparável.

## Auditoria das experiências existentes

| Experiência | Passos | Settlement | Gama CC | Leaf | Leitura |
|---|---:|---|---|---|---|
| Pedro `CC-SMART` | 34 944 | `null`, desligado na semântica atual | 0,5--1,5 | `SignalAwareRBC`, não marcado frozen | histórica, não entra na tabela nova |
| Pedro `CC-TD3` | 35 040 | ligado, 0,8/0,8/0 | 0,5--1,3 | `MATD3` frozen | histórica com settlement |
| PPO local seeds 123/456/789 | 35 040 | desligado na avaliação original | n/a | PPO residual local frozen | checkpoints reutilizáveis |
| CC+PPO v2 atual | 35 040 | desligado | output efetivo estreito, 1,0--1,075 | PPO 789 frozen | fica na tabela histórica sem settlement |
| RBCCommunity matching | 35 040 | ligado, 0,8/0,8/0 | n/a | `RBCCommunityPolicy` | referência settled, não é o SMART emparelhado |

Proveniência dos dois resolved configs do Pedro:

- `CC-SMART`: SHA-256 `9cb1bb3d3657acf8fb43dc14d67a52df5fe3e6678867fde1b121d764990bf1f9`;
- `CC-TD3`: SHA-256 `de0e7e645a102028783b605da4be13b89d64242168f7346e4897c6284157ea70`.

O resolved config standalone do SMART que produziu a imagem do Pedro ainda
não foi recebido. A afirmação de settlement desligado fica registada como
confirmação do autor, não como inferência do ficheiro `CC-SMART`.

## Identidade congelada dos dados e runtime

- Dataset path: `./datasets/citylearn_three_phase_electrical_service_demo_15min_parquet/schema.json`.
- SHA-256 do `schema.json`: `6ea6ab786bec8bc3fa2fb8e9997c965ad4808ea74ecd3a98a9924f802a811ec0`.
- SHA-256 combinado dos 31 ficheiros do dataset, com nomes relativos:
  `816c4efd49cb253caed9ce577ed64a20c89f0e9407ada8d79b3838012da5bfa5`.
- Interface: `entity`.
- Topologia: `static`.
- `central_agent: false`.
- 17 edifícios/agentes locais.
- A campanha deve usar um único commit e uma única imagem; ambos são
  registados nos payloads e no histórico quando os jobs forem preparados.

Uma divergência no hash do dataset, versão do simulador, janela ou commit
cria outra superfície e não pode ser adicionada à tabela canónica v1.

## Contrato económico

```yaml
community_market:
  enabled: true
  local_price_ratio_to_grid_import: 0.8
  intra_community_sell_ratio: 0.8
  grid_export_price: 0.0
  import_member_weights: {}
  kpis:
    community_local_traded_enabled: true
    community_self_consumption_enabled: true
```

O multiplicador do CC altera apenas o preço percebido pelo leaf. Não altera a
tarifa real, a regra de settlement nem diretamente as ações físicas.

Quando o mercado está ativo, o custo oficial é
`district_cost_community_market_settled_total_eur`. O scorecard deve também
guardar:

- `district_cost_community_market_counterfactual_total_eur`;
- `district_cost_community_market_savings_total_eur`;
- energia local comprada e vendida;
- percentagem da procura coberta pelo mercado local.

Isto permite separar melhoria física do controlador de poupança produzida
pela regra de settlement.

## Matriz experimental canónica

| Linha | Pipeline | Treino novo | Seeds | Comparador causal |
|---|---|---|---|---|
| SMART | `FixedPriceSignal(1.0) -> SignalAwareRBC` | não; replay determinístico | 123 | referência do `CC-SMART` |
| CC-SMART | `CCLevel1 -> SignalAwareRBC` | apenas o CC | 123, 456, 789 | SMART neutral |
| PPO | `FixedPriceSignal(1.0) -> PPO local frozen` | não; replay dos três checkpoints | 123, 456, 789 | SMART settled |
| CC-PPO | `CCLevel1 -> PPO local frozen` | apenas o CC | 123, 456, 789, emparelhadas | PPO neutral da mesma seed |

O SMART neutral usa a mesma implementação e os mesmos hiperparâmetros do leaf
do `CC-SMART`. O contexto `1.0` é um no-op coberto pelos testes do
`SignalAwareRBC`. Assim, a diferença SMART--CC-SMART mede apenas o CC.

O leaf PPO continua a ser o composto
`RBCSmartLocalPolicy + PPO residual-battery + safety projector + price adapter`.
Os 17 checkpoints existem para as seeds 123, 456 e 789. Durante treino e
avaliação do CC ficam frozen, determinísticos e sem observações comunitárias.
Não se volta a treinar o PPO com settlement: liga-se o settlement na
avaliação e o CC é o único componente novo que aprende sobre a comunidade.

O único treino histórico do Pedro que precisa de ser repetido para a nova
tabela é o `CC-SMART`. SMART exige apenas um replay anual determinístico com
settlement; PPO exige replay dos checkpoints; CC-PPO é uma nova experiência.

## Contrato do CC

- Algoritmo: `CCLevel1`.
- Um multiplicador global comunitário.
- Decisão a cada 4 passos, isto é, uma vez por hora.
- `c_dim: 17`.
- `price_min: 0.5`.
- `price_max: 1.3`, alinhado com o `CC-TD3` anual comparável do Pedro.
- `reference_multiplier: 1.0`.
- `policy_residual_scale: 1.0`, para não reduzir artificialmente a gama.
- Quatro passagens anuais nos jobs aprendidos.
- `deterministic_finish: true`.
- Apenas o quarto e último episódio entra no scorecard.
- BC inicial permitido: 8 760 decisões horárias recolhidas e 2 000 updates,
  com os pesos históricos do Pedro.
- Reward de mercado: `CCRewardLevel1` com `cost_aggregation: community_net`.

O objetivo primário do reward é consistente com o custo settled do distrito
quando vendas à rede valem zero. Pico, ramping, exportação e violações mantêm
os pesos históricos do Pedro; o scorecard continua a decidir promoção e não
uma soma opaca desses termos.

## Controlo estático

Cada leaf deve ser avaliado com multiplicador neutro `1.0`. O CC aprendido
deve ainda ser confrontado com pelo menos o incumbente fixo `1.05`. Qualquer
procura retrospetiva pelo melhor multiplicador no próprio ano é identificada
como oracle in-sample e não como política generalizada.

## Scorecard congelado

Perfil principal: `cc_frozen_leaf_scorecard_v1`, aplicado à superfície
settlement-on. Ordem de decisão:

1. job terminado e KPIs vindos do export do simulador;
2. EV minimum feasible `>= 0.99`;
3. violações elétricas `<= 1e-6 kWh` e zero eventos;
4. zero ciclos deferrable falhados, energia não servida `<= 1e-6 kWh` e
   service level `>= 0.99`;
5. zero violações de SOC, SOC dentro de `[0, 1]` e outage não servido
   `<= 1e-9`;
6. EV within-tolerance feasible `>= 0.40`;
7. só então comparar custo settled.

O perfil tolerante de projeção continua separado e explicitamente nomeado;
não substitui silenciosamente os hard gates estritos.

Métricas secundárias obrigatórias: importação, picos diário e absoluto,
ramping, load-factor penalty, emissões e autoconsumo solar. Regressões
relativas superiores a 1%, ou perda absoluta de autoconsumo superior a 0,005,
ficam marcadas como trade-off. Exportação, net exchange, throughput de
bateria, V2G e métricas do mercado local são sempre mostradas.

Fairness é calculada por edifício contra o comparador emparelhado: número e
percentagem de losers, pior delta local e índice de Jain sobre poupanças não
negativas. O Gini da imagem do Pedro só será reproduzido depois de recebermos
a fórmula exata; não se inventa uma definição incompatível.

## Critérios da história científica

- `PPO < SMART`: o controlador local aprendido melhora o SMART na mesma
  superfície settled.
- `CC-SMART < SMART`: o coordenador acrescenta valor ao SMART.
- `CC-PPO < PPO`: o coordenador acrescenta valor ao PPO congelado.
- `CC-PPO < CC-SMART`: o melhor leaf continua a ser melhor sob o mesmo CC.

Todos os sinais `menor que` referem-se ao custo settled e só são aceites se
os hard gates passarem. A tabela mostra também o custo contrafactual, para que
uma redução produzida apenas pelo settlement não seja atribuída ao CC.

## Limitações declaradas

- As quatro passagens repetem o mesmo ano; esta campanha mede desempenho
  anual in-sample e robustez a seeds, não generalização temporal.
- O resolved config/commit do SMART standalone do Pedro continua em falta.
- O protocolo novo adota a gama 0,5--1,3 do `CC-TD3` histórico. Esta decisão
  foi tomada antes dos primeiros treinos remotos de `CC-SMART` e `CC-PPO`;
  não altera os replays neutros SMART/PPO já concluídos.
- `SignalAwareRBC` deriva de `RBCSmartPolicy`, enquanto o PPO residual usa uma
  base estritamente local `RBCSmartLocalPolicy`; esta diferença é parte dos
  controladores comparados e fica visível no protocolo.

## Templates executáveis

Os quatro templates iniciais ficaram congelados em
`configs/experiments/ppo_cc_settlement_annual_v1/`:

- `smart_settlement_annual.yaml`;
- `cc_smart_settlement_annual_seed123.yaml`;
- `ppo_settlement_annual_seed789.yaml`;
- `cc_ppo_settlement_annual_seed789.yaml`.

O checkpoint PPO seed 789 foi reduzido de aproximadamente 809 MB para um pack
frozen de aproximadamente 5,1 MB em
`artifacts/frozen_ppo/annual_v1/seed789`. Os tensores dos 17 atores e value
nets mantêm paridade exata com os checkpoints de origem; replay, optimizers e
estado de exploração foram removidos porque o leaf não aprende nesta campanha.
Isto torna o checkpoint parte da imagem e elimina a dependência remota de
`runs/`, que é ignorado no build.

Os quatro schemas validam e os quatro pipelines constroem o ambiente real com
17 agentes e 26 ações. PPO e CC-PPO carregam o pack compacto e completam um
passo determinístico de inferência.

## Validação local e execução remota

Os quatro caminhos passaram um smoke local end-to-end com settlement, export
do scorecard e checkpoint PPO compacto. CC-SMART e CC-PPO completaram BC, uma
atualização PPO real e avaliação determinística final. A evidência e as
limitações estão em `2026-08-04_ppo_cc_settlement_local_smoke.md`.

Os replays anuais neutros da primeira wave terminaram com sucesso na imagem
`test-validate-ppo-cc-settlement-pipeline-0133d85`:

| Linha | Job | Custo settled | EV mínimo | Rede |
|---|---|---:|---:|---:|
| SMART | `b0747ffe-5a62-4e68-8218-765deffd4c78` | EUR 21 964,67 | 0,99818 | 0 kWh |
| PPO seed 789 | `18649307-5dbf-4bee-96a5-47fb5ca3531a` | EUR 20 850,00 | 0,99927 | 0 kWh |

O PPO melhora o SMART emparelhado em EUR 1 114,67 (5,08%). A evidência bruta
está em `runs/remote_results/ppo_cc_settlement_annual_v1_wave1_20260804/`.

Antes da primeira wave aprendida, a gama do CC foi alinhada com o `CC-TD3`
anual comparável do Pedro e alterada de 0,5--1,5 para 0,5--1,3. Os testes
contratuais passaram (`16 passed`) e os dois smokes reais repetidos com 1,3
confirmaram `bc_pretrain_done=true` e `ppo_update_count=1`.

Em 2026-08-04 foram submetidos os dois primeiros treinos anuais, ambos com
quatro passagens, settlement ligado e apenas o CC treinável:

| Linha | Job | Destino | Seed | Estado em 2026-08-04 22:11 UTC |
|---|---|---|---:|---|
| CC-SMART | `dba92ef6-cfcd-4fbf-81c0-c394cc3baf54` | `server` | 123 | `finished`, exit 0 |
| CC-PPO | `ee53bfda-c7f3-4870-bf1a-dc3b3fee45f0` | `deucalion` CPU | 789 | `running` |

Os configs foram enviados inline e os resolved configs preservam a gama
0,5--1,3. O preflight confirmou workers `0.5.3` livres e a imagem pronta nos
dois destinos. Manifest e monitorização ficam em
`runs/remote_configs/ppo_cc_settlement_annual_v1_wave2_cc_pmax13_20260804/`.
A expansão de seeds só acontece depois de avaliar esta wave contra os replays
neutros emparelhados.

## Tabela canónica anual com settlement (parcial)

Perfil: `cc_frozen_leaf_scorecard_v1`. A decisão exige EV mínimo >= 0,99,
EV dentro da tolerância >= 0,40, zero violações elétricas, deferrables sem
falhas, SOC em `[0, 1]` e zero outage não servido antes de comparar custo.

| Linha | Custo settled | Delta emparelhado | Hard gates | Decisão |
|---|---:|---:|---|---|
| SMART | EUR 21 964,67 | referência | `PASS_HARD_GATES` | `REFERENCE` |
| CC-SMART | EUR 21 937,95 | EUR -26,72 (-0,12%) | `PASS_HARD_GATES` | `PASS_CC_SCORECARD`, marginal |
| PPO seed 789 | EUR 20 850,00 | EUR -1 114,67 vs SMART (-5,08%) | gates anuais disponíveis passam | referência PPO |
| CC-PPO seed 789 | pendente | pendente vs PPO | pendente | `running` |

O CC-SMART preserva serviço e segurança: EV mínimo 0,99818, EV dentro da
tolerância 0,97928, zero violações elétricas, zero falhas deferrable, SOC
`0,0:0,99983` sem violações e zero outage. Reduz importação em 164,81 kWh,
exportação em 235,55 kWh, pico diário em 0,47% e ramping em 0,67%; o pico
absoluto fica igual. Emissões pioram 0,22%, abaixo do limiar de regressão de
1%, e throughput de bateria cresce 5,82%.

A melhoria settled não corresponde a uma melhoria física líquida do mesmo
tamanho: o custo contrafactual sem settlement piora EUR 27,80, enquanto a
poupança atribuída ao mercado local aumenta EUR 54,52. O saldo é a redução de
EUR 26,72. Quinze dos 17 edifícios reduzem custo local; `Building_10` piora
EUR 12,96 e `Building_15` EUR 17,12. Por isso esta linha passa formalmente mas
não é ainda evidência de um ganho robusto do CC.

Scorecard e séries finais completas:
`runs/remote_results/ppo_cc_settlement_annual_v1_wave2_cc_pmax13_20260804/scorecards/cc_smart/`.

## Atualização de 2026-08-05: campanha CC-SMART cost-focus V2 (parcial)

A imagem `experiment-add-cc-smart-cost-focus-v2-9f64c22` lançou quatro
receitas anuais com settlement, SMART congelado e gama do CC 0,5--1,3. A
campanha ainda não está encerrada, pelo que não foi adicionada ao ledger
canónico; este é o registo intermédio dos resultados efetivamente recolhidos.

| Receita | Job | Destino | Estado observado em 2026-08-05 06:55 UTC |
|---|---|---|---|
| `settled_focus_adaptive` | `d15d201c-ad30-47de-b0c4-7d504c6bd68c` | `server` | `finished`, exit 0 |
| `legacy_long_control` | `fe0bfedd-56fe-4617-a0f4-c544db477c82` | `server` | `finished`, exit 0 |
| `hybrid_physical_adaptive` | `3330e778-4da4-4d1c-81d4-e28555e071cd` | Deucalion CPU | `queued` |
| `settled_focus_regularized` | `8ca52657-4f16-4618-9be5-31b14fe9a447` | `tiago-laptop` | falha de infraestrutura em `setup:image_pull`, sem simulação |
| `settled_focus_regularized` (reposição) | `4a0c38f9-0a94-422a-a660-c6958240e8f2` | Deucalion CPU | `queued` |

Os dois jobs terminados têm os 53 ficheiros de simulação, trace de decisão e
manifest recolhidos. Depois de restaurar esses ficheiros, ambos passam os
hard gates anuais: EV mínimo e dentro da tolerância, rede, deferrables, SOC e
outage. A comparação emparelhada é:

| Linha | Custo settled | Delta vs SMART | Custo contrafactual vs SMART | Edifícios melhores | Decisão |
|---|---:|---:|---:|---:|---|
| SMART | EUR 21 964,67 | referência | referência | 0/17 | `REFERENCE` |
| CC-SMART V1 | EUR 21 937,95 | EUR -26,72 (-0,122%) | EUR +27,80 | 15/17 | `PASS_CC_SCORECARD` |
| `settled_focus_adaptive` | EUR 21 960,20 | EUR -4,46 (-0,020%) | EUR -94,99 | 2/17 | `PASS_CC_SCORECARD` |
| `legacy_long_control` | EUR 21 955,74 | EUR -8,92 (-0,041%) | EUR -103,89 | 3/17 | `PASS_CC_SCORECARD` |

As V2 produzem melhoria física real: `settled_focus_adaptive` reduz importação
em 82,47 kWh, pico diário em 0,22%, ramping em 2,80% e emissões em 0,40%; a
receita `legacy_long_control` reduz importação em 135,36 kWh, pico diário em
0,36%, ramping em 3,50% e emissões em 0,49%. Não há regressões secundárias
acima dos limiares do scorecard.

Contudo, a melhoria física não se converte integralmente em custo settled.
Face ao SMART, as duas V2 perdem respetivamente EUR 90,52 e EUR 94,97 de
poupança do mercado local. O saldo oficial fica assim limitado a EUR 4,46 e
EUR 8,92; a V1 continua a melhor versão por custo settled. Nos traces finais,
as V2 convergem para multiplicadores altos e sobretudo correlacionados com a
tarifa: a mediana fica perto de 1,29 e a sensibilidade a importação/PV cai
fortemente. Isto é consistente com menor coordenação física local e menor
benefício do settlement.

O horizonte de rollout V2 também mudou de 96 para 336 passos. Apesar de seis
episódios de treino em vez de dois, isto dá aproximadamente 156 atualizações
PPO (`6 * 8760 / 336`), contra aproximadamente 182 na V1
(`2 * 8760 / 96`). Houve mais interação total, mas não mais passos de
otimização; por isso estes resultados não demonstram que simplesmente treinar
durante mais episódios melhora o CC.

Scorecards parciais e comparação conjunta V1/V2:
`runs/remote_results/cc_smart_cost_focus_v2_annual_20260805/scorecards/`.

### Run CC-PPO bloqueada no Deucalion

O job anterior `ee53bfda-c7f3-4870-bf1a-dc3b3fee45f0` continua oficialmente
`running`, mas não mostra progresso desde `2026-08-04 23:49:34 UTC`. Ficou no
episódio 4/4, passo 7 779/35 040, `global_step=112896/140160` e 80,55%. O fim
do log contém passos normais e nenhum erro Python.

Às 06:55 UTC, o worker Deucalion permanecia online mas reportava o job em
`active_job_ids` e simultaneamente uma lista `active_jobs` vazia. O estado
Slurm `RUNNING` e o detalhe de execução também deixaram de atualizar. A
evidência aponta para estado de lifecycle órfão entre worker, Slurm e
orquestrador, e não para uma simulação apenas lenta. Como o worker aceita um
único job CPU em simultâneo, este estado mantém os dois jobs V2 de Deucalion
em fila. O job não foi parado ou ressubmetido sem autorização explícita.

### Recuperação autorizada e relançamentos em 2026-08-05

Depois de autorização explícita, o `stop` normal do job CC-PPO bloqueado foi
aceite mas permaneceu sem confirmação do worker. O job foi então reconciliado
operacionalmente para `failed` com o motivo
`ops_reconcile_stalled_deucalion_no_progress`. Esta transição libertou o slot
CPU do Deucalion sem apagar a evidência do job incompleto.

O `hybrid_physical_adaptive` entrou em execução no Deucalion. A reposição
`settled_focus_regularized` foi reencaminhada, ainda em fila, para o `server`
depois de preflight estrito da mesma imagem e começou a executar com progresso
real. A reposição anual da CC-PPO, sem alterações ao PPO congelado ou ao CC,
foi lançada no Union-INESCTEC:

| Linha | Job | Destino | Estado às 07:48 UTC |
|---|---|---|---|
| V2 `hybrid_physical_adaptive` | `3330e778-4da4-4d1c-81d4-e28555e071cd` | Deucalion CPU | `running` |
| V2 `settled_focus_regularized` | `4a0c38f9-0a94-422a-a660-c6958240e8f2` | `server` | `running` |
| CC-PPO reposição R1 | `2d79643a-967e-48ca-823c-234821bf2976` | Union-INESCTEC | `running` |

O preflight Union confirmou worker `0.5.3` online, autenticação recente e a
imagem `test-validate-ppo-cc-settlement-pipeline-0133d85` pronta. Evidência
operacional:
`runs/remote_configs/ppo_cc_settlement_annual_v1_wave2_cc_pmax13_20260804/`.

### Campanha de identificação CC-SMART V3

Os resultados V2 sugerem duas causas diferentes que precisam de ser
separadas. Primeiro, um único multiplicador global só altera os limiares
`cheap`/`expensive` do SMART; não tem a autoridade contínua e por edifício do
PPO. Segundo, o rollout V2 de 336 decisões produziu menos atualizações PPO e
as políticas finais ficaram enviesadas para multiplicadores altos.

A campanha `cc_smart_price_response_v3_annual_20260805` foi definida antes de
observar qualquer resultado e mede estas hipóteses diretamente:

| Receita | Job | Objetivo |
|---|---|---|
| fixo 0,7 | `ba61937c-c811-4272-ac1f-02c5e922bd19` | limite inferior da resposta SMART |
| fixo 0,9 | `ac59fcc0-3973-435f-9187-8b796c4071d2` | resposta conservadora abaixo do neutro |
| fixo 1,1 | `7f62487e-5748-4294-8057-cde3654fd20d` | resposta conservadora acima do neutro |
| fixo 1,3 | `da5f5f3e-560d-40ab-bb84-2a7cb9f33bf3` | limite superior da resposta SMART |
| `legacy_update_dense` | `1c1d7be4-49fc-4e90-ba7c-36e4c0e254fc` | V1 exata com aproximadamente 547 atualizações PPO |

O replay SMART neutro 1,0 existente continua a referência. Os quatro fixos
medem a margem alcançável pelo canal escalar e impedem que uma melhoria por
mero bias constante seja atribuída a coordenação aprendida. A receita densa
mantém reward, BC, regularização, gamma e rollout V1, aumentando apenas os
episódios de treino; compara aproximadamente 547 atualizações com 182 na V1 e
156 nas V2 longas.

Todos os configs validaram e os testes contratuais relacionados passaram
(`28 passed`). A imagem `experiment-add-cc-smart-cost-focus-v2-9f64c22` já
contém todo o código necessário; os configs foram enviados inline depois de
preflight Union estrito. Evidência operacional:
`runs/remote_configs/cc_smart_price_response_v3_annual_20260805/`.
