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
- `price_max: 1.5`.
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
- O `CC-TD3` histórico usa 0,5--1,3 e não é reclassificado como 0,5--1,5.
- `SignalAwareRBC` deriva de `RBCSmartPolicy`, enquanto o PPO residual usa uma
  base estritamente local `RBCSmartLocalPolicy`; esta diferença é parte dos
  controladores comparados e fica visível no protocolo.

## Próxima ação autorizável

Construir e validar os quatro templates da matriz. Depois executar primeiro o
SMART neutral settled e o smoke de paridade PPO neutral settled. Nenhum job
foi preparado ou submetido durante esta auditoria.
