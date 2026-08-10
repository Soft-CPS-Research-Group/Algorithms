# Recuperação do CC-L2 sobre PPO congelado

- Dataset: `citylearn_three_phase_electrical_service_demo_15min_parquet`
- Janela de evidência: ano completo, passos `0:35039`
- Settlement comunitário: ligado
- PPO local de referência: seed `789`, 17 folhas estritamente locais
- Código das runs rejeitadas: `4278675`
- Imagens: `fix-recover-matd3-and-train-price-aware-cc-policies-4278675`
  e variante `-union-blackwell`
- Arquivo: `2026-08-10T10:10:00+01:00`
- Baseline: PPO seed `789`, settlement-on, ano completo, EUR 21 050,00
- Gates de decisão: `phase6` (`EV mínimo >= 0.999`, precisão EV `>= 0.80`,
  violações elétricas `<= 1e-6 kWh`), seguidos do custo e scorecard físico
- Estado: V5.3 fechada; CC-L2 PPO V1 rejeitada; V2 validada por testes e smoke

## Decisões

A linha causal V5.3 fica encerrada. O melhor candidato, `hourly balanced`,
atingiu EUR 20 829,77 e manteve zero violações elétricas, mas ficou EUR 111,62
acima da melhor V5.2 diagnóstica (EUR 20 718,15). Os resultados ficam
preservados como evidência; não serão submetidas novas variantes V5.3.

A campanha `cc_level2_ppo_joint_v1` também fica rejeitada. A comparação sem
V2G foi:

| Configuração | Custo settled | Delta vs PPO EUR 21 050 | Importação | Solar self-consumption | Decisão |
|---|---:|---:|---:|---:|---|
| PPO neutro seed 789 | EUR 21 050,00 | referência | 129 871,29 kWh | 72,622% | `REFERENCE` |
| CC-L2 PPO current/storage | EUR 23 650,72 | EUR +2 600,72 | cerca de 142 MWh | cerca de 61% | `REJECT` |
| CC-L2 PPO forecasts/storage | EUR 23 585,66 | EUR +2 535,66 | cerca de 142 MWh | cerca de 62% | `REJECT` |

A variante V2G da campanha V1 não é promovida nem continuada. A recuperação
V2 foca apenas o PPO residual-battery já congelado e o sinal de preço L2.

## Jobs arquivados

| Linha | Job ID | Host | Estado |
|---|---|---|---|
| CC-L2 PPO current/storage | `a50985b2-e7a3-46b0-80e0-1a663a51ef38` | Union | finished |
| CC-L2 PPO forecasts/storage | `4d1f6ebe-b0c2-401e-9eac-84ba408ba506` | Union | finished |
| V5.3 15-min cost | `80ce0941-cb1b-4734-81a3-fc30d80420b6` | server | finished |
| V5.3 hourly cost | `3e0fed3f-430b-4875-89e1-d8d069c80f01` | server | finished |
| V5.3 hourly balanced | `245250b0-9dfb-4c4f-8440-09761a0acf0d` | server | finished |

## Causa da degradação V1

O CC não estava inerte. Os traces mostram preços médios por episódio a moverem
de aproximadamente 1,09 para 0,96--1,02, com dispersão entre edifícios. O
problema foi o contrato de treino conjunto:

1. as 17 folhas PPO foram descongeladas desde a primeira transição;
2. todas receberam a mesma recompensa comunitária, sem crédito local;
3. começaram a atualizar durante o warm-up BC do próprio CC;
4. a base local do checkpoint mudou de `RBCSmartLocalPolicy` para
   `SignalAwareRBCSmartLocal` ao mesmo tempo que os atores eram treinados;
5. a ação L2 variava a cada 15 minutos em toda a gama `[0.5, 1.3]`.

Esta combinação apagou o comportamento local forte. A importação aumentou,
o autoconsumo solar caiu cerca de 11 pontos percentuais e o throughput de
bateria caiu para aproximadamente 21--26 MWh. A diferença mínima entre a
receita com preço atual e a receita com forecasts confirma que o forecast não
era a causa dominante.

Foi ainda corrigida uma transição inválida no limite BC/PPO do CC-L2: a última
ação do professor podia entrar no rollout com log-probabilidade zero como se
fosse uma amostra da policy. Isso provocava KL artificialmente elevado e podia
bloquear a primeira atualização PPO.

## Protocolo V2

A V2 introduz gates explícitos e deixa de treinar as duas camadas em conjunto:

1. **Gate A — PPO original neutro:** reproduzir o composto original
   `RBCSmartLocalPolicy + PPO residual battery + safety projector`.
2. **Gate B — caminho de sinal neutro:** trocar apenas a base por
   `SignalAwareRBCSmartLocal`, manter multiplicador `1.0` e demonstrar paridade
   com o Gate A.
3. **Treino CC-L2:** congelar e tornar determinísticos os 17 PPOs; treinar
   exclusivamente o coordenador.
4. **Gama conservadora:** preços por edifício apenas em `[0.90, 1.00]`, com
   referência exata em `1.0` e decisões horárias.
5. **Reward alinhada:** custo settled continua dominante, com termos menores
   de pico, ramping e exportação, penalização de violações e proteção de
   serviço EV. V2G está desligado.
6. **Promoção:** nenhuma policy é promovida sem custo inferior ao PPO, hard
   gates, scorecard emparelhado e comparação por edifício.

Foram preparadas duas receitas anuais: `cost_seed123` e
`scorecard_seed456`. Ambas usam dez episódios, BC causal durante o primeiro
ano, oito anos de aprendizagem possíveis e um último ano determinístico.

## Validação local

- 64 testes focados e a suite completa de 986 testes passaram.
- Os Gates A e B foram executados em dois smokes CityLearn emparelhados de
  384 transições. Os 53 ficheiros exportados foram idênticos byte a byte,
  incluindo `exported_kpis.csv` e toda a timeseries comunitária. O caminho
  `SignalAwareRBCSmartLocal` é, portanto, exatamente neutro em `1.0`.
- Smoke CityLearn real de 384 transições por episódio terminou com os 17
  checkpoints carregados.
- BC terminou com loss `0.003970`.
- A primeira atualização genuinamente on-policy terminou com `KL=0.0001`,
  `kl_stop=False` e policy loss não nula.
- O resultado e o manifesto foram exportados sem alterar ou exportar
  checkpoints das folhas congeladas.

Configurações anuais:
`configs/experiments/cc_level2_ppo_frozen_v2/`.

Evidência remota V1/V5.3 ignorada pelo Git:
`runs/remote_results/cc_matd3_ppo_research_20260809_4278675/reconcile_20260810/`.

Smoke V2 ignorado pelo Git:
`runs/cc_level2_ppo_frozen_v2_smokes/jobs/cc-l2-ppo-frozen-v2-cost-smoke-r2/`.
