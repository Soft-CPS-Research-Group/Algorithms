# Update Para A Equipa: Baselines, MADDPG E Performance

Data: 2026-05-21.

Este repo passou a ter uma base mais solida para comparar controladores RL/MARL
contra baselines deterministicos. O objetivo deixou de ser "fazer MADDPG ganhar
a qualquer custo"; o objetivo e encontrar o melhor algoritmo para o nosso caso:
comunidade energetica com EVs, baterias, deferrables, restricoes eletricas,
custo, renovaveis e KPIs de servico.

## Versao Do Simulador

O repo esta alinhado com `softcpsrecsimulator==0.6.9`.

Esta versao e importante porque traz melhorias de performance e permite reduzir
o payload de observacoes que a reward precisa de receber. Isto reduz overhead
quando a reward nao precisa de todas as observacoes do simulador.

## Novos Baselines

Os baselines estao registados como algoritmos e podem ser usados diretamente nos
configs.

| Baseline | Objetivo | Leitura correta |
|---|---|---|
| `RandomPolicy` | Amostra todas as acoes disponiveis dentro dos bounds. | Lower bound/sanidade. Serve para mostrar que um algoritmo e melhor que aleatorio e para apanhar erros de bounds, fases, V2G, storage e deferrables. |
| `NormalNoBatteryPolicy` | Comportamento normal sem controlo de bateria. EV carrega quando chega; deferrables arrancam cedo. | Aproxima o comportamento "sem otimizacao" quando queremos isolar o efeito da bateria. |
| `NormalPolicy` | Igual ao normal, mas com bateria simples para autoconsumo. | Baseline BAU (`Business as Usual`) operacional simples; pode ser pior que no-battery se bateria tiver perdas/ciclos maus. |
| `RBCBasicPolicy` | Heuristica simples com urgencia, preco atual/previsao curta e limites. | Baseline inteligente medio. Nao deve ser oracle. |
| `RBCSmartPolicy` | Heuristica mais forte: preco, PV, picos/headroom, EV service, storage conservador e V2G quando configurado/seguro. | Baseline forte principal para comparar candidatos RL/MARL. |
| `RuleBasedPolicy` | RBC legacy EV-focused. | Mantido para compatibilidade/debug. Nao usar na maior parte das comparacoes novas. |

Configs locais principais:

```bash
python run_experiment.py --config configs/templates/baselines/random_local.yaml
python run_experiment.py --config configs/templates/baselines/normal_no_battery_local.yaml
python run_experiment.py --config configs/templates/baselines/normal_local.yaml
python run_experiment.py --config configs/templates/baselines/rbc_basic_local.yaml
python run_experiment.py --config configs/templates/baselines/rbc_smart_local.yaml
```

Para o dataset 2022 hourly/all-plus-EVs, usar as variantes:

```bash
configs/templates/baselines/*_2022_all_plus_evs_local.yaml
```

## MADDPG Melhorado

O MADDPG atual e o candidato principal implementado, mas ja ficou preparado para
comparacao mais seria:

- replay buffer compacto prealocado, para reduzir RAM e overhead de sampling;
- funcao de reward iterada; a receita `V48` e a melhor local ate agora;
- warm-start opcional com policy teacher, por exemplo `RBCSmartPolicy`;
- behavior cloning configuravel no actor;
- exploration noise configuravel por tipo de acao;
- reward normalization;
- actor/critic/lrs/batch/replay/update cadence configuraveis por YAML;
- AMP em GPU (`use_amp: true`);
- checkpoints independentes de MLflow;
- diagnosticos de treino/reward/action sampling configuraveis;
- export mais controlavel, para evitar custo desnecessario em runs longas.

Configs locais principais:

```bash
python run_experiment.py --config configs/templates/maddpg/maddpg_local.yaml
python run_experiment.py --config configs/templates/maddpg/maddpg_2022_all_plus_evs_local.yaml
```

Para runs remotas/full-year ja existem configs preparadas em:

```bash
configs/experiments/phase6_2022_full_year/
configs/experiments/phase6_15s_scaling/
configs/experiments/phase6_dataset_variants/remote_pending/
configs/experiments/phase6_algorithm_matrix/remote_pending/
```

## Reward Functions E Receita V48

Ha dois conceitos diferentes:

- `reward_function`: classe registada no repo, por exemplo
  `CostServiceCommunityFeasiblePrecisionRewardV46`;
- receita/config `V48`: combinacao que temos usado para MADDPG com essa reward,
  teacher `RBCSmartPolicy`, behavior cloning e hiperparametros especificos.

Neste momento, para novos algoritmos RL/MARL, eu usaria a reward
`CostServiceCommunityFeasiblePrecisionRewardV46` como ponto de partida. Foi a
base da receita `V48`, que teve o melhor equilibrio local entre custo, EV service
e picos. Isto nao prova que seja universalmente a melhor para todos os
algoritmos, mas e o default mais defensavel neste momento.

Variantes testadas:

| Receita | Reward base | Ideia | Leitura atual |
|---|---|---|---|
| `V48` | `CostServiceCommunityFeasiblePrecisionRewardV46` | Equilibrar custo, EV minimo aceitavel, precisao de SOC, comunidade e bateria. | Melhor candidata local. Promovida para runs longas/multi-seed. |
| `V49` | `CostServiceCommunityStorageValueRewardV49` | Reduzir penalizacao de bateria e dar mais espaco a storage/V2G. | Nao mostrou ganho suficiente; pode piorar EV. |
| `V50` | `CostServiceCommunityDeadlineValueRewardV50` | Aumentar pressao de deadline EV antes da saida. | Reduz deficit, mas tende a over-service e pior custo. |
| `V51` | `CostServiceCommunityPrecisionValueRewardV51` | Penalizar mais over-service para ficar perto do SOC alvo. | Cortou excesso, mas falhou mais o minimo EV. |
| `V52` | `CostServiceCommunityPeakDeadlineRewardV52` | Aumentar sinal de pico/export/settlement comunitario. | Piorou custo/precisao nos testes locais. |
| `V54` | Diagnostico com policy loss desligado | Quase clone do `RBCSmartPolicy`. | Provou que o actor consegue aprender o teacher; nao e a melhor receita. |
| `V55`/`V56` | Variantes com BC extra | Testar warmup/BC mais forte. | Ainda nao bateram V48 nesta configuracao. |

Comparacao local mais informativa ate agora: dataset 2022, semana curta, 8
episodios. O `RBCSmart` e a referencia forte; `V48` foi a melhor receita MADDPG.

| Controlador/receita | Custo EUR | EV min feasible | EV within feasible | Deficit SOC medio | Surplus SOC medio | Erro abs. SOC | Pico comunitario reward mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `RBCSmart` | `424.66` | `1.000` | `0.423` | `0.001` | `0.063` | `0.064` | `0.997` |
| `MADDPG V48`, 8 eps | `398.51` | `1.000` | `0.385` | `0.004` | `0.063` | `0.067` | `0.750` |
| `MADDPG V50`, 8 eps | `466.79` | `1.000` | `0.019` | `0.002` | `0.145` | `0.147` | `0.827` |
| `MADDPG V52`, 8 eps | `459.65` | `1.000` | `0.000` | `0.002` | `0.135` | `0.137` | `1.470` |
| `MADDPG V54`, 8 eps | `419.82` | `1.000` | `0.308` | `0.002` | `0.067` | `0.068` | `0.801` |
| `MADDPG V55`, 8 eps | `436.95` | `1.000` | `0.096` | `0.001` | `0.103` | `0.104` | `1.407` |
| `MADDPG V56`, 8 eps | `439.68` | `1.000` | `0.096` | `0.001` | `0.108` | `0.110` | `0.603` |
| `MADDPG V48`, 16 eps | `407.84` | `1.000` | `0.519` | `0.002` | `0.055` | `0.056` | `0.450` |

Robustez inicial da `V48` com 16 episodios e seeds `123/456/789`:

| Receita | Custo medio EUR | EV min feasible medio | EV within feasible medio | Deficit SOC medio | Erro abs. SOC medio | Pico comunitario reward mean |
|---|---:|---:|---:|---:|---:|---:|
| `MADDPG V48`, media 3 seeds | `409.74` | `1.000` | `0.474` | `0.0016` | `0.0601` | `0.444` |

Leitura curta:

- `V48` bateu o `RBCSmart` local em custo nesta janela e manteve
  `EV min feasible = 1.0`;
- `V48` tambem melhorou `EV within feasible` quando treinada mais episodios;
- aumentar peso comunitario/deadline sem preservar primeiro EV precision levou
  a over-service, pior custo ou pior precisao;
- estes resultados sao bons para escolher direcao, mas nao substituem runs
  longas/full-year/multi-seed.

## KPIs Que Tenho Usado Para Comparar

Comparacao principal:

1. primeiro garantir servico e restricoes;
2. depois custo;
3. depois uso comunitario/renovavel/picos;
4. depois eficiencia operacional de bateria/V2G/deferrables.

| KPI | Porque importa | Como ler |
|---|---|---|
| `community_cost_eur` | Custo total da comunidade. | Menor e melhor; comparar contra `RBCSmart` e BAU. |
| `cost_delta_to_bau_eur` / `cost_ratio_to_bau` | Ganho/perda contra BAU. | Negativo ou ratio < 1 e melhor. |
| `ev_min_acceptable_feasible_rate` | EV saiu com pelo menos o minimo aceitavel. | Gate de servico; deve tender para 1.0. |
| `ev_within_tolerance_feasible_rate` | EV saiu perto do SOC pedido, dentro da tolerancia. | Qualidade de otimizacao; mais exigente que minimo aceitavel. |
| `ev_departure_infeasible_count` | Quantas saidas EV falharam fisicamente/operacionalmente. | Deve ser 0 quando a janela e viavel. |
| `electrical_violation_kwh` / `electrical_violation_events` | Violacoes de fases/headroom/rede. | Deve ser 0 ou praticamente 0. |
| `battery_throughput_kwh` / `battery_throughput_ratio_to_bau` | Ciclos/uso da bateria. | Nao e sempre "menor melhor"; serve para apanhar abuso de bateria. |
| `v2g_export_kwh` | Uso de V2G. | Deve existir apenas quando faz sentido e nao sacrifica EV service. |
| `community_solar_self_consumption_rate` | Uso local/comunitario de PV. | Maior e melhor, desde que nao piore servico/rede. |
| `community_market_import_share_rate` | Dependencia de importacao externa. | Menor tende a ser melhor. |
| `peak_daily_ratio_to_bau` / `peak_all_time_ratio_to_bau` | Reducao de picos. | Ratio < 1 e melhoria face a BAU. |

## Performance E Otimizacoes

O dataset 15s e o maior problema de escala: um ano tem `2,102,400` steps por
episodio. Por isso, pequenas diferencas em segundos/step mudam dias de runtime.

| Estado | Tempo por step | 1 episodio anual 15s | 6 episodios anuais |
|---|---:|---:|---:|
| Antes das otimizacoes fortes | ~`0.108 s/step` | ~`2.6 dias` | ~`15.7 dias` |
| Estimativa intermedia que chegou a ser usada | ~`0.062 s/step` | ~`1.5 dias` | ~`9 dias` |
| Estado atual, wall-clock local | ~`0.050 s/step` | ~`29 h` | ~`7.3 dias` |
| Estado atual, mediana perfilada | ~`0.037-0.040 s/step` | ~`22-23 h` | ~`5.4-5.8 dias` |

As maiores melhorias vieram de:

- cache de encoding/entity layout;
- replay buffer compacto;
- reduzir payload da reward com `required_observation_names`;
- reduzir logs/exports/MLflow em runs longas;
- manter GPU/AMP para treino MADDPG.

Ainda assim, full-year 15s multi-episodio continua caro. Para ciencia iterativa,
o caminho recomendado e:

- treinar em janelas representativas;
- avaliar em janelas maiores/full-year;
- estudar `action_repeat`/`step_many` com cuidado;
- usar Deucalion/server para matriz de runs longas;
- comparar sempre contra baselines e BAU, nao apenas contra reward interna.

## Mensagem Principal

Temos agora uma base melhor para fazer ciencia:

- baselines mais justos e com papeis claros;
- MADDPG mais configuravel e mais eficiente;
- KPIs de EV, custo, rede, bateria, PV e comunidade mais explicitos;
- infraestrutura para scorecards remotos e analise por building;
- roadmap aberto para comparar tambem `MATD3`, `MASAC`, `IPPO`, `MAPPO` e
  `HAPPO`.

O proximo passo nao e assumir que MADDPG e o vencedor. E correr comparacoes
longas/robustas, perceber onde cada abordagem falha, e escolher o algoritmo que
resolve melhor o problema real.
