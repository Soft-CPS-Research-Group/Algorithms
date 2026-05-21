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
| `RandomPolicy` | Amostra todas as acoes disponiveis dentro dos bounds. | Lower bound/sanidade. Serve para apanhar erros de bounds, fases, V2G, storage e deferrables. |
| `NormalNoBatteryPolicy` | Comportamento normal sem controlo de bateria. EV carrega quando chega; deferrables arrancam cedo. | Aproxima o comportamento "sem otimizacao" quando queremos isolar o efeito da bateria. |
| `NormalPolicy` | Igual ao normal, mas com bateria simples para autoconsumo. | Baseline BAU operacional simples; pode ser pior que no-battery se bateria tiver perdas/ciclos maus. |
| `RBCBasicPolicy` | Heuristica simples com urgencia, preco atual/previsao curta e limites. | Baseline inteligente medio. Nao deve ser oracle. |
| `RBCSmartPolicy` | Heuristica mais forte: preco, PV, picos/headroom, EV service, storage conservador e V2G quando configurado/seguro. | Baseline forte principal para comparar candidatos RL/MARL. |
| `RuleBasedPolicy` | RBC legacy EV-focused. | Mantido para compatibilidade/debug; nao e o baseline principal. |

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
