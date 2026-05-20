# Auditoria da Bateria Estacionaria

Data: 2026-05-18.

Objetivo: perceber porque algumas runs ficam melhores sem usar a bateria, antes
de correr mais MADDPG longo.

## Artefactos

Auditoria gerada em:

- `runs/storage_audits/phase6f_storage_behavior/dataset_storage_schema.csv`
- `runs/storage_audits/phase6f_storage_behavior/storage_run_summary.csv`
- `runs/storage_audits/phase6f_storage_behavior/storage_reward_metric_summary.csv`
- `runs/storage_audits/phase6f_storage_behavior/storage_by_building_episode.csv`

Script reutilizavel:

- `scripts/audit_storage_behavior.py`

## Conclusoes

1. O dataset nao parece ter consumo parado/standby da bateria.

Nos dois datasets principais:

- `citylearn_three_phase_electrical_service_demo_15s_parquet`;
- `citylearn_challenge_2022_phase_all_plus_evs`;

as baterias estacionarias têm:

- `loss_coefficient = 0.0`;
- `capacity_loss_coefficient = 1e-05`;
- `capacity = 6.4 kWh`;
- `efficiency = 0.9`;
- `nominal_power = 5.0 kW`.

Nos exports, quando a politica deixa a bateria idle, `idle_soc_abs_delta_sum`
fica `0.0` ou praticamente zero. Portanto nao ha evidencia de a bateria estar
a gastar energia parada.

2. A bateria pode ajudar, mas depende muito da politica.

Resumo das runs existentes:

| run | carga kWh | descarga kWh | throughput kWh | custo/price exportado |
|---|---:|---:|---:|---:|
| `15s_normal_no_battery` | 0.000 | 0.000 | 0.000 | 112.505 |
| `15s_normal` | 101.850 | 63.146 | 164.996 | 113.876 |
| `15s_rbc_basic` | 90.722 | 25.447 | 116.169 | 99.495 |
| `15s_rbc_smart` | 0.000 | 0.000 | 0.000 | 97.241 |
| `2022_normal_no_battery` | 0.000 | 0.000 | 0.000 | 4729.065 |
| `2022_normal` | 8228.608 | 6881.917 | 15110.526 | 4847.261 |
| `2022_rbc_basic` | 4512.000 | 3624.000 | 8136.000 | 3805.410 |
| `2022_rbc_smart` | 0.000 | 0.000 | 0.000 | 3967.057 |

Leitura:

- `Normal` usa bateria de forma demasiado mecanica e pode piorar custo por
  perdas de ciclo.
- `RBCBasic` mostra que a bateria pode ajudar bastante, sobretudo no dataset
  2022.
- `RBCSmart` estava demasiado conservador para storage: em ambas as janelas
  analisadas ficou com bateria totalmente idle.

3. O problema mais serio estava na reward nomeada usada no MADDPG.

Os templates base usam:

- `battery_soc_min = 0.0`;
- `battery_soc_max = 1.0`;
- `scale_state_penalties_by_time_step = true`.

Mas as variantes nomeadas do benchmark, como `CostServiceCommunityServiceBandRewardV42`,
substituiam `reward_function_kwargs` por `{}`. Com isso a reward voltava aos
defaults da classe base:

- `battery_soc_min = 0.05`;
- `battery_soc_max = 0.95`;
- `scale_state_penalties_by_time_step = false`.

Isto fez a V42 penalizar uma bateria vazia e parada como se estivesse abaixo de
um limite de conforto inventado. Nos logs da `phase6f11`, logo no primeiro
snapshot:

- `battery_soc_min_limit_mean = 0.05`;
- `battery_soc_below_limit_mean = 0.05`;
- `battery_soc_violation_penalty_amount_mean = 1.5`;
- `battery_throughput_penalty_mean = 0.0`.

Ou seja: a penalizacao forte nao vinha de ciclos, vinha de SOC zero ser tratado
como violacao.

4. A penalizacao de throughput da V42 tambem era demasiado alta para procurar
valor economico na bateria.

`battery_throughput_penalty = 1.2` implica que 1 kWh de throughput pode custar
muito mais na reward do que a poupança marginal do settlement comunitario nessa
janela. Isto desencoraja a bateria antes de ela conseguir aprender arbitragem
ou autoconsumo comunitario util.

## Alteracoes Aplicadas

1. Nova reward:

- `CostServiceCommunityBatteryValueRewardV43`

Mantem a pressao de servico EV da V42, mas muda a parte de bateria:

- `battery_soc_min = 0.0`;
- `battery_soc_max = 1.0`;
- `use_observed_storage_soc_limits = true`;
- `scale_state_penalties_by_time_step = true`;
- `state_penalty_reference_seconds = 3600.0`;
- `battery_throughput_penalty = 0.02`.

Os limites `0.0..1.0` sao apenas fallback fisico quando o simulador/dataset nao
expõe limites especificos. Se vierem observacoes como
`electrical_storage_soc_min_ratio` e `electrical_storage_soc_max_ratio`, esses
limites continuam a ser respeitados e penalizados.

2. Novo variant de benchmark:

- `community_battery_value_v43_prioritized_warmtrain_phaseout_rbc_smart`

Este variant fica preparado para a proxima run MADDPG, mas ainda nao foi usado
para uma experiencia longa.

3. `RBCSmartPolicy` ajustado para storage.

Antes, a bateria so carregava em barato/PV se a rede nao estivesse stressed.
Como `stressed` incluia import alto, a politica podia bloquear carga mesmo com
headroom eletrico suficiente. Agora:

- carga em PV/preco baixo e bloqueada por low headroom, nao por import alto;
- descarga continua a responder a preco alto/import peak;
- os limites fisicos/fases continuam a ser tratados pelo clipping de headroom.

## Implicacao

Antes de nova run longa, a comparacao correta deve ser:

1. repetir baselines curtos para confirmar que `RBCSmart` passou a usar bateria
   quando faz sentido;
2. correr a nova receita MADDPG V43 em janela 15s com departures;
3. comparar contra `RBCSmart` corrigido, nao contra o RBCSmart antigo idle.

Isto deve acontecer antes de concluir que MADDPG ja bate o baseline forte.

## Atualizacao 2026-05-19 - Baselines 0.6.6 e RBCSmart 2022

Nova matriz local:

- `runs/benchmarks/phase6g_baseline_storage_check`;
- auditoria storage:
  `runs/storage_audits/phase6g_baseline_storage_check`.

Resultados principais:

| run | custo | carga kWh | descarga kWh | throughput kWh | storage idle |
|---|---:|---:|---:|---:|---:|
| `15s_normal_no_battery` | 112.505 | 0.000 | 0.000 | 0.000 | 1.000 |
| `15s_normal` | 114.035 | 94.464 | 56.955 | 151.419 | 0.788 |
| `15s_rbc_basic` | 99.453 | 82.146 | 21.828 | 103.974 | 0.660 |
| `15s_rbc_smart` | 97.105 | 58.717 | 16.900 | 75.617 | 0.815 |
| `2022_normal_no_battery` | 4729.065 | 0.000 | 0.000 | 0.000 | 1.000 |
| `2022_normal` | 4825.920 | 6346.507 | 5304.725 | 11651.233 | 0.792 |
| `2022_rbc_basic` | 3835.326 | 3916.500 | 3142.500 | 7059.000 | 0.723 |
| `2022_rbc_smart` antigo | 3894.033 | 3782.000 | 3065.000 | 6847.000 | 0.799 |

Leitura:

- `Normal` continua pior que `NormalNoBattery`: a bateria nao esta a perder
  energia parada, esta a ser usada em momentos pouco valiosos e paga perdas de
  ciclo.
- `RBCBasic` e `RBCSmart` provam que a bateria e util quando a estrategia olha
  para preco/PV/peak.
- No 15s, o `RBCSmart` atual e o baseline mais forte: menor custo, EV feasible
  dentro da tolerancia a `1.0`, zero violacoes.
- No 2022, o `RBCSmart` antigo perdia para `RBCBasic` porque carregava storage
  a preco medio mais alto (`0.131`) do que o Basic (`0.112`).

Alteracoes aplicadas depois desta auditoria:

- `RBCSmartPolicy` deixou de limitar a carga por preco barato a
  `storage_target_soc`; passa a respeitar `storage_price_charge_soc_ceiling`.
- A carga por PV deixou de bloquear a carga por preco quando `pv_charge_rate`
  e zero ou inferior a `price_charge_rate`.
- O clipping fisico de storage continua a ser feito depois da decisao da
  policy, preservando headroom/fases/SOC.
- O template `rbc_smart_local.yaml` ficou com floors de descarga `0.30`,
  recuperando o baseline 15s forte: custo `97.105`, EV feasible min `1.0`,
  EV feasible dentro da tolerancia `1.0`, sem violacoes.
- O template `rbc_smart_2022_all_plus_evs_local.yaml` passou a usar
  `price_charge_rate=0.15`, `storage_price_charge_soc_ceiling=0.85`,
  `storage_price_discharge_soc_floor=0.30` e
  `storage_peak_discharge_soc_floor=0.30`.
- O schema de config passou a preservar estes knobs e
  `deferrable_safety_margin_steps`; sem isto, alguns parametros existiam no
  YAML mas eram descartados no config resolvido.

Validacao do template 2022 atualizado:

- run: `runs/benchmarks/phase6g_rbcsmart_2022_template_verify_final`;
- auditoria storage:
  `runs/storage_audits/phase6g_rbcsmart_2022_template_verify_final`;
- custo: `3823.523`, melhor que `RBCBasic` (`3835.326`) e que o
  `RBCSmart` antigo (`3894.033`);
- EV feasible min: `1.0`;
- EV feasible dentro da tolerancia: `0.435816`;
- violacao eletrica: `0.0`.
- storage: `5783.500 kWh` de carga, `4691.000 kWh` de descarga,
  `10474.500 kWh` de throughput, sem perda de SOC em idle.

Implicacao:

- manter o `RBCSmart` 15s conservador atual;
- usar o `RBCSmart` 2022 retuned como baseline forte nesse dataset;
- quando voltarmos ao MADDPG, o baseline forte ficou mais justo e ligeiramente
  mais dificil no 2022, mas o 15s de referencia para o `f14` continua o
  `RBCSmart` de custo `97.105`.
