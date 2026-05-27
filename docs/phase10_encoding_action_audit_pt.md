# Phase 10 - Auditoria Encoding e Action Mapping EV

Data: 2026-05-27.

Fonte principal local: `runs/smoke_simulator_1_1_0/jobs/maddpg_v3_operational_entity_96_1779797979/bundle/artifact_manifest.json`.
Codigo auditado: `utils/entity_adapter.py`, `utils/wrapper_citylearn.py`, `algorithms/agents/maddpg_agent.py`.

## Resultado Curto

Nao encontrei evidencia de bug no alinhamento das actions EV. O perfil `maddpg_v3_operational` inclui os sinais essenciais para cada charger com action EV, e o wrapper encaminha as actions por nome para as tabelas entity do simulador.

O problema mais provavel nao e missing action mapping. E aprendizagem:

- antes desta auditoria o actor recebia deficit EV, deadline, capacidade e feedback, mas nao recebia um sinal explicito de excesso/surplus em relacao ao target;
- antes desta auditoria tambem nao existia uma action feature clara do tipo `max_charge_to_target_action_normalized`;
- a exploracao inicial `uniform_full_range` gera muita carga/V2G sem sentido e enche o replay com exemplos maus;
- BC/warm-start estavam desligados nas Wave 4 configs;
- reward shaping sozinho quase nao mudou o comportamento entre V46/V50/V52.

Alteracao local aplicada depois da auditoria: `maddpg_v3_operational` passou a expor `connected_ev_soc_surplus`, `connected_ev_soc_error_signed`, `incoming_ev_soc_surplus`, `incoming_ev_soc_error_signed` e `max_charge_to_required_soc_action_normalized`.

## Action Mapping

O wrapper em modo entity constroi layout com `EntityContractAdapter` e usa `self.action_names` para converter as actions do modelo para payload entity. A conversao final chama `to_entity_actions(actions, self.action_names)`.

Confirmado no manifest:

| Agent | Building | Actions |
|---:|---|---|
| 0 | `Building_1` | `electrical_storage`, `electric_vehicle_storage_charger_1_1`, `deferrable_appliance_deferrable_appliance_1` |
| 3 | `Building_4` | `electrical_storage`, `electric_vehicle_storage_charger_4_1` |
| 4 | `Building_5` | `electrical_storage`, `electric_vehicle_storage_charger_5_1` |
| 6 | `Building_7` | `electrical_storage`, `electric_vehicle_storage_charger_7_1` |
| 9 | `Building_10` | `electrical_storage`, `electric_vehicle_storage_charger_10_1` |
| 11 | `Building_12` | `electrical_storage`, `electric_vehicle_storage_charger_12_1` |
| 14 | `Building_15` | `electrical_storage`, `electric_vehicle_storage_charger_15_1`, `electric_vehicle_storage_charger_15_2` |

Todas as restantes buildings so tem `electrical_storage`, como esperado.

As action specs do simulador expostas no manifest tambem batem certo:

- building actions: `electrical_storage`, 17 ids;
- charger actions: `electric_vehicle_storage`, 8 charger ids;
- deferrable actions: `start`, 1 id.

O `EntityAdapter.to_entity_actions(...)` resolve cada posicao da action por nome e escreve na tabela certa (`building`, `charger`, `deferrable_appliance`). O MADDPG tambem consegue distinguir tipos de action por nome: EV, storage e deferrable.

## Encoding EV

Para todos os agentes com EV action, o perfil `maddpg_v3_operational` inclui os grupos importantes:

| Grupo | Sinais presentes |
|---|---|
| SOC atual | `connected_ev_soc` |
| Target | `connected_ev_required_soc_departure` |
| Deficit | `connected_ev_soc_deficit` |
| Excesso/erro | `connected_ev_soc_surplus`, `connected_ev_soc_error_signed` |
| Deadline | `connected_ev_departure_available`, `connected_ev_departure_urgency_24h`, `hours_until_departure_24h`, `time_until_departure_ratio` |
| Energia necessaria | `energy_to_required_soc_kwh`, `required_average_power_kw` |
| Capacidade de acao | `can_charge`, `can_discharge`, `available_charge_action_normalized`, `available_discharge_action_normalized` |
| Acao util ate target | `max_charge_to_required_soc_action_normalized` |
| Minimo operacional | `min_required_action_normalized`, `departure_feasibility_ratio`, `departure_energy_margin_kwh`, `max_deliverable_energy_until_departure_kwh` |
| Feedback | `last_requested_action_normalized`, `last_limited_action_normalized`, `last_applied_power_kw`, `last_projection_error_kw` |
| Clip reasons | `clip_reason_*`, incluindo SOC, power, headroom, no EV, not V2G |

As dimensoes observadas no manifest sao consistentes com a topologia:

- action dims: `[3, 1, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 1, 1, 3, 1, 1]`;
- obs dims: `[236, 143, 143, 199, 199, 143, 199, 143, 143, 199, 143, 199, 143, 143, 255, 143, 143]`.

Isto e o esperado: buildings com charger tem mais features, e `Building_15` tem dois chargers.

## Escala

A escala tambem parece util para treino:

- SOC e required SOC entram como fracoes 0-1;
- ratios e flags ficam 0-1;
- `hours_until_*` vira uma escala normalizada a 24h;
- energia e potencia usam ratios contra capacidade/power bounds;
- community/headroom/power tambem entram em ratios/minmax quando aplicavel.

Nota importante: no manifest os `environment.encoders` aparecem como `NoNormalization`, mas isso nao e bug neste modo. Com `entity_encoding.enabled=true` e `serving_observation_names=encoded`, o `EntityAdapter` ja emite o vetor codificado para o modelo. Os encoders do wrapper ficam como pass-through.

## Community Signals

O perfil nao esta a olhar so para custo/EV. Todos os agentes recebem sinais comunitarios:

- `community_net_power_kw`;
- `community_import_power_kw`;
- `community_export_power_kw`;
- `community_pv_power_kw`;
- `community_bess_power_kw`;
- `community_ev_power_kw`;
- `community_building_headroom_kw`;
- `community_building_export_headroom_kw`;
- `community_phase_headroom_kw`;
- `community_phase_export_headroom_kw`;
- `community_flexible_charge_power_capacity_kw`;
- `community_flexible_discharge_power_capacity_kw`;
- previsoes comunitarias quando o profile inclui forecasts.

Isto da informacao suficiente para objetivos de pico, import/export, autoconsumo e uso de flexibilidade. O que falta e garantir que o score/reward realmente pesa esses objetivos e que nao se escolhe o "melhor" run so por custo.

## Falhas Provaveis

1. `connected_ev_soc_deficit` era one-sided. Quando o EV ja passava o target, o actor nao recebia um `connected_ev_soc_surplus` explicito nem um erro assinado `required_soc - soc`. Corrigido localmente.
2. `min_required_action_normalized` ensina o minimo para nao falhar, mas nao ensina o maximo util para nao passar o target. Foi adicionada `max_charge_to_required_soc_action_normalized` para esse papel.
3. A Wave 4 continuou com `initial_exploration_strategy: uniform_full_range`, `warm_start_policy: null` e `actor_behavior_cloning_weight: 0.0`. O replay arranca com muita carga positiva/V2G aleatoria.
4. O reward ja penaliza servico EV, mas o comportamento V46/V50/V52 quase nao mudou. Isto sugere que reward shaping isolado tem pouco leverage neste setup de 1 episodio.
5. O problema empirico bate com isto: no melhor run inspecionado, a falha principal foi over-service. O deficit medio era baixo, mas o surplus medio era alto.

## Sobre Action Guard

Concordo que um hard action guard como solucao principal e perigoso. Se o guard corta a action final, a rede pode aprender uma realidade falsa: o critic ve consequencias do guard, mas o actor nao aprende naturalmente a escolher a action correta.

Eu usaria action guard so em tres casos:

- como diagnostico: confirmar quanto melhora se bloquearmos carga acima do target;
- como teacher/BC: gerar targets de acao seguros com RBC/guard e treinar o actor a imitar;
- como safety layer final em deployment, se for requisito operacional.

Para treino principal, prefiro:

1. features explicitas de surplus/erro assinado/acao maxima ate target;
2. warm-start ou BC de `RBCBasicPolicy`/`RBCSmartPolicy`;
3. exploracao centrada no RBC/noop, nao uniform full-range;
4. reward multi-objetivo com EV, custo, pico, import/export, autoconsumo, emissoes, throughput, V2G e deferrable.

## Recomendacao Wave 5

Nao lancar mais sweep cego. A proxima wave deve testar aprendizagem guiada:

| Variante | Objetivo |
|---|---|
| `MADDPG + v52 + RBCBasic warm-start + EV BC` | Aprender servico EV sem hard guard |
| `MATD3 + v52 + RBCBasic warm-start + EV BC` | Mesmo teste no algoritmo mais estavel da Wave 4 |
| `MADDPG + v52 + RBCSmart/RBCCommunity teacher` | Ver se melhora autoconsumo/picos sem perder EV |
| `MATD3 + features surplus/action-to-target` | Testar o novo encoding EV sem hard action guard |

Gate de decisao:

- EV min feasible perto de `1.0`;
- EV within tolerance acima de `0.10` primeiro, depois perseguir `0.445` da `RBCBasicPolicy`;
- deferrable service ratio >= `0.95`;
- V2G abaixo dos `5.8 MWh` da Wave 4;
- import kWh, peak ratios, ramping e self-consumption avaliados junto com custo.
