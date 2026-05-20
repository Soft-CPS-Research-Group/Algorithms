# Fase 6D - Resultados 0.5.2 EV Service KPIs

Data: 2026-05-14.

Simulador: `softcpsrecsimulator==0.5.2` instalado localmente em editable a partir
de `/home/tiago/dev/Simulator`.

Dataset: `citylearn_three_phase_electrical_service_demo_15s_parquet`.

Janela: 1 dia, `5760` steps de 15 segundos, seed `123`.

## Objetivo

Repetir a triagem com os KPIs EV novos do simulador `0.5.2`:

- `ev_departure_min_acceptable_rate`: KPI primario de conforto EV.
- `ev_departure_success_rate`: cumprimento estrito.
- `ev_departure_within_tolerance_rate`: accuracy simetrica.
- `ev_departure_soc_deficit_mean`.
- `ev_departure_shortfall_beyond_tolerance_mean`.
- `ev_departure_soc_surplus_mean`.
- `ev_departure_soc_absolute_error_mean`.

## Runs

Matriz completa:

- `runs/benchmarks/phase6d_052_service_kpis_15s`

Reruns de retune:

- `runs/benchmarks/phase6d_052_rbc_basic_retuned_15s`
- `runs/benchmarks/phase6d_052_rbc_smart_retuned_15s`
- `runs/benchmarks/phase6d_052_rbc_smart_safe_storage_15s`

## Resultado Principal

| policy | cost eur | grid violation kWh | EV min acceptable | EV strict | EV deficit mean | EV shortfall tol mean | storage pos/neg |
|---|---:|---:|---:|---:|---:|---:|---:|
| NormalNoBattery | 250.353 | 0.000 | 0.143 | 0.143 | 0.333 | 0.290 | 0.000/0.000 |
| Normal | 255.665 | 0.000 | 0.143 | 0.143 | 0.333 | 0.290 | 0.128/0.077 |
| RBCBasic retuned | 254.145 | 0.000 | 0.143 | 0.143 | 0.333 | 0.290 | 0.087/0.043 |
| RBCSmart safe-storage | 250.328 | 0.000 | 0.143 | 0.143 | 0.333 | 0.290 | 0.000/0.000 |
| MADDPG anti_saturation_warm_rbc_smart | 126.517 | 0.000 | 0.000 | 0.000 | 0.791 | 0.741 | 0.345/0.514 |

## Conclusoes

- Os KPIs novos do simulador estao a ser exportados e agregados corretamente.
- `RBCBasicPolicy` anterior nao era aceitavel como baseline intermédio porque
  baixava custo a custa de EV service (`EV min acceptable = 0.0`).
- `RBCBasicPolicy` retuned passou a igualar `Normal` em EV service e ficou
  ligeiramente abaixo de `Normal` em custo.
- O `RBCSmartPolicy` antigo ciclava storage demais no dataset 15s e piorava
  custo/import/shape sem melhorar EV.
- `RBCSmartPolicy` safe-storage ficou como baseline forte mais justo nesta
  janela: preserva EV service, nao cria violacoes e evita ciclo de bateria que
  nao tem beneficio claro quando `grid_export_price=0.0`.
- A variante MADDPG testada continua a falhar EV service apesar de custo baixo.
  Isto reforca que custo baixo sozinho nao e sucesso; o gate principal deve
  incluir `ev_departure_min_acceptable_rate`.

## Decisao

Para comparacoes imediatas:

- usar `RBCBasicPolicy` retuned com service target forte;
- usar `RBCSmartPolicy` safe-storage como baseline forte 15s;
- nao usar os resultados antigos da Fase 6B/6C como conclusao final de EV,
  porque ainda usavam a metrica strict/alias antiga.

## Proximo Passo

Repetir a mesma triagem no dataset `2022 all-plus-EVs` e, se a ordem dos
baselines continuar plausivel, passar para testes MADDPG isolados:

1. diagnostico pequeno 0.5.2;
2. MADDPG com reward EV tolerance-aware;
3. comparacao multi-seed contra `RBCSmartPolicy` safe-storage.

## Revisao 0.6.4

Data: 2026-05-16.

Simulador: `softcpsrecsimulator==0.6.4`, validado primeiro localmente em
editable a partir de `/home/tiago/dev/Simulator` e depois instalado via PyPI.

Contexto: o simulador corrigiu fisica sub-hourly EV/BESS/deferrables depois da
triagem `0.5.2`. Por isso, os numeros acima continuam uteis historicamente, mas
nao devem ser usados como conclusao final sobre qualidade dos RBCs.

Validacao de compatibilidade no repo Algorithms:

- `.venv/bin/python -m pytest -q`: `155 passed`;
- subset de integracao RBC/wrapper/entity/benchmark: `45 passed`;
- contrato MADDPG regenerado em
  `runs/training_contracts/local064_maddpg_profiles`;
- benchmark baseline 15s:
  `runs/benchmarks/local064_baseline_ev_check_15s`, 4/4 completed.

Resultado 15s, 1 dia, seed `123`:

| policy | cost eur | grid violation kWh | EV min acceptable | EV strict | EV deficit mean | EV shortfall tol mean | storage pos/neg |
|---|---:|---:|---:|---:|---:|---:|---:|
| NormalNoBattery | 105.998 | 0.000 | 0.714 | 0.714 | 0.101 | 0.087 | 0.000/0.000 |
| Normal | 106.703 | 0.000 | 0.714 | 0.714 | 0.101 | 0.087 | 0.125/0.100 |
| RBCBasic | 106.660 | 0.000 | 0.714 | 0.714 | 0.101 | 0.087 | 0.084/0.043 |
| RBCSmart | 105.973 | 0.000 | 0.714 | 0.714 | 0.101 | 0.087 | 0.000/0.000 |

Conclusoes novas:

- O fix do simulador resolveu a maior parte da distorcao EV: `NormalNoBattery`
  subiu de `0.143` para `0.714` em `ev_departure_min_acceptable_rate`.
- Os RBCs ja nao parecem "maus" por estrategia EV nesta janela; todos preservam
  o mesmo nivel de service EV.
- A ordem economica dos baselines 15s ficou mais plausivel: `RBCSmart`
  safe-storage e `NormalNoBattery` ficam melhores que `Normal`/`RBCBasic` quando
  storage ciclado nao traz beneficio claro.
- Ainda ha 2/7 departures abaixo do minimo aceitavel. Isto deve ser auditado
  por evento antes de mexer novamente na estrategia RBC.
- Os resultados MADDPG de `0.5.2` nao devem ser usados para decidir tuning
  longo; a comparacao MADDPG precisa de ser repetida com `0.6.4`.

## Revisao 0.6.5

Data: 2026-05-16.

Simulador: `softcpsrecsimulator==0.6.5`, instalado localmente em editable a
partir de `/home/tiago/dev/Simulator`. PyPI ainda esta em `0.6.4`, por isso o
`requirements.txt` permanece em `0.6.4` ate a release estar publicada.

Mudanca relevante: o simulador adicionou KPIs EV feasible-only para separar
falhas fisicamente impossiveis de falhas reais do controlador.

Novos KPIs rastreados pelo benchmark:

- `ev_departure_min_acceptable_feasible_rate`;
- `ev_departure_success_feasible_rate`;
- `ev_departure_within_tolerance_feasible_rate`;
- `ev_departure_count`;
- `ev_departure_target_infeasible_count`;
- `ev_departure_min_acceptable_infeasible_count`;
- `ev_departure_within_tolerance_infeasible_count`.

Decisao de leitura:

- `ev_departure_min_acceptable_feasible_rate` passa a ser o gate principal para
  comparar MADDPG contra RBCs, porque mede a qualidade do controlador apenas em
  departures fisicamente atingiveis;
- `ev_departure_min_acceptable_rate` continua a ser reportado como experiencia
  real do utilizador/cenario;
- os counts infeasible explicam se uma falha raw vem de schedule/potencia/SOC
  inicial impossivel, em vez de culpar a policy.

Os resultados numericos `0.6.4` acima devem ser repetidos com `0.6.5` antes de
decidir tuning MADDPG longo.

### Baselines 15s 0.6.5

Run:

- `runs/benchmarks/local065_baseline_ev_feasibility_15s`

Resultado 15s, 1 dia, seed `123`:

| policy | cost eur | grid violation kWh | EV feasible min acceptable | EV raw min acceptable | departures | min infeasible | EV deficit mean | EV shortfall tol mean | storage pos/neg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NormalNoBattery | 105.998 | 0.000 | 0.833 | 0.714 | 7 | 1 | 0.101 | 0.087 | 0.000/0.000 |
| Normal | 106.703 | 0.000 | 0.833 | 0.714 | 7 | 1 | 0.101 | 0.087 | 0.125/0.100 |
| RBCBasic | 106.660 | 0.000 | 0.833 | 0.714 | 7 | 1 | 0.101 | 0.087 | 0.084/0.043 |
| RBCSmart | 105.973 | 0.000 | 0.833 | 0.714 | 7 | 1 | 0.101 | 0.087 | 0.000/0.000 |

Leitura:

- dos `2/7` departures raw abaixo do minimo aceitavel, `1` e marcado como
  min-acceptable infeasible;
- ainda ha `1/6` departure feasible que falha em todos os baselines, portanto
  ha uma falha real a auditar por evento;
- a ordem dos baselines continua plausivel para esta janela: `RBCSmart` fica
  com melhor custo, zero violacoes e sem ciclar storage;
- antes de mexer na estrategia RBC, convem identificar o departure feasible que
  falha e perceber se falta prioridade/headroom/observacao ou se a policy
  chega tarde ao carregamento.
