# Auditoria do evento EV estranho na run 0.6.5

Run auditada:

- `runs/benchmarks/local065_baseline_ev_feasibility_15s`
- dataset: `citylearn_three_phase_electrical_service_demo_15s_parquet`
- foco: `NormalNoBatteryPolicy`, mas o mesmo KPI apareceu em `Normal`, `RBCBasic` e `RBCSmart`.

## KPI observado

No agregado do distrito:

- `ev_departure_count = 7`
- `ev_departure_min_acceptable_rate = 0.7142857142857143`
- `ev_departure_min_acceptable_feasible_rate = 0.8333333333333334`
- `ev_departure_min_acceptable_infeasible_count = 1`

As duas falhas brutas estao concentradas no `Building_15`:

- `building_ev_events_departure_count = 2`
- `building_ev_events_departure_min_acceptable_count = 0`
- `building_ev_events_departure_min_acceptable_feasible_count = 1`
- `building_ev_events_departure_min_acceptable_infeasible_count = 1`

## Eventos do Building 15

`charger_15_1`:

- EV: `Electric_Vehicle_5`
- ligado: step `0..2159`, `00:00:00..08:59:45`
- `EV Departure Time`: comeca em `1920` e chega a `0` no step `1920`, mas o EV so desliga no step `2160`
- SOC: `0.15 -> 0.44`
- target: `0.80`, minimo aceitavel com tolerancia 5%: `0.75`
- energia carregada no segmento: `31.36 kWh`

`charger_15_2`:

- EV: `Electric_Vehicle_6`
- ligado: step `0..2399`, `00:00:00..09:59:45`
- `EV Departure Time`: comeca em `2160` e chega a `0` no step `2160`, mas o EV so desliga no step `2400`
- SOC: `0.26 -> 0.47`
- target: `0.82`, minimo aceitavel com tolerancia 5%: `0.77`
- energia carregada no segmento: `24.67 kWh`

## O que esta estranho

O baseline `NormalNoBatteryPolicy` devia carregar imediatamente ate o EV estar cheio. A logica base faz isso:

- `NormalPolicy._compute_ev_action(...)` devolve `ev_normal_charge_rate = 1.0` enquanto o EV esta ligado e tem gap de SOC.
- `NormalNoBatteryPolicy` so desativa storage; nao devia enfraquecer o carregamento EV.

Mas as acoes reais do Building 15 alternam entre carga cheia no limite da fase e quase zero:

- `charger_15_1`: step 0 carrega `0.029166 kWh` (7 kW durante 15 s), step 1 carrega quase `0`, step 2 volta a `0.029166 kWh`, etc.
- `charger_15_2`: step 0 carrega `0.020833 kWh` (5 kW durante 15 s), step 1 carrega quase `0`, step 2 volta a `0.020833 kWh`, etc.

Isto reduz a potencia media efetiva para aproximadamente metade do que a policy pretendia.

## Causa provavel no Algorithms

A causa mais provavel esta em `RuleBasedPolicy._apply_ev_dynamic_headroom_limit(...)`.

Essa funcao usa:

- `charging_building_headroom_kw`
- `charging_phase_L1_headroom_kw`
- `charging_phase_L2_headroom_kw`
- `charging_phase_L3_headroom_kw`

para limitar a acao EV antes de a enviar ao simulador.

O problema e semantico: estes headrooms parecem representar margem residual depois da carga aplicada no step anterior. A acao do charger, no entanto, nao e delta de potencia adicional; e comando total de carga/descarga para o step. Quando o EV carregou no limite da fase no step anterior, o headroom residual seguinte fica ~0, e a policy corta a acao seguinte para ~0. No step seguinte volta a haver headroom, e a policy volta a carregar. Daqui nasce a oscilacao ON/OFF.

Para policy e MADDPG isto e perigoso: se o agente aprender a usar headroom como capacidade total disponivel, pode aprender um comportamento artificialmente intermitente.

## Questao adicional no dataset/simulador

Ha tambem uma segunda ambiguidade:

- `electric_vehicle_departure_time` chega a `0` antes do EV desligar fisicamente.
- O KPI de departure conta o evento quando o charger passa de conectado para nao conectado.

No Building 15:

- `charger_15_1`: deadline observado no step `1920`, desligamento no step `2160`.
- `charger_15_2`: deadline observado no step `2160`, desligamento no step `2400`.

Isto nao causa diretamente a subcarga, mas pode confundir policies/reward: a observacao diz que o prazo chegou a zero, mas o evento KPI so acontece uma hora depois.

## Fix aplicado no Algorithms

Foi aplicado um fix estreito em `RuleBasedPolicy._apply_ev_dynamic_headroom_limit(...)`:

- o headroom continua a limitar o comando EV;
- mas, quando existe observacao `applied_power_kw`/`commanded_power_kw` do proprio charger, a policy soma essa potencia atual ao headroom residual;
- isto evita tratar a acao EV como delta quando ela e, na pratica, comando total do charger.

Teste unitario adicionado:

- `test_rbc_ev_headroom_clip_allows_existing_charger_power`

Validacao:

- `pytest -q`: `157 passed`
- run: `runs/benchmarks/local065_ev_event_fix_15s`

Resultado no `NormalNoBatteryPolicy` depois do fix:

- `charger_15_1`: energia no primeiro segmento subiu de `31.36 kWh` para `62.70 kWh`; steps positivos `1213/2160 -> 2160/2160`; SOC final `0.44 -> 0.74`.
- `charger_15_2`: energia no primeiro segmento subiu de `24.67 kWh` para `49.33 kWh`; steps positivos `1333/2400 -> 2400/2400`; SOC final `0.47 -> 0.68`.
- `ev_departure_soc_deficit_mean`: `0.1010 -> 0.0294`.
- `ev_departure_shortfall_beyond_tolerance_mean`: `0.0868 -> 0.0152`.

O KPI `ev_departure_min_acceptable_rate` manteve-se `0.7142857`, porque o Building 15 ainda fica abaixo do minimo aceitavel nos dois eventos. A diferenca agora e que a falha ja nao vem da oscilacao artificial ON/OFF da policy; vem do limite fisico/temporal observado para aqueles chargers/fases, ou de uma pequena diferenca entre o instante da acao final e o instante em que o KPI le o SOC.

## Recomendacao restante

Do lado Algorithms:

1. Repetir todos os baselines, nao so `NormalNoBattery`, para confirmar que agora se diferenciam melhor.
2. Auditar especificamente se o ultimo comando antes do disconnect entra no SOC usado pelo KPI ou so no proximo step.
3. Decidir se `NormalNoBattery` deve tentar target 100% ou apenas target do utilizador; hoje usa `max(required_soc, ev_normal_target_soc)`.

Do lado simulador/dataset:

1. Alinhar ou documentar a diferenca entre `departure_time == 0` e o instante real de desligamento/KPI.
2. Confirmar se a feasibility dos EV departures deve considerar limites de fase/electrical service, nao so capacidade nominal do charger.
