# Auditoria dos Baselines RBC

## Estado do `RuleBasedPolicy` legacy

O `RuleBasedPolicy` e um baseline heuristico, sem treino, que recebe observacoes
raw do wrapper (`use_raw_observations = True`). Isto e intencional para manter a
politica legivel, mas significa que ele nao avalia o mesmo vetor encoded que o
MADDPG.

No modo `entity`, o agente deve ler primeiro observacoes namespaced do charger:

- `charger::<building>/<charger>::connected_state`
- `charger::<building>/<charger>::connected_ev_soc`
- `charger::<building>/<charger>::connected_ev_required_soc_departure`
- `charger::<building>/<charger>::hours_until_departure`
- `charger::<building>/<charger>::connected_ev_departure_time_step`

Os aliases legacy (`electric_vehicle_soc`, `electric_vehicle_charger_state`,
etc.) ficam apenas como fallback para compatibilidade.

## Regras do `RuleBasedPolicy`

### EV

O RBC carrega um EV quando:

- o charger esta conectado;
- o SOC atual esta abaixo do SOC requerido na saida;
- ha energia necessaria acima de `energy_epsilon`.

A intensidade de carga e baseada em:

- energia necessaria ate ao SOC requerido;
- tempo ate a saida;
- potencia maxima do charger;
- PV local atual (`pv_power_kw`, com `solar_generation` apenas como fallback
  legacy);
- janela de emergencia (`emergency_hours`);
- flexibilidade (`electric_vehicle_is_flexible` e `non_flexible_chargers`).

Ele nao usa previsoes de preco/PV, nem optimizacao global de custo. Tambem nao
faz V2G: mesmo quando o action space permite negativo, a politica EV atual nunca
devolve valores negativos.

### Storage/BESS legacy

O `RuleBasedPolicy` legacy nao controla storage. A acao `electrical_storage`
fica sempre `0.0`.

Isto torna o RBC simples e relativamente justo como baseline EV, mas fraco como
baseline para comparar gestao completa de comunidade. Um MADDPG que aprenda BESS
pode ganhar ao RBC por uma capacidade que o RBC nem tenta usar.

### Deferrables

O RBC so inicia um deferrable quando:

- esta pendente;
- nao esta a correr;
- pode iniciar;
- ainda nao falhou deadline;
- e passa pelo menos uma condicao de urgencia/slack/prioridade.

Nao tenta alinhar deferrables com PV, preco, headroom ou carga da comunidade.

## Leitura sobre justica do `RuleBasedPolicy`

O RBC e realista para EV se assumirmos que o EV comunica SOC atual, SOC alvo e
tempo ate departure. Isto e uma premissa comum em smart charging.

Nao parece ser "oracle" no sentido forte: nao usa informacao futura escondida de
precos/PV/carga. Mas usa dados de compromisso do EV (`required_soc_departure` e
departure), que devem tambem estar disponiveis nas observacoes do MADDPG para a
comparacao ser justa.

O RBC e EV-focused. E fraco ou incompleto para:

- storage;
- deferrables com custo/PV/headroom;
- coordenacao global entre casas;
- V2G;
- minimizacao de import/export com previsoes.

## Familia nova de baselines

Foram adicionadas policies separadas para comparar contra MADDPG sem misturar
cenarios:

- `RandomPolicy`: amostra todas as dimensoes de acao dentro dos bounds do
  ambiente. Controla storage, EV, V2G quando o bound permite negativo, e
  deferrables.
- `NormalPolicy`: representa um dia normal sem otimizacao. EV carrega logo
  quando chega ate ao target normal, por defeito `100%`, respeitando pelo menos
  o SOC requerido de saida, sem V2G. Deferrables arrancam no primeiro slot
  valido. BESS usa autoconsumo simples: carrega com excedente PV local e
  descarrega para cobrir import local.
- `NormalNoBatteryPolicy`: igual ao `NormalPolicy` para EV e deferrables, mas
  deixa BESS sempre em `0.0`. Serve para medir o comportamento normal sem
  qualquer controlo de bateria de casa.
- `RBCBasicPolicy`: controlo inteligente minimo. Usa urgencia e preco
  atual/previsoes curtas `1..3` para storage e deferrables, mas preserva EV
  service com target forte antes de tentar poupar custo. Nao usa V2G.
- `RBCSmartPolicy`: heuristica mais forte. Preserva EV service, pode usar V2G
  conservador quando `allow_v2g=true`, e usa PV/import/headroom apenas quando
  isso nao piora a baseline. No dataset 15s, o default seguro e nao ciclar BESS,
  porque `grid_export_price=0.0` e os testes mostraram que o ciclo simples de
  storage piorava custo/import sem melhorar EV.

Todas estas policies herdam a infraestrutura do `RuleBasedPolicy`, recebem raw
observations e exportam artefactos `rule_based` compatíveis com o manifest.

## Garantias operacionais

### Tamanho do timestep

As policies leem `seconds_per_time_step` no `attach_environment` e guardam
`step_hours`. Isto cobre os dois datasets usados atualmente:

- dataset 15s parquet: `seconds_per_time_step=15`;
- dataset 2022 all-plus-EVs: `seconds_per_time_step=3600`.

Quando a observacao EV vem em `*_departure_time_step`, o agente converte para
horas com `value * step_hours`. Quando a observacao ja vem como
`hours_until_departure`, usa esse valor diretamente. Os deferrables usam
principalmente `can_start`, `deadline_missed`, `slack_ratio` e `urgency_ratio`,
que ja sao fornecidos pelo simulador no referencial correto do dataset.

### Limites de fase

Os baselines aplicam clipping dinamico de EV e BESS quando o dataset fornece
`phase_connection` no asset. Para carregamento usam:

- `charging_building_headroom_kw`;
- `charging_phase_L1_headroom_kw`, `charging_phase_L2_headroom_kw`,
  `charging_phase_L3_headroom_kw`, conforme a fase do charger.

Para V2G usam os equivalentes de export:

- `charging_building_export_headroom_kw`;
- `charging_phase_L*_export_headroom_kw`.

Isto evita que a policy peça mais potencia controlada do que o headroom
local/fase permite. Se o dataset nao tiver `phase_connection`, a policy nao
inventa restricoes de fase; nesse caso a garantia física final continua a ser do
simulador e da reward/penalizacao de violacoes.

### SOC de saida EV

O `RBCSmartPolicy` trata o SOC alvo como prioridade forte. A taxa base de carga
usa:

- SOC atual;
- `connected_ev_required_soc_departure`;
- capacidade da bateria do EV ligado;
- potencia maxima do charger;
- horas ate departure, ja corrigidas pelo timestep.

Mesmo em preco alto ou rede sob stress, o smart RBC mantem pelo menos a taxa
necessaria para chegar ao SOC requerido, se isso for fisicamente possivel dentro
do power bound do charger e do headroom disponivel. V2G so descarrega quando
`SOC > required_soc + ev_v2g_reserve_soc`, para nao consumir a reserva necessaria
para a saida.

## Versoes recomendadas para comparacao

Usar a seguinte escada de comparacao:

- `RandomPolicy` como lower bound estocastico e teste de bounds;
- `NormalNoBatteryPolicy` como comportamento normal sem controlo de bateria;
- `NormalPolicy` como comportamento normal sem otimizacao;
- `RBCBasicPolicy` como RBC simples e realista;
- `RBCSmartPolicy` como RBC forte para a comparacao principal;
- `MADDPG` com `maddpg_v1`;
- `MADDPG` com `maddpg_v2_compact`.

## Auditorias atuais

Auditorias historicas geradas com `softcpsrecsimulator==0.4.3` instalado via PyPI:

- `runs/pipeline_contracts/rbc_static_15s_parquet_pypi043_entity_charger_fix`
- `runs/action_audits/rbc_static_15s_parquet_pypi043_entity_charger_fix`
- `runs/action_audits/rbc_static_15s_parquet_pypi043_entity_charger_fix_day1`
- `runs/pipeline_contracts/random_static_15s_parquet_pypi043`
- `runs/action_audits/random_static_15s_parquet_pypi043`
- `runs/pipeline_contracts/normal_static_15s_parquet_pypi043`
- `runs/action_audits/normal_static_15s_parquet_pypi043`
- `runs/pipeline_contracts/rbc_basic_static_15s_parquet_pypi043`
- `runs/action_audits/rbc_basic_static_15s_parquet_pypi043`
- `runs/pipeline_contracts/rbc_smart_static_15s_parquet_pypi043`
- `runs/action_audits/rbc_smart_static_15s_parquet_pypi043`
- `runs/pipeline_contracts/rbc_2022_all_plus_evs_hourly_pypi043`
- `runs/action_audits/rbc_2022_all_plus_evs_hourly_pypi043`
- `runs/pipeline_contracts/random_2022_all_plus_evs_pypi043`
- `runs/action_audits/random_2022_all_plus_evs_pypi043`
- `runs/pipeline_contracts/normal_2022_all_plus_evs_pypi043`
- `runs/action_audits/normal_2022_all_plus_evs_pypi043`
- `runs/pipeline_contracts/rbc_basic_2022_all_plus_evs_pypi043`
- `runs/action_audits/rbc_basic_2022_all_plus_evs_pypi043`
- `runs/pipeline_contracts/rbc_smart_2022_all_plus_evs_pypi043`
- `runs/action_audits/rbc_smart_2022_all_plus_evs_pypi043`

Validacao historica com `softcpsrecsimulator==0.5.0` instalado via PyPI:

- `runs/training_contracts/pypi050_maddpg_profiles`
- `runs/action_audits/pypi050_dynamic_topology_rule_based`
- `runs/action_audits/pypi050_dynamic_assets_only_rule_based`

Nota: o requisito alvo do repo foi atualizado para `softcpsrecsimulator==0.5.1`
instalado via PyPI. Os contratos MADDPG foram revalidados em
`runs/training_contracts/pypi051_maddpg_profiles`; estes action audits de
baselines ainda sao os runs historicos `0.5.0`.

Em ambos os datasets, o contrato sobe com:

- `17` agentes;
- `26` acoes;
- `8` acoes EV;
- `bounds_issues: []`.

No audit de um dia do `RuleBasedPolicy` legacy no dataset 15s, todas as acoes
de storage ficaram a `0.0`, como esperado para essa policy. As acoes EV ficaram
positivas apenas quando havia EV conectado/necessidade de carga. O Building 15
tem dois chargers e agora as duas acoes sao avaliadas com features namespaced do
respetivo charger, nao com o mesmo alias legacy.

Nos audits dos baselines novos nos datasets 15s e 2022 all-plus-EVs, as cinco
policies sobem com o mesmo contrato (`17` agentes, `26` acoes, `8` acoes EV) e
`bounds_issues: []`.

## Auditoria de papel dos baselines

Foi gerada uma auditoria de rollout para confirmar que cada policy tem o papel
certo antes de ser usada contra MADDPG:

- `runs/policy_role_audits/random_15s`
- `runs/policy_role_audits/normal_no_battery_15s`
- `runs/policy_role_audits/normal_15s`
- `runs/policy_role_audits/rbc_basic_15s`
- `runs/policy_role_audits/rbc_smart_15s`
- `runs/policy_role_audits/random_2022`
- `runs/policy_role_audits/normal_no_battery_2022`
- `runs/policy_role_audits/normal_2022`
- `runs/policy_role_audits/rbc_basic_2022`
- `runs/policy_role_audits/rbc_smart_2022`

Conclusao:

- `RandomPolicy` e apenas lower bound/sanidade. Amostra todas as acoes dentro
  dos bounds, incluindo storage, EV/V2G quando permitido e deferrables. Nao deve
  ser interpretada como comportamento humano.
- `NormalPolicy` e o comportamento "dia normal": EV sem V2G, carga imediata ate
  ao target normal, deferrables no primeiro slot valido e BESS de autoconsumo
  local simples. Nao usa preco, forecast ou coordenacao global.
- `NormalNoBatteryPolicy` e a variante mais passiva do "dia normal": EV e
  deferrables iguais ao normal, mas todas as acoes de BESS ficam a `0.0`.
- `RBCBasicPolicy` fica no meio: usa preco atual/previsoes curtas e urgencia
  `1..3` para EV, storage e deferrables, mas nao usa PV/headroom como criterio
  economico e nao faz V2G.
- `RBCSmartPolicy` e o RBC forte. Usa apenas observacoes disponiveis ao agente:
  preco atual/previsoes observadas, PV/import/headroom atuais, estado de EV,
  deadlines/urgencia de deferrables e SOC/capacidade local. Nao le carga/PV
  futuros escondidos, nem KPIs finais, nem estado interno do simulador fora do
  contrato de observacao/asset usado para bounds e fases.

Numeros agregados dos rollouts:

| policy | dataset | storage pos/neg | EV pos/neg | deferrable pos |
|---|---|---:|---:|---:|
| `RandomPolicy` | 15s | `0.498/0.496` | `0.495/0.490` | `1.000` |
| `NormalNoBatteryPolicy` | 15s | `0.000/0.000` | `0.818/0.000` | `0.000` |
| `NormalPolicy` | 15s | `0.125/0.069` | `0.818/0.000` | `0.000` |
| `RBCBasicPolicy` | 15s | `0.092/0.074` | `0.473/0.000` | `0.000` |
| `RBCSmartPolicy` | 15s | `0.243/0.174` | `0.862/0.000` | `0.000` |
| `RandomPolicy` | 2022 | `0.497/0.503` | `0.500/0.500` | `1.000` |
| `NormalNoBatteryPolicy` | 2022 | `0.000/0.000` | `0.433/0.000` | `0.042` |
| `NormalPolicy` | 2022 | `0.144/0.114` | `0.433/0.000` | `0.042` |
| `RBCBasicPolicy` | 2022 | `0.107/0.114` | `0.133/0.000` | `0.042` |
| `RBCSmartPolicy` | 2022 | `0.301/0.308` | `0.468/0.001` | `0.042` |

Nota: no dataset 15s, a acao deferrable aparece muito pouco porque a janela
observada quase nao expõe passos `pending + can_start` para esse ativo. Isto
nao e problema da policy; e uma caracteristica da janela/dataset auditado.

## Revisao EV Service Pos-Fase 6B

A matriz de 1 dia da Fase 6B mostrou que `RBCBasicPolicy` e `RBCSmartPolicy`
eram bons baselines economicos, mas fracos como baselines de servico EV: podiam
adiar carga ate ao ponto de falhar SOC de saida.

Foi corrigido:

- ambos calculam uma taxa minima de carga ate a partida com base em deficit SOC,
  capacidade EV, potencia do charger e horas ate departure;
- essa taxa minima e sempre preservada antes de aplicar preco/PV/headroom;
- existe margem configuravel (`ev_service_margin_rate`) e buffer de deadline
  (`ev_deadline_buffer_hours`) para evitar falhas por discretizacao/headroom;
- `RBCSmartPolicy` so usa V2G quando o EV ja nao tem deficit e ainda tem margem
  SOC/tempo suficiente.

Isto torna o RBCSmart mais justo como baseline forte: ele continua sem usar
informacao oracle, mas ja nao deve ganhar custo por deixar EVs mal servidos.

Retune posterior `0.5.1`:

- `RBCBasicPolicy`: `ev_service_target_soc=0.95`, floor EV moderado. Continua a
  ser baseline intermédio economico.
- `RBCSmartPolicy`: `ev_service_target_soc=1.0`, floor EV forte. Passa a ser o
  baseline forte de servico.

No full-day 15s `runs/benchmarks/phase6c_15s_rbc_service_retune_seed123`,
`RBCSmartPolicy` atingiu o mesmo EV success/deficit que `NormalPolicy`
(`1/7`, deficit medio `0.333`) com custo `262.018`.

## Revisao 0.5.2 Com KPI EV Minimo Aceitavel

Com `softcpsrecsimulator==0.5.2`, a metrica primaria EV passou a ser
`ev_departure_min_acceptable_rate`.

Runs:

- matriz original: `runs/benchmarks/phase6d_052_service_kpis_15s`;
- `RBCBasic` retuned: `runs/benchmarks/phase6d_052_rbc_basic_retuned_15s`;
- `RBCSmart` safe-storage:
  `runs/benchmarks/phase6d_052_rbc_smart_safe_storage_15s`.

Resultado 15s, 1 dia, seed `123`:

| policy | cost eur | EV min acceptable | EV strict | EV deficit mean | storage pos/neg |
|---|---:|---:|---:|---:|---:|
| NormalNoBattery | 250.353 | 0.143 | 0.143 | 0.333 | 0.000/0.000 |
| Normal | 255.665 | 0.143 | 0.143 | 0.333 | 0.128/0.077 |
| RBCBasic retuned | 254.145 | 0.143 | 0.143 | 0.333 | 0.087/0.043 |
| RBCSmart safe-storage | 250.328 | 0.143 | 0.143 | 0.333 | 0.000/0.000 |

Conclusao:

- `RBCBasicPolicy` ja nao sacrifica EV service face a `NormalPolicy`;
- `RBCSmartPolicy` ficou mais seguro no dataset 15s ao evitar ciclo de BESS que
  empiricamente piorava custo/import;
- ainda falta repetir esta triagem no dataset 2022 antes de chamar a escada de
  baselines final.

## Revisao 0.6.4 Apos Fixes Fisicos

O simulador `0.6.4` corrigiu a fisica sub-hourly de EV/BESS/deferrables.
Isto muda a interpretacao dos resultados anteriores: o mau EV service de
`0.5.2` nao era apenas estrategia RBC; havia perda artificial de SOC EV em 15s.

Validado primeiro contra `/home/tiago/dev/Simulator` instalado em editable e
depois com a wheel PyPI:

- `softcpsrecsimulator==0.6.4`;
- `pytest -q`: `155 passed`;
- contrato MADDPG regenerado em
  `runs/training_contracts/local064_maddpg_profiles`;
- benchmark baseline 15s em
  `runs/benchmarks/local064_baseline_ev_check_15s`.

Resultado 15s, 1 dia, seed `123`:

| policy | cost eur | EV min acceptable | EV strict | EV deficit mean | EV shortfall tol mean | storage pos/neg |
|---|---:|---:|---:|---:|---:|---:|
| NormalNoBattery | 105.998 | 0.714 | 0.714 | 0.101 | 0.087 | 0.000/0.000 |
| Normal | 106.703 | 0.714 | 0.714 | 0.101 | 0.087 | 0.125/0.100 |
| RBCBasic | 106.660 | 0.714 | 0.714 | 0.101 | 0.087 | 0.084/0.043 |
| RBCSmart | 105.973 | 0.714 | 0.714 | 0.101 | 0.087 | 0.000/0.000 |

Leitura:

- `NormalNoBatteryPolicy` agora comporta-se como esperado para "dia normal":
  carregar EV logo que ligado melhora muito o service face ao resultado antigo.
- `RBCBasicPolicy` e `RBCSmartPolicy` ja nao ganham custo a custa de EV service
  nesta janela.
- `RBCSmartPolicy` continua essencialmente safe-storage no dataset 15s; isso e
  aceitavel enquanto storage ciclado nao demonstrar beneficio.
- Os 2/7 departures que ainda falham podem ser infeasiveis por potencia/janela
  ou limites fisicos. Antes de afinar RBC outra vez, convem gerar auditoria por
  departure: SOC inicial, SOC alvo, horas ligado, kWh carregado, potencia maxima
  possivel e headroom.
- A baseline nativa do simulador `BusinessAsUsualAgent` aparece agora nos KPIs
  exportados como referencia BAU, mas a comparacao principal deste repo ainda
  usa as nossas policies `NormalNoBattery`, `Normal`, `RBCBasic` e `RBCSmart`.

## Revisao 0.6.5 Com Feasibility EV

O simulador `0.6.5` adicionou KPIs de feasibility para departures EV. Isto
permite separar:

- experiencia bruta do utilizador/cenario:
  `ev_departure_min_acceptable_rate`;
- qualidade justa do controlador em eventos fisicamente atingiveis:
  `ev_departure_min_acceptable_feasible_rate`;
- diagnostico de schedule/dados:
  `ev_departure_min_acceptable_infeasible_count`,
  `ev_departure_target_infeasible_count` e
  `ev_departure_within_tolerance_infeasible_count`.

Implicacao para a auditoria RBC:

- se os `2/7` departures que falhavam em `0.6.4` forem marcados como
  infeasible, entao as policies estavam essencialmente corretas e o KPI raw
  deve ser lido como limitacao do cenario;
- se houver falhas no `ev_departure_min_acceptable_feasible_rate`, entao ainda
  ha problema real na estrategia RBC ou na prioridade EV;
- a partir daqui, o gate principal para comparar MADDPG contra RBCs deve ser o
  feasible rate, mantendo o raw rate no relatório para experiencia de utilizador.

Triagem 15s local `0.6.5`:

- run: `runs/benchmarks/local065_baseline_ev_feasibility_15s`;
- 4/4 baselines completas, 0 falhas.

| policy | cost eur | EV feasible min acceptable | EV raw min acceptable | departures | min infeasible | storage pos/neg |
|---|---:|---:|---:|---:|---:|---:|
| NormalNoBattery | 105.998 | 0.833 | 0.714 | 7 | 1 | 0.000/0.000 |
| Normal | 106.703 | 0.833 | 0.714 | 7 | 1 | 0.125/0.100 |
| RBCBasic | 106.660 | 0.833 | 0.714 | 7 | 1 | 0.084/0.043 |
| RBCSmart | 105.973 | 0.833 | 0.714 | 7 | 1 | 0.000/0.000 |

Conclusao temporaria:

- `1/7` departure e infeasible para minimo aceitavel;
- `1/6` departure feasible ainda falha em todas as policies;
- isto aponta menos para "RBCSmart esta errado" e mais para auditar o evento
  especifico: chegada, target, janela, potencia, fase/headroom e se o
  controlador inicia carga cedo o suficiente.

## Revisao 0.6.6 / Fase 6G - Storage e RBCSmart 2022

Depois dos fixes do simulador `0.6.6`, a matriz completa de baselines foi
repetida nos dois datasets principais:

- benchmark: `runs/benchmarks/phase6g_baseline_storage_check`;
- auditoria storage: `runs/storage_audits/phase6g_baseline_storage_check`.

Resumo de custo:

| dataset | policy | cost eur | EV feasible min | EV feasible tolerance | leitura |
|---|---|---:|---:|---:|---|
| 15s | NormalNoBattery | 112.505 | 1.000 | 0.000 | dia normal sem bateria |
| 15s | Normal | 114.035 | 1.000 | 0.000 | bateria piora por ciclos pouco valiosos |
| 15s | RBCBasic | 99.453 | 1.000 | 1.000 | baseline medio forte |
| 15s | RBCSmart | 97.105 | 1.000 | 1.000 | baseline forte atual |
| 2022 | NormalNoBattery | 4729.065 | 1.000 | 0.049 | dia normal sem bateria |
| 2022 | Normal | 4825.920 | 1.000 | 0.049 | bateria piora por eficiencia/ciclos |
| 2022 | RBCBasic | 3835.326 | 1.000 | 0.436 | baseline medio forte |
| 2022 | RBCSmart antigo | 3894.033 | 1.000 | 0.436 | demasiado conservador/mal calibrado |

Conclusoes:

- nao ha evidencia de consumo parado da bateria: quando a policy deixa storage
  idle, o SOC nao drena de forma relevante;
- `Normal` com bateria ser pior que `NormalNoBattery` e plausivel, porque a
  estrategia mecanica paga perdas de ciclo sem olhar para preco/PV/peak;
- `RBCBasic` e `RBCSmart` demonstram que storage pode ajudar quando a decisao e
  minimamente contextual;
- no 15s, o `RBCSmart` ja era o baseline forte;
- no 2022, o `RBCSmart` antigo carregava a bateria a preco medio pior que
  `RBCBasic`, logo precisava de retune.

Alteracoes aplicadas:

- `RBCSmartPolicy` passou a permitir carga por preco ate
  `storage_price_charge_soc_ceiling`, sem ficar limitada por
  `storage_target_soc`;
- a rama de carga por PV ja nao bloqueia carga por preco quando `pv_charge_rate`
  e zero ou inferior a `price_charge_rate`;
- o clipping fisico continua depois da decisao da policy, preservando limites
  de SOC, headroom e fases;
- `rbc_smart_local.yaml` foi alinhado para floors de descarga `0.30`, mantendo
  o resultado forte no 15s: custo `97.105`, EV feasible min `1.0`, EV feasible
  dentro da tolerancia `1.0` e zero violacoes;
- `rbc_smart_2022_all_plus_evs_local.yaml` foi retuned para
  `price_charge_rate = 0.15`, `storage_price_charge_soc_ceiling = 0.85`,
  `storage_price_discharge_soc_floor = 0.30` e
  `storage_peak_discharge_soc_floor = 0.30`.
- `utils/config_schema.py` passou a preservar estes knobs de storage e
  `deferrable_safety_margin_steps`; antes, alguns valores existiam no template
  mas eram descartados no config resolvido.

Validacao do template 2022 atualizado:

- run: `runs/benchmarks/phase6g_rbcsmart_2022_template_verify_final`;
- auditoria storage:
  `runs/storage_audits/phase6g_rbcsmart_2022_template_verify_final`;
- custo: `3823.523`;
- melhor que `RBCBasic` (`3835.326`) e que o `RBCSmart` antigo (`3894.033`);
- `ev_departure_min_acceptable_feasible_rate = 1.0`;
- `ev_departure_within_tolerance_feasible_rate = 0.435816`;
- `electrical violation = 0.0`.
- storage: `5783.500 kWh` de carga, `4691.000 kWh` de descarga,
  `10474.500 kWh` de throughput, sem drift de SOC quando idle.

Estado para comparacao MADDPG:

- `Random` continua a ser apenas sanity/lower bound;
- `NormalNoBattery` representa dia normal sem controlo de bateria;
- `Normal` representa dia normal com bateria mecanica, nao uma estrategia
  otimizada;
- `RBCBasic` e o baseline intermedio;
- `RBCSmart` e o baseline forte por dataset: conservador no 15s, retuned no
  2022.
