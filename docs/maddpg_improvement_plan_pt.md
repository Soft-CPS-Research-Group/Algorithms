# Plano de Melhoria MADDPG

Este documento e o quadro de trabalho para melhorar o MADDPG sem voltar a
misturar responsabilidades entre simulador, wrapper, baselines e algoritmo.

Principio base: o MADDPG deve aprender a partir das observacoes, das acoes e da
reward. Nao devemos meter regras prescritivas no wrapper para forcar bons KPIs.

## Objetivo Inicial

O objetivo continua a ser ter um MADDPG multi-agente que:

- controle todos os recursos disponiveis por building, sem atalhos no wrapper;
- tenha um agente por casa/building, com observacoes e acoes corretas;
- aprenda a partir de observacoes, reward, replay, exploracao e arquitetura;
- cumpra servico EV/deferrables e limites fisicos antes de otimizar custo;
- melhore KPIs de custo, import, picos, uso de renovaveis e violacoes;
- seja comparado contra baselines justos: `Random`, `NormalNoBattery`,
  `Normal`, `RBCBasic`, `RBCSmart`;
- exporte artefactos utilizaveis no repo de inference.

Regra de responsabilidade:

- simulador define fisica, datasets, observacoes, acoes e KPIs;
- wrapper adapta contrato/encoding/export, sem ensinar comportamento;
- baselines sao referencias heuristicas, nao treino;
- MADDPG deve aprender comportamento por RL/MARL.

## Estado

- Data de referencia: 2026-05-17.
- Simulador PyPI fixado em `requirements.txt`: `softcpsrecsimulator==0.6.6`.
- Ambiente atual de trabalho: `.venv` com package PyPI;
  `citylearn.__version__ == 0.6.6`.
- Interface principal: `entity`.
- Topologia para MADDPG: `static`.
- Dataset 15s principal: `citylearn_three_phase_electrical_service_demo_15s_parquet`.
- Dataset horario principal: `citylearn_challenge_2022_phase_all_plus_evs`.
- Perfis MADDPG suportados: `maddpg_v1`, `maddpg_v2_compact`.
- Perfil MADDPG por defeito nos templates: `maddpg_v2_compact`.

Ficheiros de apoio:

- Hipoteses e caminhos MADDPG: `docs/maddpg_hypotheses_pt.md`.
- Auditoria tecnica fase 3: `docs/maddpg_phase3_audit_pt.md`.
- Plano de experiencias fase 3.5: `docs/maddpg_phase35_experiments_pt.md`.
- Resultados fase 6B: `docs/maddpg_phase6b_triage_pt.md`.
- Resultados fase 6C: `docs/maddpg_phase6c_results_pt.md`.
- Resultados fase 6D: `docs/maddpg_phase6d_results_pt.md`.
- Auditoria dos baselines: `docs/rbc_baseline_audit_pt.md`.
- Auditoria evento EV/headroom 0.6.5: `docs/ev_departure_event_audit_065_pt.md`.
- Resultados fase 6E: `docs/maddpg_phase6e_results_pt.md`.
- Resultados fase 6E.1: `docs/maddpg_phase6e1_results_pt.md`.
- Resultados fases 6F+: `docs/maddpg_phase6f_results_pt.md`.

## Fotografia Atual

Fase 6F.1 esta concluida como primeira matriz de custo sob gate EV.

O que esta solido:

- contrato `entity + static` para MADDPG;
- dois perfis de observacao: `maddpg_v1` e `maddpg_v2_compact`;
- reward com componentes logados;
- diagnosticos de acoes/reward/MADDPG;
- harness de benchmarks;
- KPIs EV feasible-only do simulador `0.6.6` integrados no benchmark;
- bug dos baselines EV com headroom residual corrigido no Algorithms.
- Fase 6E executada em 15s e 2022 com `10/10` jobs completos e `0` falhas.
- Fase 6E.1 executada em 15s e 2022 com `10/10` jobs completos e `0` falhas.
- Fase 6F ja repetiu diagnostico MADDPG pequeno, matriz curta 15s e janela 15s
  com departures reais usando simulador `0.6.6`.
- reward EV corrigida para nao tratar departure desconhecido
  (`departure_time_step = -1`) como departure falhado em todos os passos.
- baselines deterministicos cumprem `ev_departure_min_acceptable_feasible_rate`
  em ambos os datasets.
- Fase 6G revalidou storage/baselines: nao ha evidencia de perda de SOC em
  idle; `Normal` com bateria piora por estrategia/ciclos, enquanto
  `RBCBasic` e `RBCSmart` mostram valor real de storage.
- `RBCSmart` 2022 ficou retuned com `price_charge_rate=0.15`,
  `storage_price_charge_soc_ceiling=0.85` e floors de descarga `0.30`; isto
  bate `RBCBasic` no 2022 mantendo EV feasible e sem violacoes.
- O schema de config agora preserva os knobs novos de storage/deferrables dos
  baselines, evitando drift entre YAML, config resolvido e policy.
- As variantes MADDPG com professor `RBCBasicPolicy`/`RBCSmartPolicy` agora
  recebem automaticamente os hiperparametros do baseline do mesmo dataset; a
  exploracao inicial e o behavior cloning deixam de usar defaults desalinhados
  face ao baseline de comparacao.
- MADDPG ja tem uma receita que cumpre EV feasible no 15s:
  `ev_service_v2g_guard_warm_rbc_basic`.
- Replay ponderado por reward ja existe como variante configuravel:
  `RewardWeightedMultiAgentReplayBuffer`.
- CUDA ja funcionou no ambiente local com PyTorch `2.5.1+cu121` e GPU
  `NVIDIA GeForce RTX 4080 Laptop GPU`; confirmar novamente antes de nova run
  MADDPG longa.
- Fase 6F.1 executada em GPU com `5/5` jobs completos e `0` falhas.
- MADDPG manteve `ev_departure_min_acceptable_feasible_rate = 1.0` nas
  variantes testadas.
- Melhor variante MADDPG atual em custo:
  `service_guard_v2_prioritized_low_bc_decay_warm_rbc_basic`
  com custo `69.737`, ainda pior que `RBCSmart` (`60.402`).
- Fase 6F.2 implementou a reward experimental
  `CostServiceCommunityBandRewardV4`, alinhada com settlement comunitario e
  penalizacao de sobre-servico EV.
- Fase 6F.2 executada em GPU no 15s com `5/5` jobs completos e `0` falhas.
- Melhor V4 atual:
  `community_band_v4_prioritized_tiny_bc_decay_warm_rbc_basic`, custo `69.794`,
  mantendo `ev_departure_min_acceptable_feasible_rate = 1.0`.
- Fase 6F.3 implementou teacher behavior cloning separado da acao executada no
  replay, com alvo `warm_start_policy`.
- Run GPU longa 6F.3 ficou parcial (`95.8333%`) por `CUDA error: unspecified
  launch failure`; os logs sao uteis para diagnostico, mas nao contam como
  benchmark final.
- Teacher BC reduziu EV negative fraction e EV service penalty face a V43
  anterior, mas nao resolveu o evento dominante do Building 15/agente 14.
- Fase 6F.4/6F.5 criou V44/V45, estabilizando critic e tornando a pressao EV
  phase-aware no Building 15.
- Fase 6F.7 adicionou actor pretraining configuravel, para que a avaliacao
  deterministica nao dependa do professor durante o rollout de treino. A
  primeira variante melhorou 1 episodio, mas degradou no terceiro quando a
  policy loss ficou demasiado forte.
- Fase 6F.10-6F.14 testou teacher-clone EV focus. O melhor marco atual e
  `community_feasible_service_v45_teacher_clone_ev_focus_rbc_smart` com 3
  episodios de treino + 1 avaliacao deterministica: custo `87.460`,
  `EV feasible min = 1.0`, `EV within tolerance feasible = 0.6`, EV V2G medio
  `0.0`.
- Fase 6H.3 substituiu o professor "acidental" por um professor explicito
  `RBCSmart` mais suave para aprendizagem:
  `community_feasible_service_v45_teacher_clone_ev_learning_teacher_rbc_smart`.
  Resultado 15s, 3 treino + 1 avaliacao: custo `87.808`,
  `EV feasible min = 1.0`, `EV within_tolerance_feasible = 0.6`, erro EV
  absoluto medio `0.0654`, EV V2G medio `0.0`. A variante recupera o bom custo
  do `f14` e melhora o erro EV absoluto, mas ainda nao bate o `RBCSmart` em
  precisao EV (`within_tolerance_feasible = 1.0`, erro absoluto `0.0299`).
- Fase 6H.4 testou event replay EV generico e foi rejeitada: custo `87.945`,
  `EV feasible min = 0.8`, `EV within_tolerance_feasible = 0.4`.
- Fase 6H.5 reforcou BC EV e targets zero/idle sem event replay generico.
  Resultado 15s, 5 treino + 1 avaliacao: custo `85.172`,
  `EV feasible min = 1.0`, `EV within_tolerance_feasible = 0.8`, erro EV
  absoluto medio `0.0484`, EV V2G medio `0.0`.
- Fase 6H.6 criou a reward `CostServiceCommunityFeasiblePrecisionRewardV46`
  com caps em deficits EV usados no treino, over-service mais forte e critic
  target clip `25`. Resultado 15s, 5 treino + 1 avaliacao: custo `83.761`,
  `EV feasible min = 1.0`, `EV within_tolerance_feasible = 0.8`, erro EV
  absoluto medio `0.0473`, EV V2G medio `0.0`. E o melhor marco MADDPG atual
  em custo e mantem o gate EV, mas ainda nao atinge a precisao EV do professor.

O que ainda nao esta solido:

- melhorar banda de SOC EV sem perder o custo ja obtido com 6H.6;
- tuning multi-episodio/multi-seed;
- validacao final do bundle no inference.
- confirmar o ganho de custo do `6H.6` em mais seeds/datasets;
- reduzir uso excessivo de storage no MADDPG;
- melhorar `ev_departure_within_tolerance_feasible_rate`, porque a V4 cumpre o
  minimo/target mas nao sai dentro da banda do SOC pedido.
- estabilizar critic em runs longas sem depender demasiado de clipping; V46 ja
  reduziu o target clip para `25`, mas ainda precisa de validacao;
- fazer o actor bater tambem a precisao EV do `RBCSmart`
  (`within_tolerance_feasible = 1.0`).

Conclusao pratica: a infraestrutura, baselines, CUDA, logs e benchmark ja estao
coerentes. O gargalo atual e de aprendizagem/reward tradeoff: manter servico EV
e baixar custo sem voltar a descarregar EVs ou abusar de storage.

## Checklist Macro

- [x] Fase 1: congelar contrato atual de treino/export.
- [x] Fase 2: criar caso controlado de aprendizagem MADDPG.
- [x] Fase 3: auditar internals do MADDPG.
- [x] Fase 3.5: melhorias MADDPG implementadas; avaliacao adiada para benchmarks.
- [x] Fase 4: melhorar logs de treino e diagnostico minimo.
- [x] Fase 5: ajustar reward base com componentes observaveis.
- [x] Fase 6A: criar harness curto e repetivel para comparar baselines/MADDPG.
- [x] Fase 6B: triagem 15s de 1 dia contra baselines e variantes MADDPG.
- [x] Fase 6C: corrigir saturacao do actor e servico EV antes de treino longo.
- [x] Fase 6D: repetir triagem com KPIs EV 0.5.2 e reward alinhada com
  tolerancia de servico.
- [x] Fase 6D.1: validar compatibilidade local com simulador 0.6.4 e repetir
  baselines 15s apos fixes fisicos EV/BESS/deferrables.
- [x] Fase 6D.2: alinhar benchmark com KPIs EV feasible do simulador 0.6.5.
- [x] Fase 6D.3: auditar evento EV estranho 0.6.5 e corrigir clipping de
  headroom residual nos baselines.
- [x] Fase 6E: repetir baselines pos-fix em 15s e 2022.
- [x] Fase 6E.1: corrigir/calibrar estrategia dos baselines heuristicos.
- [x] Fase 6F: repetir diagnostico/tuning MADDPG contra baselines confiaveis.
- [x] Fase 6F.1: testar custo do MADDPG mantendo `EV feasible min = 1.0`.
- [x] Fase 6F.2: nova calibracao reward/config para reduzir custo sem perder
  o gate EV. V4 implementada/testada; ainda nao bate `RBCSmart`.
- [x] Fase 6F.3: teacher BC e replay com acoes do professor implementados; run
  longa parcial usada para diagnostico, nao benchmark final.
- [x] Fase 6F.4/6F.5: V44/V45 implementadas para critic mais estavel e schedule
  EV phase-aware.
- [x] Fase 6F.7: actor pretraining avaliado; policy loss cedo demais degradou
  EV service.
- [x] Fase 6F.10-6F.14: teacher-clone EV focus avaliado; `f14` e o melhor
  marco MADDPG atual no 15s.
- [x] Fase 6G: auditar storage e retunar `RBCSmart` 2022 com schema/config
  alinhados.
- [x] Fase 6H.3: professor explicito de aprendizagem testado; recupera custo
  baixo e melhora erro EV face ao `f14`, mas ainda nao atinge precisao EV do
  `RBCSmart`.
- [x] Fase 6H.4: event replay EV generico testado e rejeitado.
- [x] Fase 6H.5: strong BC EV testado; melhor marco MADDPG anterior.
- [x] Fase 6H.6: V46 testada; melhor marco MADDPG atual em custo, mantendo
  `EV feasible min = 1.0`.
- [ ] Fase 7: benchmark KPI completo contra baselines.
- [ ] Fase 8: validar bundle real no repo de inference.

## Proxima Ordem Logica

1. Tomar a 6H.6/V46 como baseline MADDPG atual.
2. Melhorar a precisao de banda EV feasible (`within_tolerance_feasible`) antes
   de otimizar mais custo.
3. Rever storage depois da EV precision: 6H.6 baixa custo, mas continua a usar
   descarga de storage e precisa de auditoria por beneficio real.
4. So depois testar fine-tune RL muito lento: policy loss pequena, BC EV forte,
   anti-V2G ativo, e sem voltar a `ev_band`/`balanced`.
5. Se fine-tune degradar EV service, atacar replay event-aware por janelas EV/
   deferrable/grid, mas so com classificacao feasible/infeasible.
6. Escalar MADDPG:
   1 building -> 2 buildings -> 17 buildings, sempre comparando com os baselines.
8. Testar variantes controladas:
   replay ponderado com menor prioridade, redes menores/maiores, LayerNorm,
   critic maior, exploration schedule, reward weights e eventualmente
   TD3-style.
9. Fase 7:
   benchmark final multi-seed contra baselines.
10. Fase 8:
   validar export/inference.

## Fase 1 - Contrato Atual Congelado

Status: concluido.

Revisao apos `softcpsrecsimulator==0.5.1`: continua valido. O contrato foi
regenerado em `runs/training_contracts/pypi051_maddpg_profiles`; dimensoes,
bounds e manifests continuam consistentes para `maddpg_v1` e
`maddpg_v2_compact`.

Revisao local apos `softcpsrecsimulator==0.5.2`: o simulador passou a expor
KPIs EV separados para cumprimento estrito, minimo aceitavel, tolerancia
simetrica, deficit, shortfall alem da tolerancia, surplus e erro absoluto. Os
dois schemas principais declaram explicitamente:

```json
"ev_departure_within_tolerance": 0.05,
"ev_departure_service_tolerance": 0.05
```

Isto nao altera dimensoes de observacao/acao, mas altera o contrato de KPI e a
interpretacao de sucesso EV.

Revisao apos `softcpsrecsimulator==0.6.4`:

- validado primeiro em editable local e depois instalado via PyPI;
- import atual vem de `.venv/lib/python3.10/site-packages/citylearn`;
- `pytest -q`: `155 passed`;
- subset de integracao citado pelo simulador:
  `tests/test_rbc_agent.py tests/test_baseline_policies.py
  tests/test_benchmark_agents.py tests/test_wrapper_action_clipping.py
  tests/test_entity_adapter.py tests/test_wrapper_entity_mode.py`: `45 passed`;
- matriz de contrato regenerada em
  `runs/training_contracts/local064_maddpg_profiles`;
- 4 variantes validas: datasets `15s` e `2022`, perfis `maddpg_v1` e
  `maddpg_v2_compact`;
- os novos KPIs BAU do simulador aparecem no export, mas os KPIs primarios
  usados pelo benchmark continuam a ser os EV/custo/rede ja rastreados.

Revisao local apos `softcpsrecsimulator==0.6.5`:

- instalado em editable a partir de `/home/tiago/dev/Simulator`;
- `citylearn.__version__ == 0.6.5`;
- sem alteracoes de schema, observacoes ou acoes;
- altera o contrato de KPI EV: passam a existir racios feasible-only para
  departures cujo target/minimo aceitavel/tolerancia eram fisicamente atingiveis;
- o KPI primario de qualidade do controlador passa a ser
  `ev_departure_min_acceptable_feasible_rate`;
- `ev_departure_min_acceptable_rate` continua importante, mas mede experiencia
  bruta do utilizador/cenario, incluindo schedules fisicamente impossiveis.

Revisao apos `softcpsrecsimulator==0.6.6`:

- validado primeiro em editable local e depois instalado via PyPI;
- `citylearn.__version__ == 0.6.6`;
- corrige a classificacao feasible dos eventos EV problematicos do 15s;
- Fases 6E.1, 6F e 6F.1 usam este contrato de KPI;
- `requirements.txt` esta fixado em `softcpsrecsimulator==0.6.6`.

Objetivo: antes de mexer em aprendizagem, deixar claro o contrato que o MADDPG
recebe e exporta. A partir daqui, qualquer alteracao a observacoes, encoding,
acoes, reward ou manifest deve ser explicita e versionada.

### Configs Oficiais MADDPG

Usar estes templates como fonte principal:

- `configs/templates/maddpg/maddpg_local.yaml`
- `configs/templates/maddpg/maddpg_2022_all_plus_evs_local.yaml`

Ambos usam:

- `algorithm.name: MADDPG`
- `simulator.interface: entity`
- `simulator.topology_mode: static`
- `simulator.entity_encoding.enabled: true`
- `simulator.entity_encoding.normalization: minmax_space`
- `simulator.entity_encoding.profile: maddpg_v2_compact`
- `simulator.entity_encoding.clip: true`
- `simulator.reward_function: CostHardConstraintReward`
- `simulator.wrapper_reward.enabled: false`

Notas:

- `maddpg_v2_compact` e o perfil default para treino normal.
- `maddpg_v1` fica suportado para comparacao e debugging, mas nao e o default.
- MADDPG em `entity + dynamic` continua fora de contrato; dynamic fica para RBC
  e smoke/debug ate haver agente dynamic-ready.

### Contrato de Datasets

Dataset 15s parquet:

- Config: `configs/templates/maddpg/maddpg_local.yaml`
- Schema: `datasets/citylearn_three_phase_electrical_service_demo_15s_parquet/schema.json`
- `seconds_per_time_step: 15`
- `num_agents: 17`
- `total_actions: 26`
- `ev_actions: 8`

Dataset 2022 all-plus-EVs:

- Config: `configs/templates/maddpg/maddpg_2022_all_plus_evs_local.yaml`
- Schema: `datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json`
- `seconds_per_time_step: 3600`
- `num_agents: 17`
- `total_actions: 26`
- `ev_actions: 8`

### Contrato de Observacoes

Auditoria atual:

- `runs/training_contracts/pypi051_maddpg_profiles`

Resumo validado:

| dataset | profile | obs encoded range | action dims | manifest | bounds issues |
|---|---|---:|---|---|---:|
| 15s parquet | `maddpg_v1` | `111-225` | `[3,1,1,2,2,1,2,1,1,2,1,2,1,1,3,1,1]` | valido | `0` |
| 15s parquet | `maddpg_v2_compact` | `71-131` | `[3,1,1,2,2,1,2,1,1,2,1,2,1,1,3,1,1]` | valido | `0` |
| 2022 all-plus-EVs | `maddpg_v1` | `96-204` | `[3,1,1,2,2,1,2,1,1,2,1,2,1,1,3,1,1]` | valido | `0` |
| 2022 all-plus-EVs | `maddpg_v2_compact` | `56-110` | `[3,1,1,2,2,1,2,1,1,2,1,2,1,1,3,1,1]` | valido | `0` |

Ficheiros principais da auditoria:

- `runs/training_contracts/pypi051_maddpg_profiles/matrix_summary.csv`
- `runs/training_contracts/pypi051_maddpg_profiles/agent_contract_summary.csv`
- `runs/training_contracts/pypi051_maddpg_profiles/kpi_contract.json`

### Contrato de Acoes

O contrato atual tem sempre:

- 17 agentes, um por building.
- 26 acoes totais.
- 8 acoes EV.
- Cada agente controla apenas as acoes disponiveis no respetivo building.
- Bounds sao lidos do ambiente/action space.
- V2G e permitido apenas onde o action bound permite negativo.
- O wrapper continua responsavel por converter a lista de acoes por agente para
  payload entity do simulador.

Action dimensions atuais:

```text
[3, 1, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 1, 1, 3, 1, 1]
```

Isto significa:

- Alguns buildings tem so BESS/storage.
- Alguns tem BESS + EV.
- Buildings com 3 acoes tem BESS + EV + deferrable, ou BESS + multiplos EVs
  conforme o schema.

### Contrato de Export/Manifest

Validado pela matriz:

- `manifest_valid: True`
- `agent_artifact_count: 17`
- `agent_format: onnx`

Isto garante que:

- existe um artefacto por agente;
- a topologia exportada bate com dimensoes de observacao/acao;
- nomes de observacoes/acoes ficam disponiveis para inference;
- o perfil de encoding fica descrito no contrato.

### Baselines Congelados Para Comparacao Futura

A escada de comparacao passa a ser:

1. `RandomPolicy`
2. `NormalNoBatteryPolicy`
3. `NormalPolicy`
4. `RBCBasicPolicy`
5. `RBCSmartPolicy`
6. `MADDPG` com `maddpg_v1`
7. `MADDPG` com `maddpg_v2_compact`

O `RuleBasedPolicy` legacy fica no repo para compatibilidade/debug, mas nao e
baseline principal de comparacao.

Docs relacionadas:

- `docs/rbc_baseline_audit_pt.md`

### Validacoes Feitas

Com `softcpsrecsimulator==0.5.1` instalado via PyPI:

- `.venv/bin/python -m pytest tests/test_dataset_config.py tests/test_entity_adapter.py tests/test_wrapper_entity_mode.py tests/test_replay_buffer.py tests/test_maddpg_exploration.py tests/test_maddpg_checkpointing.py::test_maddpg_update_uses_terminated_or_truncated_for_done tests/test_mlflow_sampling.py -q`
  -> `27 passed`
- `.venv/bin/python -m pytest -q` -> `132 passed`
- `scripts/audit_training_contract_matrix.py --output-dir runs/training_contracts/pypi051_maddpg_profiles --matrix-name pypi051_maddpg_profiles`
  -> 4 variantes validas, `bounds_issue_count=0`, manifests validos
- `scripts/run_maddpg_phase2_diagnostic.py --output-dir runs/maddpg_diagnostics/phase2_building1_15s_maddpg_v2_compact_051_pypi_smoke --episodes 1 --steps 128 --batch-size 32 --random-exploration-steps 32 --actor-layers 64,32 --critic-layers 128,64`
  -> 1 agente, replay buffer `127`, updates feitos, `critic_loss 2.0313 -> 0.0241`
- Semantica deferrable `0.5.1` confirmada no simulador: com `can_start=1`,
  action `0.4` fica OFF, action `0.6` faz start, action `nan` fica OFF.

Validacao local com simulador editable `0.5.2`:

- `.venv/bin/python -m pytest tests/test_benchmark_agents.py tests/test_reward_functions.py tests/test_dataset_config.py -q`
  -> `21 passed`
- `.venv/bin/python -m pytest -q` -> `155 passed`
- `scripts/audit_training_contract_matrix.py --output-dir runs/training_contracts/local052_maddpg_profiles --matrix-name local052_maddpg_profiles`
  -> 4 variantes validas, `bounds_issue_count=0`, manifests validos, KPIs
  0.5.2 incluidos no contrato
- `git diff --check` -> limpo

Validacoes historicas com `softcpsrecsimulator==0.5.0`:

- `.venv/bin/python -m pytest` -> `132 passed`
- `scripts/audit_training_contract_matrix.py` -> 4 variantes validas
- `runs/action_audits/pypi050_dynamic_topology_rule_based` -> 256 steps, 0 issues
- `runs/action_audits/pypi050_dynamic_assets_only_rule_based` -> 256 steps, 0 issues
- `git diff --check` -> limpo

### Regra de Mudanca

A partir deste ponto:

- mudancas em observacoes/encodings devem criar ou alterar explicitamente um
  perfil, por exemplo `maddpg_v3_*`;
- mudancas em reward devem ser versionadas/configuraveis e acompanhadas por logs
  de componentes;
- mudancas em actions/wrapper so devem acontecer por alteracao real do contrato
  do simulador, nao para ensinar o MADDPG por regras;
- qualquer comparacao KPI deve indicar dataset, perfil, reward e config exatos.

## Fase 2 - Caso Controlado MADDPG

Status: concluido como diagnostico inicial.

Revisao apos `softcpsrecsimulator==0.5.1`: continua valido como prova de que o
pipeline aprende em pequeno. Foi corrido tambem um smoke PyPI `0.5.1` em
`runs/maddpg_diagnostics/phase2_building1_15s_maddpg_v2_compact_051_pypi_smoke`.
Antes de benchmarks longos, vale a pena repetir o diagnostico 3x256 com o PyPI
`0.5.1` para ter numeros diretamente comparaveis.

Revisao local `0.5.2`: como a reward EV foi alinhada com
`ev_departure_service_tolerance`, convem repetir pelo menos o diagnostico
1-building `3x256` antes de retomar tuning serio. Nao e necessario repetir a
auditoria estrutural inteira se a Fase 1 confirmar que as dimensoes nao mudaram.

Objetivo: provar que o MADDPG aprende num problema pequeno antes de escalar.

Ferramenta criada:

- `scripts/run_maddpg_phase2_diagnostic.py`

O script nao altera wrapper nem MADDPG. Ele carrega uma config MADDPG real,
cria um subset temporario em memoria do schema com 1 ou 2 buildings, corre o
stack atual e guarda:

- `diagnostic_config.resolved.yaml`
- `diagnostic_schema.json`
- `logs/metrics.jsonl`
- `diagnostic.log`
- `summary.json`
- `README.md`

Runs executados:

- `runs/maddpg_diagnostics/phase2_building1_15s_maddpg_v2_compact_3x256`
- `runs/maddpg_diagnostics/phase2_building1_2_15s_maddpg_v2_compact_3x256`

Config comum:

- dataset: `citylearn_three_phase_electrical_service_demo_15s_parquet`
- perfil: `maddpg_v2_compact`
- episodios: `3`
- steps por episodio: `256`
- batch size: `64`
- warm-up/exploracao inicial: `64` steps
- actor: `[128, 64]`
- critic: `[256, 128]`

### Resultado 1 Building

- buildings: `Building_1`
- agentes: `1`
- observacoes encoded: `[102]`
- acoes: `[3]`
- action names: `electrical_storage`,
  `electric_vehicle_storage_charger_1_1`,
  `deferrable_appliance_deferrable_appliance_1`
- replay buffer final: `765`
- actor delta L2: `[2.2294375896453857]`
- update records no log: `44`
- critic loss: `2.2324 -> 0.0063`
- reward por episodio: `-321.7118 -> -50.7813 -> -42.8811`

### Resultado 2 Buildings

- buildings: `Building_1`, `Building_2`
- agentes: `2`
- observacoes encoded: `[102, 56]`
- acoes: `[3, 1]`
- action names agente 0: `electrical_storage`,
  `electric_vehicle_storage_charger_1_1`,
  `deferrable_appliance_deferrable_appliance_1`
- action names agente 1: `electrical_storage`
- replay buffer final: `765`
- actor delta L2: `[2.083838939666748, 2.069765567779541]`
- update records no log: `44`
- critic loss: `2.1093 -> 0.0074`
- reward por episodio agente 0: `-387.1024 -> -136.1645 -> -134.1689`
- reward por episodio agente 1: `-233.7106 -> -18.3780 -> -14.9126`

### Conclusao Fase 2

O MADDPG atual consegue preencher replay buffer, fazer updates, alterar pesos dos
actors e melhorar a reward num caso pequeno repetido. Isto e uma prova util de
que o pipeline basico de treino esta vivo.

Isto ainda nao prova:

- bons KPIs;
- generalizacao;
- estabilidade com 17 agentes;
- reward correta;
- exploracao adequada;
- ausencia de saturacao de acoes.

Observacao importante: nos logs, depois do warm-up, varias acoes aproximam-se
ou batem frequentemente nos bounds (`1.0`, `-1.0` ou `0.0`). Isto nao e
necessariamente bug, mas deve entrar na Fase 3/4 como sinal a auditar com
histogramas e metricas de saturacao.

## Fase 3 - Auditoria Interna MADDPG

Status: concluido como auditoria inicial.

Revisao apos `softcpsrecsimulator==0.5.1`: continua valida. A diferenca principal
e que o contrato dos deferrables deixou de ser o risco principal; agora o risco
e a policy/exploracao continua do MADDPG ficar perto do threshold `0.5`.

Revisao local `0.5.2`: nao ha nova exigencia arquitetural no MADDPG. A mudanca
e de KPI/reward: a gate de EV deixa de depender de success strict e passa a
usar minimo aceitavel como KPI primario de conforto.

Documento detalhado:

- `docs/maddpg_phase3_audit_pt.md`

Verificar:

- scaling do actor para bounds de acao;
- critic centralizado com obs/actions de todos os agentes;
- replay buffer: shapes, sampling, done/truncated, warmup;
- target networks e soft update;
- exploracao inicial;
- normalizacao de rewards;
- gradient clipping;
- logs de actor loss, critic loss, Q-values;
- agentes com poucas acoes/dados.

Resumo:

- arquitetura MADDPG principal esta alinhada: actor local por agente, critic
  centralizado, replay conjunto, targets e action scaling para bounds reais;
- riscos principais: acoes one-sided/threshold, exploracao uniforme,
  critic loss agregada, ausencia de reward normalization, logs insuficientes,
  buffer prioritized incompativel e checkpoint parcial;
- a Fase 3.5 deve transformar estes riscos em experiencias isoladas.

Validacao focada:

- `.venv/bin/python -m pytest tests/test_replay_buffer.py tests/test_maddpg_exploration.py tests/test_maddpg_checkpointing.py::test_maddpg_update_uses_terminated_or_truncated_for_done tests/test_mlflow_sampling.py -q`
  -> `10 passed`

## Fase 3.5 - Plano de Melhorias MADDPG

Status: implementacao base concluida; avaliacao comparativa adiada para a
matriz de benchmarks.

Documento detalhado:

- `docs/maddpg_phase35_experiments_pt.md`

Objetivo: transformar os findings da Fase 3 em experiencias controladas antes
de mexer em reward/logging/tuning de forma larga.

Revisao apos `softcpsrecsimulator==0.5.1`: o contrato dos deferrables esta
corrigido no simulador. Nao e preciso mexer no wrapper nem nos RBCs. A Fase 3.5
passa a focar a policy continua do MADDPG, a exploracao e a estabilidade do
critic.

Ordem curta atual:

1. actor init no-op-aware, exploracao no-op-centered, warm-start com RBC/Normal
   e critic update por agente ja estao configuraveis;
2. replay buffer incompativel ja falha cedo em MADDPG;
3. checkpoint RL ja guarda exploracao/RNG;
4. logs minimos de actions/start delay e internals MADDPG ja estao
   configuraveis;
5. reward normalization ja esta configuravel e ligada nos templates MADDPG;
6. comparar as variantes configuraveis uma de cada vez na fase de benchmarks;
7. so depois testar redes menores/maiores e LayerNorm.

## Fase 4 - Logs de Treino

Status: diagnostico minimo implementado.

Disponivel via `tracking.*` nos templates MADDPG:

- `action_diagnostics_enabled`;
- `action_diagnostics_detail: summary | per_action`;
- `training_diagnostics_enabled`;
- `training_diagnostics_detail: summary | per_agent`.

Ja fica logado:

- episode reward e step reward;
- actor loss e critic loss;
- Q mean/std/min/max;
- action mean/std/saturation;
- buffer size;
- exploration noise/sigma;
- gradient norms;
- EV action positive/negative/idle ratio;
- storage action positive/negative/idle ratio;
- deferrable ON/OFF ratio;
- deferrable start delay quando `pending + can_start`.

Tambem fica logado via componentes da reward:

- custo/import/export local;
- penalizacoes de grid/power outage;
- penalizacao de EV service/departure;
- penalizacao de deferrable service;
- limites e penalizacao de bateria;
- termos comunitarios de import/pico/export.

Fica para benchmark:

- confirmar se estes componentes batem certo com os KPIs exportados;
- comparar a distribuicao de acoes com os resultados fisicos;
- decidir se os pesos da reward precisam de nova calibracao.

## Fase 5 - Reward

Status: implementacao base concluida; revista localmente para `0.5.2`; pesos
ainda sujeitos a benchmark.

Objetivo: reward informativa, nao prescritiva.

Componentes a rever:

- custo/import/export;
- violacoes de rede;
- EV departure deficit;
- deferrable deadlines/urgency;
- abuso de bateria;
- suavidade de acoes, se necessaria.

### Decisao De Arquitetura Reward

O MADDPG continua a precisar de uma reward escalar por agente. A solucao nao
deve ser "uma reward global unica para todos" nem "so reward local":

- so local: cada casa otimiza custo proprio, mas pode criar picos comunitarios;
- so global: o sinal fica ruidoso e o credit assignment fica fraco;
- hibrida local + comunitaria: cada agente recebe sinal local claro e uma parte
  partilhada dos objetivos da comunidade.

Forma recomendada:

```text
r_i =
  local_cost_i
  - local_constraints_i
  - local_service_deficits_i
  - battery_abuse_i
  + shared_community_signal
```

Onde `shared_community_signal` deve ser configuravel e normalmente dividido por
agente ou com peso pequeno para nao destruir o credit assignment.

### Principios

- A reward nao deve mandar "carrega agora" ou "descarrega agora".
- A reward deve dizer o que e bom/mau:
  - menor custo/import;
  - maior autoconsumo renovavel quando isso nao viola servico;
  - menor pico/import comunitario;
  - zero violacoes de electrical service;
- EV sair com SOC requerido;
- deferrables cumprir deadline;
- bateria nao ser abusada.
- Restrições duras continuam com penalizacao forte.
- Objetivos economicos/renovaveis/comunitarios devem ter pesos moderados.
- Todos os componentes devem ser logados antes de mudar pesos.

### Estado Implementado

`CostHardConstraintReward` agora expoe componentes e usa uma reward hibrida
local + comunitaria, sem regras prescritivas de acao:

- `local_cost_reward`;
- `local_import_energy`;
- `local_export_energy`;
- `local_import_cost`;
- `local_export_credit`;
- `service_violation_penalty`;
- `power_outage_penalty`;
- `battery_safety_penalty`;
- `ev_service_penalty`;
- `ev_soc_deficit_sum`;
- `ev_soc_service_target_sum`;
- `ev_soc_shortfall_beyond_tolerance_sum`;
- `ev_soc_strict_gap_within_tolerance_sum`;
- `ev_soc_surplus_sum`;
- `ev_soc_absolute_error_sum`;
- `ev_schedule_min_soc_required_sum`;
- `ev_schedule_soc_deficit_sum`;
- `ev_schedule_deficit_penalty`;
- `ev_departure_window_penalty`;
- `ev_departure_missed_penalty_amount`;
- `deferrable_service_penalty`;
- `deferrable_deadline_missed_penalty_amount`;
- `deferrable_urgency_penalty_amount`;
- `community_import_penalty`;
- `community_peak_import_penalty`;
- `community_export_penalty`;
- `community_shared_penalty`;
- `reward_total`;
- componentes comunitarios agregados.

Alteracoes principais:

- a partir do simulador `0.5.2`, a reward distingue deficit estrito
  (`required_soc - soc`) de shortfall alem da tolerancia de servico
  (`required_soc - ev_departure_service_tolerance - soc`);
- a penalizacao dura de departure/window/missed usa o minimo aceitavel pelo
  utilizador; o deficit estrito continua como sinal denso configuravel para
  guiar o agente ate ao alvo ideal;
- limites de SOC da bateria sao lidos das observacoes quando existem
  (`soc_min_ratio` e, quando o simulador expuser, `soc_max_ratio`);
- a penalizacao EV principal deixou de punir qualquer deficit cedo demais e
  passou a usar uma trajetoria minima viavel: se ainda ha tempo para atingir o
  SOC minimo aceitavel carregando a potencia maxima, nao ha penalizacao de schedule;
  se o SOC atual fica abaixo do minimo necessario para cumprir a saida, a
  penalizacao cresce automaticamente a medida que o prazo se aproxima;
- a penalizacao forte de EV missed departure continua como evento de falha de
  servico;
- os templates usam fallback fisico `battery_soc_min: 0.0` e
  `battery_soc_max: 1.0`, evitando inventar uma banda de conforto quando o
  simulador nao expõe limites; quando existem observacoes de minimo/maximo, sao
  esses limites que devem ser respeitados;
- `export_credit_ratio` passou a `0.0` nos templates comparaveis, porque o
  dataset 15s declara `grid_export_price: 0.0` e nao queremos premiar export
  artificial;
- termo comunitario linear de import e termo quadratico de pico ficam
  configuraveis e divididos por agente nos templates;
- throughput da bateria continua penalizavel separadamente, para evitar abuso
  mesmo quando SOC esta dentro dos limites fisicos.

Tambem ficou configuravel o escalamento temporal de penalizacoes densas/de
estado:

```yaml
simulator:
  reward_function_kwargs:
    scale_state_penalties_by_time_step: true
    state_penalty_reference_seconds: 3600.0
```

Isto e importante no dataset de 15 segundos. Sem esta escala, uma penalizacao de
estado como SOC fora do intervalo e aplicada 240 vezes por hora, enquanto custo
e import/export ja estao em energia por step. Com `seconds_per_time_step=3600`,
a escala e `1.0`, portanto o dataset horario fica equivalente.

O wrapper loga estes componentes quando:

```yaml
tracking:
  reward_diagnostics_enabled: true
  reward_diagnostics_detail: summary # ou per_agent
```

O MADDPG tambem tem normalizacao opcional da reward usada no target do critic:

```yaml
algorithm:
  exploration:
    params:
      reward_normalization: true
      reward_normalization_clip: 10.0
      reward_normalization_epsilon: 1.0e-8
```

Isto nao altera a reward do simulador nem os KPIs; apenas normaliza o sinal de
treino dentro do MADDPG. E importante porque os dois datasets tem escalas muito
diferentes: no 15s os custos por step sao pequenos, enquanto no 2022 horario
podem aparecer export/import de grande magnitude.

### Smokes Reward V2

Foram corridos dois smokes curtos de aprendizagem MADDPG, nao RBC:

- `runs/maddpg_diagnostics/phase5_reward_schedule_ev_building1_15s_smoke`
- `runs/maddpg_diagnostics/phase5_reward_schedule_ev_building1_2022_hourly_smoke`

Resultados relevantes:

- ambos tiveram replay suficiente, updates e `actor_delta_l2 > 0`;
- a exploracao usou warm-up aleatorio ate 32 steps e depois Gaussian noise;
- `final_sigma` ficou em cerca de `0.1201`;
- no 15s, a bateria deixou de dominar a reward por SOC minimo inventado e o EV
  nao foi penalizado enquanto ainda tinha tempo para cumprir a saida;
- no 2022, os maiores termos negativos continuam a vir de EV service/departure
  quando a policy ainda esta exploratoria/nao treinada, agora separados em
  schedule deficit, window deficit e missed departure.

Isto valida que o pipeline aprende em smoke, mas ainda nao valida bons KPIs.
Bons KPIs exigem rollout/treino mais longo e comparacao contra baselines.

## Fase 6 - Rebaseline e Tuning RL

Status: fases 6A, 6B, 6C, 6D, 6D.1, 6D.2, 6D.3 e 6E concluidas. A Fase
6E.1 e a proxima.

Motivo para nao saltar ja para tuning longo MADDPG: as conclusoes antigas de EV
service ficaram contaminadas por mudancas reais no simulador e por um bug nas
policies baseline que usavam headroom residual como se fosse capacidade total
do charger. A matriz 6E ja correu; agora a prioridade e calibrar a estrategia
dos baselines, porque `RBCBasic` e `RBCSmart` ainda nao sao referencias finais.

### Fase 6A - Harness Curto de Comparacao

Status: concluido.

Objetivo: deixar pronto um comando repetivel para comparar, em janelas curtas,
os baselines e variantes MADDPG sem fazer ainda afirmacoes finais de KPIs.

Script:

```bash
python scripts/run_phase6a_benchmark.py
```

O script gera uma config por run em `generated_configs/`, corre
`run_experiment.py` quando nao esta em `--dry-run`, e agrega os resultados em:

- `benchmark_summary.csv`;
- `benchmark_summary.json`;
- `README.md`;
- `jobs/<job_id>/` com os outputs normais do runner.

Matriz default:

- datasets: `15s`, `2022`;
- baselines: `RandomPolicy`, `NormalNoBatteryPolicy`, `NormalPolicy`,
  `RBCBasicPolicy`, `RBCSmartPolicy`;
- MADDPG: variante `current` por defeito;
- seeds: `123` por defeito.

Variantes MADDPG configuraveis:

- `current`: template atual;
- `v1`: encoding `maddpg_v1`;
- `noop_centered`: exploracao inicial centrada em no-op;
- `noop_actor`: inicializacao do actor perto de no-op;
- `warm_rbc_basic`: warm-start de exploracao com `RBCBasicPolicy`;
- `per_agent_critic`: critic update por agente.

Comando de smoke ja validado:

```bash
.venv/bin/python scripts/run_phase6a_benchmark.py \
  --output-dir runs/benchmarks/phase6a_smoke_15s_random_maddpg \
  --dataset 15s \
  --agent random \
  --agent maddpg \
  --maddpg-variant current \
  --seed 123 \
  --steps 32 \
  --episodes 1 \
  --batch-size 16 \
  --random-exploration-steps 8 \
  --metric-interval 8
```

Resultado do smoke: 2 runs completas, 0 falhas, KPIs exportados e colunas de
reward/action/MADDPG presentes no `benchmark_summary.csv`.

Revisao local `0.5.2`: `DEFAULT_KPIS` passou a separar:

- `ev_departure_min_acceptable_rate` como KPI primario de conforto;
- `ev_departure_success_rate` como cumprimento estrito;
- `ev_departure_within_tolerance_rate` como accuracy simetrica;
- `ev_departure_soc_deficit_mean`;
- `ev_departure_shortfall_beyond_tolerance_mean`;
- `ev_departure_soc_surplus_mean`;
- `ev_departure_soc_absolute_error_mean`;
- `ev_departure_tolerance_ratio`.

Revisao local `0.6.5`: `DEFAULT_KPIS` passa tambem a rastrear:

- `ev_departure_min_acceptable_feasible_rate`;
- `ev_departure_success_feasible_rate`;
- `ev_departure_within_tolerance_feasible_rate`;
- `ev_departure_count`;
- `ev_departure_target_infeasible_count`;
- `ev_departure_min_acceptable_infeasible_count`;
- `ev_departure_within_tolerance_infeasible_count`.

O gate EV do benchmark usa `ev_departure_min_acceptable_feasible_rate` quando
existe. Para runs antigas, faz fallback para `ev_departure_min_acceptable_rate`,
depois `ev_departure_success_feasible_rate` e finalmente
`ev_departure_success_rate`.

Validacoes adicionais:

- `tests/test_phase6a_benchmark.py` cobre dry-run, geracao de configs,
  summary e parametros MADDPG configuraveis;
- `runs/benchmarks/phase6a_dry_run_full_matrix` gerou a matriz default completa
  em dry-run: 12 runs planeadas, 0 falhas.

Interpretacao:

- serve para confirmar comparabilidade, contratos, KPIs, reward components e
  diagnosticos;
- nao serve ainda para dizer que uma policy e melhor em performance final;
- para conclusoes finais, usar janelas longas, multiplas seeds e a matriz
  selecionada de hipoteses em `docs/maddpg_hypotheses_pt.md`.

### Fase 6B - Triagem 15s de 1 Dia

Antes de tuning longo, a triagem 15s de 1 dia mostrou um bloqueio claro.

Resultado documentado em `docs/maddpg_phase6b_triage_pt.md`.

Resumo:

- 9/9 runs completas, 0 falhas;
- janela 15s de 1 dia, 5760 steps, seed 123;
- 7 EV departures, 1 ciclo deferrable, 0 violacoes de electrical service;
- `Normal` e `NormalNoBattery` so cumprem 1/7 EV departures;
- `RBCBasic` e `RBCSmart` cumprem 0/7 EV departures;
- todas as variantes MADDPG cumprem 0/7 EV departures;
- `MADDPG current`, `v1`, `noop_centered` e `warm_rbc_basic` fazem updates
  estaveis, mas acabam com forte saturacao de acoes;
- `maddpg_v1` nao resolveu nada e saturou ainda mais;
- `noop_centered` e `warm_rbc_basic` melhoram o buffer inicial, mas nao
  impedem a deriva do actor para extremos.

Conclusao: nao vale correr treino longo multi-seed com as variantes atuais antes
de atacar saturacao do actor e servico EV.

### Fase 6C - Anti-Saturacao e EV Service

Status: implementacao inicial feita e benchmark 15s executado.

Objetivo: corrigir o modo de falha observado na fase 6B sem meter regras no
wrapper. As mudancas devem ficar configuraveis no algoritmo/reward e logadas em
componentes separadas.

Implementado:

- `RBCBasicPolicy` e `RBCSmartPolicy` passaram a preservar uma taxa minima de
  carga EV ate ao SOC de saida, antes de otimizar preco/PV;
- `RBCSmartPolicy` so usa V2G quando nao ha deficit EV e existe margem SOC/tempo
  ate a partida;
- MADDPG tem `actor_update_interval` configuravel para delayed actor update;
- MADDPG tem `target_policy_smoothing` configuravel no target do critic;
- MADDPG tem regularizacao configuravel de magnitude/saturacao de acoes do
  actor, calculada em espaco normalizado por bounds;
- logs/metricas MADDPG incluem perdas de policy, regularizacao e saturacao;
- `scripts/run_phase6a_benchmark.py` ganhou variantes `anti_saturation` e
  `anti_saturation_warm_rbc_basic`.

Ainda a confirmar por benchmark:

- repetir MADDPG com reward densa EV ativa;
- testar warm-start com `RBCSmartPolicy` retuned;
- validar se `1/7` departures e limite fisico da janela 15s.

Resultado: `docs/maddpg_phase6c_results_pt.md`.

### Fase 6D - Repetir Triagem 0.5.2

Status: concluido para 15s; falta repetir no dataset 2022.

Objetivo: repetir uma matriz curta/medio prazo com simulador `0.5.2` local para
confirmar que:

- os novos KPIs aparecem no `exported_kpis.csv`;
- `RBCSmart` melhora em `ev_departure_min_acceptable_rate`;
- `Normal`, `RBCBasic`, `RBCSmart` ficam ordenados de forma plausivel;
- MADDPG nao e avaliado apenas por strict success;
- a reward EV nao e dominada por penalizacao dura quando o EV esta dentro da
  tolerancia de servico.

Comando minimo sugerido:

```bash
.venv/bin/python scripts/run_phase6a_benchmark.py \
  --output-dir runs/benchmarks/phase6d_052_service_kpis_15s \
  --dataset 15s \
  --agent normal_no_battery \
  --agent normal \
  --agent rbc_basic \
  --agent rbc_smart \
  --agent maddpg \
  --maddpg-variant anti_saturation_warm_rbc_smart \
  --seed 123 \
  --steps-15s 5760 \
  --episodes 1 \
  --batch-size 64 \
  --random-exploration-steps 256 \
  --metric-interval 240
```

Se isto estiver coerente, repetir para `2022` numa janela curta e depois
multi-seed.

Smoke local ja feito com simulador editable `0.5.2`:

```bash
.venv/bin/python scripts/run_phase6a_benchmark.py \
  --output-dir runs/benchmarks/phase6d_052_rbc_smart_smoke \
  --dataset 15s \
  --agent rbc_smart \
  --seed 123 \
  --steps-15s 5760 \
  --episodes 1 \
  --metric-interval 240
```

Resultado:

- 1/1 run completa, 0 falhas;
- `exported_kpis.csv` inclui os novos KPIs `departure_min_acceptable`,
  `departure_shortfall_beyond_tolerance`, `departure_soc_surplus`,
  `departure_soc_absolute_error` e `departure_tolerance`;
- o agregador leu:
  - `ev_departure_min_acceptable_rate: 0.143`;
  - `ev_departure_success_rate: 0.143`;
  - `ev_departure_within_tolerance_rate: 0.143`;
  - `ev_departure_soc_deficit_mean: 0.333`;
  - `ev_departure_shortfall_beyond_tolerance_mean: 0.29`;
  - `ev_departure_soc_surplus_mean: 0.005`;
  - `ev_departure_soc_absolute_error_mean: 0.338`;
  - `ev_departure_tolerance_ratio: 0.05`.

Isto prova o contrato KPI/export local `0.5.2`, mas ainda nao prova melhoria de
policy.

Resultado documentado:

- `docs/maddpg_phase6d_results_pt.md`

Resumo 15s:

| policy | cost eur | EV min acceptable | EV strict | EV deficit mean |
|---|---:|---:|---:|---:|
| NormalNoBattery | 250.353 | 0.143 | 0.143 | 0.333 |
| Normal | 255.665 | 0.143 | 0.143 | 0.333 |
| RBCBasic retuned | 254.145 | 0.143 | 0.143 | 0.333 |
| RBCSmart safe-storage | 250.328 | 0.143 | 0.143 | 0.333 |
| MADDPG anti_saturation_warm_rbc_smart | 126.517 | 0.000 | 0.000 | 0.791 |

Conclusao: a escada dos baselines ficou mais justa para esta janela. O MADDPG
ainda reduz custo a custa de EV service, portanto nao esta pronto para treino
longo/final sem nova intervencao em reward/exploracao/treino.

### Fase 6D.1 - Revisao 0.6.4

Status: concluido para compatibilidade e baselines 15s; falta repetir MADDPG e
2022.

O simulador `0.6.4` inclui fixes fisicos importantes:

- EVs deixam de herdar perdas estacionarias artificiais em datasets sub-hourly;
- `loss_coefficient` passa a ter semantica horaria;
- SOC de EV/BESS e carregamento com acao zero passam a ficar consistentes;
- bounds/observacoes/KPIs/export foram hardenizados;
- o simulador adicionou baseline nativa `BusinessAsUsualAgent` e KPIs
  `business_as_usual`, `delta_to_business_as_usual` e
  `ratio_to_business_as_usual`.

Validacoes locais:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/audit_training_contract_matrix.py \
  --output-dir runs/training_contracts/local064_maddpg_profiles \
  --matrix-name local064_maddpg_profiles
```

Resultados:

- `155 passed`;
- matriz de contrato: 4 variantes, 0 falhas;
- nao foi preciso alterar wrapper, encoders, MADDPG ou reward para compatibilidade
  com `0.6.4`.

Triagem baseline 15s:

```bash
.venv/bin/python scripts/run_phase6a_benchmark.py \
  --output-dir runs/benchmarks/local064_baseline_ev_check_15s \
  --dataset 15s \
  --agent normal_no_battery \
  --agent normal \
  --agent rbc_basic \
  --agent rbc_smart \
  --seed 123 \
  --steps-15s 5760 \
  --episodes 1 \
  --metric-interval 240
```

Resultado 15s, 1 dia, seed `123`:

| policy | cost eur | EV min acceptable | EV strict | EV deficit mean | EV shortfall tol mean |
|---|---:|---:|---:|---:|---:|
| NormalNoBattery | 105.998 | 0.714 | 0.714 | 0.101 | 0.087 |
| Normal | 106.703 | 0.714 | 0.714 | 0.101 | 0.087 |
| RBCBasic | 106.660 | 0.714 | 0.714 | 0.101 | 0.087 |
| RBCSmart | 105.973 | 0.714 | 0.714 | 0.101 | 0.087 |

Conclusao:

- a falha EV extrema de `0.5.2` era em grande parte artefacto fisico do
  simulador, nao estrategia dos baselines;
- `NormalNoBattery` passou de `0.143` para `0.714` em
  `ev_departure_min_acceptable_rate`;
- os 2/7 departures restantes devem ser auditados como possivel infeasibilidade
  de janela/potencia/headroom, antes de culpar RBC ou reward;
- os resultados MADDPG antigos com EV service mau devem ser repetidos contra
  `0.6.4` antes de qualquer tuning longo.

### Fase 6D.2 - Revisao 0.6.5

Status: alinhamento de KPI concluido; a auditoria do evento estranho foi
separada para a Fase 6D.3.

O simulador `0.6.5` adiciona KPIs EV feasible-only. Isto resolve a ambiguidade
que tinhamos no `0.6.4`: quando `ev_departure_min_acceptable_rate = 0.714`,
nao sabiamos pelo KPI agregado se os `2/7` restantes eram falha do controlador
ou eventos fisicamente impossiveis.

Decisao de avaliacao:

- KPI primario de qualidade do controlador:
  `ev_departure_min_acceptable_feasible_rate`;
- KPI primario de experiencia real do utilizador/cenario:
  `ev_departure_min_acceptable_rate`;
- diagnostico de dataset/schedule:
  `ev_departure_min_acceptable_infeasible_count`,
  `ev_departure_target_infeasible_count` e
  `ev_departure_within_tolerance_infeasible_count`;
- strict target e within-tolerance continuam secundarios para perceber quao
  perto o controlador fica do SOC pedido.

Implicacao:

- os baselines 15s e 2022 devem ser corridos outra vez com `0.6.5`
  (`15s` ja concluido; falta `2022`);
- se o feasible rate for `1.0` e o raw continuar `0.714`, os eventos restantes
  sao problema de cenario/viabilidade, nao de policy;
- se o feasible rate tambem ficar abaixo de `1.0`, entao ha falha real de
  policy/reward/exploracao a investigar.

Triagem baseline 15s ja corrida:

- run: `runs/benchmarks/local065_baseline_ev_feasibility_15s`;
- 4/4 completas, 0 falhas;
- todas as policies ficaram com:
  - `ev_departure_min_acceptable_feasible_rate = 0.833`;
  - `ev_departure_min_acceptable_rate = 0.714`;
  - `ev_departure_count = 7`;
  - `ev_departure_min_acceptable_infeasible_count = 1`;
  - `ev_departure_soc_deficit_mean = 0.101`;
  - `ev_departure_shortfall_beyond_tolerance_mean = 0.087`.

Leitura: havia de facto `1` departure raw infeasible, mas ainda existe `1/6`
departure feasible que falha em todos os baselines. Antes de mudar reward ou RBC
outra vez, auditar esse evento especifico por charger/building.

### Fase 6D.3 - Auditoria Evento EV/Headroom

Status: concluido.

Documento detalhado:

- `docs/ev_departure_event_audit_065_pt.md`

Evento auditado:

- run original: `runs/benchmarks/local065_baseline_ev_feasibility_15s`;
- building afetado: `Building_15`;
- chargers afetados: `charger_15_1` e `charger_15_2`;
- sintoma: todos os baselines tinham
  `ev_departure_min_acceptable_feasible_rate = 0.833` e
  `ev_departure_min_acceptable_rate = 0.714`.

Finding principal:

- os baselines carregavam EV no limite de fase num step;
- no step seguinte liam headroom residual quase zero;
- como a action do charger e comando total, nao delta, o baseline cortava a
  acao seguinte quase para zero;
- isto criava um padrao artificial ON/OFF e reduzia a potencia media efetiva.

Fix aplicado no Algorithms:

- `RuleBasedPolicy._apply_ev_dynamic_headroom_limit(...)` continua a respeitar
  headroom;
- mas agora soma a potencia ja aplicada pelo proprio charger
  (`applied_power_kw`/`commanded_power_kw`) ao headroom residual;
- teste unitario:
  `test_rbc_ev_headroom_clip_allows_existing_charger_power`.

Validacao:

- `.venv/bin/python -m pytest -q` -> `157 passed`;
- run pos-fix: `runs/benchmarks/local065_ev_event_fix_15s`.

Resultado pos-fix no `NormalNoBatteryPolicy`:

| charger | energia antes | energia depois | SOC antes | SOC depois | steps positivos antes | steps positivos depois |
|---|---:|---:|---:|---:|---:|---:|
| `charger_15_1` | 31.36 kWh | 62.70 kWh | 0.44 | 0.74 | 1213/2160 | 2160/2160 |
| `charger_15_2` | 24.67 kWh | 49.33 kWh | 0.47 | 0.68 | 1333/2400 | 2400/2400 |

KPIs agregados:

- `ev_departure_soc_deficit_mean`: `0.1010 -> 0.0294`;
- `ev_departure_shortfall_beyond_tolerance_mean`: `0.0868 -> 0.0152`;
- `ev_departure_min_acceptable_rate` manteve `0.714`;
- `ev_departure_min_acceptable_feasible_rate` manteve `0.833`.

Leitura:

- a falha artificial da policy foi corrigida;
- o KPI de sucesso nao mudou porque o Building 15 continua abaixo do minimo
  aceitavel nos dois eventos;
- agora a falha residual parece ligada a limite fisico/temporal, detalhe do
  instante de departure, ou feasibility que nao considera limites de fase;
- antes de culpar MADDPG/reward, temos de repetir todos os baselines pos-fix.

### Fase 6E - Repetir Baselines Pos-Fix

Status: executada, mas gate de qualidade dos baselines nao passou.

Objetivo: reconstruir a base de comparacao depois do fix de headroom EV.

Matriz minima:

- datasets: `15s`, `2022`;
- policies: `RandomPolicy`, `NormalNoBatteryPolicy`, `NormalPolicy`,
  `RBCBasicPolicy`, `RBCSmartPolicy`;
- seed inicial: `123`;
- depois, se estiver coerente, repetir multi-seed.

Comando executado:

```bash
.venv/bin/python scripts/run_phase6a_benchmark.py \
  --output-dir runs/benchmarks/phase6e_065_baselines_post_headroom_fix \
  --dataset 15s \
  --dataset 2022 \
  --agent random \
  --agent normal_no_battery \
  --agent normal \
  --agent rbc_basic \
  --agent rbc_smart \
  --seed 123 \
  --episodes 1 \
  --full-window \
  --metric-interval 240
```

Resultado:

- output: `runs/benchmarks/phase6e_065_baselines_post_headroom_fix`;
- `10/10` jobs completos;
- `0` falhas;
- detalhes: `docs/maddpg_phase6e_results_pt.md`.

Conclusoes:

- `NormalNoBattery` representa bem o dia normal sem bateria e ficou como
  baseline mais limpo;
- EV service nos deterministicos ficou forte, especialmente no 2022;
- no 15s ainda ha gargalo EV no Building 15, com uma falha feasible e um evento
  infeasible;
- `Random` serve apenas como sanity/lower bound, nao como baseline de custo;
- `Normal` com bateria nao melhora custo nos runs 6E;
- `RBCBasic` reduz custo no 2022, mas com penalizacao forte de bateria;
- `RBCSmart` evita penalizacao de bateria, mas nao melhora custo de forma clara;
- `RBCBasic` e `RBCSmart` introduzem penalizacao deferrable no 15s.

Gate para MADDPG:

Ainda nao passar para comparacao MADDPG final. Primeiro fazer Fase 6E.1.

### Fase 6E.1 - Calibrar Baselines

Status: concluida.

Objetivo: tornar `Normal`, `RBCBasic` e `RBCSmart` uma escada justa e
fisicamente coerente.

Gates minimos:

- EV deterministicos:
  `ev_departure_min_acceptable_feasible_rate >= NormalNoBattery - 1e-6`;
- deferrables:
  `deferrable_service_penalty_mean == 0` ou tradeoff explicitamente
  documentado;
- bateria:
  `battery_safety_penalty_mean` baixo e sem abuso de ciclos;
- `RBCBasic`:
  melhorar custo ou pico face a `NormalNoBattery` sem degradar servico;
- `RBCSmart`:
  melhorar pelo menos custo, pico, PV/self-consumption ou bateria face a
  `RBCBasic`, sem ser oracle.

Trabalho provavel:

- deferrables: regra de ultimo arranque seguro;
- `Normal`: tornar storage conservador ou manter `NormalNoBattery` como
  principal baseline normal;
- `RBCBasic`: limitar arbitragem por preco e ciclos de bateria;
- `RBCSmart`: ativar PV/preco/headroom de forma util, mas com limites de SOC e
  throughput;
- repetir a matriz 6E num novo output dir.

### Fase 6F - MADDPG Diagnostico e Tuning

Status: concluida como diagnostico e primeiro tuning MADDPG.

Base executada depois da Fase 6E.1:

1. repetir diagnostico pequeno:
   - 1 building;
   - 2 buildings;
   - `maddpg_v2_compact`;
   - reward atual;
   - KPIs `0.6.6`;
2. correr MADDPG 15s contra baselines pos-fix:
   - `current`;
   - `noop_centered`;
   - `anti_saturation`;
   - `anti_saturation_warm_rbc_basic`;
   - eventualmente `anti_saturation_warm_rbc_smart`;
3. escolher uma receita curta;
4. so depois fazer tuning mais caro.

Resultado:

- reward corrigida para departure EV desconhecido;
- `current` ainda mostra saturacao excessiva;
- `anti_saturation` reduz saturacao;
- behavior cloning inicial a partir de `RBCBasicPolicy` reduz EV discharge mas
  sozinho nao resolve EV service;
- `ev_v2g_service_penalty` resolveu o problema principal de servico:
  `ev_service_v2g_guard_warm_rbc_basic` cumpriu
  `ev_departure_min_acceptable_feasible_rate = 1.0` na janela 15s com
  departures reais;
- `RewardWeightedMultiAgentReplayBuffer` esta implementado e a variante
  priorizada tambem cumpriu EV feasible, com custo menor que o guard uniforme
  nesta run curta;
- custo ainda nao bate `RBCBasic`/`RBCSmart`, portanto a fase 6F.1 focou
  otimizar custo mantendo o gate EV.

Hipoteses de tuning:

- reduzir `priority_fraction` no replay ponderado para nao sobre-amostrar
  eventos infeasible;
- aumentar episodios antes de mexer em LSTM;
- testar critic maior e actor moderado;
- testar LayerNorm;
- ajustar learning rates separados;
- rever peso de `ev_v2g_service_penalty` vs custo;
- explorar schedule de noise e warmup;
- TD3-style tricks se necessario;
- LSTM apenas se houver evidencia de problema de memoria temporal.

### Fase 6F.1 - Baixar Custo Mantendo EV

Status: concluida como primeira matriz de custo; problema ainda aberto.

Gate fixo:

- `ev_departure_min_acceptable_feasible_rate == 1.0`;
- `electrical_service_violation_total_kwh` perto de zero;
- sem regressao forte em `ev_departure_soc_deficit_mean`.

Run principal:

- `runs/benchmarks/phase6f1_066_maddpg_15s_cost_service_gpu`
- `5/5` jobs completos, `0` falhas;
- `pytest -q`: `173 passed`.

Resultado curto:

- `RBCSmart`: custo `60.402`, EV feasible min `1.0`;
- `service_guard_v2_warm_rbc_basic`: custo `72.263`, EV feasible min `1.0`;
- `service_guard_v2_prioritized_low_bc_decay_warm_rbc_basic`: custo `69.737`,
  EV feasible min `1.0`;
- `cost_balanced_v3_prioritized_low_bc_decay_warm_rbc_basic`: custo `80.917`,
  EV feasible min `1.0`.

Conclusao:

- melhor MADDPG atual em custo:
  `service_guard_v2_prioritized_low_bc_decay_warm_rbc_basic`;
- ainda nao bate `RBCSmart`;
- V3 reduziu deficit EV mas piorou custo, portanto nao e a receita principal;
- proxima fase deve atacar sobre-servico EV e storage caro, nao aumentar rede
  ou meter LSTM.

### Fase 6F.2 - Reward/Config V4 Para Custo

Status: primeira iteracao concluida.

Implementado/testado:

- `CostServiceCommunityBandRewardV4`;
- settlement comunitario aproximado como custo de treino;
- penalizacao de EV acima da banda do SOC pedido;
- replay `priority_fraction=0.15`;
- behavior cloning mais leve (`0.03 -> 0.005`);
- run: `runs/benchmarks/phase6f2_066_maddpg_15s_community_band_gpu`.

Resultado:

- melhor V4: `community_band_v4_prioritized_tiny_bc_decay_warm_rbc_basic`;
- custo `69.794`;
- EV feasible min `1.0`;
- ainda pior que `RBCSmart` (`60.402`);
- `within_tolerance_feasible=0.0`, logo a V4 cumpre minimo/target mas nao fica
  suficientemente perto do SOC pedido;
- storage continua demasiado ativo face ao `RBCSmart`.

Proxima fase 6F.3:

- reforcar disciplina de storage;
- calibrar penalizacao de over-service EV perto de departure;
- repetir a melhor candidata em 2-3 seeds antes de mexer em rede/LSTM.

### Fase 6F.10-6F.15 - Teacher Clone e Fine-Tune

Status: `f14` aceite como melhor marco; `f15` e `f16` rejeitadas.

Melhor marco atual:

- `community_feasible_service_v45_teacher_clone_ev_focus_rbc_smart`;
- run: `runs/benchmarks/phase6f14_ev_focus_cpu_3train_1eval_15s`;
- custo `87.460`;
- `EV feasible min = 1.0`;
- `EV within_tolerance_feasible = 0.6`;
- EV V2G medio `0.0`.

Tentativa rejeitada:

- `community_feasible_service_v45_teacher_clone_ev_focus_slow_finetune_rbc_smart`;
- run: `runs/benchmarks/phase6f15_slow_finetune_cpu_1train_1eval_15s`;
- custo `93.481`;
- `EV feasible min = 0.8`;
- `EV within_tolerance_feasible = 0.2`;
- EV V2G medio `0.0133`.

Conclusao:

- reabrir policy loss, mesmo com peso pequeno e warmup longo, ainda quebra o
  servico EV;
- nao correr `f15` mais tempo sem protecao adicional;
- event-aware BC/replay foi testado em `f16` e tambem quebrou o gate EV:
  - run: `runs/benchmarks/phase6f16_event_focus_cpu_1train_1eval_15s`;
  - custo `90.405`;
  - `EV feasible min = 0.8`;
  - `EV within_tolerance_feasible = 0.4`;
- nao correr mais variantes derivadas de `f16` sem rever a estrategia;
- proximo passo deve ser uma reuniao curta de decisao antes de mais treino
  longo.

### Fase 6H.3-6H.8 - Professor Explicito, Strong BC, V46 e V48

Status: 6H.8 aceite como melhor marco MADDPG atual; 6H.7 rejeitada.

Professor de aprendizagem:

- `RBCSmartPolicy` mais suave que o baseline final;
- EV sem V2G;
- carga EV menos agressiva;
- storage com peso baixo no BC.

Resultados 15s:

| Variante | Custo | EV feasible min | EV dentro tolerancia feasible | Erro abs EV | EV V2G |
|---|---:|---:|---:|---:|---:|
| `RBCSmart` learning teacher | 89.010 | 1.000 | 1.000 | 0.0300 | 0.0000 |
| `6H.3` professor explicito | 87.808 | 1.000 | 0.600 | 0.0654 | 0.0000 |
| `6H.4` event replay EV | 87.945 | 0.800 | 0.400 | 0.1002 | 0.0000 |
| `6H.5` strong BC EV | 85.172 | 1.000 | 0.800 | 0.0484 | 0.0000 |
| `6H.6` V46 precision | 83.761 | 1.000 | 0.800 | 0.0473 | 0.0000 |
| `6H.7` V47 over-service guard | 86.448 | 1.000 | 0.800 | 0.0540 | 0.0000 |
| `6H.8` V48 zero-band | 83.702 | 1.000 | 1.000 | 0.0330 | 0.0000 |

Conclusoes:

- strong BC EV melhora a precisao sem quebrar o gate feasible;
- event replay EV generico piora e fica rejeitado;
- 6H.6 bate o professor suave em custo, mas ainda nao bate em precisao de SOC;
- os dois failures min-acceptable crus do Building 15 sao target-infeasible
  segundo o KPI oficial;
- a V46 reduziu ligeiramente surplus/erro absoluto EV face a 6H.5;
- V47 apertou demasiado over-service via reward/BC e piorou custo/erro, por
  isso fica rejeitada;
- V48 manteve a reward V46 e reforcou moderadamente BC de EV zero/idle;
- V48 eliminou o erro feasible restante: todos os eventos EV target-feasible
  ficaram dentro da tolerancia;
- o critic ficou mais controlado com target clip `25`, mas ainda falta validar
  multi-seed/multi-dataset.

Proxima fase:

- promover V48 a baseline MADDPG candidato;
- repetir V48 em 2-3 seeds no 15s;
- correr V48 no dataset 2022;
- rever storage por beneficio real, porque o custo baixo ainda vem com uso
  relevante de bateria;
- so testar policy loss muito fraca quando V48 estiver robusta em seed/dataset.

## Fase 7 - Benchmark KPI Completo

Status: por fazer.

Comparar:

- `RandomPolicy`
- `NormalNoBatteryPolicy`
- `NormalPolicy`
- `RBCBasicPolicy`
- `RBCSmartPolicy`
- `MADDPG maddpg_v1`
- `MADDPG maddpg_v2_compact`

KPIs minimos:

- custo total liquidado da comunidade;
- import/export;
- violacoes de electrical service;
- EV departure minimo aceitavel, strict success, tolerancia simetrica, deficit,
  shortfall alem da tolerancia e erro absoluto;
- deferrable deadline success;
- bateria: throughput e violacoes SOC.

## Fase 8 - Inference

Status: por fazer.

Validar no repo de inference:

- manifest;
- nomes de observacoes;
- perfil de encoding;
- nomes e bounds de acoes;
- carregamento de 17 ONNX actors;
- predicao end-to-end com payload entity.
