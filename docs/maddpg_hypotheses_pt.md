# Hipoteses MADDPG

Data: 2026-05-17.

Objetivo: manter uma memoria clara das hipoteses tecnicas que queremos testar
para melhorar o MADDPG, sem misturar isto com regras no wrapper. O principio
mantem-se: o comportamento deve vir de observacoes, reward, arquitetura,
exploracao e treino.

## Base Atual

Ja temos uma primeira matriz solida:

- `maddpg_v1` vs `maddpg_v2_compact`;
- exploracao uniforme vs `noop_centered`;
- actor normal vs `noop_actor_initialization`;
- treino normal vs warm-start com `NormalPolicy` ou `RBCBasicPolicy`;
- critic update `joint_mean` vs `per_agent`;
- reward normalization;
- reward EV schedule-aware;
- actor/critic sizes configuraveis;
- action diagnostics, reward components e diagnosticos internos MADDPG.

Esta matriz cobre os riscos principais: escala de observacoes, semantica de
acoes one-sided, exploracao inicial, credit assignment, estabilidade do critic e
sinal da reward.

## Estado Atual

Os baselines ja foram corrigidos/calibrados o suficiente para servirem como
escada de comparacao. A Fase 6F voltou ao MADDPG com simulador `0.6.6`, corrigiu
a reward para departure EV desconhecido e provou que o loop de treino aprende
no caso pequeno.

A Fase 6F.1 correu em GPU:

- `runs/benchmarks/phase6f1_066_maddpg_15s_cost_service_gpu`
- `5/5` jobs completos, `0` falhas;
- `pytest -q`: `173 passed`.

Conclusao atual:

- o MADDPG ja consegue manter `ev_departure_min_acceptable_feasible_rate = 1.0`;
- ainda nao bate `RBCSmart` em custo;
- melhor candidato atual:
  `service_guard_v2_prioritized_low_bc_decay_warm_rbc_basic`
  (`69.737` vs `RBCSmart` `60.402`);
- `cost_balanced_v3_prioritized_low_bc_decay_warm_rbc_basic` reduziu deficit EV,
  mas ficou caro (`80.917`), por isso nao e a receita principal.

Implicacao: resultados MADDPG antigos antes de `0.6.6` e antes da reward EV
corrigida devem ser lidos como historicos. O foco agora e reduzir custo sem
perder o gate EV, nao mexer ja em LSTM ou arquiteturas grandes.

A Fase 6F.2 implementou e testou `CostServiceCommunityBandRewardV4`:

- run: `runs/benchmarks/phase6f2_066_maddpg_15s_community_band_gpu`;
- `5/5` jobs completos;
- melhor V4: `community_band_v4_prioritized_tiny_bc_decay_warm_rbc_basic`;
- custo `69.794`, melhor que a V2 equivalente nesta run (`75.022`), mas ainda
  pior que `RBCSmart` (`60.402`);
- EV feasible min manteve `1.0`;
- `within_tolerance_feasible=0.0` nas V4, indicando que cumprir minimo/target
  ainda nao e o mesmo que sair perto do SOC pedido;
- storage continua a ser o maior desvio face ao `RBCSmart`.

Nota apos auditoria de bateria em 2026-05-18:

- nao ha evidencia de consumo parado da bateria nos datasets principais
  (`loss_coefficient = 0.0` e SOC idle estavel);
- `RBCSmart` antigo estava demasiado conservador e deixava storage sempre idle
  nas janelas analisadas;
- as rewards nomeadas V4/V41/V42 nao herdavam os `reward_function_kwargs` dos
  templates, logo a V42 usava `battery_soc_min = 0.05` e penalizava bateria
  vazia/parada como violacao;
- foi criada `CostServiceCommunityBatteryValueRewardV43`, que mantem servico EV
  da V42 mas usa limites fisicos de bateria (`0..1`) apenas como fallback; se o
  simulador expuser `soc_min_ratio`/`soc_max_ratio`, esses limites continuam a
  mandar;
- proxima comparacao MADDPG deve usar `RBCSmart` corrigido e uma receita V43,
  nao os resultados antigos contra RBCSmart idle.

Nota apos Fase 6F.3 em 2026-05-18:

- foi implementado teacher behavior cloning separado da acao executada no replay;
- `actor_behavior_cloning_source: warm_start_policy` usa a policy de warm-start
  deterministica como alvo, em vez de clonar a propria acao ruidosa executada;
- a run GPU longa parcial com teacher BC reduziu EV negative fraction e EV
  service penalty face a V43 anterior, mas nao resolveu o caso Building 15 e
  voltou a mostrar instabilidade do critic;
- a prioridade imediata passa a ser estabilizar critic e auditar eventos EV
  especificos, nao aumentar cegamente a rede ou passar ja para LSTM.

Nota apos V45/actor pretraining em 2026-05-19:

- phase-out do professor nao basta, porque `predict(deterministic=True)` usa o
  actor puro e nao a policy professor;
- o problema observado passou a ser: o rollout de treino pode estar razoavel,
  mas o actor final ainda nao imitou/aprendeu a politica que cumpre EV service;
- foi adicionada uma schedule configuravel para a policy loss do actor, para
  deixar behavior cloning dominar no inicio e so depois dar peso total ao
  gradiente RL;
- a nova hipotese a testar e `V45 + RBCSmart teacher + actor pretraining`;
- sucesso tecnico minimo: avaliacao deterministica final deve manter EV service
  no Building 15 sem depender do professor durante `predict()`.
- primeira tentativa degradou quando a policy loss efetiva chegou perto de
  `0.84`; isto sugere rampa mais lenta/persistente, nao abandono da hipotese.

## KPIs EV 0.6.5

Com `softcpsrecsimulator==0.6.5`, a hipotese principal de EV service deve ser
avaliada com dois niveis:

- `ev_departure_min_acceptable_feasible_rate`: qualidade justa do controlador,
  excluindo departures fisicamente impossiveis;
- `ev_departure_min_acceptable_rate`: experiencia real do utilizador/cenario,
  incluindo schedules impossiveis.

Para diagnostico, acompanhar sempre:

- `ev_departure_min_acceptable_infeasible_count`;
- `ev_departure_target_infeasible_count`;
- `ev_departure_within_tolerance_infeasible_count`;
- `ev_departure_soc_deficit_mean`;
- `ev_departure_shortfall_beyond_tolerance_mean`.

Isto evita otimizar o MADDPG contra falhas que nenhuma policy conseguiria
resolver por potencia/janela/SOC inicial.

Nota apos `0.6.6`: estes KPIs continuam a ser a base. O gate justo primario para
MADDPG e `ev_departure_min_acceptable_feasible_rate`.

## Hipotese Principal

Antes de mudar para algoritmos novos, devemos provar:

1. se o MADDPG aprende em caso pequeno;
2. se falha apenas quando aumenta para 17 agentes;
3. se a falha vem de reward, exploracao, replay, arquitetura ou escala.

Se nao aprende no caso pequeno, o problema provavelmente e implementacao,
reward ou exploracao. Se aprende no pequeno mas nao no completo, o problema
provavelmente e escala multi-agent, observacoes, replay ou credit assignment.

## Replay Buffer

### 1. MultiAgentReplayBuffer Atual

Estado: adequado para MADDPG base.

O buffer atual preserva transicoes conjuntas entre agentes:

- observacoes de todos os agentes no mesmo timestep;
- acoes de todos os agentes no mesmo timestep;
- rewards por agente;
- next observations;
- done/truncated.

Isto e essencial para centralized critic. Nao devemos trocar por um buffer
single-agent.

### 2. Event-aware Replay

Prioridade: alta se o MADDPG aprender no pequeno mas falhar no caso completo.

Ideia: continuar a guardar transicoes conjuntas, mas amostrar mais vezes passos
com eventos raros/importantes:

- EV perto da saida;
- EV missed departure;
- deferrable deadline/urgency;
- grid/phase violation;
- picos comunitarios;
- bateria muito perto de limites;
- acoes saturadas.

Motivo: o replay uniforme pode diluir eventos decisivos em episodios longos. A
reward EV e deferrable tem eventos que podem ser raros; se o critic quase nao os
ve, aprende devagar ou aprende uma politica indiferente.

Risco: se exagerarmos no peso dos eventos, o agente pode otimizar apenas casos
raros e piorar custo normal. Deve ser configuravel e comparado contra replay
uniforme.

Estado apos 6F.3: ja existe replay ponderado por reward total, mas ainda nao
existe replay verdadeiramente event-aware por componente. O proximo passo
provavel e priorizar janelas EV/deferrable/grid com base em componentes de
reward ou diagnosticos de estado. Isto deve passar apenas informacao de treino
ao algoritmo, nao regras prescritivas ao wrapper.

### 3. Multi-agent Prioritized Replay

Prioridade: media.

So faz sentido se for multi-agent de verdade:

- prioridade por transicao conjunta;
- TD-error agregado ou por agente;
- importance-sampling weights;
- preservacao do alinhamento temporal entre agentes.

O `PrioritizedReplayBuffer` single-agent nao serve para o MADDPG atual.

### 4. N-step Returns

Prioridade: media/baixa.

Pode ajudar quando a consequence vem atrasada, como EV departure e deferrable
deadline. Mas em multi-agent com time steps diferentes e episodios longos deve
ser implementado com cuidado.

## Exploracao

Hipoteses ja configuraveis:

- `uniform_full_range`;
- `noop_centered`;
- warm-start com policy;
- `noop_actor_initialization`.
- teacher behavior cloning a partir de `warm_start_policy`.

O que medir:

- action saturation;
- EV charge/V2G sign ratio;
- deferrable ON/OFF e start delay;
- storage charge/discharge/idle;
- primeiros episodios antes/depois de warm-up.

Hipotese principal: `noop_centered` ou warm-start deve produzir um buffer inicial
menos destrutivo do que exploracao uniforme total.

Hipotese apos 6F.3: clonar a policy professor deterministica e melhor do que
clonar a acao executada com ruido. Isto ajuda EV service, mas nao substitui uma
reward correta nem resolve critic instavel.

Hipotese apos 6F.7: o professor deve tambem dominar a fase inicial da loss do
actor, nao apenas a acao executada. Caso contrario, o actor pode aprender contra
um critic ainda fraco e perder a politica de servico EV quando a avaliacao fica
deterministica.

Hipotese refinada: para episodios longos e 17 agentes, a policy loss nao deve
chegar perto de peso total antes de a policy conseguir reproduzir servico EV
critico. Testar uma variante slow com BC EV mais forte e policy loss limitada/
rampada lentamente.

## Critic

Hipoteses:

- `joint_mean`: comportamento MADDPG atual;
- `per_agent`: update separado por critic;
- critic maior que actor;
- LayerNorm no critic/actor.

Motivo: com 17 agentes, a media conjunta pode esconder instabilidade de agentes
especificos e reduzir o gradiente efetivo de cada critic.

## Redes

Primeira vaga:

- MLP menor;
- MLP maior;
- critic maior que actor;
- LayerNorm.

Deixar para mais tarde:

- LSTM/GRU, apenas se houver evidencia de falta de memoria temporal;
- attention critic/GNN critic, apenas numa segunda geracao;
- actor partilhado com building-id/agent-id embedding, se as casas parecerem
  suficientemente semelhantes.

## Reward

Estado atual:

- reward hibrida local + comunitaria;
- custo/import;
- grid/power violation;
- bateria com limites observados/fisicos;
- throughput de bateria;
- EV schedule-aware;
- EV missed departure;
- deferrable deadline/urgency;
- comunidade import/pico configuravel.

Hipoteses a calibrar:

- `ev_connected_deficit_penalty`, que agora fica ligado nos templates para dar
  sinal denso enquanto o EV esta conectado;
- peso de `ev_schedule_deficit_penalty`;
- peso de `ev_departure_missed_penalty`;
- peso de `ev_v2g_service_penalty`, que penaliza V2G quando o EV ainda esta
  abaixo do target de servico;
- peso de `community_import_penalty`;
- peso de `community_peak_import_penalty`;
- peso de `battery_throughput_penalty`;
- eventual suavizacao de acoes, se os logs mostrarem chattering.

Hipotese nova para 6F.2:

- uma V4 deve preservar penalizacao forte de deficit EV factivel;
- deve reduzir incentivo a sobre-servir EV muito acima do minimo aceitavel;
- deve tornar custo/throughput de storage mais visivel, porque as variantes
  6F.1 carregam storage/EV de forma cara para cumprir servico;
- deve manter `ev_v2g_service_penalty`, mas talvez so com peso alto quando ha
  risco real de falhar minimo aceitavel.
- deve alinhar melhor a reward com o KPI economico comunitario: a energia
  local entre membros e comprada a preco reduzido no simulador, enquanto export
  residual para rede pode valer zero. A reward V4 deve aproximar este settlement
  em vez de olhar apenas para import local bruto.

## Algoritmos Derivados

So depois da primeira matriz:

- MATD3 / TD3-style: twin critics, delayed actor update, target policy
  smoothing;
- behavior cloning previo com RBC/Normal, seguido de fine-tune RL;
- action head especifica para deferrables se a policy continua ficar indecisa
  perto do threshold;
- replay ponderado por reward/evento, ja implementado como
  `RewardWeightedMultiAgentReplayBuffer`;
- event-aware replay explicito por componentes de reward como melhoria futura,
  se o replay ponderado por reward for demasiado grosseiro.

## Receita Anti-Saturacao Atual

Implementacao disponivel para teste em `run_phase6a_benchmark.py`:

- `anti_saturation`: no-op actor initialization, `per_agent` critic update,
  delayed actor update, target policy smoothing e regularizacao L2/saturacao;
- `anti_saturation_warm_rbc_basic`: igual, mas a exploracao inicial usa
  `RBCBasicPolicy` revisto;
- `ev_service_v2g_guard_warm_rbc_basic`: adiciona behavior cloning inicial e
  reward anti-V2G quando o EV esta abaixo do target de servico;
- `ev_service_v2g_guard_prioritized_warm_rbc_basic`: igual, mas troca o replay
  uniforme por `RewardWeightedMultiAgentReplayBuffer`.
- `service_guard_v2_warm_rbc_basic`: usa o perfil nomeado
  `CostServiceGuardRewardV2`;
- `service_guard_v2_prioritized_low_bc_decay_warm_rbc_basic`: usa replay
  ponderado por penalizacao negativa, prioridade baixa e decay de behavior
  cloning;
- `cost_balanced_v3_prioritized_low_bc_decay_warm_rbc_basic`: usa
  `CostServiceCostBalancedRewardV3`; reduziu deficit EV, mas ficou caro.
- `community_band_v4_warm_rbc_basic`: usa
  `CostServiceCommunityBandRewardV4`, treinando custo de settlement comunitario
  aproximado e penalizando sobre-servico EV acima da banda do target;
- `community_band_v4_prioritized_tiny_bc_decay_warm_rbc_basic`: igual a V4,
  mas com replay ponderado mais baixo (`priority_fraction=0.15`) e behavior
  cloning mais leve (`0.03 -> 0.005`) para reduzir a dependencia do RBC.

Estas opcoes continuam dentro do MADDPG/DDPG-style. Ainda nao sao MATD3 completo:
nao ha twin critics nem minimo entre critics. O objetivo e testar primeiro se a
deriva para extremos diminui e se o EV service melhora antes de aumentar a
complexidade.

## Ordem Recomendada

1. Partir de `service_guard_v2_prioritized_low_bc_decay_warm_rbc_basic`, porque
   foi a melhor MADDPG 6F.1 em custo mantendo EV feasible.
2. Criar/testar uma reward/config V4 orientada a custo sob gate EV:
   settlement comunitario, menos sobre-servico, mais disciplina de storage,
   V2G guard contextual.
3. Testar `priority_fraction` menor (`0.10-0.20`) e BC decay mais rapido.
4. Se uma variante ficar perto de `RBCSmart`, repetir 2-3 seeds no 15s.
5. So depois testar LayerNorm/critic maior.
6. Se custo continuar pior que RBC mantendo EV service, testar TD3-style.
7. So depois considerar LSTM ou attention critic.

## Atualizacao Apos 6F.14

Resultado novo:

- `community_feasible_service_v45_teacher_clone_ev_focus_rbc_smart`;
- 3 episodios de treino + 1 avaliacao deterministica;
- custo `87.460`, melhor que `RBCSmart` (`97.241`) no 15s;
- `EV feasible min = 1.0`, portanto nao perdeu o gate principal;
- `EV within tolerance feasible = 0.6`, ainda pior que `RBCSmart` (`1.0`);
- EV V2G medio `0.0`.

Hipotese que ganhou forca:

- o MADDPG consegue aprender uma politica util se primeiro for estabilizado por
  BC forte do professor, sobretudo nas dimensoes EV;
- o problema nao era falta de capacidade bruta da rede, era instabilidade do
  fine-tune RL antes de o actor ter uma politica de servico EV minimamente
  segura.

Hipotese que fica rejeitada por agora:

- pressionar mais a banda EV com `ev_band`, `ev_balanced` ou
  `ev_guarded_band` logo no clone de 1 episodio baixa custo, mas reabre falhas
  em EV factiveis. Estas variantes nao devem ser base principal.

Proxima hipotese a testar:

- partir do `f14` e introduzir policy loss pequena e lenta, com BC EV ainda
  alto, para tentar melhorar custo/settlement sem baixar
  `EV feasible min = 1.0`;
- se o `within_tolerance_feasible` nao subir, a melhoria deve ir para BC/replay
  orientado a eventos de departure, nao para LSTM/rede maior.

## Atualizacao Apos 6F.15

Teste feito:

- `community_feasible_service_v45_teacher_clone_ev_focus_slow_finetune_rbc_smart`;
- 1 episodio de treino + 1 avaliacao deterministica;
- custo `93.481`;
- `EV feasible min = 0.8`;
- `EV within tolerance feasible = 0.2`;
- EV V2G medio voltou a ser positivo (`0.0133`).

Hipotese rejeitada:

- policy loss pequena e lenta nao e suficiente para proteger EV service quando
  o actor ainda depende muito do professor.

Hipotese que fica ativa:

- o proximo ganho deve vir de replay/BC orientado a eventos EV criticos:
  janelas com deficit, alta urgencia ou potencia media requerida ate departure;
- so depois de melhorar a imitacao nesses estados faz sentido reabrir policy
  loss RL.

## Atualizacao Apos 6F.16

Teste feito:

- `community_feasible_service_v45_teacher_clone_ev_focus_event_rbc_smart`;
- policy loss a `0.0`;
- BC EV mais forte;
- replay priorizado por acoes EV positivas e por observacoes EV criticas;
- 1 episodio de treino + 1 avaliacao deterministica.

Resultado:

- custo `90.405`;
- `EV feasible min = 0.8`;
- `EV within tolerance feasible = 0.4`;
- EV V2G medio `0.0053`.

Hipotese rejeitada:

- dar mais peso generico aos eventos EV no replay nao resolve sozinho.

Leitura:

- o problema parece estar menos em "nao ver eventos EV" e mais em estabilidade
  da politica/qualidade do target nas casas criticas, sobretudo Building 15 e
  Building 4;
- `f14` continua a ser a melhor base, porque foi a unica que manteve
  `EV feasible min = 1.0` com custo abaixo do `RBCSmart`.

## Atualizacao Apos 6H.3

Teste feito:

- `community_feasible_service_v45_teacher_clone_ev_learning_teacher_rbc_smart`;
- professor `RBCSmart` explicito, menos agressivo que o baseline final;
- BC EV forte, BC storage fraco, policy loss desligada;
- 3 episodios de treino + 1 avaliacao deterministica.

Resultado:

- custo `87.808`;
- `EV feasible min = 1.0`;
- `EV within_tolerance_feasible = 0.6`;
- EV deficit medio `0.0298`;
- EV surplus medio `0.0356`;
- EV erro absoluto medio `0.0654`;
- EV V2G medio `0.0`.

Hipotese reforcada:

- separar "professor de baseline" de "professor de aprendizagem" e correto.
  O professor de aprendizagem deve ser mais suave e mais facil de imitar, nao
  necessariamente o melhor baseline heuristico final.

Hipotese parcialmente validada:

- aumentar o peso de zero-target/idle EV reduziu sobre-servico face ao f14
  antigo, mas nao chega para bater `RBCSmart` em precisao EV.

Hipotese rejeitada por agora:

- simplesmente alinhar o professor com o `RBCSmart` final nao melhora o MADDPG;
  a run alinhada ficou pior em custo e em banda EV.

Proxima hipotese:

- antes de reabrir policy loss, melhorar imitacao/priority apenas nas janelas
  EV realmente criticas, sobretudo Building 15, e rever a parte de storage para
  evitar descarga final sem beneficio claro.

## Atualizacao Apos 6H.4

Teste feito:

- `community_feasible_service_v45_teacher_clone_ev_learning_teacher_event_rbc_smart`;
- replay/BC mais focado em eventos EV;
- policy loss desligada;
- 3 episodios de treino + 1 avaliacao deterministica.

Resultado:

- custo `87.945`;
- `EV feasible min = 0.8`;
- `EV within_tolerance_feasible = 0.4`;
- EV deficit medio `0.0570`;
- EV surplus medio `0.0431`;
- EV erro absoluto medio `0.1002`;
- EV V2G medio `0.0`.

Hipotese rejeitada:

- prioridade generica por eventos EV nao e suficiente e pode piorar o problema.

Leitura:

- o replay event-aware provavelmente misturou eventos infeasible do Building 15
  com eventos feasible onde a politica ainda precisava de precisao;
- antes de voltar a usar event replay, e preciso distinguir:
  eventos target-infeasible, eventos feasible em deficit, eventos feasible em
  over-service e janelas normais.

## Atualizacao Apos 6H.5

Teste feito:

- mesma variante base da 6H.3:
  `community_feasible_service_v45_teacher_clone_ev_learning_teacher_rbc_smart`;
- 5 episodios de treino + 1 avaliacao deterministica;
- BC EV mais forte;
- maior peso em targets positivos e targets zero/idle;
- replay priorizado por acao EV do professor, sem event replay generico;
- policy loss desligada.

Resultado:

- custo `85.172`;
- `EV feasible min = 1.0`;
- `EV within_tolerance_feasible = 0.8`;
- EV deficit medio `0.0301`;
- EV surplus medio `0.0182`;
- EV erro absoluto medio `0.0484`;
- EV V2G medio `0.0`.

Hipotese validada:

- reforcar BC EV e targets zero melhora a precisao de SOC sem quebrar o gate
  feasible.

Hipotese nova:

- o MADDPG ja consegue bater o professor suave em custo, mas ainda nao reproduz
  a banda EV do professor; o foco deve ser precision tuning, nao mais custo.

Risco tecnico que ficou mais claro:

- o critic continua a saturar em valores negativos extremos perto/abaixo de
  `-35`. Isto sugere que eventos raros/infeasible continuam a dominar a escala
  de treino mesmo quando os KPIs oficiais os excluem do denominador feasible.

Proxima hipotese:

- criar uma variante de reward/replay que continue a reportar todos os eventos,
  mas reduza ou classifique separadamente o impacto de target-infeasible no
  treino do critic;
- manter policy loss desligada ate `EV within_tolerance_feasible` chegar a
  `1.0` no 15s;
- depois testar fine-tune RL lento apenas se o clone estiver estavel.

## Atualizacao V46 Implementada

Implementado:

- `CostServiceCommunityFeasiblePrecisionRewardV46`;
- variante:
  `community_feasible_precision_v46_teacher_clone_ev_learning_teacher_rbc_smart`;
- caps configuraveis:
  `ev_schedule_deficit_cap_soc` e
  `ev_departure_window_shortfall_cap_soc`;
- over-service mais forte (`ev_over_service_tolerance = 0.02`,
  `ev_over_service_penalty = 420`);
- critic target clip da variante a `25`.

Hipotese a testar:

- se a 6H.5 falhou apenas por excesso de carga feasible e por critic dominado
  por deficits unrecoverable, a V46 deve manter `EV feasible min = 1.0`,
  aumentar `EV within_tolerance_feasible` para perto de `1.0` e manter custo
  baixo.

## Atualizacao Apos 6H.6

Teste feito:

- `community_feasible_precision_v46_teacher_clone_ev_learning_teacher_rbc_smart`;
- 5 episodios de treino + 1 avaliacao deterministica;
- strong BC EV igual ao 6H.5;
- reward V46 com caps em deficits EV usados no treino;
- penalizacao mais forte de over-service EV;
- critic target clip a `25`;
- policy loss desligada.

Resultado:

- custo `83.761`;
- `EV feasible min = 1.0`;
- `EV within_tolerance_feasible = 0.8`;
- EV deficit medio `0.0304`;
- EV surplus medio `0.0169`;
- EV erro absoluto medio `0.0473`;
- EV V2G medio `0.0`.

Hipotese validada parcialmente:

- reduzir a pressao de deficits unrecoverable no critic e apertar over-service
  melhora custo e ligeiramente o erro EV sem quebrar o gate feasible.

Hipotese ainda por validar:

- V46 ainda nao chega a `EV within_tolerance_feasible = 1.0`;
- a falha feasible restante e over-service no `Building 5 / charger_5_1`, nao
  deficit;
- o proximo ajuste deve focar over-service feasible e nao voltar a aumentar
  carga EV de forma generica.

Risco tecnico:

- o ganho de custo continua ligado a uso relevante de storage. No episodio de
  avaliacao, V46 carregou `63.067 kWh`, descarregou `40.981 kWh` e fez
  `104.048 kWh` de throughput. E ligeiramente melhor que 6H.5, mas ainda
  precisa de validacao multi-seed/dataset antes de ser considerado policy final.

Proxima hipotese:

- manter V46 como novo baseline MADDPG;
- testar uma V47 pequena com penalizacao/BC focada em over-service feasible;
- so depois testar policy loss muito fraca, com BC EV forte e monitorizacao do
  gate EV em cada episodio.
