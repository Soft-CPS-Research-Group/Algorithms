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

Antes de testar mais hipoteses MADDPG, a prioridade e fechar baselines
confiaveis. Depois da auditoria `0.6.5`, descobrimos que os baselines EV
estavam a oscilar artificialmente por causa do uso de headroom residual como
limite de comando total do charger. O fix esta no Algorithms e ficou
documentado em:

- `docs/ev_departure_event_audit_065_pt.md`

A matriz pos-fix correu em:

- `docs/maddpg_phase6e_results_pt.md`

Implicacao: resultados MADDPG antigos devem ser lidos como historicos. A
comparacao justa deve ser refeita depois de calibrar `Normal`, `RBCBasic` e
`RBCSmart`, porque a Fase 6E mostrou que EV service esta forte mas storage e
deferrables ainda nao estao bons o suficiente para baselines finais.

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

O que medir:

- action saturation;
- EV charge/V2G sign ratio;
- deferrable ON/OFF e start delay;
- storage charge/discharge/idle;
- primeiros episodios antes/depois de warm-up.

Hipotese principal: `noop_centered` ou warm-start deve produzir um buffer inicial
menos destrutivo do que exploracao uniforme total.

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
- peso de `community_import_penalty`;
- peso de `community_peak_import_penalty`;
- peso de `battery_throughput_penalty`;
- eventual suavizacao de acoes, se os logs mostrarem chattering.

## Algoritmos Derivados

So depois da primeira matriz:

- MATD3 / TD3-style: twin critics, delayed actor update, target policy
  smoothing;
- behavior cloning previo com RBC/Normal, seguido de fine-tune RL;
- action head especifica para deferrables se a policy continua ficar indecisa
  perto do threshold;
- event-aware replay como proxima melhoria mais provavel.

## Receita Anti-Saturacao Atual

Implementacao disponivel para teste em `run_phase6a_benchmark.py`:

- `anti_saturation`: no-op actor initialization, `per_agent` critic update,
  delayed actor update, target policy smoothing e regularizacao L2/saturacao;
- `anti_saturation_warm_rbc_basic`: igual, mas a exploracao inicial usa
  `RBCBasicPolicy` revisto.

Estas opcoes continuam dentro do MADDPG/DDPG-style. Ainda nao sao MATD3 completo:
nao ha twin critics nem minimo entre critics. O objetivo e testar primeiro se a
deriva para extremos diminui e se o EV service melhora antes de aumentar a
complexidade.

## Ordem Recomendada

1. Calibrar baselines heuristicos na Fase 6E.1.
2. Repetir matriz de baselines pos-fix em 15s e 2022.
3. Confirmar ranking e coerencia fisica dos baselines.
4. Repetir diagnostico MADDPG pequeno com KPIs `0.6.5`.
5. Comparar matriz MADDPG base em caso pequeno.
6. Se aprender no pequeno, escalar gradualmente para 17 agentes.
7. Se falhar no completo, implementar event-aware replay.
8. Depois testar LayerNorm/critic maior.
9. So depois considerar MATD3, LSTM ou attention critic.
