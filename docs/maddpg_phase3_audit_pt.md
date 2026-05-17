# Auditoria Fase 3 - MADDPG

Data: 2026-05-13.

Objetivo: auditar a implementacao atual do MADDPG antes de mexer em reward,
logs ou tuning. Esta fase nao tenta melhorar KPIs; tenta perceber se o
algoritmo esta correto, onde ha risco tecnico e que experiencias fazem sentido
na Fase 3.5.

## Fontes Verificadas

- `algorithms/agents/maddpg_agent.py`
- `algorithms/utils/replay_buffer.py`
- `algorithms/utils/networks.py`
- `utils/wrapper_citylearn.py`
- `configs/templates/maddpg/maddpg_local.yaml`
- `runs/training_contracts/pypi051_maddpg_profiles/matrix_summary.csv`
- `runs/maddpg_diagnostics/phase2_building1_15s_maddpg_v2_compact_3x256/summary.json`
- `runs/maddpg_diagnostics/phase2_building1_2_15s_maddpg_v2_compact_3x256/summary.json`

## Conclusao Curta

O esqueleto MADDPG esta vivo e alinhado no essencial:

- ha um actor por agente/building;
- os actors recebem apenas a observacao local encoded do respetivo agente;
- cada critic recebe observacoes e acoes concatenadas de todos os agentes;
- o replay buffer preserva transicoes conjuntas entre agentes;
- target networks existem e fazem soft update;
- as acoes sao escaladas para os bounds reais do ambiente;
- o caso controlado da Fase 2 mostrou updates, alteracao de pesos e melhoria de
  reward.

Nao encontrei um erro estrutural do tipo "critic nao e centralizado" ou
"agentes nao sao realmente independentes". Encontrei, no entanto, riscos fortes
para aprendizagem no caso completo:

- acoes one-sided inicializam perto do meio do intervalo, nao perto de no-op;
- exploracao inicial uniforme pode ser demasiado agressiva para EVs,
  deferrables e baterias;
- critic loss e calculada agregando todos os critics, o que reduz a escala do
  gradiente de cada critic quando ha muitos agentes;
- nao ha reward normalization;
- logs atuais ainda nao mostram Q-values, action saturation ou gradient norms;
- `PrioritizedReplayBuffer` esta registado mas nao e compativel com o MADDPG
  atual;
- checkpoint/resume nao guarda estado completo de exploracao.

## Auditoria Por Tema

### 1. Actor Output e Action Scaling

Estado: parcialmente bom, com risco de semantica de acao.

O actor termina em `tanh`, logo produz valores em `[-1, 1]`. O MADDPG depois
escala esses valores para os bounds reais do `action_space`:

```text
scaled = low + 0.5 * (raw_action + 1.0) * (high - low)
```

Isto e usado em:

- predicao deterministica;
- predicao com ruido;
- target actions;
- actor loss;
- export ONNX via `ActionScaledActor`.

Ponto positivo: o treino e a inferencia usam a mesma transformacao de bounds.
Isto evita o erro classico de treinar em `[-1, 1]` e servir noutra escala.

Risco principal: em acoes com bounds `[0, 1]`, um actor inicial com output perto
de `0.0` produz acao perto de `0.5`, nao perto de `0.0`. Para deferrables e
alguns modos de carga, isto pode significar uma politica inicial que ja atua
bastante, mesmo antes de aprender. Em acoes `[-1, 1]`, output `0.0` e
naturalmente no-op; em `[0, 1]`, nao e.

Sinal observado na Fase 2: depois do warm-up, varias acoes aproximam-se ou
batem em `1.0`, `-1.0` ou `0.0`. Isto pode ser aprendizagem legitima, mas tambem
pode ser efeito de reward/action scaling/exploracao.

Decisao para Fase 3.5:

- testar inicializacao no-op-aware dos actors para acoes one-sided;
- comparar contra a inicializacao atual;
- medir action saturation por acao antes de mexer em reward.

### 1.1. Acao dos Deferrables

Estado: risco de contrato mitigado no simulador `0.5.1`; fica risco de
exploracao/aprendizagem.

No simulador `softcpsrecsimulator==0.5.1`, a acao
`deferrable_appliance_*`:

- tem bounds `[0, 1]`;
- e interpretada como comando pratico ON/OFF de inicio, nao como potencia
  continua;
- so arranca se houver ciclo pendente e se o ciclo puder comecar no time step
  atual;
- usa `trigger_threshold=0.5` por defeito;
- arranca apenas quando `action_value > trigger_threshold`;
- trata valores invalidos (`nan`, `inf`) como OFF;
- depois de arrancar, o load profile completo fica agendado nos time steps
  futuros;
- durante o ciclo running, a acao seguinte nao modula a potencia desse ciclo.

O problema original do `0.5.0` era `trigger_threshold=0.0`, que fazia quase
qualquer valor positivo arrancar o ciclo. Isso fica corrigido no contrato
`0.5.1`.

Consequencia pratica ainda relevante para MADDPG:

- exploracao uniforme em `[0, 1]` ainda produz ON em cerca de metade das
  amostras quando `can_start=True`;
- actor inicial em acoes `[0, 1]` continua a produzir valores perto de `0.5`,
  ou seja, mesmo junto ao threshold;
- o algoritmo pode aprender uma politica indecisa se muitas saidas ficarem perto
  de `0.5`, especialmente com ruido de exploracao.

Observacoes disponiveis para aprender isto:

- `pending`;
- `running`;
- `can_start`;
- `urgency_ratio`;
- `slack_ratio`;
- `hours_until_latest_start_24h`;
- `hours_until_deadline_24h`;
- `priority`;
- `must_run`;
- `remaining_duration_steps_day_ratio`;
- `remaining_energy_kwh_cycle_ratio`.

Portanto, do lado das observacoes o perfil `maddpg_v2_compact` tem contexto
suficiente. O risco maior agora esta em como a exploracao e a policy continua
lidam com o threshold `0.5`.

Decisao para Fase 3.5:

- manter a correcao no simulador, nao no wrapper;
- deixar o MADDPG aprender a saida continua que cruza o threshold;
- testar inicializacao no-op-aware e exploracao no-op-centered para gerar
  exemplos claros de OFF e ON;
- so testar discretizacao/cabeca binaria dentro do agente depois de haver logs
  que mostrem indecisao em torno de `0.5`;
- adicionar/confirmar teste de simulador/dataset para `0.0`, abaixo do
  threshold, acima do threshold e `nan`/`inf`.

### 2. Critic Centralizado

Estado: alinhado com MADDPG.

Cada critic e construido com:

- `global_state_size = sum(observation_dimensions)`;
- `global_action_size = sum(action_dimensions)`.

No update, o codigo concatena:

- `global_state = torch.cat(states, dim=1)`;
- `global_next_state = torch.cat(next_states, dim=1)`;
- `global_actions = torch.cat(actions_all, dim=1)`;
- `global_next_actions = torch.cat(target_actor_actions, dim=1)`.

Depois calcula um Q por critic/agente. Isto esta alinhado com a ideia de
centralized training, decentralized execution.

Risco: a loss dos critics e uma MSE agregada sobre todos os critics:

```text
critic_loss = mse_loss(q_expected, q_targets).mean()
```

Com 17 agentes, o gradiente efetivo de cada critic fica escalado pela media
global. Isto nao e necessariamente bug matematico, mas muda a escala do update
versus treinar cada critic com a sua propria MSE. Pode explicar a necessidade de
learning rate de critic mais alto e pode afetar estabilidade.

Decisao para Fase 3.5:

- testar critic update por agente, mantendo a mesma arquitetura;
- alternativa mais pequena: usar soma/media controlada por config e comparar;
- medir TD error por agente.

### 3. Replay Buffer

Estado: bom para `MultiAgentReplayBuffer`.

O buffer guarda transicoes conjuntas:

```text
(state_tensors, action_tensors, reward_tensors, next_state_tensors, done_tensor)
```

Na amostragem, preserva alinhamento temporal entre agentes. Isto e importante:
o critic centralizado precisa de estados/acoes da mesma transicao, nao batches
independentes por agente.

Shapes esperadas:

- `states[i]`: `[batch_size, obs_dim_i]`
- `actions[i]`: `[batch_size, action_dim_i]`
- `rewards[i]`: `[batch_size, 1]`
- `dones`: `[num_agents, batch_size, 1]`

Ponto a corrigir: `PrioritizedReplayBuffer` esta no registry do MADDPG, mas a
interface dele nao bate com a interface multi-agent usada no update atual. Se
alguem trocar a config para `PrioritizedReplayBuffer`, deve falhar ou comportar-
se mal. Portanto, neste momento, o buffer suportado de facto e
`MultiAgentReplayBuffer`.

Decisao para Fase 3.5:

- ou remover/desativar `PrioritizedReplayBuffer` da config MADDPG;
- ou implementar uma versao prioritized multi-agent com a mesma interface de
  `MultiAgentReplayBuffer`;
- adicionar teste que impeca escolher buffers incompativeis silenciosamente.

### 4. Done, Terminated e Truncated

Estado: correto para o contrato atual.

O `done` armazenado e:

```text
done = terminated or truncated
```

Existe teste dedicado para isto. Como o ambiente usa fim de episodio global,
um `done` partilhado por todos os agentes faz sentido.

Risco baixo: se no futuro houver agentes com episodios/ativos independentes por
asset, este contrato teria de mudar. Para os datasets atuais, esta aceitavel.

### 5. Warm-up e Exploracao Inicial

Estado: funcional, mas com risco de desenho.

Ha duas variaveis:

- `random_exploration_steps`: ate quando `predict()` devolve acoes uniformes
  aleatorias nos bounds;
- `end_initial_exploration_time_step`: quando o wrapper passa
  `initial_exploration_done=True` e o update pode realmente treinar.

Nos templates atuais estao iguais (`960`), o que e coerente. No entanto, o codigo
nao obriga estes valores a manterem uma relacao segura. Se forem diferentes,
pode haver:

- acoes gaussianas sem update;
- ou updates enquanto as acoes ainda sao totalmente random.

Outro risco: exploracao uniforme em todo o action space nao respeita semantica.
Isto nao e "regra no wrapper", mas e uma escolha forte do agente. Para V2G,
deferrables e baterias, uniform random pode criar muito comportamento destrutivo
e encher o buffer com transicoes pouco informativas.

Decisao para Fase 3.5:

- testar exploracao inicial centrada em no-op + ruido, comparada com uniform;
- testar schedules de sigma por tipo de acao/bound;
- validar que `random_exploration_steps <= end_initial_exploration_time_step`
  ou documentar claramente a excecao.

### 6. Target Networks e Soft Update

Estado: alinhado.

Targets sao inicializados com os mesmos pesos dos modelos locais e atualizados
por soft update:

```text
target = (1 - tau) * target + tau * local
```

O update acontece quando o wrapper passa `update_target_step=True`. Nos templates
locais, `target_update_interval` e pequeno (`2`), com `tau=0.001`.

Risco: `tau=0.001` com update interval e training cadence pode ser demasiado
lento ou adequado dependendo da escala da reward. Sem Q-value logs nao da para
afirmar.

Decisao para Fase 3.5:

- testar `tau` e `target_update_interval` so depois de termos logs de Q/TD
  error ou pelo menos diagnostics comparaveis.

### 7. Reward Normalization

Estado: ausente.

O MADDPG usa a reward diretamente na target:

```text
q_target = reward + gamma * q_next * (1 - done)
```

Nao ha normalizacao, clipping ou running stats no agente. A reward atual pode
misturar custo, penalizacoes de rede, EV deficit, bateria e deferrables. Mesmo
que a funcao de reward esteja correta, a escala pode dominar o critic e tornar
o actor instavel.

Decisao para Fase 3.5/Fase 5:

- separar duas coisas:
  - logs de componentes da reward para perceber dominancia;
  - reward normalization/clipping no agente como experiencia controlada.

### 8. Gradient Clipping

Estado: existe.

O codigo usa:

- `clip_grad_norm_` nos critics com `max_norm=1.0`;
- `clip_grad_norm_` em cada actor com `max_norm=1.0`.

Ponto positivo: isto protege contra explosao de gradientes.

Risco: nao ha logs de gradient norm antes/depois do clipping. Se o clipping
estiver sempre ativo, podemos estar a esconder problema de reward scale ou LR.

Decisao para Fase 4:

- logar gradient norm por actor/critic;
- logar percentagem de updates em que houve clipping relevante.

### 9. Logs Atuais

Estado: insuficiente para diagnostico fino.

O MADDPG loga:

- actor loss;
- critic loss;
- training step time;
- alguns logs textuais.

Isto ajuda, mas nao chega para perceber porque nao aprende no caso completo.
Faltam:

- Q mean/std/min/max;
- TD error por agente;
- action mean/std/min/max;
- action saturation por acao;
- replay buffer size;
- sigma/exploration phase;
- reward components;
- gradient norms.

Decisao: isto e a Fase 4, mas a Fase 3 confirma que nao devemos tentar tuning
sem estes sinais.

### 10. Agentes Com Poucas Acoes

Estado: funcional, com risco de eficiencia/estabilidade.

Nos datasets principais temos 17 agentes e 26 acoes totais. Muitos agentes tem
apenas 1 acao, mas cada critic recebe o input global completo.

Para `maddpg_v2_compact`:

- dataset 15s: global obs `1466`, global actions `26`;
- dataset 2022: global obs `1187`, global actions `26`.

Com as redes default `[1024, 512, 256]`:

- 15s: actors totalizam cerca de `12.7M` parametros;
- 15s: cada critic tem cerca de `2.19M` parametros;
- 15s: 17 critics totalizam cerca de `37.1M` parametros;
- total aproximado do modelo 15s: quase `50M` parametros.

Isto nao e automaticamente errado, mas e pesado para treinar em CPU e pode ser
sample-inefficient. Um agente com 1 acao tem actor grande e critic global grande,
mesmo que o controlo local seja simples.

Decisao para Fase 3.5:

- testar redes menores no diagnostico controlado e no caso 17 agentes curto;
- testar critic maior que actor, mas nao necessariamente ambos enormes;
- considerar parameter sharing apenas depois de medir se os agentes sao
  suficientemente semelhantes.

### 11. Checkpoint e Resume

Estado: parcial.

O checkpoint guarda:

- actors;
- critics;
- targets;
- optimizers;
- replay buffer.

Nao guarda explicitamente:

- `sigma`;
- `exploration_step`;
- RNG states;
- global wrapper step;
- estatisticas futuras de normalizacao, se forem adicionadas.

Isto significa que resume de treino e util para pesos/buffer, mas nao e um
resume completo do processo RL. Se retomarmos uma corrida longa, a exploracao
pode recomecar com estado diferente.

Decisao para Fase 3.5:

- adicionar estado de exploracao/checkpoint completo antes de confiar em resumes
  longos;
- ou declarar resume como "warm-start/fine-tune", nao reproducao exata.

## Veredito Fase 3

O MADDPG nao parece partido na arquitetura principal. A principal hipotese passa
a ser:

1. o stack aprende em pequeno porque a implementacao base funciona;
2. no caso completo, a combinacao de action semantics, exploracao, reward scale,
   tamanho das redes, multi-agent non-stationarity e falta de logs deve estar a
   impedir aprendizagem KPI robusta.

Portanto, a Fase 3.5 deve ser uma fase de experiencias pequenas e isoladas,
nao uma mudanca grande de reward/rede de uma vez.

## Proposta Para Fase 3.5

Plano vivo detalhado:

- `docs/maddpg_phase35_experiments_pt.md`

Prioridade alta:

1. Validar em rollout que os deferrables `0.5.1` geram starts abaixo/acima do
   threshold como esperado e logar start delay.
2. Comparar critic loss agregada atual contra critic update por agente.
3. Testar inicializacao no-op-aware para acoes `[0, 1]`.
4. Testar exploracao inicial no-op-centered em vez de uniform full-range.
5. Marcar `PrioritizedReplayBuffer` como nao suportado para MADDPG ou implementar
   a interface multi-agent.
6. Melhorar checkpoint para guardar `sigma` e `exploration_step`.

Prioridade media:

1. Testar redes menores: actor `[256, 128]`, critic `[512, 256]`.
2. Testar LayerNorm apos as camadas lineares.
3. Testar reward normalization no agente, mantendo reward function igual.
4. Testar `tau` e target update interval depois de termos Q/TD logs.

Prioridade baixa agora:

1. LSTM/GRU. Pode fazer sentido, mas so depois de provar que o problema e falta
   de memoria e nao reward/exploracao/buffer.
2. Parameter sharing. Pode ajudar, mas muda bastante o desenho e deve vir depois
   dos testes basicos.
3. TD3 completo. E promissor, mas primeiro devemos confirmar os problemas mais
   simples.
