# Fase 3.5 - Plano de Experiencias MADDPG

Data: 2026-05-13.

Objetivo: transformar a auditoria da Fase 3 em melhorias pequenas,
controladas e comparaveis. A implementacao base das variantes esta pronta; a
avaliacao completa fica para a matriz de benchmarks, agora que reward e logging
ja estao observaveis.

Regras:

- nao meter regras prescritivas no wrapper;
- comportamento aprendido deve vir do agente, da reward ou de configs;
- cada experiencia deve ser isolada;
- cada resultado deve indicar dataset, perfil de observacoes, reward, seed e
  baselines usados;
- primeiro medir sinais internos, depois decidir mudancas maiores.

## Estado Apos Simulador 0.5.1

O risco original dos deferrables mudou.

No `softcpsrecsimulator==0.5.0`, a acao deferrable arrancava com qualquer valor
positivo por causa de `trigger_threshold=0.0`. Isso podia enviesar qualquer
agente continuo para "start now".

No `softcpsrecsimulator==0.5.1`, o contrato esta melhor:

- action space continua `[0, 1]`;
- comando pratico e ON/OFF;
- `action <= 0.5` e OFF;
- `action > 0.5` e ON quando `can_start=True`;
- `nan`/`inf` contam como OFF;
- depois de iniciar, o ciclo segue o perfil fixo e a acao ja nao modula potencia.

Portanto, nao precisamos de alterar wrapper nem RBCs. O risco que sobra e do
MADDPG: policy continua e exploracao podem ficar demasiado perto do threshold
`0.5`, ou gerar muitos starts aleatorios durante warm-up.

## Ordem Recomendada

1. Melhorar logs/diagnostico minimo para actions e starts. Feito.
2. Medir comportamento atual com `0.5.1` e reward observavel. Feito em smoke.
3. Testar inicializacao no-op-aware na matriz final.
4. Testar exploracao no-op-centered na matriz final.
5. Testar critic update por agente na matriz final.
6. Bloquear buffers incompativeis com MADDPG. Feito.
7. Melhorar checkpoint de estado RL. Feito.
8. Reward normalization esta implementada e ligada nos templates MADDPG.
9. So depois testar redes menores/maiores e LayerNorm.

## Implementado Nesta Iteracao

Defaults continuam equivalentes ao comportamento anterior:

- `initial_exploration_strategy: uniform_full_range`;
- `noop_actor_initialization: false`;
- `critic_update_mode: joint_mean`;
- `replay_buffer.class: MultiAgentReplayBuffer`.

Novas opcoes configuraveis em `algorithm.exploration.params`:

- `initial_exploration_strategy`: `uniform_full_range`, `noop_centered` ou
  `policy`;
- `warm_start_policy`: `NormalPolicy`, `NormalNoBatteryPolicy`,
  `RBCBasicPolicy`, `RBCSmartPolicy`, `RuleBasedPolicy` ou `RandomPolicy`;
- `warm_start_policy_hyperparameters`: parametros opcionais para essa policy;
- `warm_start_policy_noise_scale`: ruido opcional em cima da policy;
- `noop_noise_scale`: escala do ruido em torno da action no-op;
- `deferrable_on_probability`: probabilidade de amostrar ON em deferrables
  durante `noop_centered`;
- `deferrable_trigger_threshold`: threshold usado pela exploracao, default
  `0.5`;
- `noop_actor_initialization`: inicializar actor perto do no-op real;
- `noop_actor_initialization_epsilon`: distancia minima ao bound para evitar
  saturacao total;
- `critic_update_mode`: `joint_mean` ou `per_agent`.

Tambem ficou feito:

- `PrioritizedReplayBuffer` passa a falhar cedo em MADDPG, porque e
  single-agent;
- checkpoint MADDPG passa a guardar/restaurar `sigma`, `exploration_step` e RNG
  states;
- o wrapper passa raw/encoded observation context para agentes que implementem
  `set_observation_context`, permitindo warm-start com RBC sem mudar a regra de
  controlo no wrapper.
- logs compactos de action diagnostics no wrapper, configuraveis por
  `tracking.action_diagnostics_enabled` e `tracking.action_diagnostics_detail`;
- logs internos MADDPG para critic/actor/Q-values/gradients/buffer/sigma,
  configuraveis por `tracking.training_diagnostics_enabled` e
  `tracking.training_diagnostics_detail`;
- quando MLflow esta desligado, os diagnosticos de step/action continuam no
  `logs/metrics.jsonl`; com MLflow ativo, vao para MLflow.

Smokes de validacao apos a reward base:

- `runs/maddpg_diagnostics/phase5_reward_schedule_ev_building1_15s_smoke`
- `runs/maddpg_diagnostics/phase5_reward_schedule_ev_building1_2022_hourly_smoke`

Ambos confirmaram updates de MADDPG, noise de exploracao ativo e alteracao dos
pesos do actor. Estes smokes nao sao conclusao de KPI; servem para confirmar
que as variantes podem ser comparadas sem o pipeline estar partido.

Nota EV reward:

- a reward EV nao deve obrigar "carrega agora";
- o sinal principal e deadline-aware: calcula o SOC minimo necessario agora para
  ainda chegar ao SOC requerido ate a saida, assumindo potencia maxima restante;
- enquanto o EV ainda consegue cumprir a saida, nao ha penalizacao de schedule
  so por estar abaixo do required SOC final;
- quando esse minimo deixa de ser cumprido, a penalizacao cresce a medida que a
  saida se aproxima;
- missed departure continua a ser penalizacao forte de falha de servico.

Exemplo de warm-start com RBC:

```yaml
algorithm:
  exploration:
    params:
      initial_exploration_strategy: policy
      warm_start_policy: RBCBasicPolicy
      warm_start_policy_deterministic: true
      warm_start_policy_noise_scale: 0.0
```

Exemplo de exploracao no-op-centered:

```yaml
algorithm:
  exploration:
    params:
      initial_exploration_strategy: noop_centered
      noop_noise_scale: 0.12
      deferrable_on_probability: 0.2
      deferrable_trigger_threshold: 0.5
```

## Experiencia 3.5.1 - Logs Minimos de Acoes

Prioridade: maxima.

Problema:

- sem histogramas de acoes, nao sabemos se o MADDPG esta saturado, indeciso ou
  simplesmente a explorar mal;
- sem start delay dos deferrables, nao sabemos se a policy aprende a esperar.

Implementar/logar:

- action mean/std/min/max por agente e por action name;
- percentagem perto do low/high;
- percentagem de deferrable actions `<= 0.5` e `> 0.5`;
- start delay dos deferrables quando `pending + can_start`;
- EV charge/V2G sign ratio;
- storage charge/discharge/idle ratio.

Status: implementado como metricas compactas, amostradas pelo mesmo intervalo
de step metrics.

Configs:

```yaml
tracking:
  action_diagnostics_enabled: true
  action_diagnostics_detail: per_action  # ou summary
  action_saturation_tolerance: 0.01
  action_idle_tolerance: 0.02
  training_diagnostics_enabled: true
  training_diagnostics_detail: per_agent # ou summary
```

Metricas novas principais:

- `Action/all_mean`, `Action/all_std`, `Action/near_low_fraction`,
  `Action/near_high_fraction`;
- `Action/storage_positive_fraction`, `Action/storage_negative_fraction`,
  `Action/storage_idle_fraction`;
- `Action/ev_positive_fraction`, `Action/ev_negative_fraction`,
  `Action/ev_idle_fraction`;
- `Action/deferrable_on_fraction`, `Action/deferrable_off_fraction`;
- `Deferrable/pending_can_start_count`,
  `Deferrable/start_when_available_count`,
  `Deferrable/start_delay_steps_mean`;
- `MADDPG/average_critic_loss`, `MADDPG/average_actor_loss`,
  `MADDPG/q_expected_*`, `MADDPG/q_target_*`,
  `MADDPG/critic_grad_norm_*`, `MADDPG/actor_grad_norm_*`,
  `MADDPG/replay_buffer_size`, `MADDPG/exploration_sigma`.

Criterio de sucesso:

- conseguimos responder, por action name, se a policy esta a usar a acao ou a
  bater num bound;
- conseguimos ver se deferrables arrancam sempre no primeiro slot, nunca
  arrancam, ou variam com observacoes.

## Experiencia 3.5.2 - Inicializacao No-op-aware

Prioridade: alta.

Problema:

- actor inicial com output `0.0` vira action `0.5` em bounds `[0, 1]`;
- para deferrables isto fica em cima do threshold;
- para outras acoes one-sided isto tambem nao e uma politica inicial passiva.

Plano:

- implementado: opcao config no MADDPG para inicializar a ultima camada do actor para
  aproximar a action inicial do no-op real;
- para `[0, 1]`, inicializar perto de `0.0`, mas sem saturar totalmente;
- aplicar depois de `attach_environment`, porque precisa de bounds reais;
- nao aplicar quando se carrega checkpoint.

Comparar:

- baseline atual;
- no-op-aware.

Metricas:

- primeira action deterministica por action name;
- action saturation;
- start delay deferrables;
- reward no diagnostico 1/2 buildings;
- rollout curto 17 agentes.

## Experiencia 3.5.3 - Exploracao No-op-centered

Prioridade: alta.

Problema:

- `random_exploration_steps` usa uniform full-range;
- isso cria muito comportamento destrutivo em BESS/EV/V2G/deferrables;
- com deferrables `0.5.1`, uniform `[0, 1]` ainda liga cerca de metade das vezes
  quando `can_start=True`.

Plano:

- implementado: manter `uniform_full_range` como modo atual;
- implementado: adicionar `noop_centered` por config;
- amostrar ruido em torno do no-op real de cada action bound;
- para deferrables, garantir amostras claras de OFF e ON sem forcar sempre ON;
- manter tudo dentro do agente.

Comparar:

- buffer inicial com uniform;
- buffer inicial com no-op-centered.

Metricas:

- action distribution;
- start delay;
- critic loss inicial;
- reward nos primeiros episodios.

## Experiencia 3.5.4 - Critic Update Por Agente

Prioridade: alta.

Problema:

- hoje a critic loss e uma media global sobre todos os critics;
- com 17 agentes, isto pode reduzir a escala efetiva do gradiente por critic;
- tambem pode esconder um agente instavel atras da media.

Plano:

- implementado: adicionar config `critic_update_mode`;
- implementado: manter `joint_mean` como default atual;
- implementado: permitir `per_agent`, com loss e optimizer step por critic;
- manter critic centralizado com obs/actions globais.

Metricas:

- critic loss por agente;
- TD error por agente;
- actor loss por agente;
- estabilidade em 1/2 buildings e depois 17 agents curto.

## Experiencia 3.5.5 - Buffer Suportado de Verdade

Prioridade: alta para robustez.

Problema:

- `MultiAgentReplayBuffer` esta alinhado com MADDPG;
- `PrioritizedReplayBuffer` existe no config/registry, mas a interface e
  single-agent e nao serve o update atual.

Plano curto:

- implementado: bloquear `PrioritizedReplayBuffer` em MADDPG com erro claro.

Plano medio:

- implementar `MultiAgentPrioritizedReplayBuffer` se houver necessidade real.

Teste:

- config invalida falha cedo;
- buffer multi-agent preserva alinhamento temporal de obs/actions/rewards.

## Experiencia 3.5.6 - Checkpoint Completo

Prioridade: media.

Problema:

- checkpoint guarda modelos, targets, optimizers e replay;
- nao guarda `sigma`, `exploration_step` nem RNG states.

Plano:

- implementado: guardar/restaurar estado de exploracao;
- implementado: guardar RNG states de Python, NumPy e Torch;
- documentar diferenca entre resume exato e warm-start.

## Experiencia 3.5.7 - Reward Normalization

Prioridade: alta; implementado depois dos logs/componentes de reward.

Problema:

- reward pode misturar custo, rede, EV deficit, bateria e deferrables em escalas
  diferentes;
- o critic recebe a escala diretamente.

Plano:

- implementado: running reward normalization opcional no MADDPG;
- manter reward function igual;
- logar reward raw e reward usada no target.

Configuracao:

```yaml
algorithm:
  exploration:
    params:
      reward_normalization: true
      reward_normalization_clip: 10.0
      reward_normalization_epsilon: 1.0e-8
```

O estado da normalizacao fica no checkpoint.

## Experiencia 3.5.8 - Redes Menores e LayerNorm

Prioridade: media.

Problema:

- com `maddpg_v2_compact`, o caso 15s completo fica perto de 50M parametros;
- pode ser pesado e sample-inefficient.

Plano:

- testar actor `[256, 128]`;
- testar critic `[512, 256]`;
- testar LayerNorm como variante, nao default imediato;
- manter critic maior que actor.

## Fora de Prioridade Agora

- LSTM/GRU: so depois de provar que o problema e memoria temporal.
- Parameter sharing: interessante, mas muda bastante o desenho.
- TD3 completo: promissor, mas deve vir depois de resolver logs, exploracao e
  critic update.
