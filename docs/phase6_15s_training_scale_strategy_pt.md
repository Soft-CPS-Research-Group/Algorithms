# Estrategia Para Escalar Treino 15s

Data: 2026-05-21.

Objetivo: definir como tornar viavel treino RL/MARL com dados de 15 segundos
sem depender apenas de "mais horas de GPU".

## Problema Numerico

Com os numeros da fase `softcpsrecsimulator==0.6.9`, apos caches, replay compacto e integracao:

- MADDPG V48 em GPU, 15s/update60, micro-run: mediana perfilada perto de
  `0.036-0.040 s/step`, com wall-clock de job perto de `0.05-0.06 s/step`
  quando inclui inicializacao/export;
- o custo recorrente do `step` baixa quando a reward declara
  `required_observation_names`, porque o simulador deixa de montar o payload
  completo para a reward;
- smoke local 15s/64 steps, `CostServiceCommunityPeakDeadlineRewardV52`:
  `reward_observations_time` desceu de cerca de `1.29 ms/step` para
  `0.32 ms/step`; `step` medio desceu de `8.62 ms` para `7.25 ms` no caso
  sem treino;
- configs longas devem manter BAU e series temporais desligadas durante treino,
  ligando exports so no episodio final quando for preciso auditar;
- baseline/simulador sem treino continua na ordem de dezenas de ms por step;
- 2022 horario full-year: `8760` steps;
- 15s full-year: `365 * 24 * 3600 / 15 = 2,102,400` steps.

Atualizacao `softcpsrecsimulator==1.0.2`, 2026-05-24:

```text
No dataset 2022 all-plus-EVs local, o MADDPG v3 operational passou a usar um
plano compilado de encoding e uma via direta entity->MADDPG:
  model_observation_encoding: ~79 ms/step -> ~6.4 ms/step
  entity_layout:              ~10.5 ms/step -> ~1.1 ms/step
  model_observation_encoding
    na via direta:            ~0.0 ms/step
  step perfilado mediano:     ~106 ms/step -> ~20.7 ms/step
  loop sem profiler:          ~27.5 ms/step
```

Se treinarmos MADDPG em todos os steps 15s, mesmo com estas otimizacoes:

- 1 episodio anual: cerca de `16h` no loop local sem profiler, antes de medir
  hardware remoto;
- 6 episodios anuais: cerca de `4 dias` por seed, ainda caro para matriz
  grande.

Isto nao e uma estrategia aceitavel.

## Diagnostico Do Bottleneck

O simulador/wrapper sem treino custa aproximadamente dezenas de ms por step,
dependendo do dataset, payload entity e export.

MADDPG ainda tem custo relevante no algoritmo, sobretudo em steps de update:

- replay sampling;
- transfer CPU -> GPU;
- critic target para todos os agentes;
- critic update por agente;
- actor update por agente;
- behavior cloning/teacher;
- target networks;
- metricas/diagnosticos.

Mas mesmo que o treino fique muito mais rapido, um ano completo 15s continua
caro se o simulador tiver de executar 2.1M chamadas Python `env.step`.

## Plano Em Camadas

### Camada 1 - Medir Melhor

Antes de otimizar as cegas, adicionar profiling configuravel:

- no wrapper:
  - tempo de `predict`;
  - tempo de conversao de acoes;
  - tempo de `env.step`;
  - tempo de entity encoding;
  - tempo de reward shaping;
  - tempo de `agent.update`;
  - tempo de logging/progress/checkpoint;
- no MADDPG:
  - tempo de `replay.push`;
  - tempo de `sample`;
  - tempo de transfer para GPU;
  - tempo de target critics;
  - tempo de critic update;
  - tempo de actor update;
  - tempo de BC extra;
  - tempo de metricas.

Isto deve ser logado de X em X steps, nao a cada step.

### Camada 2 - Ganhos Sem Mudar O Problema

Estas mudancas nao alteram o ambiente nem a semantica das acoes.

Config/treino:

- aumentar `steps_between_training_updates` para 5, 10, 20 ou 60 em 15s;
- aumentar `actor_update_interval`;
- manter `target_update_interval` menos frequente;
- usar `use_amp: true` em GPU, testando estabilidade;
- reduzir redes para variante 15s, por exemplo `256-128` ou `512-256`;
- reduzir batch se sampling/transfer dominar, ou aumentar batch se GPU estiver
  subutilizada;
- desligar diagnosticos detalhados em long-runs, mantendo summary raro.

Implementacao:

- replay buffer em ring buffer pre-alocado em vez de `deque` de listas de
  tensores. Estado: implementado como replay compacto prealocado;
- sampling prioritizado sem recalcular probabilidades sobre todo o buffer a cada
  update;
- cache de bounds/masks de acoes como tensores no device, em vez de recriar
  `torch.as_tensor` repetidamente;
- reduzir copias `float64 -> float32` onde possivel;
- opcional: `torch.compile` configuravel para actors/critics com shapes estaveis.

Impacto esperado:

- pode reduzir bastante o custo do MADDPG por step;
- nao resolve sozinho o custo de simular um ano 15s completo.

### Camada 3 - Abstracao Temporal De Controlo

Esta e provavelmente a peca principal para 15s.

Ideia: o simulador continua em 15s, mas o agente nao precisa decidir/aprender em
todos os 15s. Exemplo:

- decisao a cada 5 minutos: repetir a mesma acao por 20 steps internos;
- decisao a cada 15 minutos: repetir a mesma acao por 60 steps internos.

Isto nao e uma regra prescritiva para melhorar KPI; e uma decisao de escala
temporal do controlador. O agente continua a aprender por observacoes/reward,
mas com um intervalo de controlo realista.

Versao simples no Algorithms:

- `action_repeat: N`;
- o wrapper/runner aplica a mesma acao durante N steps;
- soma/mediana/agrega rewards;
- guarda uma transicao no replay por macro-step;
- progress/logging contam macro-step e env-step.

Limite desta versao:

- ainda chama `env.step` N vezes em Python;
- reduz treino/replay/logging, mas nao elimina custo do simulador.

Versao melhor no simulador:

- adicionar `env.step_many(action, n)` ou `action_repeat` nativo;
- o simulador executa N substeps internamente;
- devolve observacao final, reward agregado e eventos;
- evita grande parte do overhead Python/wrapper.

Isto e provavelmente necessario para full-year 15s ficar confortavel.

### Camada 4 - Treino Por Janelas Representativas

Nao devemos treinar sempre em ano completo 15s.

Treino recomendado:

- janelas EV-heavy;
- janelas com alta solar;
- janelas com baixa solar;
- janelas com precos altos/baixos;
- janelas com picos comunitarios;
- janelas com Building 15/fases/headroom criticos;
- janelas de transicao dia/noite.

Exemplos:

- 7 dias 15s por episodio;
- 14 dias 15s por episodio;
- amostragem de dias/semanas com curriculum;
- oversampling de eventos raros: EV departure, deferrable deadlines,
  grid/headroom stress.

Depois:

- avaliar em full-year 15s, possivelmente com action-repeat;
- nao usar full-year 15s como loop normal de treino em todas as experiencias.

### Camada 5 - Pretraining/Imitation

Para reduzir amostras RL:

- gerar trajetorias `RBCSmart`/`Normal` uma vez;
- treinar actor por behavior cloning/offline;
- usar RL apenas para fine-tuning;
- priorizar janelas onde RBCSmart falha ou custa demasiado.

Isto encaixa com o que ja existe:

- warm-start policy;
- BC actor loss;
- behavior action priority;
- replay com sinais de EV.

Mas deve ser tratado como inicializacao/aprendizagem, nao regra escondida no
wrapper.

### Camada 6 - Arquitetura Mais Eficiente

O MADDPG atual tem critic por agente. Com 17 agentes isto e caro:

- targets: 17 critics;
- update: 17 critics;
- actor loop: 17 critics.

Possiveis variantes:

- centralized critic multi-head:
  - uma backbone global;
  - uma saida Q por agente;
  - reduz forward/backward repetidos;
- shared actor trunk por tipo de agente/asset;
- critic maior mas unico, actors menores;
- MATD3 multi-head como alternativa se DDPG instavel;
- MASAC se exploracao/off-policy estocastica ajudar, mas so depois de termos
  perfil de custo.

Isto ja altera arquitetura, mas continua dentro da filosofia MARL.

### Camada 7 - Reduzir Tempo Ao Nivel Da IA

Estas mudancas reduzem tempo de treino/aprendizagem sem mudar o simulador:

1. Delayed actor mais agressivo:
   - manter critic a aprender mais frequentemente;
   - atualizar actor de 6 em 6, 10 em 10 ou 20 em 20 updates;
   - reduz custo de actor, que e caro com muitos agentes;
   - risco: policy melhora mais devagar se exagerarmos.

2. Critic multi-head/shared backbone:
   - em vez de 17 critics independentes;
   - uma backbone centralizada sobre obs+actions globais;
   - 17 heads Q, uma por agente;
   - deve reduzir forwards/backwards repetidos;
   - continua MADDPG-style e centralized critic.

3. Target critic multi-head:
   - mesmo racional para target networks;
   - `q_targets_next` passaria a vir de um forward multi-head;
   - grande ganho potencial nos steps de update.

4. Actor menor por tipo de ativo:
   - edificios com 1 acao nao precisam do mesmo actor que edificios com
     storage+EV+deferrable;
   - testar `128-64` em actors simples e manter critic maior;
   - ganho em predict/update e possivelmente menos overfit.

5. Pretraining offline/BC mais forte:
   - gerar dataset RBCSmart/Normal uma vez;
   - treinar actors sem chamar simulador;
   - depois fine-tune RL;
   - reduz horas gastas em exploracao fraca.

6. Curriculum temporal:
   - primeiro janelas curtas com EV events;
   - depois semanas;
   - depois avaliacao anual;
   - evita que a aprendizagem gaste horas em periodos pouco informativos.

7. Replay/event sampling mais inteligente:
   - nao e "escolher o que queremos" de forma prescritiva;
   - e garantir que eventos raros aparecem no batch:
     - EV departure;
     - deadline deferrable;
     - stress de headroom/fase;
     - picos comunitarios;
   - pode reduzir muitos episodios desperdicados.

8. `torch.compile` opcional:
   - testar apenas com shapes estaveis;
   - pode ajudar actor/critic;
   - tem overhead inicial e pode complicar debugging/checkpoints.

Prioridade recomendada:

1. delayed actor/update schedule;
2. replay/event sampling;
3. pretraining offline/BC;
4. critic multi-head;
5. actor por tipo de ativo;
6. `torch.compile`.

## Caminho Recomendado

Ordem pragmatica:

1. Resolver `stale_status` para long-runs remotos. Estado: implementado no
   `job_orchestrator_agent`; falta deploy da alteracao quando se quiser
   corrigir producao.
2. Adicionar profiling detalhado configuravel. Estado: implementado no wrapper
   e no MADDPG via `tracking.runtime_profiling_enabled`.
3. Criar configs 15s com:
   - `steps_between_training_updates` alto;
   - `actor_update_interval` maior;
   - `use_amp: true`;
   - redes menores;
   - diagnostics raros.
   Estado: configs em `configs/experiments/phase6_15s_scaling/`.
4. Implementar replay buffer pre-alocado/ring buffer. Estado: implementado.
5. Testar MADDPG V48 otimizado em 15s 1 dia e 7 dias.
6. Implementar `action_repeat`/macro-step no Algorithms.
7. Pedir/implementar no simulador `step_many` para action-repeat nativo.
8. Treinar por janelas representativas.
9. Avaliar em full-year 15s com control interval realista.
10. So depois comparar algoritmos alternativos em 15s.

## Criterio De Sucesso

Para 15s, um caminho e aceitavel se:

- permite treino de janelas relevantes em horas, nao dias;
- permite avaliacao full-year 15s em tempo razoavel;
- mantem EV service, limites e deferrables;
- mostra ganho comunitario/custo/pico contra baselines;
- nao depende de atalhos prescritivos no wrapper.

## Decisao Importante

Tentar treinar MADDPG em todos os steps 15s de um ano completo nao e realista
com a arquitetura atual.

O caminho cientificamente defensavel e:

- controlador em escala temporal realista;
- treino por janelas representativas/eventos;
- avaliacao full-year;
- otimizacoes de replay/rede/profiling;
- eventual suporte nativo do simulador para action-repeat em bloco.
