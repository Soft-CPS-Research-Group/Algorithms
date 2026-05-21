# Perfil Local De Runtime 15s MADDPG

Data: 2026-05-21.

Objetivo: perceber onde se gasta tempo antes de lançar long-runs no Deucalion.
GPU local usada: RTX 4080 Laptop via PyTorch CUDA.

## Configs Usadas

Configs geradas em `runs/local_profiling/`:

- `phase6_15s_runtime_probe`: 1024 steps, update a cada 20 steps.
- `phase6_15s_runtime_probe_aligned`: 400 steps, profiling alinhado com update20.
- `phase6_15s_runtime_probe_per_step`: 40 steps, profiling a cada step.
- `phase6_15s_runtime_probe_update60`: 400 steps, update a cada 60 steps.

Estas runs servem para performance, nao para conclusoes de KPI.

## Resultado Principal

Antes da cache de encodings:

- 400 steps update20: `43.02s`, cerca de `0.1076 s/step`.
- 40 steps sem treino real: `4.05s`, cerca de `0.1013 s/step`.

Depois de cachear encodings entre `predict` e `update`:

- 400 steps update20: `23.15s`, cerca de `0.0579 s/step`.
- 40 steps sem treino real: `2.23s`, cerca de `0.0558 s/step`.

Depois de cachear tambem layout estatico do contrato entity:

- 400 steps update20: `16.93s`, cerca de `0.0423 s/step`.
- 40 steps sem treino real: `1.62s`, cerca de `0.0405 s/step`.

Depois das otimizacoes seguras adicionais:

- 400 steps update60, job completo com export: `22.81s`, cerca de `0.0570 s/step` wall-clock total.
- nos steps perfilados, mediana `Runtime/step_perf_seconds`: `0.0368s`.
- `action_prepare`: cerca de `0.00016s`.
- `replay_push`: cerca de `0.00031s`.
- `replay_sample`: cerca de `0.00283s`.

Ganho total: cerca de 61% na micro-run update20 face ao ponto inicial.

## Bottleneck Antes Da Cache

O profiling mostrou que `MADDPG.update()` em si era barato quando ainda nao
havia batch suficiente:

- `MADDPG/training_step_perf_seconds`: cerca de `0.0004s`.

Mas o wrapper media:

- `Runtime/agent_update_seconds`: cerca de `0.047s`.

Conclusao: o tempo estava a ser gasto antes de entrar realmente no treino,
sobretudo em reencoding das observacoes para passar ao agente.

## O Que Foi Corrigido

1. Cache de acoes teacher da warm-start policy:
   - evita recalcular `RBCSmartPolicy` no `update` quando o `predict` ja a
     calculou para o mesmo observation context.
   - ganho pequeno no total, mas remove duplicacao desnecessaria.

2. Scheduling de update/target update:
   - passou a usar `global_step`;
   - antes usava `self.time_step`, que nao era o contador local fiavel neste
     loop.

3. Cache de encodings:
   - evita codificar as mesmas observacoes no `predict` e voltar a codificar
     no `update`;
   - permite reutilizar a `next_obs` codificada no `predict` seguinte quando o
     objeto de observacao e o mesmo.

4. Cache de layout entity/encoder:
   - evita recriar encoders quando a topologia nao muda;
   - cacheia bounds/espaços/edges de assets por topologia;
   - cacheia nomes/espacos/origens das observacoes em topologia estatica;
   - cacheia o layout de nomes/filtros do encoder `maddpg_v2_compact`.

5. Cache de rotas de acao entity:
   - resolve nomes de acao para tabelas/linhas/colunas apenas quando a
     topologia ou os nomes de acao mudam;
   - remove parsing repetido de strings em todos os steps;
   - mantem exatamente o mesmo payload de acao para o simulador.

6. Replay buffer com lista circular:
   - substitui `deque` por lista circular para amostragem O(1) por indice;
   - preserva o contrato de checkpoints e `sample`;
   - ajuda sobretudo no replay ponderado quando o buffer fica grande.

7. Caches menores:
   - cache de bounds de acoes no wrapper;
   - cache da assinatura de `replay_buffer.push`, evitando `inspect.signature`
     em todos os steps.

## Perfil Depois Da Cache

40 steps sem treino real:

- total: `0.0558 s/step`;
- `predict`: cerca de `0.0026s`;
- `env.step`: cerca de `0.0103s`;
- entity layout/observation conversion apos step: cerca de `0.0160s`;
- `agent_update` sem treino real: cerca de `0.0239s`;
- `MADDPG.update` interno: cerca de `0.0004s`.

400 steps update20:

- total: `0.0579 s/step`;
- em steps de update, `agent_update` ainda sobe para cerca de `0.078s`;
- dentro do treino real:
  - actor update: mediana cerca de `0.0376s`;
  - critic update: mediana cerca de `0.0160s`;
  - replay sample: cerca de `0.0031s`;
  - target compute: cerca de `0.0036s`.

400 steps update60:

- total: `0.0561 s/step`;
- pouco melhor que update20 nesta janela.

Conclusao: depois da cache, reduzir update20 para update60 quase nao muda o
runtime total. O custo fixo por step 15s domina.

## Perfil Depois Do Cache Entity

40 steps sem treino real:

- total: `0.0405 s/step`;
- `predict`: cerca de `0.0026s`;
- `env.step`: cerca de `0.0111s`;
- entity layout apos step: cerca de `0.0031s`;
- `agent_update` sem treino real: cerca de `0.0203s`;
- `MADDPG.update` interno: cerca de `0.0005s`.

400 steps update20:

- total: `0.0423 s/step`;
- `predict`: cerca de `0.0038s` nos steps logados;
- `env.step`: cerca de `0.0107s`;
- entity layout apos step: cerca de `0.0031s`;
- em steps de update, `agent_update` mediana cerca de `0.088s`;
- dentro do treino real:
  - actor update: mediana cerca de `0.0419s`;
  - critic update: mediana cerca de `0.0167s`.

400 steps update60:

- total: `0.0407 s/step`;
- apenas ligeiramente melhor que update20 nesta janela;
- confirma que, depois das caches, reduzir a frequencia de update ajuda, mas
  nao e o maior salto enquanto o simulador continuar a executar todos os
  substeps 15s individualmente.

## Perfil Depois Das Otimizacoes Seguras Adicionais

400 steps update60, job completo:

- wall-clock total: `22.81s`, incluindo inicializacao/export;
- mediana dos steps perfilados: `0.0368s`;
- `predict`: cerca de `0.0021s`;
- `action_prepare`: cerca de `0.00016s`;
- `env.step`: cerca de `0.0103s`;
- entity layout apos step: cerca de `0.0031s`;
- `agent_update`: mediana cerca de `0.0201s`, maximo cerca de `0.0945s`
  quando ha treino real;
- `replay_push`: cerca de `0.00031s`;
- `replay_sample`: cerca de `0.00283s`;
- actor update no step de treino perfilado: cerca de `0.0401s`;
- critic update no step de treino perfilado: cerca de `0.0165s`.

Conclusao: estas otimizacoes deixam a parte de conversao de acoes e replay
mais limpa, mas nao mudam a conclusao principal: o custo fixo por step 15s e
os updates actor/critic continuam a dominar.

## Implicacoes Para 15s

Com `0.042 s/step`, um ano 15s tem cerca de `2,102,400` steps:

- 1 episodio anual: cerca de `24.7h`;
- 6 episodios anuais: cerca de `6.2 dias` por seed.

Isto ja e muito melhor que a estimativa anterior, mas continua pesado para
iteracao cientifica.

## Proximas Otimizacoes Provaveis

1. Action repeat / macro-step no Algorithms:
   - decidir a cada 5 ou 15 minutos;
   - reduzir replay/update/logging por fator 20 ou 60;
   - ainda chama `env.step` internamente se for implementado so no Algorithms.

2. `step_many` ou action-repeat nativo no simulador:
   - necessario para reduzir tambem overhead Python de `env.step`;
   - provavelmente indispensavel para full-year 15s confortavel.

3. Otimizar encoding:
   - precomputar transforms por feature;
   - reduzir loops Python por agente/feature;
   - evitar conversoes repetidas `float64/float32`;
   - idealmente fundir entity layout + feature encoding num caminho vetorizado.

4. Replay em macro-step ou replay lazy:
   - guardar menos transicoes quando o controlador so decide de N em N steps;
   - ou guardar raw/compact e codificar por batch no sample, se compensar.

5. Arquitetura MADDPG:
   - actor update e critic update ainda dominam nos steps de treino;
   - para muitos agentes, avaliar critic multi-head/shared backbone.

## Decisao Pratica

Para long-runs 15s, nao vale a pena apenas trocar update20 por update60 e
esperar grande ganho. O proximo salto real vem de reduzir custo fixo por step:
action-repeat/macro-step e/ou suporte nativo no simulador.
