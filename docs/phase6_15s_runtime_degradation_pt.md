# Auditoria De Degradacao Runtime 15s

Data: 2026-05-21.

Objetivo: perceber se o treino fica progressivamente mais lento com mais
steps, e se ha sinal de bottleneck de memoria em runs longas.

## Run Local 1 Dia 15s

Config temporaria:

- dataset: `citylearn_three_phase_electrical_service_demo_15s_parquet`;
- steps: `5760` (`1` dia a `15s`);
- algoritmo: `MADDPG`, receita v48/update60;
- profiling: a cada `256` steps;
- MLflow: desligado;
- progress: desligado para nao contaminar medicao;
- GPU: RTX 4080 Laptop.

Resultado wall-clock:

- tempo total: `4m46.90s`;
- media wall-clock: `0.0498 s/step`;
- max RSS do processo: `2.44 GB`;
- output do job: `40 MB`.

Nota: o wall-clock inclui inicializacao e export final. As metricas
`Runtime/*` medem os steps perfilados dentro do loop.

## Tendencia Ao Longo Do Episodio

Comparacao por terços dos pontos perfilados:

| Bloco | Steps | `step_perf` medio | `step_perf` mediano | `env.step` medio | `entity layout` medio | `agent_update` medio |
|---|---:|---:|---:|---:|---:|---:|
| Inicio | 256-2048 | `0.0378s` | `0.0384s` | `0.0108s` | `0.0030s` | `0.0199s` |
| Meio | 2304-4096 | `0.0496s` | `0.0397s` | `0.0112s` | `0.0031s` | `0.0300s` |
| Fim | 4352-5632 | `0.0396s` | `0.0395s` | `0.0112s` | `0.0031s` | `0.0202s` |

O bloco do meio tem media maior por causa de um step de treino perfilado com
`actor/critic update`; a mediana fica estavel.

Deltas entre fim e inicio:

- `Runtime/step_perf_seconds`: `+0.0018s`;
- `Runtime/env_step_seconds`: `+0.0004s`;
- `Runtime/observation_encoding_seconds`: `+0.00004s`;
- `Runtime/agent_update_seconds`: `+0.00023s`;
- GPU PyTorch allocated/reserved: `0 MB` de aumento.

Conclusao: nesta escala de 1 dia 15s, nao ha degradacao temporal relevante.
O custo por step fica essencialmente estavel.

## Memoria Observada

Durante a run:

- GPU PyTorch allocated: `325 MB`, constante;
- GPU PyTorch reserved: `368 MB`, constante;
- RAM do sistema subiu cerca de `2 p.p.`, mas isto e uma metrica global da
  maquina e nao permite concluir fuga de memoria no processo;
- `/usr/bin/time -v` reportou max RSS do processo: `2.44 GB`.

Isto indica que a GPU nao e o gargalo de memoria nesta run curta. O risco
principal esta no replay em CPU quando a run fica longa.

## Micro-Benchmark Do Replay Antes Do Compacto

Foi feito um benchmark sintetico com as dimensoes reais da run:

- `17` agentes;
- soma de observacoes codificadas: `1466`;
- soma de acoes: `26`;
- `RewardWeightedMultiAgentReplayBuffer`;
- capacidade configurada: `150000`;
- batch: `128`.

Payload numerico bruto por transicao:

- cerca de `3002` floats;
- `~11.7 KB/transicao`;
- `~1.72 GB` para `150000` transicoes, sem overhead Python/Torch.

Memoria real medida ao fazer push de transicoes sinteticas:

| Entradas | RSS |
|---:|---:|
| 1 | `374 MB` |
| 1k | `451 MB` |
| 5k | `758 MB` |
| 10k | `1142 MB` |
| 15k | `1526 MB` |
| 20k | `1910 MB` |
| 25k | `2294 MB` |

Estimativa grosseira:

- crescimento observado: `~76-77 KB/transicao`;
- replay cheio a `150000`: pode acrescentar `~11.5 GB` de RSS;
- com simulador/modelos/export, esperar algo na ordem de `12-14 GB` por
  processo nesta receita.

Esta estimativa e consistente com o facto de o replay atual guardar muitos
tensores pequenos por transicao. O payload numerico e menor; o overhead de
objetos `torch.Tensor` e listas e que pesa.

## Replay Compacto Implementado

Foi implementado replay compacto prealocado em `algorithms/utils/replay_buffer.py`:

- `MultiAgentReplayBuffer` mantem o mesmo contrato publico:
  - `push`;
  - `sample`;
  - `sample_with_behavior_actions`;
  - `get_state`;
  - `set_state`;
- `RewardWeightedMultiAgentReplayBuffer` mantem sampling ponderado e prioridades;
- checkpoints antigos com `buffer` de transicoes continuam suportados;
- o retorno para os agentes continua a ser lista de tensores por agente;
- a semantica de `done`, rewards, behavior actions e transicoes conjuntas foi
  preservada.

Estrutura nova:

- arrays NumPy `float32` prealocados por agente:
  - `states[agent]`;
  - `actions[agent]`;
  - `behavior_actions[agent]`;
  - `next_states[agent]`;
- arrays conjuntos:
  - `rewards`;
  - `dones`;
  - `priorities`.

### Memoria Depois Do Replay Compacto

Mesmo benchmark sintetico, `capacity=150000`, dimensoes reais:

| Entradas | RSS antes | RSS depois |
|---:|---:|---:|
| 1 | `374 MB` | `376 MB` |
| 1k | `451 MB` | `387 MB` |
| 5k | `758 MB` | `432 MB` |
| 10k | `1142 MB` | `488 MB` |
| 15k | `1526 MB` | `545 MB` |
| 20k | `1910 MB` | `602 MB` |
| 25k | `2294 MB` | `658 MB` |

Leitura:

- aos `25k`, reduziu cerca de `1.6 GB`;
- o crescimento ficou perto do payload real, em vez de crescer com overhead de
  milhoes de objetos Python/Torch;
- extrapolando para `150k`, o replay deve ficar na ordem de poucos GB, nao
  `>10 GB`.

Tradeoff:

- `push` sintetico subiu de cerca de `267 us` para `388 us`;
- em troca, a RAM fica muito mais controlada;
- no loop real, `replay_push` continua sub-ms e nao e o bottleneck principal.

### Sampling Depois Do Replay Compacto

| Entradas | `sample_with_behavior_actions` antes | depois | `_sample_indices` antes | depois |
|---:|---:|---:|---:|---:|
| 1k | `2.77 ms` | `0.55 ms` | `0.07 ms` | `0.05 ms` |
| 5k | `3.06 ms` | `0.74 ms` | `0.19 ms` | `0.10 ms` |
| 10k | `3.30 ms` | `0.80 ms` | `0.34 ms` | `0.16 ms` |
| 25k | `3.66 ms` | `0.99 ms` | `0.80 ms` | `0.34 ms` |

Conclusao:

- sampling ficou cerca de `3-5x` mais rapido;
- o custo O(N) das probabilidades no replay ponderado continua la, mas menor;
- se o buffer chegar a `150k`, ainda pode valer a pena sum-tree/Fenwick tree.

### Micro-Run Real Depois Do Replay Compacto

400 steps 15s, update60:

- wall-clock total: `22.80s`, praticamente igual a antes;
- `replay_push`: cerca de `0.00057s`;
- `replay_sample`: cerca de `0.00086s`;
- `env.step`: cerca de `0.0102s`;
- `entity layout`: cerca de `0.0030s`;
- actor/critic continuam a dominar quando ha update real.

Leitura:

- a mudanca nao acelera muito runs curtas;
- a mudanca e sobretudo para tornar runs longas viaveis em RAM;
- tambem reduz o custo de sampling quando o replay fica maior.

## Tempo De Sampling

Sampling sintetico:

| Entradas | `sample_with_behavior_actions` medio | `_sample_indices` medio |
|---:|---:|---:|
| 1k | `2.77 ms` | `0.07 ms` |
| 5k | `3.06 ms` | `0.19 ms` |
| 10k | `3.30 ms` | `0.34 ms` |
| 25k | `3.66 ms` | `0.80 ms` |

Conclusao:

- o sampling cresce com o tamanho do replay, mas ainda nao e dramatico a
  `25k`;
- a parte ponderada recalcula pesos/probabilidades sobre todo o buffer, logo e
  O(N);
- a `150k`, isto pode chegar a varios ms por update, mas o maior risco e
  memoria, nao compute.

## Conclusao

Nao ha evidencia de que o simulador ou encoding fiquem progressivamente mais
lentos dentro de um dia 15s.

Ha evidencia de que o replay atual pode tornar-se o principal bottleneck de
memoria em runs longas:

- cresce linearmente;
- guarda muitos tensores pequenos;
- capacidade `150000` pode exigir mais de `10 GB` so para replay;
- GPU memory fica estavel porque o replay esta em CPU.

## Proximas Otimizacoes Possiveis

1. Se o replay ponderado continuar caro:
   - Fenwick tree / sum-tree para weighted sampling;
   - mesma distribuicao conceptual, menor custo por sample;
   - mais complexo, deixar para depois do replay compacto.

2. Configurar capacidade por escala:
   - para full-year 15s, `capacity=150000` e plausivel, mas caro;
   - testar `50000`, `75000`, `100000` antes de assumir `150000`.

3. Otimizacoes ao nivel do algoritmo:
   - delayed actor update mais agressivo;
   - critic update por minibatch mais eficiente;
   - shared critic backbone ou critic multi-head;
   - reduzir actor/critic width para 15s se nao perder KPI;
   - `torch.compile` opcional nos actors/critics;
   - pretraining/BC offline antes de interagir com o simulador.

4. Otimizacoes de escala temporal:
   - action repeat/macro-step;
   - `step_many` no simulador;
   - janelas representativas de treino;
   - avaliacao anual completa apenas para candidatos.
