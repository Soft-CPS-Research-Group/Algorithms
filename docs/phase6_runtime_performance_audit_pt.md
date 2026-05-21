# Auditoria de Performance Runtime

Data: 2026-05-21.

Objetivo: ter nocao concreta de quanto custa correr simulacoes/treino antes de
submeter mais experiencias longas.

## Leitura Curta

- Baselines 2022 ano completo ja acabaram e custam minutos, nao horas.
- MADDPG ano completo custa horas, mas nao porque o simulador em si seja lento:
  o custo dominante passa a ser treino/replay/redes.
- No Deucalion GPU, MADDPG V48 esta em cerca de `0.39-0.43 s/step`.
- A 6 episodios anuais, isto da uma expectativa de cerca de `5.6-6.3 h` por
  seed, se o worker/orchestrator nao falhar.
- Dois jobs MADDPG full-year falharam por `stale_status`, nao por timeout Slurm.
  Isto aponta para problema operacional/heartbeat/sync do worker, nao para uma
  conclusao de performance do algoritmo.

## Numeros Observados

### Baselines 2022 Full-Year

Todos usam `8760` steps, 1 episodio.

| Job | Host | Duracao | s/step |
|---|---:|---:|---:|
| `fullyear-2022-random-server` | server | 435.93s | 0.0498 |
| `fullyear-2022-normal_no_battery-server` | server | 438.76s | 0.0501 |
| `fullyear-2022-normal-server` | server | 435.60s | 0.0497 |
| `fullyear-2022-rbc_basic-deucalion` | Deucalion CPU | 778.25s | 0.0888 |
| `fullyear-2022-rbc_smart-deucalion` | Deucalion CPU | 776.29s | 0.0886 |

Leitura: para baselines, simulador+wrapper+policy/export ficam entre
`0.05-0.09 s/step`. Isto e a base de custo do ambiente.

### Baselines 2022 Short-Window Antigos

Os configs antigos tinham `2000` steps.

| Job | Host | Duracao | s/step |
|---|---:|---:|---:|
| `cpu-baseline-2022-random` | Deucalion CPU | 195.82s | 0.0979 |
| `cpu-baseline-2022-normal_no_battery` | Deucalion CPU | 193.83s | 0.0969 |
| `cpu-baseline-2022-normal` | Deucalion CPU | 214.27s | 0.1071 |
| `cpu-baseline-2022-rbc_basic` | Deucalion CPU | 217.79s | 0.1089 |
| `cpu-baseline-2022-rbc_smart` | Deucalion CPU | 220.77s | 0.1104 |

Leitura: nao ha sinal de degradacao por usar mais steps. O full-year ficou ate
ligeiramente melhor por step, provavelmente por amortizar overhead fixo.

### MADDPG

| Job | Host | Janela | Steps esperados | Duracao/progresso | s/step aprox. |
|---|---:|---:|---:|---:|---:|
| `main-v48-15s-seed123-gpu` | Deucalion GPU | 5760 x 6 | 34560 | 13337s | 0.386 |
| `main-v48-2022-seed123-gpu` antigo | Deucalion GPU | 2000 x 6 | 12000 | parado aos 1363s | nao conclusivo |
| `fullyear-2022-maddpg-v48-seed123` | Deucalion GPU | 8760 x 6 | 52560 | `stale_status` aos 32256 steps | 0.393 |
| `fullyear-2022-maddpg-v48-seed456` | Deucalion GPU | 8760 x 6 | 52560 | `stale_status` aos 1536 steps | 0.66 inicial / 0.42 estavel |
| `fullyear-2022-maddpg-v48-seed789` | Deucalion GPU | 8760 x 6 | 52560 | em execucao, 17518 steps | 0.388 |
| `server-full-2022-no-v2g-maddpg` antigo | server CPU | 2000 x 6 | 12000 | 15063s | 1.255 |

Leitura: GPU e essencial para MADDPG. O server CPU fica cerca de `3x` mais lento
que Deucalion GPU para esta configuracao.

## Bottleneck

Baseline full-year server:

- cerca de `0.05 s/step`.

MADDPG full-year Deucalion GPU:

- cerca de `0.39-0.43 s/step`.

Logo, para MADDPG, aproximadamente `75-85%` do tempo por step vem do lado
algoritmico: `update`, replay buffer, forward/backward dos actores/critics,
behavior cloning/teacher, target networks e diagnosticos associados.

O simulador/wrapper continua a importar, mas nao e o principal gargalo quando
MADDPG esta a treinar.

## Degradacao Com O Tempo

Nao ha evidencia forte de degradacao por step nos dados atuais:

- logs MADDPG mostram `Step Duration` tipicamente `0.41-0.44s` depois do arranque;
- o primeiro bloco pode ser mais irregular (`0.64s`) por warmup, inicializacao,
  replay ainda pequeno e overhead de arranque;
- episodio completo do seed789: episodio 2 demorou `3338.10s`, ou seja
  `0.381s/step`.

Risco ainda aberto: o replay buffer cresce durante o treino (`capacity=200000`,
run full-year com `52560` transitions), portanto memoria e sampling devem ser
monitorizados em runs mais longas ou com mais episodios. Nos dados atuais ainda
nao ha sinal claro de slowdown progressivo.

## Falhas Operacionais

Os jobs `fullyear-2022-maddpg-v48-seed123` e `seed456` falharam no orchestrator
com `stale_status`. O estado Slurm reportado ainda era `RUNNING` perto da falha,
e os logs tinham progresso normal antes de deixar de atualizar.

Isto sugere:

- nao foi timeout de 24h;
- nao parece ser explosao obvia de tempo por step;
- pode ser heartbeat/sync/log polling do worker ou stale TTL demasiado agressivo
  para jobs longos.

Antes de confiar em long-runs, convem:

- confirmar se o Slurm job ainda continuou/orfanizou depois do `stale_status`;
- aumentar tolerancia de stale/heartbeat para jobs longos;
- garantir sync periodico de progress/checkpoints;
- considerar requeue/resume a partir de checkpoint quando houver falha operacional.

## Regras Praticas Para Planeamento

Estimativas atuais:

- baseline 2022 full-year: `7-13 min`, dependendo host;
- MADDPG 2022 full-year, 6 episodios, GPU: `~6 h` por seed;
- MADDPG 2022 full-year, 6 episodios, CPU server: `~18 h` por seed;
- 15s 1 dia x 6 episodios em GPU: `~3.7 h`.

Para experimentos futuros:

- baselines podem correr no server/CPU sem problema;
- MADDPG e comparadores com treino devem ir para GPU;
- para debug rapido, usar short-window;
- para conclusao, usar sempre ano completo 2022 ou explicitar claramente que e
  short-window;
- antes de aumentar episodios/seeds, resolver ou mitigar `stale_status`.
