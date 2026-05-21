# Phase 6 Reward Scout MADDPG

Data: 2026-05-21.

Objetivo: testar rapidamente variantes de reward/config antes de gastar runs
longas no Deucalion. Estas runs nao substituem treino longo nem multi-seed; so
servem para aceitar/rejeitar direcoes.

## Variantes Testadas

- `V48`: baseline MADDPG atual, `CostServiceCommunityFeasiblePrecisionRewardV46`
  com teacher RBCSmart learning.
- `V49`: `CostServiceCommunityStorageValueRewardV49`; reduz penalizacao de
  throughput e aumenta sinal comunitario para dar mais espaco a bateria/V2G.
- `V50`: `CostServiceCommunityDeadlineValueRewardV50`; aumenta pressao de
  deadline de EV antes da saida.
- `V51`: `CostServiceCommunityPrecisionValueRewardV51`; tenta equilibrar V50
  com penalizacao mais forte de over-service para ficar mais perto do SOC alvo.
- `V52`: `CostServiceCommunityPeakDeadlineRewardV52`; adiciona pressao mais
  explicita de settlement/pico/export comunitario.
- `V53`: diagnostico V50 com zero-target BC mais forte; testou se o excesso de
  carga vinha de pouca penalizacao em acoes EV quando o teacher queria zero.
- `V54`: diagnostico quase clone do RBCSmart, com policy loss desligado.
- `V55`: reward V52 com warmup extra de behavior cloning.
- `V56`: reward V46/V48 com warmup extra de behavior cloning e policy loss
  pequeno, para testar finetune depois de imitar.

## Runs Locais

### 15s, janela sem EV departures

Comando base:

```bash
.venv/bin/python scripts/run_phase6a_benchmark.py \
  --output-dir runs/benchmarks/phase6_reward_scout_20260520_234608 \
  --dataset 15s \
  --agent rbc_smart \
  --agent maddpg \
  --maddpg-variant community_feasible_precision_v48_zero_band_teacher_clone_ev_learning_teacher_rbc_smart \
  --maddpg-variant community_storage_value_v49_teacher_clone_ev_balanced_rbc_smart \
  --maddpg-variant community_deadline_value_v50_teacher_clone_ev_balanced_rbc_smart \
  --seed 123 --episodes 2 --deterministic-finish --steps 768 \
  --random-exploration-steps 128 --metric-interval 64 --batch-size 64 \
  --buffer-capacity 50000 --actor-layers 128,64 --critic-layers 256,128 \
  --sigma 0.12 --fail-fast
```

Resultado:

| Variante | Custo comunidade EUR | EV departures | Violacoes | Acao EV positiva | Storage ativo | Leitura |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| RBCSmart | 40.49 | 0 | 0.0 | 0.837 | 0.000 | baseline sem eventos EV |
| V48 | 22.66 | 0 | 0.0 | 0.978 | 0.000 | melhor custo nesta janela |
| V49 | 40.20 | 0 | 0.0 | 0.957 | 0.003 | nao mostrou ganho de storage |
| V50 | 41.31 | 0 | ~0.0 | 0.957 | 0.000 | reward mais pesada sem ganho aqui |

Conclusao: esta janela so valida custo/acoes. Nao tem eventos EV, portanto nao
serve para escolher reward de departure.

### 15s, janela com EV departures

Comando base:

```bash
.venv/bin/python scripts/run_phase6a_benchmark.py \
  --output-dir runs/benchmarks/phase6_reward_event_scout_2ep_20260521_000200 \
  --dataset 15s \
  --agent rbc_smart \
  --agent maddpg \
  --maddpg-variant community_feasible_precision_v48_zero_band_teacher_clone_ev_learning_teacher_rbc_smart \
  --maddpg-variant community_storage_value_v49_teacher_clone_ev_balanced_rbc_smart \
  --maddpg-variant community_deadline_value_v50_teacher_clone_ev_balanced_rbc_smart \
  --seed 123 --episodes 2 --deterministic-finish --steps 960 --start 1800 \
  --random-exploration-steps 128 --metric-interval 64 --batch-size 64 \
  --buffer-capacity 50000 --actor-layers 128,64 --critic-layers 256,128 \
  --sigma 0.12 --fail-fast
```

Resultado:

| Variante | Custo comunidade EUR | EV departures | EV min feasible | EV within feasible | Deficit medio SOC | Violacoes | Leitura |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RBCSmart | 30.18 | 3 | 0.000 | 0.000 | 0.355 | 0.0 | janela dificil/infeasible |
| V48 | 29.91 | 3 | 0.000 | 0.000 | 0.358 | ~0.0 | iguala custo, EV similar |
| V49 | 29.49 | 3 | 0.000 | 0.000 | 0.361 | 0.0 | melhor custo, EV ligeiramente pior |
| V50 | 29.90 | 3 | 0.000 | 0.000 | 0.358 | 0.0 | deadline reward nao se distingue nesta janela |

Conclusao: boa janela de stress, mas nao boa para escolher EV reward porque os
eventos sao maioritariamente marcados como infeasible.

### 2022, semana curta com muitos EV departures

Comando base:

```bash
.venv/bin/python scripts/run_phase6a_benchmark.py \
  --output-dir runs/benchmarks/phase6_reward_2022_scout_20260521_000300 \
  --dataset 2022 \
  --agent rbc_smart \
  --agent maddpg \
  --maddpg-variant community_feasible_precision_v48_zero_band_teacher_clone_ev_learning_teacher_rbc_smart \
  --maddpg-variant community_storage_value_v49_teacher_clone_ev_balanced_rbc_smart \
  --maddpg-variant community_deadline_value_v50_teacher_clone_ev_balanced_rbc_smart \
  --seed 123 --episodes 2 --deterministic-finish --steps 168 \
  --random-exploration-steps 48 --metric-interval 24 --batch-size 64 \
  --buffer-capacity 50000 --actor-layers 128,64 --critic-layers 256,128 \
  --sigma 0.12 --fail-fast
```

V51 foi validado num segundo smoke equivalente:

```bash
.venv/bin/python scripts/run_phase6a_benchmark.py \
  --output-dir runs/benchmarks/phase6_reward_2022_v51_scout_20260521_000400 \
  --dataset 2022 --agent rbc_smart --agent maddpg \
  --maddpg-variant community_precision_value_v51_teacher_clone_ev_precise_rbc_smart \
  --seed 123 --episodes 2 --deterministic-finish --steps 168 \
  --random-exploration-steps 48 --metric-interval 24 --batch-size 64 \
  --buffer-capacity 50000 --actor-layers 128,64 --critic-layers 256,128 \
  --sigma 0.12 --fail-fast
```

Resultado:

| Variante | Custo comunidade EUR | EV min feasible | EV within feasible | Deficit medio SOC | Surplus medio SOC | Erro abs. medio | Leitura |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RBCSmart | 424.66 | 1.000 | 0.423 | 0.001 | 0.063 | 0.064 | melhor EV service |
| V48 | 339.07 | 0.750 | 0.327 | 0.072 | 0.067 | 0.138 | melhor custo, EV insuficiente |
| V49 | 385.73 | 0.846 | 0.154 | 0.046 | 0.097 | 0.143 | meio-termo fraco |
| V50 | 425.08 | 0.923 | 0.038 | 0.028 | 0.131 | 0.159 | menos deficit, mas over-service alto |
| V51 | 309.53 | 0.538 | 0.212 | 0.109 | 0.049 | 0.157 | corta over-service, mas falha minimo |

### 2022, semana curta com 8-16 episodios

Depois dos smokes de 2 episodios, a leitura mudou bastante: algumas variantes
precisam de mais treino para copiar a distribuicao de acoes do teacher.

| Variante | Episodios | Custo EUR | EV min feasible | EV within feasible | Deficit SOC | Surplus SOC | Erro abs. SOC | Pico comunitario reward mean | Leitura |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RBCSmart | 8 | 424.66 | 1.000 | 0.423 | 0.001 | 0.063 | 0.064 | 0.997 | referencia |
| V48 | 8 | 398.51 | 1.000 | 0.385 | 0.004 | 0.063 | 0.067 | 0.750 | melhor custo que RBC, EV ok |
| V50 | 8 | 466.79 | 1.000 | 0.019 | 0.002 | 0.145 | 0.147 | 0.827 | over-service forte |
| V52 | 8 | 459.65 | 1.000 | 0.000 | 0.002 | 0.135 | 0.137 | 1.470 | piorou pico e precisao |
| V54 | 8 | 419.82 | 1.000 | 0.308 | 0.002 | 0.067 | 0.068 | 0.801 | prova que clone aprende, mas fica atras de V48 |
| V55 | 8 | 436.95 | 1.000 | 0.096 | 0.001 | 0.103 | 0.104 | 1.407 | reward comunitaria forte desestabiliza EV precision |
| V56 | 8 | 439.68 | 1.000 | 0.096 | 0.001 | 0.108 | 0.110 | 0.603 | baixa pico, mas perde custo/precision |
| V48 | 16 | 407.84 | 1.000 | 0.519 | 0.002 | 0.055 | 0.056 | 0.450 | melhor candidato local: custo, EV e pico |

Robustez inicial V48/16 episodios:

| Seed | Custo EUR | EV min feasible | EV within feasible | Deficit SOC | Surplus SOC | Erro abs. SOC | Pico comunitario reward mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 123 | 407.84 | 1.000 | 0.519 | 0.0015 | 0.0549 | 0.0565 | 0.450 |
| 456 | 410.45 | 1.000 | 0.500 | 0.0017 | 0.0593 | 0.0611 | 0.437 |
| 789 | 410.93 | 1.000 | 0.404 | 0.0015 | 0.0612 | 0.0626 | 0.446 |
| media | 409.74 | 1.000 | 0.474 | 0.0016 | 0.0585 | 0.0601 | 0.444 |

Leitura:

- `V48` e a melhor receita local neste momento. Com mais episodios melhora
  tambem a precisao EV e o sinal de pico comunitario.
- `V48` manteve o minimo EV em 3 seeds e ficou consistentemente abaixo do custo
  do RBCSmart local; a precisao EV ainda varia, mas a media fica acima do
  RBCSmart nesta janela.
- `V50`, `V52`, `V55` e `V56` mostram que aumentar peso comunitario/deadline
  sem preservar primeiro a politica EV leva a over-service e pior custo.
- `V54` provou que o actor consegue aprender a aproximar o teacher quando ha
  episodios suficientes; o problema nao parece ser arquitetura base incapaz.
- as metricas globais de acao EV podem ser enganadoras porque contam chargers
  sem EV ligado. Foi adicionado diagnostico `Action/ev_connected_*` para runs
  futuras.

## Decisao

- promover `V48` como candidata local principal para runs longas/multi-seed;
- manter `V54` como diagnostico de clone/teacher, nao como melhor candidata;
- nao promover `V50`, `V52`, `V55` ou `V56` nesta configuracao: cumprem o
  minimo EV, mas degradam custo e precisao;
- usar os resultados remotos para confirmar em 15s e variantes no_v2g /
  multi_charger;
- proxima melhoria tecnica deve focar policy improvement conservador depois de
  uma fase BC forte, ou comparar com algoritmos on-policy/off-policy alternativos
  em vez de continuar a aumentar pesos de reward.

## Observacoes

- Runs curtas continuam uteis para direcao, mas nao provam convergencia.
- A comparacao EV deve usar sempre `ev_min_acceptable_feasible_rate` e
  `ev_within_tolerance_feasible_rate` em conjunto.
- Reduzir custo sem EV service nao chega.
- Melhorar target precision so com over-service penalty pode empurrar o agente
  para sub-carregar; a configuracao de teacher/BC tambem precisa de evoluir.
