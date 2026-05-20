# Fase 6B - Triagem MADDPG 15s

Data: 2026-05-14.

Objetivo: correr uma matriz real, mas ainda controlada, para perceber se as
hipoteses MADDPG atuais merecem treino longo. Isto nao e benchmark final.

## Run

Output:

- `runs/benchmarks/phase6b_15s_full_day_seed123`

Comando:

```bash
.venv/bin/python scripts/run_phase6a_benchmark.py \
  --output-dir runs/benchmarks/phase6b_15s_full_day_seed123 \
  --dataset 15s \
  --agent random \
  --agent normal_no_battery \
  --agent normal \
  --agent rbc_basic \
  --agent rbc_smart \
  --agent maddpg \
  --maddpg-variant current \
  --maddpg-variant v1 \
  --maddpg-variant noop_centered \
  --maddpg-variant warm_rbc_basic \
  --seed 123 \
  --steps-15s 5760 \
  --episodes 1 \
  --batch-size 64 \
  --buffer-capacity 50000 \
  --random-exploration-steps 960 \
  --metric-interval 240 \
  --actor-layers 128,64 \
  --critic-layers 256,128
```

Escopo:

- dataset `citylearn_three_phase_electrical_service_demo_15s_parquet`;
- 1 dia completo de 15 segundos: 5760 steps;
- 1 seed: 123;
- 7 EV departures no periodo;
- 1 ciclo deferrable servido;
- 0 kWh de violacao de electrical service em todas as policies;
- 9 runs completas, 0 falhas.

## Resultados Principais

| Policy | Cost EUR | EV success | EV deficit mean | EV charge kWh | V2G kWh | Deferrable service | Action near extremes final |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | 81.517 | 0.000 | 0.727 | 408.233 | 210.929 | 1.000 | 0.038 |
| NormalNoBattery | 250.353 | 0.143 | 0.333 | 1494.876 | 0.000 | 1.000 | 0.269 |
| Normal | 255.665 | 0.143 | 0.333 | 1494.876 | 0.000 | 1.000 | 0.269 |
| RBCBasic | 140.229 | 0.000 | 0.422 | 747.374 | 0.000 | 1.000 | 0.077 |
| RBCSmart | 154.218 | 0.000 | 0.409 | 779.696 | 0.000 | 1.000 | 0.077 |
| MADDPG current | 92.081 | 0.000 | 0.603 | 451.748 | 160.803 | 1.000 | 0.731 |
| MADDPG v1 | 88.848 | 0.000 | 0.651 | 450.438 | 168.043 | 1.000 | 0.923 |
| MADDPG noop_centered | 129.460 | 0.000 | 0.634 | 719.519 | 102.409 | 1.000 | 0.769 |
| MADDPG warm_rbc_basic | 88.919 | 0.000 | 0.707 | 469.538 | 121.361 | 1.000 | 0.731 |

Notas:

- `Action near extremes final` e a soma de `near_low_fraction` e
  `near_high_fraction` no ultimo log.
- EV success e calculado sobre 7 departures.
- Baixo custo sem EV success nao e bom resultado; pode significar under-service,
  V2G abusivo ou pouca carga util.

## Leitura

### 1. A matriz correu bem tecnicamente

O pipeline esta funcional para esta fase:

- configs geradas;
- `run_experiment.py` executou todos os jobs;
- exports de KPIs existem;
- logs de reward/action/MADDPG existem;
- ONNX/manifests foram exportados para as runs MADDPG;
- nao houve violacoes de electrical service.

### 2. A janela de 1 dia e util, mas ainda nao prova aprendizagem final

Esta janela e melhor do que um "curto" arbitrario porque inclui:

- eventos EV;
- deadline/service de deferrable;
- variacao de preco/carga/PV;
- 960 steps de warm-up e updates posteriores.

Mas continua a ser uma unica passagem. Nao devemos concluir que um MADDPG
"aprendeu bem" ou "aprendeu mal definitivamente"; devemos concluir quais os
problemas que aparecem antes de gastar treino longo.

### 3. Os baselines ainda nao sao oracles fortes para EV

Resultados EV:

- `Normal` e `NormalNoBattery` so cumprem 1/7 departures;
- `RBCBasic` e `RBCSmart` cumprem 0/7 departures;
- `Random` tambem cumpre 0/7.

Isto significa que:

- o custo do `Random` e baixo porque nao presta bom servico EV;
- `RBCBasic`/`RBCSmart` sao uteis como baselines economicos, mas nao como
  referencia forte de servico EV;
- antes do benchmark final, os RBCs devem ser revistos para garantir SOC de
  saida quando possivel.

### 4. MADDPG tem updates estaveis, mas comportamento mau

O critic nao explodiu:

- losses finais pequenas;
- reward normalization ativa;
- Q-values/TD error sem sinais grosseiros de divergencia.

O problema observado e outro:

- EV success 0/7 em todas as variantes;
- alto V2G nas variantes MADDPG;
- action saturation muito forte no final;
- baixo custo aparece acompanhado de EV under-service.

Isto sugere que o MADDPG esta a otimizar um caminho facil de custo/energia e
nao esta a aprender servico EV de forma aceitavel nesta configuracao.

### 5. `maddpg_v1` nao e melhoria simples

`maddpg_v1` teve custo ligeiramente menor que `current`, mas:

- EV success continua 0;
- EV deficit medio piora;
- saturacao final e a pior da matriz: cerca de 92%.

Conclusao: nao vale priorizar `v1` para treino longo nesta fase.

### 6. `noop_centered` melhora o inicio, mas nao resolve

Durante warm-up, `noop_centered` produziu acoes muito mais suaves do que
exploracao uniforme.

No final:

- EV success continua 0;
- custo pior que `current`/`v1`;
- saturacao final continua alta.

Conclusao: `noop_centered` e melhor como exploracao inicial do que random
full-range, mas nao resolve sozinho a deriva do actor para extremos.

### 7. `warm_rbc_basic` confirma a ideia, mas tambem nao chega

O warm-start com `RBCBasicPolicy` funcionou tecnicamente:

- a policy foi carregada;
- o buffer inicial recebeu acoes estruturadas;
- inicio com menor saturacao.

No final:

- EV success continua 0;
- custo parecido com `v1`;
- saturacao final continua alta;
- deferrable service cumpre, mas com atraso medio de 2.25h.

Conclusao: warm-start e uma boa ferramenta, mas nao chega sem corrigir a
dinamica do actor/reward.

## Hipoteses Atualizadas

Prioridade alta antes de treino longo:

1. Controlar saturacao do actor.
2. Reforcar reward EV para servico de saida sem ser apenas penalizacao tardia.
3. Rever RBCBasic/RBCSmart para garantir EV SOC de saida quando fisicamente
   possivel.
4. Reduzir abuso de V2G/throughput quando o EV ainda tem deficit para saida.
5. Separar benchmark diagnostico de export bundle para reduzir outputs pesados.

Hipoteses tecnicas a testar:

- actor action regularization configuravel no algoritmo;
- penalty suave de magnitude/variacao de acoes na reward, com componentes
  logadas separadamente;
- delayed actor update;
- target policy smoothing / MATD3-style;
- critic maior ou LayerNorm, mas isto parece menos urgente que saturacao;
- event-aware replay para EV departures, depois de estabilizar o actor;
- warm-start com RBC melhorado, nao com RBCBasic atual.

Nao recomendado ainda:

- correr treino longo multi-seed com as variantes atuais;
- promover `maddpg_v1` como perfil principal;
- usar custo isolado como metrica de sucesso.

## Proximo Passo

Fase 6C:

1. Implementar uma variante configuravel para reduzir saturacao do actor.
2. Adicionar reward/action regularization com logs separados.
3. Rever RBC EV service antes de o usar como warm-start forte.
4. Repetir uma matriz pequena de 1 dia apenas com:
   - `RBCBasic` revisto;
   - `RBCSmart` revisto;
   - `MADDPG current`;
   - `MADDPG current + anti-saturation`;
   - `MADDPG current + anti-saturation + warm-start`.

## Seguimento Implementado

Foi implementada uma primeira resposta direta a estas conclusoes:

- `RBCBasicPolicy` e `RBCSmartPolicy` foram revistos para nunca ficarem abaixo
  da taxa minima de servico EV quando existe deficit de SOC e a carga ainda e
  fisicamente possivel;
- `RBCSmartPolicy` passou a proteger melhor V2G: so descarrega quando nao ha
  deficit, ha margem SOC adicional e a partida nao esta demasiado proxima;
- MADDPG ganhou delayed actor update, target policy smoothing e regularizacao
  configuravel de acoes extremas;
- a matriz de benchmark passou a suportar `anti_saturation` e
  `anti_saturation_warm_rbc_basic`.

O proximo resultado a escrever deve ser uma Fase 6C com a mesma janela de 1 dia,
comparando RBC revisto, MADDPG atual e as variantes anti-saturacao.
