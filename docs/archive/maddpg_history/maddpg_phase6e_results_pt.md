# Fase 6E - Baselines Pos-Fix Headroom

Data: 2026-05-17.

## Contexto

Objetivo: repetir os baselines em janela completa depois do fix de headroom
residual nos EVs e depois das mudancas de KPI EV do simulador `0.6.5`.

Ambiente usado:

- simulador local: `/home/tiago/dev/Simulator`;
- commit simulador: `8881bd82`;
- `citylearn.__version__`: `0.6.5`;
- import efetivo: `/home/tiago/dev/Simulator/citylearn/__init__.py`;
- `requirements.txt` deste repo ainda aponta para `softcpsrecsimulator==0.6.4`;
- testes focados apos reinstalar simulador local: `46 passed`.

Comando:

```bash
.venv/bin/python scripts/run_phase6a_benchmark.py \
  --output-dir runs/benchmarks/phase6e_065_baselines_post_headroom_fix \
  --dataset 15s \
  --dataset 2022 \
  --agent random \
  --agent normal_no_battery \
  --agent normal \
  --agent rbc_basic \
  --agent rbc_smart \
  --seed 123 \
  --episodes 1 \
  --full-window \
  --metric-interval 240
```

Resultado da matriz:

- `10/10` jobs completos;
- `0` falhas;
- output: `runs/benchmarks/phase6e_065_baselines_post_headroom_fix`;
- resumo: `runs/benchmarks/phase6e_065_baselines_post_headroom_fix/benchmark_summary.csv`.

## KPI EV Primario

Para comparar controladores, usar primeiro:

- `ev_departure_min_acceptable_feasible_rate`.

Este KPI mede se o controlador cumpriu o minimo aceitavel nos eventos
fisicamente atingiveis. Tambem devemos acompanhar:

- `ev_departure_min_acceptable_rate`, que mede experiencia bruta do cenario;
- `ev_departure_success_feasible_rate`;
- `ev_departure_success_rate`;
- `ev_departure_soc_deficit_mean`;
- `ev_departure_shortfall_beyond_tolerance_mean`;
- contagens de infeasible.

## Resultados 15s

| policy | cost EUR | grid viol kWh | EV min feasible | EV min raw | EV success feasible | EV success raw | EV deficit mean | EV shortfall > tol | battery safety reward | deferrable reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | 61.22 | 0.00 | 0.000 | 0.000 | 0.000 | 0.000 | 0.6267 | 0.5767 | 0.0020 | 0.0000 |
| NormalNoBattery | 112.51 | 0.00 | 0.833 | 0.714 | 0.833 | 0.714 | 0.0294 | 0.0152 | 0.0000 | 0.0000 |
| Normal | 113.88 | 0.00 | 0.833 | 0.714 | 0.833 | 0.714 | 0.0294 | 0.0152 | 0.0003 | 0.0000 |
| RBCBasic | 113.69 | 0.00 | 0.833 | 0.714 | 0.833 | 0.714 | 0.0294 | 0.0152 | 0.0003 | 0.0255 |
| RBCSmart | 112.48 | 0.00 | 0.833 | 0.714 | 0.833 | 0.714 | 0.0294 | 0.0152 | 0.0000 | 0.0255 |

Leitura:

- O EV service dos baselines deterministicos ficou alinhado depois do fix de
  headroom: todos carregam o que conseguem carregar.
- O raw EV min acceptable fica em `0.714` por causa de `7` departures totais,
  com `1` evento fisicamente infeasible e outro evento feasible que ainda falha
  no Building 15.
- O feasible EV min acceptable fica em `0.833`, logo ainda ha uma falha real
  em `6` eventos fisicamente atingiveis.
- `Random` e mais barato so porque nao presta servico EV; nao e baseline de
  custo, e apenas lower bound/sanity.
- `NormalNoBattery` e a referencia mais limpa para "dia normal" no 15s.
- `Normal` piora ligeiramente custo por uso simples da bateria.
- `RBCBasic` nao melhora custo e ainda introduz penalizacao deferrable.
- `RBCSmart` fica quase igual a `NormalNoBattery` em custo, mas tambem tem
  penalizacao deferrable; portanto ainda nao esta bom como baseline "smart".

## Resultados 2022

| policy | cost EUR | grid viol kWh | EV min feasible | EV min raw | EV success feasible | EV success raw | EV deficit mean | EV shortfall > tol | battery safety reward | deferrable reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | 2788.85 | 0.00 | 0.152 | 0.151 | 0.124 | 0.123 | 0.3763 | 0.3330 | 0.2728 | 0.0000 |
| NormalNoBattery | 4729.07 | 0.00 | 1.000 | 0.995 | 0.986 | 0.975 | 0.0007 | 0.0003 | 0.0000 | 0.0000 |
| Normal | 4847.26 | 0.00 | 1.000 | 0.995 | 0.986 | 0.975 | 0.0007 | 0.0003 | 0.0310 | 0.0000 |
| RBCBasic | 4365.20 | 0.00 | 1.000 | 0.995 | 0.986 | 0.975 | 0.0007 | 0.0003 | 0.1922 | 0.0000 |
| RBCSmart | 4752.62 | 0.00 | 1.000 | 0.995 | 0.986 | 0.975 | 0.0007 | 0.0003 | 0.0000 | 0.0000 |

Leitura:

- EV service esta forte nos deterministicos: `1.000` feasible min acceptable.
- Raw min acceptable fica `0.995` porque ha eventos fisicamente infeasible:
  `5` min acceptable infeasible e `8` target infeasible.
- `RBCBasic` reduz custo face a `NormalNoBattery`, mas faz isso com muita
  agressividade na bateria: `battery_safety_penalty_mean ~= 0.1922` e
  throughput/ciclos muito altos.
- `RBCSmart` evita a penalizacao de bateria, mas nao melhora custo; fica perto
  de `NormalNoBattery`.
- `Normal` com bateria simples piora custo e introduz penalizacao de bateria.

## Coisas Estranhas / Pontos a Auditar

### 1. Building 15 no 15s continua a ser o gargalo EV

No `15s`, os dois chargers do Building 15 continuam a explicar as falhas EV.
Exemplo em `NormalNoBattery`:

- `charger_15_1`: desconecta as `09:00`, SOC aproximadamente `0.74`, required
  `0.80`;
- `charger_15_2`: desconecta as `10:00`, SOC aproximadamente `0.68`, required
  `0.82`.

Isto aparece mesmo com acao de carga positiva continua, portanto nao e o bug
antigo de clipping ON/OFF.

Ponto estranho: a observacao/export `EV Departure Time` chega a `0` antes do
disconnect real e fica em `0` durante bastante tempo enquanto o EV continua
ligado e a carregar. Isto pode ser semantica intencional ("deadline atingido")
ou ruido de export/dataset, mas deve ficar marcado porque afeta o sinal de
tempo para o agente.

### 2. Random nao pode ser usado como baseline de custo

O `Random` tem custo menor em alguns cenarios, mas falha EV service. Portanto:

- serve para sanity/lower bound;
- nao serve para dizer que uma policy e boa por custo;
- qualquer comparacao deve usar custo condicionado a servico EV/deferrable.

### 3. Storage ainda nao esta bem calibrado nos baselines

`Normal` e `RBCBasic` usam bateria de forma demasiado grosseira:

- no `15s`, `Normal` piora custo face a `NormalNoBattery`;
- no `2022`, `RBCBasic` baixa custo mas paga muita penalizacao de bateria;
- `RBCSmart` praticamente nao usa bateria nestes runs, evitando penalizacao mas
  tambem nao entregando vantagem clara.

Isto significa que os baselines ainda nao formam uma escada limpa.

### 4. Deferrables no 15s

`RBCBasic` e `RBCSmart` introduzem penalizacao deferrable no `15s`, enquanto
`NormalNoBattery` e `Normal` nao. Isto sugere que a regra de adiar por preco/PV
esta a esperar demais para a janela de flexibilidade.

## Decisao da Fase 6E

A Fase 6E esta executada, mas a qualidade dos baselines ainda nao passa o gate.

Podemos aceitar:

- `Random` como sanity/lower bound;
- `NormalNoBattery` como baseline "dia normal sem bateria";
- EV KPI feasible-only como KPI primario.

Ainda nao devemos aceitar como comparacao final:

- `Normal` com bateria;
- `RBCBasic`;
- `RBCSmart`.

Antes de voltar a comparar MADDPG a serio, precisamos de uma Fase 6E.1 para
corrigir/calibrar os baselines heuristicos.

## Fase 6E.1 Proposta

Objetivo: tornar os baselines uma escada justa e fisicamente coerente.

Gates minimos:

- todos os deterministicos devem manter
  `ev_departure_min_acceptable_feasible_rate >= NormalNoBattery - 1e-6`;
- `deferrable_service_penalty_mean` deve ficar `0` ou explicitamente justificado
  por tradeoff de custo;
- `battery_safety_penalty_mean` deve ficar baixo e documentado;
- `RBCBasic` deve melhorar custo ou pico face a `NormalNoBattery` sem degradar
  EV/deferrables;
- `RBCSmart` deve melhorar pelo menos uma dimensao relevante face a `RBCBasic`
  sem esconder penalizacoes fortes noutros KPIs.

Alteracoes a considerar:

- Deferrables: usar regra de ultimo arranque seguro, nao apenas cheap/PV ou
  urgencia tarde. Se a janela esta a fechar, start tem de ser `1`.
- Normal storage: tornar mais conservador ou remover do baseline normal
  principal; manter `NormalNoBattery` como referencia pura.
- RBCBasic storage: reduzir ciclos, usar banda de conforto SOC e limitar
  arbitragem por preco para nao trocar custo por abuso de bateria.
- RBCSmart storage: ativar uso de PV/preco/headroom de forma mais clara, mas
  com limites de throughput e SOC.
- RBCSmart EV: V2G deve continuar conservador e so quando ha margem de servico.

Depois repetir exatamente a mesma matriz em novo output dir.
