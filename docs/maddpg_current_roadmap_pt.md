# Roadmap Atual MADDPG

Data: 2026-05-20.

Este e o documento curto de trabalho. Os documentos antigos continuam validos
como historico e evidencia, mas a ordem futura deve ser decidida a partir daqui.

## Objetivo

Chegar a um controlador MADDPG/MARL que:

- controle EVs, baterias e deferrables disponiveis nos datasets;
- cumpra limites fisicos e requisitos de servico, sobretudo EV departure;
- bata ou se aproxime de `RBCSmart` em custo/comunidade;
- use informacao comunitaria de forma real, nao apenas heuristica local;
- exporte artefactos utilizaveis no repo de inference.

Regra permanente: nao meter comportamento prescritivo no wrapper. Melhorias de
comportamento devem vir de observacoes, reward, replay, exploracao, arquitetura
ou algoritmo.

## Estado Atual

Base tecnica:

- simulador: `softcpsrecsimulator==0.6.7`;
- interface: `entity`;
- topologia MADDPG: `static`;
- datasets principais:
  - `citylearn_three_phase_electrical_service_demo_15s_parquet`;
  - `citylearn_challenge_2022_phase_all_plus_evs`;
- variantes locais/remotas:
  - `no_v2g`;
  - `multi_charger`;
- perfil principal de observacoes MADDPG: `maddpg_v2_compact`;
- candidato MADDPG atual: `V48`.

O que ja esta solido:

- baselines `Random`, `NormalNoBattery`, `Normal`, `RBCBasic`, `RBCSmart`;
- reward V46/V48 com EV service e precision;
- teacher/warm-start/BC configuraveis;
- replay ponderado por reward;
- logs de reward/action/training;
- CUDA local validado;
- Deucalion/server integrados;
- scorecard remoto preparado.

O que ainda nao esta provado:

- V48 robusta em multi-seed;
- V48 no dataset 2022;
- se V2G esta a dificultar aprendizagem;
- se `multi_charger` quebra por escala de acoes/fases;
- se storage esta a criar ganho real ou atalho caro;
- se o critic atual chega ou precisa de MATD3.

## Runs Remotas Em Curso

Imagem: `sha-969d417`.

Grupos submetidos:

- V48 original V2G-capable em GPU:
  - `15s seed123`;
  - `2022 seed123`;
- baselines full no Deucalion CPU:
  - `Random`;
  - `NormalNoBattery`;
  - `Normal`;
  - `RBCBasic`;
  - `RBCSmart`;
- smokes e full variants no server:
  - `no_v2g`;
  - `multi_charger`.

Registos locais:

- `runs/remote_configs/phase6_remote_2026_05_20/submitted_jobs_2026_05_20_sha969d417.csv`;
- `runs/remote_configs/phase6_remote_2026_05_20/submitted_cpu_jobs_2026_05_20_sha969d417.csv`;
- `runs/remote_configs/phase6_remote_2026_05_20/submitted_server_variant_full_jobs_2026_05_20_sha969d417.csv`.

## Proximo Passo Imediato

Quando as runs terminarem, recolher tudo:

```bash
.venv/bin/python scripts/collect_remote_results.py \
  --jobs-file runs/remote_configs/phase6_remote_2026_05_20/submitted_jobs_2026_05_20_sha969d417.csv \
  --jobs-file runs/remote_configs/phase6_remote_2026_05_20/submitted_cpu_jobs_2026_05_20_sha969d417.csv \
  --jobs-file runs/remote_configs/phase6_remote_2026_05_20/submitted_server_variant_full_jobs_2026_05_20_sha969d417.csv \
  --output-dir runs/remote_results/phase6j_sha969d417
```

Depois gerar scorecard:

```bash
.venv/bin/python scripts/build_phase6_remote_scorecard.py \
  --summary-csv runs/remote_results/phase6j_sha969d417/summary.csv \
  --output-dir runs/remote_results/phase6j_sha969d417
```

Ficheiros a ler:

- `runs/remote_results/phase6j_sha969d417/scorecard.md`;
- `runs/remote_results/phase6j_sha969d417/scorecard.csv`;
- logs dos jobs marcados como `reject_*` ou `not_finished`.

## Arvore de Decisao

### Caso 1 - V48 original passa bem

Sinal:

- `candidate_strong` ou `candidate_near_cost`;
- EV feasible bom;
- EV within tolerance bom;
- custo igual/melhor ou perto de `RBCSmart`;
- sem violacoes.

Acao:

- correr seeds `456` e `789`;
- nao mudar algoritmo ainda;
- preparar Fase 7 benchmark final.

### Caso 2 - V48 cumpre EV mas custo piora

Sinal:

- EV feasible/precision aceitaveis;
- custo pior que `RBCSmart`;
- battery throughput alto ou uso estranho de storage.

Acao:

- V49 focada em storage discipline;
- rever reward comunitaria de storage;
- comparar contra `NormalNoBattery` para medir valor real da bateria;
- nao mexer em EV se EV ja estiver bom.

### Caso 3 - V48 tem bom custo mas EV precision fraca

Sinal:

- custo bom;
- `ev_min_acceptable_feasible_rate` bom;
- `ev_within_tolerance_feasible_rate` abaixo de `RBCSmart`.

Acao:

- V49 focada em EV target-band;
- reforcar BC zero/idle quando EV ja esta dentro da banda;
- evitar voltar a uma penalizacao tipo V47 se piorar custo.

### Caso 4 - `no_v2g` melhora muito

Sinal:

- `no_v2g` supera original em custo/EV;
- original mostra descarga EV/V2G instavel.

Acao:

- V50 com curriculum V2G:
  - fase inicial praticamente sem descarga EV;
  - liberar V2G apenas com margem de servico;
  - talvez head separada charge/discharge.

### Caso 5 - `multi_charger` quebra

Sinal:

- original passa;
- `multi_charger` perde EV/grid/custo.

Acao:

- auditar observacoes por charger/fase;
- testar heads por tipo de asset;
- action scaling por charger;
- so depois attention/GNN.

### Caso 6 - critic instavel

Sinal:

- Q-values extremos;
- critic loss divergente;
- policy degrada quando reduz BC;
- resultados muito diferentes por seed.

Acao:

- implementar MATD3/MADDPG-TD3:
  - twin critics;
  - target policy smoothing;
  - delayed actor update;
  - logs Q1/Q2.

### Caso 7 - problema claramente comunitario

Sinal:

- todos cumprem EV/local;
- custo comunitario ainda fica pior;
- picos/import/export nao melhoram;
- storage/EV nao aproveitam excedente comunitario.

Acao:

- reward comunitaria mais fiel:
  - settlement comunitario;
  - autoconsumo local/comunitario;
  - valor de energia partilhada;
  - penalizacao de picos comunitarios;
- critic com sinais globais mais fortes;
- attention/GNN se agregados nao forem suficientes.

## Comparadores Futuros

Ordem recomendada:

1. `MATD3/MADDPG-TD3`: se critic/Q for o gargalo.
2. `MAPPO`: comparador MARL forte, on-policy, mais caro.
3. `IPPO`: baseline PPO simples por agente.
4. `MASAC`: se exploracao continuous for o gargalo.
5. Heads por tipo de asset: se `multi_charger` quebrar.
6. Attention critic/GNN: se escala/topologia/comunidade forem o problema.
7. `QMIX`, `VDN`, `COMA`, `DQN/Rainbow`: baixa prioridade no problema completo
   porque as acoes principais sao continuas.

Detalhe: `docs/marl_algorithm_comparators_pt.md`.

## Plano Futuro Concreto

1. Ler scorecard remoto.
2. Escolher uma unica direcao V49/V50, nao varias ao mesmo tempo.
3. Implementar a menor mudanca que responde ao sinal observado.
4. Correr smoke local curto.
5. Correr benchmark curto 15s/2022.
6. Se passar, submeter remoto.
7. So promover para Fase 7 depois de multi-seed.
8. Validar inference/bundle na Fase 8.

## Documentos de Suporte

Documentos atuais de decisao:

- `docs/maddpg_current_roadmap_pt.md`;
- `docs/maddpg_phase6j6k_remote_decision_pt.md`;
- `docs/marl_algorithm_comparators_pt.md`.

Historico e evidencia:

- `docs/archive/maddpg_history/maddpg_improvement_plan_pt.md`;
- `docs/archive/maddpg_history/maddpg_hypotheses_pt.md`;
- `docs/archive/maddpg_history/maddpg_phase6f_results_pt.md`;
- `docs/archive/maddpg_history/rbc_baseline_audit_pt.md`;
- `docs/archive/maddpg_history/storage_battery_audit_pt.md`;
- `docs/archive/maddpg_history/ev_departure_event_audit_065_pt.md`.

Contrato/export/inference:

- `docs/inference_bundle.md`;
- `docs/entity_interface_playbook_pt.md`.

## O Que Nao Fazer Agora

- nao implementar MAPPO/MASAC antes de ler V48 remoto;
- nao mexer no wrapper para melhorar KPI;
- nao mudar reward e algoritmo ao mesmo tempo;
- nao lançar seeds `456/789` se seed `123` falhar por bug claro;
- nao tirar conclusoes de smokes curtos como KPI final;
- nao otimizar para `no_v2g` e esquecer o dataset original V2G-capable.
