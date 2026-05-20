# Fase 6J/6K - Remote Scorecard e Decisao MADDPG

Data: 2026-05-20.

Objetivo: preparar a leitura das runs remotas `sha-969d417` sem decidir novas
alteracoes antes dos resultados. Esta fase nao muda wrapper nem simulador; serve
para transformar resultados remotos em decisoes tecnicas para a proxima receita
MADDPG.

## Inputs Remotos

Submissoes principais:

- `runs/remote_configs/phase6_remote_2026_05_20/submitted_jobs_2026_05_20_sha969d417.csv`
- `runs/remote_configs/phase6_remote_2026_05_20/submitted_cpu_jobs_2026_05_20_sha969d417.csv`
- `runs/remote_configs/phase6_remote_2026_05_20/submitted_server_variant_full_jobs_2026_05_20_sha969d417.csv`

Grupos:

- V48 original V2G-capable em GPU: `15s` e `2022`, seed `123`;
- baselines full em CPU: `Random`, `NormalNoBattery`, `Normal`, `RBCBasic`,
  `RBCSmart`;
- variantes de diagnostico:
  - `no_v2g`: charger EV sem descarga;
  - `multi_charger`: mais carregadores e stress de fases/acoes.

## Recolha

Quando as runs acabarem:

```bash
.venv/bin/python scripts/collect_remote_results.py \
  --jobs-file runs/remote_configs/phase6_remote_2026_05_20/submitted_jobs_2026_05_20_sha969d417.csv \
  --jobs-file runs/remote_configs/phase6_remote_2026_05_20/submitted_cpu_jobs_2026_05_20_sha969d417.csv \
  --jobs-file runs/remote_configs/phase6_remote_2026_05_20/submitted_server_variant_full_jobs_2026_05_20_sha969d417.csv \
  --output-dir runs/remote_results/phase6j_sha969d417
```

Isto escreve:

- `summary.csv`;
- `summary.json`;
- uma pasta por job com status, config resolvido, logs tail e KPI CSV quando
  existir.

## Scorecard

Depois da recolha:

```bash
.venv/bin/python scripts/build_phase6_remote_scorecard.py \
  --summary-csv runs/remote_results/phase6j_sha969d417/summary.csv \
  --output-dir runs/remote_results/phase6j_sha969d417
```

Outputs:

- `scorecard.csv`;
- `scorecard.md`.

O scorecard infere:

- dataset: `15s` ou `2022`;
- variante: `original`, `no_v2g`, `multi_charger`;
- policy: `Random`, `NormalNoBattery`, `Normal`, `RBCBasic`, `RBCSmart`,
  `MADDPG_v48`;
- track: `smoke`, `baseline`, `full`;
- seed.

Tambem calcula, quando o baseline existir:

- `cost_delta_to_rbcsmart_eur`;
- `cost_delta_to_rbcsmart_pct`;
- `ev_within_tolerance_delta_to_rbcsmart`.

## Gates de Decisao

Gates default do scorecard:

- `ev_min_acceptable_feasible_rate >= 0.999`;
- `ev_within_tolerance_feasible_rate >= 0.80`;
- `electrical_violation_kwh <= 1e-6`;
- `community_cost_eur <= RBCSmart` para candidato forte;
- custo ate `5%` acima do `RBCSmart` pode ser `candidate_near_cost`, mas nao e
  vencedor.

Decisoes automaticas:

- `candidate_strong`: promove a receita para mais seeds;
- `candidate_cost_ok_precision_watch`: custo bom, mas falta afinar precision EV;
- `candidate_near_cost`: vale tuning incremental, mas nao substituir ainda;
- `reject_ev_service`: nao mexer em custo/rede antes de corrigir EV;
- `reject_grid_violation`: investigar action bounds/phase/headroom;
- `reject_cost`: EV esta aceitavel, mas a receita nao ganha em custo;
- `awaiting_rbcsmart_baseline`: recolher de novo quando o baseline terminar.

## Leitura 6J

Ordem de leitura quando os resultados chegarem:

1. Confirmar que todos os smokes `sha-969d417` terminaram sem erros.
2. Confirmar que os baselines full terminaram e que `RBCSmart` continua uma
   referencia coerente.
3. Comparar `MADDPG_v48 original` contra `RBCSmart` no `15s`.
4. Comparar `MADDPG_v48 original` contra `RBCSmart` no `2022`.
5. Ler `no_v2g`:
   - se melhora muito custo/EV, o problema principal esta no espaco de acao
     bidirecional EV/V2G;
   - se nao melhora, V2G nao e o bloqueio dominante.
6. Ler `multi_charger`:
   - se quebra EV/grid, o problema esta em escala de acoes, fases/headroom ou
     observacoes por carregador;
   - se fica estavel, a arquitetura atual aguenta maior dimensao de acoes.
7. So depois decidir V49/V50.

## Plano 6K - Caminhos V49/V50

Nao implementar tudo em paralelo. Escolher conforme o scorecard.

### Caso A - V48 ganha ou fica muito perto

Sinal:

- `candidate_strong` ou `candidate_near_cost` nos originais;
- EV feasible e EV precision aceitaveis;
- sem violacoes.

Acao:

- correr seeds `456` e `789`;
- manter reward V46/V48;
- tuning minimo:
  - exploration phaseout ligeiramente mais longo;
  - `actor_policy_loss_weight` ainda baixo;
  - observar critic/Q antes de mudar rede.

### Caso B - EV bom, custo pior

Sinal:

- EV feasible bom;
- EV within tolerance razoavel;
- custo pior que `RBCSmart`;
- bateria/EV throughput alto.

Acao V49:

- storage discipline:
  - reforcar regularizacao de throughput storage;
  - penalizar ciclos sem valor comunitario claro;
  - manter SOC min/max do simulador;
- nao aumentar penalizacao EV se EV ja esta bom;
- comparar contra `NormalNoBattery` para saber se a bateria esta a ajudar ou a
  destruir custo.

### Caso C - Custo bom, EV precision fraca

Sinal:

- custo bate ou aproxima `RBCSmart`;
- `ev_min_acceptable_feasible_rate` bom;
- `ev_within_tolerance_feasible_rate` abaixo do `RBCSmart`.

Acao V49:

- BC EV mais focado em target band, nao em over-service;
- aumentar peso zero/idle quando EV ja esta dentro da banda;
- manter caps de deficit da V46;
- nao voltar a V47 se ela voltar a piorar custo.

### Caso D - `no_v2g` melhora muito

Sinal:

- `no_v2g` bate original em custo/EV;
- original mostra V2G/descarga EV ou instabilidade.

Acao V50:

- manter dataset original V2G-capable, mas treinar com curriculum:
  - fase 1: EV discharge praticamente bloqueado por noise/BC/regularizacao;
  - fase 2: liberar V2G so quando SOC/service margin estiver confortavel;
  - avaliar se uma head separada EV charge/discharge ajuda.

### Caso E - `multi_charger` quebra

Sinal:

- smokes passam mas full perde EV/grid;
- erros concentrados em casas/fases com muitos chargers.

Acao V50:

- actor heads por tipo de asset ou por grupo de acoes;
- scaling/regularizacao por charger;
- critic maior ou LayerNorm apenas se os logs mostrarem Q/critic instavel;
- auditar observacoes locais por charger antes de mudar reward.

### Caso F - Deucalion/server divergentes

Sinal:

- mesma config com resultados incompatíveis entre host/imagem;
- logs mostram device/versao diferente.

Acao:

- congelar ambiente antes de nova fase;
- comparar `config.resolved.yaml`, image tag, simulator version e device logs;
- nao tirar conclusoes de RL ate resolver reproducibilidade.

## O Que Nao Fazer Ja

- Nao passar para LSTM sem prova de que o problema e memoria temporal.
- Nao meter regras no wrapper.
- Nao trocar MADDPG por outro algoritmo antes de fechar V48 multi-seed.
- Nao otimizar contra uma unica janela se a 2022 ou `multi_charger` quebrar.
