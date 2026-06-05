# Update Para A Equipa: Baselines, RL/MARL E Proxima Ronda

Data: 2026-06-04.

Este documento e o snapshot operacional atual. Historico de waves antigas e
scorecards obsoletos foi removido do repo; resultados brutos ficam em `runs/`.

## Estado Atual

- Simulador: `softcpsrecsimulator==1.5.3`.
- Dataset principal anual: `citylearn_challenge_2022_phase_all_plus_evs`.
- Interface principal: `entity`.
- Topologia neural suportada: `static`.
- Topologia dynamic suportada: baselines/RBCs com `RuleBased` policies.
- Custo oficial: vem do simulador. Nao recalculamos custo agregado do lado dos
  algoritmos.

Quando o mercado comunitario esta ativo, o custo principal deve ser
`district_cost_community_market_settled_total_eur`. Caso contrario, usar
`district_cost_total_control_eur`. Import/export/self-consumption continuam
importantes, mas como KPIs de diagnostico e qualidade da politica.

## Baselines

Os baselines ficam fechados por agora e registados como algoritmos:

| Baseline | Papel |
|---|---|
| `RandomPolicy` | Sanidade/lower bound. Nao e candidato serio. |
| `NormalNoBatteryPolicy` | BAU sem bateria. Isola efeito de storage. |
| `NormalPolicy` | BAU com bateria simples. |
| `RBCBasicPolicy` | Heuristica simples para referencia minima. |
| `RBCSmartPolicy` | Baseline local forte: PV, preco, EV service, storage, V2G seguro, picos/headroom e deferrables. |
| `RBCCommunityPolicy` | Baseline comunitario forte: usa sinais comunitarios, mercado local, import/export, surplus, EV/storage/deferrables e picos. |
| `RuleBasedPolicy` | Legacy/debug. Nao usar como baseline principal nova. |

Validacao feita antes deste snapshot:

- `RBCBasicPolicy`, `RBCSmartPolicy` e `RBCCommunityPolicy` correm em
  `entity + dynamic`.
- Smoke local passou atravessando evento real de topologia no step `1440` do
  dataset `citylearn_three_phase_dynamic_asset_changes_demo_15s_parquet`.
- Os dynamic datasets expostos localmente incluem os bundles necessarios:
  `entity_core_electrical`, `entity_community_operational`,
  `entity_forecasts_existing`, `entity_forecasts_derived`,
  `entity_temporal_derived`, `entity_action_feedback`.
- O bundle dynamic exportado inclui actions por agente e o charger dinamico
  `charger_2_dyn_1`.

## MADDPG E MATD3

`MADDPG` e `MATD3` continuam validos para o dataset anual/static. Ambos passaram
smoke local de 256 steps com a recipe W6
`w6_flex_ev_gate_repair_mid_bc`.

Nao usar `MADDPG`/`MATD3` em `entity + dynamic`: o schema rejeita isto de
proposito porque a topologia muda durante runtime e estes agentes assumem layout
fixo.

Nao lancar os templates crus como runs competitivas. Os templates servem de
base; as runs reais devem vir de `scripts/generate_phase10_w6_configs.py`,
porque as recipes W6 ativam reward comunitaria, teacher/warm-start, BC e replay
ponderado.

## Reward Atual

Ponto de partida para W6:

- `CostServiceCommunityFeasiblePrecisionRewardV46`;
- recipes W6 com `community_settlement_cost_weight > 0`;
- `export_credit_ratio = 0`;
- custo local desligado nas rewards comunitarias herdadas;
- penalizacoes de pico, EV service, battery throughput e V2G calibradas por
  recipe.

Isto nao e literalmente o KPI final do simulador por step, mas esta alinhado com
a logica corrigida: import grid custa, energia local comunitaria custa menos, e
export para grid nao da credito. Se o simulador passar a expor custo/settlement
por step nas observacoes, a reward deve ser simplificada para consumir esse
sinal nativo.

## Scorecard

Usar `docs/community_optimization_success_scorecard_pt.md` como gate:

- EV minimo aceitavel quase perfeito;
- zero violacoes eletricas;
- deferrables sem falhas relevantes;
- SOC de storage dentro de limites;
- custo oficial do simulador;
- import/export/self-consumption/picos;
- battery throughput e V2G como diagnostico, nao como objetivo isolado.

## Proxima Ronda

Depois do commit, push e nova imagem/SIF:

1. Smoke remoto curto de `RBCSmartPolicy`, `RBCCommunityPolicy`, `MADDPG` e
   `MATD3`.
2. Baselines full-year no dataset anual/static para fixar a nova escala de custo
   com Simulator 1.5.3.
3. Baselines dynamic do Gustavo/Jorge no server, nao em GPU, usando o dataset
   dynamic inteiro.
4. Runs W6 para `MADDPG` e `MATD3` apenas se os smokes e baselines baterem
   certo.
5. Nao gastar Deucalion em PPO/SAC longos ate haver sinal claro de que a reward
   e os RBCs estao fechados.

## Higiene Do Repo

- `docs/` guarda estado atual, contratos e decisoes.
- `runs/` guarda outputs locais/remotos e pode ser limpo sem afetar o commit.
- `configs/experiments/` deve conter apenas configs ativos de uma ronda nova.
- Configs remotas executadas e scorecards historicos nao devem ser versionados.
