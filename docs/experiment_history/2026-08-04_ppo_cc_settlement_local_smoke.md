# Smoke local end-to-end do protocolo PPO/CC com settlement

- Campanha: `ppo_cc_settlement_smoke_v1`
- Arquivo: `2026-08-04T19:00:46+01:00`
- Árvore de código executada: `3e1c5fb-dirty` (apenas o gerador/testes deste
  smoke ainda não estavam commitados)
- Imagem: nenhuma; ambiente `.venv` local
- Dataset: `citylearn_three_phase_electrical_service_demo_15min_parquet`
- Janela: `0:384`, 385 amostras e 384 transições de 15 minutos
- Settlement: ligado em todas as quatro linhas (`0.8/0.8/0.0`)
- Perfil: `cc_frozen_leaf_scorecard_v1`
- Evidência: smoke de integração; não é evidência anual nem promoção de policy

## Resultado técnico

Os quatro pipelines canónicos passaram o runner real de ponta a ponta e
produziram `result.json`, `summary.json`, manifest, timeseries e
`exported_kpis.csv`. Os dois pipelines PPO carregaram o pack compacto seed 789.

Nos dois pipelines com CC foram exercidas as três fases necessárias:

1. 96 decisões de recolha BC e 2 000 atualizações BC;
2. um rollout completo de 96 decisões e uma atualização PPO real;
3. uma passagem final determinística, da qual vieram os KPIs e o scorecard.

Os checkpoints finais confirmam `bc_pretrain_done=true` e
`ppo_update_count=1`. O `CC-SMART` terminou a atualização com
`pg=-0.3430`, `v=18.8775`, `ent=1.4189`; o `CC-PPO` com `pg=0.0214`,
`v=12.1287`, `ent=1.4175`. Em ambos, `kl_stop=false`.

## Scorecard do smoke

| Par | Custo settled de referência | Custo settled com CC | Delta | Gates duros | Decisão curta |
|---|---:|---:|---:|---|---|
| SMART / CC-SMART | EUR 226,2157 | EUR 225,2388 | EUR -0,9770 (-0,43%) | PASS / PASS | `PASS_CC_SCORECARD` |
| PPO / CC-PPO | EUR 211,3752 | EUR 218,5426 | EUR +7,1674 (+3,39%) | PASS / PASS | `REJECT_COST` |

Todos os quatro tiveram EV minimum `1.0`, EV within tolerance `0.962963`, zero
violações elétricas, zero ciclos deferrable falhados e zero energia deferrable
por servir. Os KPIs settled oficiais, contrafactual, poupança do mercado e
trocas locais foram exportados com valores não nulos.

No par SMART, o CC também reduziu importação, ramping e emissões e aumentou
ligeiramente o autoconsumo solar nesta janela. No par PPO, melhorou pico diário,
ramping, load factor e emissões, mas piorou custo, importação, pico absoluto e
autoconsumo. Isto demonstra que o scorecard completo está ligado; não permite
concluir desempenho porque o CC recebeu apenas uma atualização numa janela de
quatro dias reutilizada para treino e avaliação.

## Problemas que o smoke apanhou

- O primeiro gerador funcionava quando importado pelos testes, mas falhava em
  execução direta por não inicializar o caminho do repositório.
- Uma janela de 384 amostras CityLearn produz apenas 383 transições; foi
  corrigida para 385 amostras/384 transições.
- Com apenas duas passagens e `deterministic_finish=true`, a segunda era
  avaliação e o wrapper não chamava `update()`. A receita ficou explicitamente
  BC -> treino -> avaliação em três passagens.

Os dois últimos pontos são agora invariantes testados pelo gerador. Os quatro
templates anuais permaneceram byte-equivalentes ao gerador canónico e não
foram alterados.

## Evidência local ignorada pelo Git

- Configs derivados: `runs/local_configs/ppo_cc_settlement_smoke_v1/`
- SMART: `runs/ppo_cc_settlement_smoke_v1/jobs/ppo-cc-smoke-385-smart`
- PPO: `runs/ppo_cc_settlement_smoke_v1/jobs/ppo-cc-smoke-385-ppo-seed789`
- CC-SMART: `runs/ppo_cc_settlement_smoke_v1/jobs/ppo-cc-smoke-385-train-cc-smart-seed123`
- CC-PPO: `runs/ppo_cc_settlement_smoke_v1/jobs/ppo-cc-smoke-385-train-cc-ppo-seed789`
- Scorecards: `runs/ppo_cc_settlement_smoke_v1/scorecards/`

## Próximo gate

A suite completa passou antes do commit. O próximo passo é fazer push e esperar
pela imagem commit-specific. Só depois escolher o host com `/hosts` e `/queue`,
fazer preflight estrito e lançar as quatro linhas anuais remotas. Os números
deste documento não entram na tabela anual final.
