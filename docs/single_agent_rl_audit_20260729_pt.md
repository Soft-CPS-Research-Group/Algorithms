# PPO e TD3 single-agent: implementacao e auditoria

Data: 2026-07-29

## Estado final desta fase

PPO e TD3 estritamente single-agent estao integrados, exercitados no simulador
real e cobertos pela suite. Isso nao significa que as receitas atuais estejam
validadas para promocao: na campanha diagnostica abaixo, nenhuma politica RL
passou todos os gates. O RBCSmart contemporaneo foi a unica referencia a
passar.

Esta conclusao substitui a leitura anterior, demasiado forte, de que os runs
curtos "confirmavam aprendizagem". Eles confirmavam wiring; os novos runs
mostram tambem aprendizagem mensuravel, mas ainda insuficiente e insegura para
promocao.

## Contrato dos agentes

- Uma stage `PPO` ou `TD3` com `count: 17` e composta como `Ensemble` de 17
  learners. Cada learner recebe exatamente uma observacao, acao e reward local.
- `PPO` usa actor Gaussiano pre-`tanh`, log-probabilidade com correcao do
  Jacobiano, value function local e rollouts on-policy.
- `TD3` reutiliza a reducao MATD3 com `num_agents=1`: estado/acao conjunta
  tornam-se locais, mantendo twin critics, delayed actor update e target policy
  smoothing.
- Checkpoints e ONNX sao separados por membro (`agent_N`) e conservam os
  indices globais no manifest.

Templates canonicos:

- `configs/templates/rl/ppo_distributed_local.yaml`
- `configs/templates/rl/td3_distributed_local.yaml`

Configs desta campanha:

- `configs/experiments/single_agent_rl_20260729/ppo_bc_pilot_seed123.yaml`
- `configs/experiments/single_agent_rl_20260729/td3_bc_pilot_seed123.yaml`
- `configs/experiments/single_agent_rl_20260729/rbcsmart_matching_pilot.yaml`

## Correcoes feitas durante a validacao

1. Transicoes controladas ou misturadas pelo teacher deixaram de entrar no
   policy loss, entropy e KL de PPO; value loss e BC continuam a poder aprender
   com elas.
2. Os updates BC extra foram movidos para depois do update PPO. Antes, alteravam
   o actor antes de calcular o rácio contra `old_log_probs`; o primeiro KL
   on-policy observado caiu de `2.99` para `0.0169` depois da correcao.
3. O BC passou a poder ponderar targets EV positivos, targets EV idle/zero e os
   raros comandos positivos de deferrables.
4. PPO ganhou um replay persistente apenas para demonstracoes do teacher. Esse
   replay nunca alimenta o objetivo on-policy e e preservado em checkpoint.
5. TD3/MADDPG rejeitam configuracoes silenciosamente inoperantes: teacher BC,
   warm-start ou residual sem teacher valido passam agora a falhar cedo.
6. A reward `LocalScorecardGuardRewardV2` passou a suportar a construcao em duas
   fases do simulador 1.5.6, em que `env_metadata` e inicialmente `None`.
7. O auditor local passou a exportar custo oficial, EV, rede, deferrables, SOC,
   outage, picos, ramping, solar, throughput, V2G e taxas charge/idle ligadas,
   com decisoes explicitas de gate.

## Superficie de comparacao

- dataset: `citylearn_three_phase_electrical_service_demo_15min_parquet`;
- schema SHA-256:
  `6ea6ab786bec8bc3fa2fb8e9997c965ad4808ea74ecd3a98a9924f802a811ec0`;
- simulador: `softcpsrecsimulator==1.5.6`;
- interface/topologia: `entity`, estatica, encoding
  `maddpg_v3_operational`;
- mercado comunitario: ativo e explicitamente igual em todos os runs;
- janela: passos `0:1023`, 1024 amostras de 15 minutos (10.67 dias);
- seed RL: `123`;
- treino curto: 4 episodios + 1 avaliacao deterministica;
- treino longo: 12 episodios + 1 avaliacao deterministica;
- baseline: RBCSmart na mesma janela/configuracao;
- custo: `district_cost_community_market_settled_total_eur` exportado pelo
  simulador, sem recalculo local.

Perfil de gate: `phase10_w6_adapted_local_v1`:

- EV minimo viavel `>= 0.99`;
- EV dentro da tolerancia `>= 0.40`;
- energia/eventos de violacao eletrica iguais a zero;
- zero ciclos deferiveis falhados e zero energia deferivel nao servida;
- SOC de storage exportado em `[0, 1]`;
- zero outage normalizado.

## Scorecard da campanha diagnostica

| Politica/receita | Treino | Custo EUR | Ratio BAU | EV min. | EV tol. | Rede kWh/eventos | Def. feitos/falhados | V2G kWh | Decisao |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RBCSmart matching | n/a | 600.66 | 0.879 | 1.000 | 0.987 | 0 / 0 | 11 / 0 | 0.00 | `PASS_LEARNING_GATES` |
| PPO r1 | 3 ep. | 592.63 | 0.867 | 0.592 | 0.263 | 0 / 0 | 0 / 11 | 0.66 | `REJECT_ev_service+deferrable_service` |
| PPO order-fixed | 4 ep. | 663.37 | 0.970 | 0.868 | 0.329 | 0 / 0 | 11 / 0 | 63.03 | `REJECT_ev_service` |
| PPO demo replay | 4 ep. | 641.74 | 0.939 | 0.947 | 0.855 | 0.120 / 2 | 11 / 0 | 84.50 | `REJECT_ev_service+electrical` |
| PPO demo replay longo | 12 ep. | 608.94 | 0.891 | 0.961 | 0.961 | 2.748 / 48 | 11 / 0 | 40.69 | `REJECT_ev_service+electrical` |
| TD3 r1 | 3 ep. | 621.76 | 0.910 | 0.803 | 0.566 | 0 / 0 | 11 / 0 | 69.86 | `REJECT_ev_service` |
| TD3 balanceado | 4 ep. | 587.98 | 0.860 | 0.921 | 0.895 | 0 / 0 | 11 / 0 | 8.15 | `REJECT_ev_service` |
| TD3 balanceado longo | 12 ep. | 598.72 | 0.876 | 0.961 | 0.895 | 0.582 / 10 | 11 / 0 | 0.22 | `REJECT_ev_service+electrical` |

Todos os runs da tabela tiveram zero violacoes SOC e zero outage. Um custo
inferior ao RBCSmart nao compensa a falha de EV ou rede.

## Leitura tecnica

- O RBCSmart carrega em apenas 16.3% dos instantes em que um EV esta ligado e
  fica idle em 83.7%. PPO/TD3 iniciais carregavam em 68--73% e usavam V2G em
  15% ou mais, explicando simultaneamente custo, imprecisao e falhas EV.
- Ponderar charge e idle reduziu fortemente o erro; o replay de demonstracoes
  resolveu parte do esquecimento PPO entre segmentos da janela.
- Mais treino levou PPO e TD3 a `0.961` no gate EV e aproximou o custo do
  RBCSmart, mas introduziu pressao eletrica. Isto e um limite real das receitas
  atuais, nao motivo para arredondar o gate.
- Nao foram lancadas seeds adicionais nem avaliacao full-year: a regra de
  campanha e nao promover uma receita que falha o screen seed-123.

## Validacao tecnica

- suite completa final: `605 passed`, `28 warnings` conhecidas;
- runs reais CityLearn concluidos a 100%, com checkpoints, ONNX e manifests;
- PPO verificado em runtime com teacher `policy_eligible=0` e actor on-policy
  `policy_eligible=1`;
- replay de demonstracoes PPO observado a crescer `256 -> 512 -> 768` e
  restaurado por checkpoint em teste;
- scorecard final:
  `runs/analysis/single_agent_rl_validation_20260729/scorecard.csv`.

Job IDs principais:

- baseline: `rbcsmart-matching-pilot-20260729-r1`;
- PPO: `ppo-bc-pilot-s123-20260729`,
  `ppo-bc-rareweighted-s123-20260729`,
  `ppo-bc-balanced-s123-20260729`,
  `ppo-bc-orderfixed-s123-20260729`,
  `ppo-bc-demoreplay-s123-20260729` e
  `ppo-bc-demoreplay-long-s123-20260729`;
- TD3: `td3-bc-pilot-s123-20260729`,
  `td3-bc-balanced-s123-20260729` e
  `td3-bc-balanced-long-s123-20260729`;
- falha diagnostica preservada: `rbcsmart-matching-pilot-20260729`.

## Limites e proximo passo

Esta e evidencia diagnostica adaptada repetidamente a uma unica janela e seed;
nao e validacao estatistica independente nem resultado anual. O source commit
era `ff5bba3`, mas o worktree estava dirty e as implementacoes/configs ainda nao
estavam publicadas numa imagem imutavel.

Nao escalar estas receitas diretamente. O proximo ciclo deve primeiro adicionar
um mecanismo explicito de seguranca/servico que preserve headroom eletrico e EV
minimo (por exemplo residual sobre teacher com limites semanticos, ou projection
shield differentiable/registrada), voltar a passar este screen e so depois usar
duas janelas sazonais held-out, seeds `123/456/789` e RBCs correspondentes.

`AgentTransformerPPO` e o branch Transformer-MATD3 nao fazem parte destes
resultados strict PPO/TD3 e mantem problemas P0 separados; nao usar os seus runs
historicos como evidencia desta campanha.

## Continuacao: seguranca local e oraculo MILP

Ainda em 2026-07-29 foi executada uma segunda iteracao, preservando exatamente
o dataset, schema, janela `0:1023`, seed `123`, interface, mercado, export e
perfil de gates da campanha diagnostica. Esta iteracao resolve o screen local,
mas nao altera a conclusao sobre validacao estatistica: a janela foi reutilizada
adaptativamente e as demonstracoes conhecem o futuro dessa mesma janela.

### Alteracoes desta iteracao

1. A reward `IndividualScorecardAlignedRewardV3` alinha o objetivo individual
   com o settlement comunitario exportado e mantem penalizacoes explicitas para
   EV, rede, storage e deferrables.
2. PPO e TD3 receberam um safety shield semantico local. Comandos deferrable sao
   atomicos/binarios; limites eletricos so ficam ativos depois de existir um
   envelope local positivo observado; se um comando de servico do teacher nao
   couber, a bateria estacionaria e colocada idle antes de preservar o servico.
3. EV e deferrables usam RBCSmart como teacher de servico em runtime. O actor
   continua a decidir a bateria estacionaria. Estes artefactos sao marcados
   runtime-only/non-deployable enquanto a dependencia nao for incorporada no
   bundle de serving.
4. O PPO passou a conservar o latente exato pre-`tanh` usado na amostragem para
   calcular o log-probability on-policy, evitando a inversao numericamente
   aproximada da acao.
5. Foi adicionado `FixedServiceOracleReplayPolicy`, registado no runner, para
   reproduzir schedules do oraculo tanto em validacao CityLearn como em
   demonstracoes BC de PPO/TD3.

### Oraculo implementado

O problema atual otimiza as 17 baterias estacionarias com os servicos EV e
deferrable fixos ao comportamento RBCSmart. Como as baterias deste dataset sao
identicas, o solver usa uma agregacao exata e expande depois o schedule para as
17 acoes fisicas.

- custo RBCSmart reconstruido: `EUR 600.6564`;
- custo sem bateria estacionaria, mantendo os restantes servicos:
  `EUR 639.5078`;
- lower bound certificado do modelo linear: `EUR 470.2145`;
- schedule conservador no modelo: `EUR 520.5926`;
- replay real no CityLearn: `EUR 521.4482`, com todos os gates passados.

O erro modelo--simulador do schedule conservador foi apenas `EUR 0.8556`. O
intervalo `EUR 470.21--521.45` e util para este subproblema, mas **nao e um
certificado do otimo global**: EV/deferrables estao fixos e o modelo ainda nao
representa conjuntamente todas as restricoes de fase/rede do simulador.

### Scorecard comparavel da continuacao

| Politica/receita | Custo EUR | Delta vs RBCSmart | EV min./tol. | Rede kWh/eventos | Def. feitos/falhados | Decisao |
|---|---:|---:|---:|---:|---:|---|
| RBCSmart matching | 600.66 | 0.00 | 1.000 / 0.987 | 0 / 0 | 11 / 0 | `PASS_LEARNING_GATES` |
| PPO + safety/service teacher | 602.07 | +1.42 | 1.000 / 0.987 | 0 / 0 | 11 / 0 | `PASS_LEARNING_GATES` |
| TD3 + safety/service teacher | 604.99 | +4.33 | 1.000 / 0.987 | 0 / 0 | 11 / 0 | `PASS_LEARNING_GATES` |
| MILP fixed-service replay | 521.45 | -79.21 | 1.000 / 0.987 | 0 / 0 | 11 / 0 | `PASS_LEARNING_GATES` |
| PPO + MILP storage BC | 562.65 | -38.00 | 1.000 / 0.987 | 0 / 0 | 11 / 0 | `PASS_LEARNING_GATES` |
| TD3 + MILP storage BC | 556.30 | -44.36 | 1.000 / 0.987 | 0 / 0 | 11 / 0 | `PASS_LEARNING_GATES` |

O TD3 ficou `EUR 6.35` abaixo do PPO e `EUR 34.85` acima do replay MILP; o PPO
ficou `EUR 41.20` acima do replay. Dois smokes anteriores foram rejeitados e
preservados: um safety shield que interpretava headroom ausente como saturacao
produziu servico invalido, e a primeira variante hibrida PPO ainda falhou o gate
EV. Estes custos nao foram promovidos.

### Evidencia desta continuacao

- scorecard conjunto:
  `runs/analysis/single_agent_rl_oracle_iteration_20260729/scorecard.csv`;
- problema/certificado/schedule MILP:
  `runs/analysis/fixed_service_battery_oracle_20260729/`;
- replay CityLearn: `fixed-service-oracle-replay-pilot-20260729`;
- PPO: `ppo-hybrid-safe-smoke-v3-s123-20260729` e
  `ppo-oracle-bc-smoke-s123-20260729`;
- TD3: `td3-hybrid-safe-smoke-v2-s123-20260729` e
  `td3-oracle-bc-smoke-s123-20260729`;
- testes focados finais desta iteracao: `86 passed`;
- suite completa final: `646 passed`, `28 warnings` conhecidas.

O proximo gate e gerar demonstracoes por janela sem usar o futuro da avaliacao,
treinar em janelas separadas e repetir em janelas sazonais held-out com seeds
`123/456/789`. Em paralelo, o oraculo deve ser alargado a EV, deferrables e
restricoes de rede/fase para poder responder a pergunta do otimo global.

Atualizacao 2026-07-31: o alargamento total-energy foi implementado e validado
por replay; “otimo” fica limitado ao modelo linear fornecido, nao a toda a
dinamica nao linear do CityLearn. A campanha sazonal/multi-seed continua por
fazer antes de congelar PPO/TD3.

## Correcao de taxonomia: benchmark estritamente por edificio

A campanha anterior usava learners separados, mas ainda incluia observacoes
comunitarias e settlement comunitario na reward. A continuacao estritamente
local desliga o mercado, usa `building_local_v1` e avalia custo/gates em cada
casa. Os resultados canonicos e a fronteira de claims estao arquivados em
`docs/experiment_history/2026-07-30_strict_building_local_ppo_milp.md`.

Neste contrato, o PPO assistido passou 17/17 gates e reduziu o custo de
`EUR 673.45` para `EUR 643.14`, vencendo RBCSmart nas 17 casas. A avaliacao sem
professor de servico foi executada separadamente e rejeitada (15/17 gates,
`EUR 700.30`). Os 17 MILPs independentes battery-only/fixed-service deram um
replay de `EUR 629.36`, tambem com 17/17 casas melhores que RBCSmart. O desenho
dos MILPs full-house e full-community entretanto implementados esta em
`docs/milp_local_community_contract_20260729_pt.md`.
