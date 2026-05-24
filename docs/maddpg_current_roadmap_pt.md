# Roadmap Atual RL/MARL

Data: 2026-05-22.

Este e o documento vivo para orientar o trabalho daqui para a frente. O nome do
ficheiro ainda menciona MADDPG porque esse foi o ponto de partida, mas o objetivo
real ja nao e "encontrar o melhor MADDPG". O objetivo e encontrar o melhor
controlador para este caso: comunidade energetica com EVs, baterias,
deferrables, fases, limites fisicos, custo, autoconsumo e requisitos de servico.

Os documentos antigos continuam em `docs/archive/maddpg_history/` como
historico/evidencia. Este documento deve ser a fonte principal para decidir o
proximo passo.

## Objetivo Real

Encontrar, validar e exportar um controlador RL/MARL que:

- controle todos os ativos disponiveis nos datasets: EVs, baterias e
  deferrables;
- respeite limites fisicos, fases/headroom e requisitos de servico;
- cumpra EV departure de forma util para a pessoa:
  - pelo menos o SOC minimo aceitavel;
  - idealmente dentro da banda de tolerancia do target;
- reduza custo e importacao externa;
- aumente uso de energia renovavel local/comunitaria;
- reduza picos comunitarios;
- seja melhor ou pelo menos competitivo contra baselines fortes;
- funcione nos datasets principais e variantes;
- exporte artefactos compativeis com inference.

MADDPG e apenas o primeiro candidato serio porque ja existe, ja corre, ja exporta
e ja foi bastante instrumentado. Se os resultados mostrarem que outro algoritmo e
melhor para o problema, o roadmap deve mudar para esse algoritmo.

Regra permanente: nao meter comportamento prescritivo no wrapper. O controlador
deve melhorar por observacoes, reward, exploracao, replay, arquitetura,
algoritmo e treino.

## Estado Atual

Base tecnica:

- simulador: `softcpsrecsimulator==1.0.2`;
- interface: `entity`;
- topologia MADDPG: `static`;
- datasets principais:
  - `citylearn_three_phase_electrical_service_demo_15s_parquet`;
  - `citylearn_challenge_2022_phase_all_plus_evs`;
- variantes ja preparadas:
  - `no_v2g`;
  - `multi_charger`;
- perfil principal de observacoes: `maddpg_v3_operational`;
- candidato atual: `MADDPG V48`;
- imagem remota atual: a reconstruir a partir do requirement `softcpsrecsimulator==1.0.2`.
- performance recente:
  - replay compacto prealocado em NumPy/torch;
  - entity layout cacheado por topologia/layout;
  - encoding `maddpg_v3_operational` compilado e com bloco `minmax`
    vetorizado;
  - rewards do repo declaram payload minimo de observacoes para o simulador;
  - configs novas exportam KPIs/series so no episodio final quando aplicavel;
  - BAU export fica desligado nas configs de treino/comparacao atuais;
  - `step_many` existe no simulador 1.0.2, mas ainda fica em estudo antes de entrar no treino.

Perfil local curto, 2026-05-24, Simulator 1.0.2, 2022 all-plus-EVs, 17 agentes:

```text
MADDPG v3 operational:
  mediana step perfilado:              ~20.7 ms
  env.step:                            ~12.4 ms
  entity->maddpg_v3 direto:            ~7.2 ms
  model_observation_encoding:          ~0.0 ms
  update neural MADDPG real:           ~0.6 ms
  loop local sem profiler, 168 steps:  ~27.5 ms/step

O antigo custo de `entity layout` caiu de ~10.5 ms para ~1.1 ms com o plano de
fontes agrupado. Nos runs neurais sem raw context, o wrapper agora passa
diretamente `entity payload -> maddpg_v3`, deixando o custo de encoding dentro
da fase entity-model-observation e removendo o re-encoding no `update`.
```

O que ja esta solido:

- baselines `Random`, `NormalNoBattery`, `Normal`, `RBCBasic`, `RBCSmart`;
- agentes comparadores registados e executaveis:
  - `MATD3`;
  - `MASAC`;
  - `IPPO`;
  - `MAPPO`;
  - `HAPPO`;
- reward com EV service, EV precision, custo, rede e bateria;
- rewards exploratorias locais:
  - `V49` para dar mais valor a storage/comunidade;
  - `V50` para reforcar EV deadline;
  - `V51` para testar precisao perto do SOC alvo;
  - `V52` para reforcar objetivo comunitario/pico;
  - `V55`/`V56` para testar BC warmup e policy finetune conservador;
- teacher/warm-start/BC configuraveis;
- BC warmup extra configuravel no actor;
- replay ponderado por reward;
- replay compacto prealocado para reduzir RAM em runs longas 15s;
- logs de reward/action/training;
- diagnostico de acoes EV condicionadas a EV ligado (`Action/ev_connected_*`);
- CUDA local validado;
- Deucalion/server integrados;
- scorecard remoto preparado para `MADDPG`, `MATD3`, `MASAC`, `IPPO`, `MAPPO`
  e `HAPPO`;
- datasets e variantes locais/remotas preparados.
- matriz remota pendente dos comparadores:
  `configs/experiments/phase6_algorithm_matrix/remote_pending/`.

O que ainda nao esta provado:

- se `MADDPG V48` e robusto em multi-seed;
- se `MADDPG V48` generaliza bem para 2022;
- se V2G ajuda ou dificulta aprendizagem;
- se `multi_charger` quebra por escala de acoes/fases;
- se storage cria ganho real ou apenas custo/throughput inutil;
- se a melhoria de EV deadline deve vir mais de reward ou de teacher/BC;
- se o critic atual chega ou se `MATD3` melhora;
- se `MASAC` explora melhor que DDPG-style;
- se um algoritmo on-policy como `MAPPO`/`IPPO`/`HAPPO` e mais estavel;
- se a aprendizagem comunitaria precisa de attention/GNN.

## Criterios de Sucesso

Um candidato so deve ser considerado "bom" se for avaliado contra baselines e
nao apenas contra a propria curva de reward.

O scorecard congelado para a matriz atual esta em
`docs/community_optimization_success_scorecard_pt.md`. A matriz remota oficial
2022 full-year submetida em 2026-05-22 esta documentada em
`docs/phase6_2022_scorecard_remote_submission_pt.md`.

KPIs principais:

- custo total/comunitario;
- importacao/exportacao comunitaria;
- pico comunitario;
- autoconsumo solar/comunitario quando o KPI existir;
- community-market import share quando o KPI existir;
- `ev_min_acceptable_feasible_rate`;
- `ev_within_tolerance_feasible_rate`;
- deficits medios/maximos de EV departure;
- violacoes de rede/fase/headroom;
- deferrables servidos dentro da janela;
- battery throughput e ciclos;
- energia renovavel local/comunitaria utilizada;
- estabilidade entre seeds.

Comparacao principal:

- contra `RBCSmart` para custo/operacao inteligente;
- contra `NormalNoBattery` e `Normal` para perceber se bateria ajuda ou estraga;
- contra `RBCBasic` para ver ganho sobre heuristica simples;
- contra `Random` apenas como sanidade/lower bound.

Regras de decisao:

- nao aceitar reducao de custo se destruir EV service;
- nao aceitar bom EV service se criar violacoes;
- nao aceitar melhoria aparente se for so uma seed;
- nao aceitar storage se throughput/ciclos forem desproporcionados;
- nao aceitar algoritmo novo se nao exportar ou nao encaixar em inference.

## Reward Scout Local

Foram testadas V49-V56 em smokes curtos e runs locais de 8/16 episodios para
nao esperar pelas runs remotas.
Detalhe em `docs/maddpg_phase6_reward_scout_pt.md`.

Leitura curta:

- `V48` passou a candidata local principal: no 2022/16 episodios e seeds
  `123/456/789` bateu `RBCSmart` em custo, manteve
  `ev_min_acceptable_feasible_rate=1.0` e reduziu o sinal de pico comunitario;
- `V48` multi-seed local: custo medio 409.74 EUR,
  `ev_within_tolerance_feasible_rate` medio 0.474 e pico comunitario reward
  mean medio 0.444;
- `V50`/`V52` cumprem minimo EV mas fazem over-service e pioram custo;
- `V55`/`V56` mostram que BC extra/policy finetune podem baixar algum sinal de
  pico, mas ainda degradam custo e EV precision nesta configuracao;
- `V54` fica como diagnostico de clone/teacher, nao como melhor candidata.

Proxima iteracao local deve evitar aumentar pesos de reward comunitaria sem
preservar primeiro a politica EV. O caminho mais promissor e levar a V48 para
runs remotas/longas, comparar contra datasets variantes, e so depois testar
policy improvement muito conservador ou algoritmos alternativos.

## Runs Remotas Em Curso

Imagem: `sha-969d417`.

Correcao importante, 2026-05-21:

- os configs remotos `remote_20260520_*_2022_*` usavam apenas `0..1999`
  horas, cerca de 83 dias, e nao devem ser tratados como resultados anuais;
- os configs remotos `remote_20260520_*_15s_*` usavam um dia de 15 segundos
  repetido por episodios, e nao devem ser tratados como prova final;
- os jobs curtos ainda ativos foram parados/cancelados;
- a matriz correta para o dataset 2022 original esta agora em
  `configs/experiments/phase6_2022_full_year/`.

Nova matriz 2022 anual submetida:

- dataset: `citylearn_challenge_2022_phase_all_plus_evs_data_2026_05_20`;
- janela: `simulation_start_time_step=0`,
  `simulation_end_time_step=8759`, `episode_time_steps=8760`;
- baselines anuais, 1 episodio:
  - `Random`;
  - `NormalNoBattery`;
  - `Normal`;
  - `RBCBasic`;
  - `RBCSmart`;
- `MADDPG V48`, 6 episodios anuais por seed:
  - seed `123`;
  - seed `456`;
  - seed `789`.

Registos locais:

- `configs/experiments/phase6_2022_full_year/README.md`;
- `runs/remote_configs/phase6_2022_full_year_2026_05_21/submitted_full_year_jobs_2026_05_21_sha969d417.csv`;
- `runs/remote_configs/phase6_remote_2026_05_20/submitted_jobs_2026_05_20_sha969d417.csv`;
- `runs/remote_configs/phase6_remote_2026_05_20/submitted_cpu_jobs_2026_05_20_sha969d417.csv`;
- `runs/remote_configs/phase6_remote_2026_05_20/submitted_server_variant_full_jobs_2026_05_20_sha969d417.csv`.

Quando terminarem, recolher e gerar scorecards primeiro da matriz anual 2022:

```bash
.venv/bin/python scripts/run_phase6_remote_analysis.py \
  --jobs-file runs/remote_configs/phase6_2022_full_year_2026_05_21/submitted_full_year_jobs_2026_05_21_sha969d417.csv \
  --output-dir runs/remote_results/phase6_2022_full_year_sha969d417
```

Os ficheiros antigos de 2026-05-20 podem continuar a servir para diagnostico,
mas nao para conclusao principal:

```bash
.venv/bin/python scripts/run_phase6_remote_analysis.py \
  --jobs-file runs/remote_configs/phase6_remote_2026_05_20/submitted_jobs_2026_05_20_sha969d417.csv \
  --jobs-file runs/remote_configs/phase6_remote_2026_05_20/submitted_cpu_jobs_2026_05_20_sha969d417.csv \
  --jobs-file runs/remote_configs/phase6_remote_2026_05_20/submitted_server_variant_full_jobs_2026_05_20_sha969d417.csv \
  --output-dir runs/remote_results/phase6j_sha969d417
```

Ficheiros a ler:

- `runs/remote_results/phase6j_sha969d417/scorecard.md`;
- `runs/remote_results/phase6j_sha969d417/scorecard.csv`;
- `runs/remote_results/phase6j_sha969d417/building_scorecard.md`;
- `runs/remote_results/phase6j_sha969d417/building_scorecard.csv`;
- logs dos jobs marcados como `reject_*`, `not_finished` ou `pending`.

Performance/tempo de execucao:

- auditoria atual: `docs/phase6_runtime_performance_audit_pt.md`;
- estrategia de escala para 15s:
  `docs/phase6_15s_training_scale_strategy_pt.md`;
- auditoria de degradacao/memoria 15s:
  `docs/phase6_15s_runtime_degradation_pt.md`;
- leitura atual: tempo por step ficou estavel numa run local de 1 dia 15s; o
  risco de memoria do replay foi mitigado com replay compacto, embora
  actor/critic e o numero total de steps continuem a dominar runtime;
- regra pratica: baselines 2022 full-year custam minutos; MADDPG 2022
  full-year em GPU custa cerca de 6h por seed;
- antes de submeter mais long-runs, resolver/mitigar falhas `stale_status` no
  worker/orchestrator.

## Fase 6J - Ler Evidencia Atual

Objetivo: fechar a fotografia do que ja correu antes de mudar mais codigo.

Perguntas:

- `MADDPG V48` bate ou chega perto de `RBCSmart`?
- o ganho vem de EV, bateria, V2G, comunidade ou acaso?
- o dataset 15s e o 2022 contam a mesma historia?
- `no_v2g` melhora estabilidade?
- `multi_charger` quebra?
- as falhas estao concentradas no Building 15/fases/headroom?
- storage esta a ser util ou esta a gerar throughput caro?

Output esperado:

- tabela por dataset/variant/policy/seed;
- tabela por building para localizar falhas de EV, fases, bateria, solar e
  deferrables;
- decisao `promote`, `iterate_maddpg`, `switch_to_matd3`, `test_mappo`,
  `test_masac`, `fix_data_or_baseline`;
- lista curta de bugs/estranhezas a auditar.

## Fase 6K - Matriz de Decisao de Algoritmos

Objetivo: decidir se continuamos em MADDPG ou se abrimos outro algoritmo.

### Se MADDPG V48 estiver forte

Acao:

- correr seeds `456` e `789`;
- manter algoritmo;
- fazer so ajustes pequenos em reward/exploracao;
- preparar benchmark final.

Racional:

- se o algoritmo ja e competitivo, trocar cedo aumenta risco e custo sem ganho
  claro.

### Se custo/EV forem bons mas instaveis por seed

Acao:

- testar `MATD3`/`MADDPG-TD3`;
- twin critics;
- target policy smoothing;
- delayed actor update;
- logs Q1/Q2;
- manter o resto o mais igual possivel.

Racional:

- e a evolucao mais natural do MADDPG;
- ataca overestimation e critic instavel;
- mantem off-policy e export semelhante.

### Se MADDPG for sensivel demais a reward/exploracao

Acao:

- testar `MASAC`;
- critic centralizado se possivel;
- entropia para exploracao;
- comparar com e sem teacher/warm-start.

Racional:

- SAC costuma explorar melhor em continuous control;
- pode descobrir estrategias de EV/storage/V2G que DDPG nao encontra.

### Se MADDPG/MATD3 forem demasiado instaveis

Acao:

- testar `MAPPO`;
- centralized value/critic;
- actors descentralizados;
- comparar contra `IPPO`.

Racional:

- MAPPO e um comparador MARL forte e defensavel;
- PPO tende a ser mais estavel, embora mais caro em samples.

### Se precisarmos de baseline RL simples

Acao:

- testar `IPPO`;
- uma policy PPO por agente;
- sem critic centralizado.

Racional:

- ajuda a perceber se o problema precisa mesmo de MARL complexo;
- baseline honesto para paper/tese.

### Se multi_charger/topologia forem o problema

Acao:

- primeiro testar heads por tipo de ativo:
  - storage;
  - EV charger;
  - deferrable;
- depois attention critic;
- depois GNN se for preciso generalizar topologias.

Racional:

- `multi_charger` pode falhar por arquitetura de output, nao por algoritmo;
- attention/GNN so devem entrar quando MLP/heads forem insuficientes.

## Algoritmos Candidatos

Prioridade atual:

1. `MADDPG V48/V49`: candidato implementado e instrumentado.
2. `MATD3 / MADDPG-TD3`: implementado; primeira evolucao se critic/Q for gargalo.
3. `MASAC`: implementado; candidato se exploracao continuous for gargalo.
4. `MAPPO`: implementado; principal comparador MARL forte.
5. `IPPO`: implementado; baseline RL robusto e simples.
6. `HAPPO`: implementado; comparador on-policy sequencial por agente.
7. Heads por tipo de ativo: trabalho futuro se `multi_charger` quebrar.
8. Attention critic: trabalho futuro se credit assignment comunitario ficar fraco.
9. GNN policy/critic: trabalho futuro se generalizacao/topologia passar a ser objetivo central.
10. `QMIX/VDN/COMA/DQN`: baixa prioridade no problema completo por causa das
    acoes continuas.

Detalhe tecnico: `docs/marl_algorithm_comparators_pt.md`.

## Fase 6L - Reward Comunitaria e Credit Assignment

Objetivo: garantir que o algoritmo aprende comunidade, nao apenas casa isolada.

Trabalhos a fazer:

- separar reward em componentes logadas:
  - custo/import;
  - export/venda;
  - autoconsumo local;
  - autoconsumo comunitario;
  - picos comunitarios;
  - EV service;
  - EV precision;
  - deferrables;
  - network/headroom;
  - battery comfort/throughput;
- confirmar se o settlement comunitario do simulador esta refletido no KPI;
- decidir se a reward total deve ser:
  - shared reward comunitaria;
  - reward local + termo comunitario;
  - reward por ativo;
  - mistura ponderada por agente;
- testar pesos sem criar regras prescritivas.

Hipotese importante:

- se o custo comunitario e o autoconsumo comunitario nao estiverem bem
  representados na reward, o agente pode cumprir EVs e ainda assim perder para
  `RBCSmart`.

## Fase 6M - Storage, V2G e Baterias

Objetivo: perceber quando baterias/EV discharge ajudam mesmo.

Trabalhos a fazer:

- comparar `NormalNoBattery`, `Normal`, `RBCSmart`, `MADDPG` e `no_v2g`;
- medir battery throughput vs poupanca;
- medir EV discharge perto de departure;
- garantir limites min/max de bateria quando existirem;
- manter conforto/servico acima de custo;
- testar reward com menor penalizacao de ciclos se a bateria estiver a criar
  valor comunitario real;
- testar curriculum V2G se descarga EV prejudicar aprendizagem.

Decisao possivel:

- se `no_v2g` ganha consistentemente, treinar primeiro sem V2G e so depois
  liberar descarga como fase curricular.

## Fase 6N - Experiencias Longas e Seeds

Objetivo: sair de evidencia pontual para robustez.

Matriz minima antes de chamar algo "bom":

- dataset 15s original;
- dataset 2022 original;
- variante `no_v2g`;
- variante `multi_charger`;
- pelo menos seeds `123`, `456`, `789` para candidatos finais;
- baselines sempre presentes no mesmo dataset/variant.

Ordem:

1. smoke local;
2. run curta diagnostica;
3. run remota seed123;
4. se passar, multi-seed;
5. so depois benchmark final.

## Fase 6O - Matriz Curta de Algoritmos

Objetivo: enquanto se espera pelas runs longas, deixar pronta a comparacao entre
familias de algoritmos.

Configs preparadas:

- `configs/experiments/phase6_algorithm_matrix/remote_pending/`;
- `MATD3`, `MASAC`, `IPPO`, `MAPPO`, `HAPPO`;
- datasets `15s` e `2022`;
- variants `original`, `no_v2g`, `multi_charger`;
- reward `CostServiceCommunityFeasiblePrecisionRewardV46`;
- encoding `maddpg_v3_operational`;
- seed `123`;
- MLflow desligado;
- export de KPIs ativo.

Regras:

- nao submeter estas configs na imagem `sha-969d417`, porque essa imagem nao
  inclui todos os commits dos comparadores;
- primeiro recolher o scorecard das runs atuais;
- depois criar nova imagem Docker/SIF a partir do commit atual;
- correr primeiro `MATD3` e `MASAC` nos datasets originais;
- usar `IPPO`/`MAPPO`/`HAPPO` como comparadores de estabilidade;
- so promover para long run quem passar EV service, rede e custo competitivo.

Metrica comunitaria passa a ser parte da leitura, nao detalhe secundario:

- reduzir custo continua importante;
- mas tambem interessa reduzir picos;
- reduzir net exchange/import externo;
- aumentar autoconsumo solar/comunitario;
- perceber se a bateria/V2G cria valor comunitario ou apenas throughput.

## Fase 7 - Benchmark Final

Objetivo: comparar candidatos finais em matriz limpa.

Candidatos provaveis:

- melhor `MADDPG`/`MATD3`;
- `MASAC`;
- `MAPPO`;
- `IPPO`;
- `HAPPO`;
- `RBCSmart`;
- `NormalNoBattery`;
- `Normal`;
- `RBCBasic`;
- `Random`.

Requisitos:

- mesmos datasets;
- mesmas seeds quando aplicavel;
- mesmos KPIs;
- scorecard unico;
- tabela de custo, EV service, EV precision, picos, violacoes, storage e tempo
  de treino;
- logs suficientes para explicar porque ganhou/perdeu.

## Fase 8 - Export e Inference

Objetivo: garantir que o melhor controlador nao fica preso ao treino.

Validar:

- `artifact_manifest.json`;
- nomes de observacoes encoded;
- nomes/bounds de acoes;
- compatibilidade com repo de inference;
- reproducibilidade do encoding profile;
- ONNX/model export por agente;
- configs resolvidas;
- ausencia de dependencia em MLflow para inference.

## Fase 9 - Produto/Tese

Objetivo: transformar resultados em conclusao defensavel.

Perguntas finais:

- qual algoritmo ganhou e por que razao?
- quanto melhora contra `RBCSmart`?
- o ganho e em custo, EV service, picos, renovaveis ou tudo?
- o ganho e robusto a dataset/seed/topologia?
- onde ainda falha?
- o algoritmo e operacionalmente viavel?
- o custo de treino compensa?
- consegue exportar para inferencia real?

## O Que Nao Fazer Agora

- nao assumir que MADDPG tem de ser o vencedor;
- nao implementar varios algoritmos novos antes de ler o scorecard remoto;
- nao trocar reward e algoritmo ao mesmo tempo sem controlo;
- nao mexer no wrapper para melhorar KPI;
- nao tirar conclusoes finais de smokes curtos;
- nao aceitar `RBCSmart` como oracle perfeito sem continuar a auditar;
- nao ignorar comunidade/autoconsumo se o objetivo e energia comunitaria;
- nao otimizar so para `no_v2g` e esquecer o caso original V2G-capable.

## Documentos de Suporte

Entrada/indice:

- `docs/README.md`.

Documentos atuais de decisao:

- `docs/maddpg_current_roadmap_pt.md`;
- `docs/maddpg_phase6j6k_remote_decision_pt.md`;
- `docs/phase6_remote_analysis_pipeline_pt.md`;
- `docs/marl_algorithm_comparators_pt.md`.
- `docs/rl_marl_algorithm_matrix_pt.md`.

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
