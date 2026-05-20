# Roadmap Atual RL/MARL

Data: 2026-05-20.

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

- simulador: `softcpsrecsimulator==0.6.7`;
- interface: `entity`;
- topologia MADDPG: `static`;
- datasets principais:
  - `citylearn_three_phase_electrical_service_demo_15s_parquet`;
  - `citylearn_challenge_2022_phase_all_plus_evs`;
- variantes ja preparadas:
  - `no_v2g`;
  - `multi_charger`;
- perfil principal de observacoes: `maddpg_v2_compact`;
- candidato atual: `MADDPG V48`;
- imagem remota atual: `sha-969d417`.

O que ja esta solido:

- baselines `Random`, `NormalNoBattery`, `Normal`, `RBCBasic`, `RBCSmart`;
- reward com EV service, EV precision, custo, rede e bateria;
- teacher/warm-start/BC configuraveis;
- replay ponderado por reward;
- logs de reward/action/training;
- CUDA local validado;
- Deucalion/server integrados;
- scorecard remoto preparado;
- datasets e variantes locais/remotas preparados.

O que ainda nao esta provado:

- se `MADDPG V48` e robusto em multi-seed;
- se `MADDPG V48` generaliza bem para 2022;
- se V2G ajuda ou dificulta aprendizagem;
- se `multi_charger` quebra por escala de acoes/fases;
- se storage cria ganho real ou apenas custo/throughput inutil;
- se o critic atual chega ou precisa de MATD3;
- se um algoritmo on-policy como MAPPO e mais estavel;
- se uma abordagem com entropia como MASAC explora melhor;
- se a aprendizagem comunitaria precisa de attention/GNN.

## Criterios de Sucesso

Um candidato so deve ser considerado "bom" se for avaliado contra baselines e
nao apenas contra a propria curva de reward.

KPIs principais:

- custo total/comunitario;
- importacao/exportacao comunitaria;
- pico comunitario;
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

## Runs Remotas Em Curso

Imagem: `sha-969d417`.

Grupos submetidos:

- `MADDPG V48` original V2G-capable em GPU:
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

Quando terminarem, recolher:

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
- logs dos jobs marcados como `reject_*`, `not_finished` ou `pending`.

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

- implementar `MATD3`/`MADDPG-TD3`;
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

- implementar `MAPPO`;
- centralized value/critic;
- actors descentralizados;
- comparar contra `IPPO`.

Racional:

- MAPPO e um comparador MARL forte e defensavel;
- PPO tende a ser mais estavel, embora mais caro em samples.

### Se precisarmos de baseline RL simples

Acao:

- implementar `IPPO`;
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
2. `MATD3 / MADDPG-TD3`: primeira evolucao se critic/Q for gargalo.
3. `MAPPO`: principal comparador MARL forte.
4. `IPPO`: baseline RL robusto e simples.
5. `MASAC`: candidato se exploracao continuous for gargalo.
6. Heads por tipo de ativo: importante se `multi_charger` quebrar.
7. Attention critic: se credit assignment comunitario ficar fraco.
8. GNN policy/critic: se generalizacao/topologia passar a ser objetivo central.
9. `HAPPO/HATRPO`: interessante academicamente, mas mais caro.
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

## Fase 7 - Benchmark Final

Objetivo: comparar candidatos finais em matriz limpa.

Candidatos provaveis:

- melhor `MADDPG`/`MATD3`;
- `MAPPO` se implementado;
- `IPPO` se implementado;
- `MASAC` se implementado;
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
