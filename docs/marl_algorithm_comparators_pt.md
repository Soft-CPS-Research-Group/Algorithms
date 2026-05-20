# Comparadores e Alternativas ao MADDPG

Data: 2026-05-20.

Objetivo: listar algoritmos/modelos que fazem sentido comparar com o MADDPG no
problema OPEVA, sem mudar a filosofia principal: o agente aprende por
observacoes, reward, replay/exploracao e arquitetura. Nada disto implica regras
no wrapper.

## Contexto do Problema

O ambiente tem caracteristicas que condicionam muito a escolha:

- varios agentes, tipicamente um por casa/building;
- acoes continuas para storage e EV chargers;
- acoes quase binarias para deferrables;
- rewards com objetivos locais e comunitarios;
- episodios longos, especialmente no dataset 15s;
- eventos raros/importantes: EV departure, deadlines, headroom/fases, picos;
- observacoes estruturadas por entidades;
- interesse em export/inference depois do treino.

Isto favorece algoritmos que suportem:

- continuous control;
- multi-agent credit assignment;
- centralized training / decentralized execution, quando possivel;
- boa estabilidade com rewards ruidosas e escalas diferentes;
- policy exportavel por agente.

## Tier 1 - Comparadores Mais Relevantes

### 1. MATD3 / MADDPG-TD3

Prioridade: muito alta. Estado: implementado como `MATD3`.

Ideia:

- manter a base MADDPG;
- trocar critic simples por twin critics;
- usar target policy smoothing;
- usar delayed actor updates;
- reduzir overestimation do Q.

Porque faz sentido:

- e a evolucao mais natural do que ja temos;
- continua em continuous action;
- mexe pouco na interface;
- ataca diretamente instabilidade de critic/Q-values, que ja apareceu nas fases
  anteriores;
- foi implementado incrementalmente mantendo o contrato de agente/export.

Risco:

- nao resolve sozinho reward mal calibrada;
- mais custo de compute por update;
- precisa de bons logs de Q1/Q2 e target.

Quando testar:

- se V48 falhar por critic instavel;
- se logs mostrarem Q-values extremos ou actor a perseguir artefactos.

Veredicto:

- primeiro comparador tecnico a testar antes de algoritmos totalmente novos.

### 2. IPPO / PPO Independente Por Agente

Prioridade: alta como baseline robusto. Estado: implementado como `IPPO`.

Ideia:

- cada casa treina uma policy PPO propria;
- sem centralized critic global;
- os outros agentes aparecem como parte do ambiente.

Porque faz sentido:

- e simples de entender e comparar;
- PPO costuma ser estavel;
- serve como baseline RL forte e honesto;
- bom para saber se o problema e mesmo MARL/critic centralizado ou se uma policy
  por casa ja chega.

Risco:

- sample inefficient;
- ignora explicitamente credit assignment comunitario;
- pode aprender estrategias locais que pioram comunidade;
- episodios 15s longos podem tornar treino caro.

Quando testar:

- depois de termos o pipeline de scorecard;
- como comparador de robustez, nao necessariamente como candidato final.

Veredicto:

- bom baseline RL; nao espero que seja o melhor, mas ajuda a diagnosticar.

### 3. MAPPO

Prioridade: alta. Estado: implementado como `MAPPO`.

Ideia:

- PPO multi-agent com centralized value/critic durante treino;
- actors descentralizados por agente.

Porque faz sentido:

- e um dos baselines MARL mais fortes e usados;
- mais estavel que MADDPG em muitos problemas;
- lida melhor com non-stationarity que IPPO;
- pode usar observacoes globais no critic e observacoes locais no actor.

Risco:

- on-policy: precisa de muitas amostras;
- com 15s pode ser caro;
- clipping/PPO com acoes continuas e action bounds precisa de cuidado;
- deferrables binarios continuam a ser um detalhe hibrido.

Quando testar:

- se MADDPG/MATD3 continuarem instaveis;
- se quisermos um comparador MARL moderno e defensavel para paper/tese.

Veredicto:

- principal algoritmo alternativo a MADDPG para benchmark serio.

### 4. MASAC / Multi-Agent SAC

Prioridade: alta/media.

Ideia:

- actor-critic off-policy com entropia;
- explora melhor que DDPG/MADDPG;
- pode usar centralized critic.

Porque faz sentido:

- continuous control;
- off-policy, logo mais sample efficient que PPO/MAPPO;
- entropia ajuda exploracao em acoes continuas;
- pode ser melhor para bater RBCSmart se a exploracao atual estiver demasiado
  presa ao professor.

Risco:

- implementacao mais complexa;
- tuning de temperatura/entropy pode ser sensivel;
- acoes binarizadas dos deferrables podem ser estranhas com Gaussian policy;
- export/inference fica mais complexo que deterministic actor, embora possivel.

Quando testar:

- se o problema principal parecer exploracao;
- se `no_v2g` mostrar que a politica precisa de explorar mais, mas sem destruir
  EV service.

Veredicto:

- muito interessante, mas depois de MATD3/MAPPO.

## Tier 2 - Boas Ideias, Mas Com Mais Custo

### 5. HAPPO / HATRPO

Prioridade: media.

Ideia:

- variantes multi-agent de PPO/TRPO com updates sequenciais por agente;
- tentam reduzir problemas de non-stationarity entre policies.

Porque faz sentido:

- teoricamente mais alinhado com multi-agent cooperative control que IPPO;
- pode ser mais estavel que MAPPO em alguns cenarios.

Risco:

- implementacao bem mais exigente;
- menos simples de explicar/operacionalizar;
- on-policy, portanto caro;
- pode nao trazer ganho suficiente face a MAPPO para justificar agora.

Quando testar:

- se MAPPO parecer promissor mas instavel;
- se precisarmos de comparador MARL forte para investigacao.

Veredicto:

- bom candidato academico, nao o primeiro para engenharia.

### 6. MAAC / Attention Critic

Prioridade: media.

Ideia:

- actor por agente;
- critic centralizado com atencao sobre outros agentes/entidades.

Porque faz sentido:

- o problema tem muitos agentes e entidades;
- nem todos os agentes importam igualmente em todos os passos;
- pode ajudar credit assignment comunitario sem concatenar tudo de forma cega.

Risco:

- arquitetura nova;
- mais dificil exportar/debugar;
- precisa de auditoria cuidadosa das observacoes globais vs locais.

Quando testar:

- se multi_charger ou 17 buildings mostrarem problemas de escala;
- se critic centralizado atual ficar demasiado grande/ruidoso.

Veredicto:

- bom caminho de arquitetura depois de confirmar que a dificuldade e escala.

### 7. GNN Policy/Critic

Prioridade: media.

Ideia:

- usar grafo de comunidade/buildings/assets;
- message passing entre casas, chargers, baterias, PV e district.

Porque faz sentido:

- o simulador ja tem entidade/topologia;
- energia comunitaria e fases sao naturalmente grafos;
- poderia generalizar melhor para mais casas/carregadores.

Risco:

- maior mudanca arquitetural;
- export/inference mais exigente;
- precisa de contrato estavel para edges/topologia;
- dynamic topology ainda nao e boa combinacao com MADDPG atual.

Quando testar:

- se o objetivo passar a ser generalizacao entre comunidades/topologias;
- se multi_charger mostrar que MLP por agente nao escala.

Veredicto:

- promissor para tese/produto futuro, nao para a proxima v49.

## Tier 3 - Menos Indicados Para Este Caso

### 8. QMIX / VDN

Prioridade: baixa.

Porque nao e primeira escolha:

- foram desenhados sobretudo para acoes discretas;
- aqui temos storage/EV continuous control;
- discretizar carregamento e bateria pode perder qualidade ou explodir o espaco
  de acoes.

Quando poderia fazer sentido:

- se criarmos uma versao discretizada pequena do problema;
- se deferrables forem o foco principal.

Veredicto:

- bom comparador academico em caso discreto, fraco para este ambiente completo.

### 9. COMA

Prioridade: baixa.

Porque nao:

- mais orientado a acoes discretas;
- credit assignment interessante, mas menos natural para continuous EV/storage;
- provavelmente mais trabalho que beneficio.

Veredicto:

- nao priorizar.

### 10. DQN / Rainbow

Prioridade: muito baixa.

Porque nao:

- acoes continuas tornam isto pouco adequado;
- discretizacao grosseira seria injusta contra RBC/MADDPG.

Veredicto:

- nao usar para o problema completo.

## Arquiteturas Que Podem Melhorar O Mesmo Algoritmo

Estas nao sao necessariamente novos algoritmos, mas podem entrar como variantes
MADDPG/MATD3/MAPPO.

### Heads Por Tipo de Ativo

Prioridade: alta se `multi_charger` quebrar.

Ideia:

- um trunk comum por agente;
- heads separadas para:
  - storage;
  - EV chargers;
  - deferrables.

Beneficio:

- cada tipo de acao tem escala/semantica diferente;
- ajuda a evitar que EV/deferrable/storage lutem pela mesma saida final.

### Action Distribution Hibrida

Prioridade: media.

Ideia:

- continuous para storage/EV;
- Bernoulli/logit para deferrables.

Beneficio:

- deferrable agora e comando ON/OFF por threshold;
- policy continua a conseguir aprender, mas uma head binaria poderia ser mais
  natural.

Risco:

- muda bastante training/export;
- para MADDPG deterministic, isto nao e trivial.

### LayerNorm / FeatureNorm Interna

Prioridade: media/alta.

Ideia:

- LayerNorm nos MLPs do actor/critic;
- estabilizar escalas de features e Q-values.

Beneficio:

- baixo custo;
- pode ajudar nos datasets 15s vs 2022.

Risco:

- pode piorar se ja tivermos normalization suficiente nos encoders.

### Recurrent Policy / LSTM / GRU

Prioridade: media/baixa por agora.

Porque pode fazer sentido:

- problema temporal;
- EV departure e deferrable deadlines dependem de historico.

Porque nao comecar aqui:

- ja temos observacoes de tempo, SOC, departure, can_start, etc.;
- LSTM aumenta muito custo/debug;
- se reward/replay/critic ainda forem o gargalo, LSTM so mascara o problema.

Veredicto:

- testar apenas se o scorecard mostrar falhas de memoria temporal que nao
  aparecem nas observacoes atuais.

## Recomendacao de Ordem

1. Fechar scorecard V48 remoto.
2. Se V48 estiver perto:
   - correr seeds `456/789`;
   - nao mudar algoritmo ainda.
3. Se critic/Q parecer instavel:
   - implementar `MATD3`/`MADDPG-TD3`.
4. Se quisermos comparador MARL forte:
   - implementar `MAPPO`.
5. Se exploracao for o bloqueio:
   - testar `MASAC`.
6. Se multi_charger quebrar:
   - testar heads por tipo de ativo antes de trocar algoritmo.
7. Se escala/topologia for o problema:
   - estudar attention critic ou GNN.

## Matriz Resumida

| Candidato | Prioridade | Tipo | Pros | Riscos |
|---|---:|---|---|---|
| MATD3 / MADDPG-TD3 | Muito alta | Off-policy CTDE | minimo delta ao MADDPG, melhora critic | mais compute por update |
| IPPO | Alta | On-policy independente | baseline RL simples/estavel | ignora credit assignment comunitario |
| MAPPO | Alta | On-policy CTDE | comparador MARL forte | sample inefficient |
| MASAC | Alta/media | Off-policy entropy | melhor exploracao continuous | implementacao/tuning mais dificil |
| HAPPO/HATRPO | Media | On-policy MARL | bom comparador academico | complexo e caro |
| Attention critic | Media | Arquitetura CTDE | escala melhor com agentes | debug/export mais dificil |
| GNN policy/critic | Media | Arquitetura grafo | bom para comunidades/topologia | grande mudanca |
| QMIX/VDN | Baixa | Value factorization discreta | util se discretizar | pouco adequado a continuous |
| COMA | Baixa | Policy gradient discreto | credit assignment | pouco adequado aqui |
| DQN/Rainbow | Muito baixa | Discreto | simples em toy problems | nao serve para EV/storage continuo |
