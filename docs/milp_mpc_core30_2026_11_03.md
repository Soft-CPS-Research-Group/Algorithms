# MILP com horizonte móvel no Core30 — 03/11/2026

**Relatório histórico.** A [referência física alinhada](milp_physics_reference_core30_2026_11_03.md) substitui os resultados desta aproximação conservadora. Os valores abaixo mantêm-se para rastreabilidade; os seus limites duais não constituem limites do Simulator.

`MILP` passa a identificar o controlador comunitário que volta a otimizar em cada passo. `TotalOracleReplayPolicy` e os templates de planos semanais permanecem como oráculo offline; o MILP individual conserva-se para diagnóstico, fora da comparação principal.

## Contrato e política

O agente usa o mesmo runner, observações raw da interface entity estática, ações por casa, avaliação determinística e exports finais. Não foi introduzido um ramo MILP no runner. A conversão de unidades reutiliza a validação do replay estrito.

Em cada passo de 15 minutos, o agente constrói um problema com os próximos 96 intervalos, ou os intervalos restantes se o episódio terminar antes. O fornecedor de previsões lê apenas essa fatia para a extração; preços, cargas, PV e eventos fora dela não entram na otimização. Os ficheiros podem estar em cache, mas um teste altera os valores futuros fora da janela e confirma um problema idêntico. Esta primeira versão usa previsão perfeita das 24 horas, não um previsor treinado.

O estado inicial do problema vem das observações atuais, em unidades SOC de razão segundo o contrato entity. As perdas passivas podem colocá-lo abaixo da reserva de descarga; essa energia é preservada, sem ser elevada artificialmente à reserva. As restrições dos estados futuros obrigam à reposição por carga real. Um ensaio inicial de julho revelou este caso, corrigido e coberto por testes de bateria e EV antes da repetição completa. A política conserva os ciclos iniciados, verifica que o Simulator os aplicou e incorpora a cauda do perfil com a fase elétrica correta, inclusive períodos de potência zero. A cauda não provoca novo comando de arranque. O plano futuro é descartado após executar a primeira ação e volta a ser calculado no passo seguinte.

A função objetivo é custo comunitário, depois de minimizar défices EV. O custo não contém penalização de pico. Mantêm-se compra externa ao preço integral, partilha a 80%, venda externa a zero e PV próprio sem custo operacional. Core30 não permite V2G.

A reserva terminal das baterias é fixa: 50% da capacidade física. Para sessões EV que atravessam o fim da janela usa-se o alvo anunciado pelo utilizador como reserva, sujeito à fase de serviço viável. Não se lê a energia futura do Community. O horizonte não aumenta no último dia. Uma janela pequena sem capacidade para atingir a reserva pode ser inviável; não há substituição silenciosa por RBC nem execução de um plano semanal.

Os limites físicos conservadores do modelo são distintos dos limiares estratégicos 20–90% dos RBCs. Todas as comparações devem mostrar a energia final por ativo e não atribuir automaticamente toda a diferença económica à inteligência da política.

## Solver e tempo por decisão

Uma reotimização pode envolver duas chamadas ao HiGHS: serviço e custo. O default é uma thread, tolerância relativa 0,1% e limite de dez segundos por fase. Um plano viável devolvido ao atingir o limite é utilizado com o estado e gap explícitos. Sem plano viável, o job falha; não existe fallback oculto.

O adaptador partilhado controla o pool global de threads do HiGHS. Chamadas no mesmo processo são protegidas por lock e a alteração do orçamento reinicia o pool; processos de jobs distintos continuam independentes. Os defaults dos oráculos antigos permanecem automáticos. Um teste alterna os orçamentos sem perder o resultado nem tornar chamadas subsequentes inválidas.

A métrica principal é `Runtime/predict_seconds`, recolhida pelo ciclo comum em todos os passos para MILP, BAU, Basic, Smart e Community. Inclui preparação da decisão e contexto comum, mas exclui o passo do Simulator e a exportação final. O MILP regista ainda preparação do problema, resolução, estados medidos, hash do problema, estado/gap do solver e ações. Mediana, P95, máximo e soma devem ser reportados; uma média amortizada de um plano semanal não é latência online.

## Protocolo experimental

Core30 Normal v3.3, Simulator 2.0.0, seed 22 e avaliação 6763. Janelas [35040,35712) e [52512,53184), 1–7 janeiro e 1–7 julho de 2024; resets independentes. MLflow desligado, exportação final e BAU nativo desativado. BAU principal = NormalTargetPolicy com bateria. Os baselines fechados não foram alterados.

Intel Core i9-13980HX, Linux, SciPy 1.15.1 / HiGHS 1.8.0. Janeiro usa CPU lógico 2 e julho CPU 4, núcleos físicos P distintos, com uma thread HiGHS. Controladores de cada mês executam sequencialmente; a repetição corrigida do MILP usa a mesma afinidade. Parte dos ensaios coexistiu com testes nos E cores 16–23. São latências do percurso comum neste host partilhado, não microbenchmarks isolados. O tempo inclui a conversão/captura comum, por isso não deve ser interpretado como custo exclusivo da regra RBC. Os limites de tempo podem também mudar a solução incumbente entre máquinas.

Foram aplicados os critérios: serviço mínimo viável ≥99,9%, precisão EV ≥80%, cobertura completa, zero ciclos perdidos, zero eventos elétricos e energia de violação ≤1e−6 kWh, SOC físico em [0,1] ±1e−6. A energia terminal por bateria e EV ainda ligado foi recolhida. Os custos abaixo pertencem apenas a execuções que passaram estes critérios.

## Oráculo offline

Os resultados semanais anteriores foram preservados. Para esta comparação calculou-se adicionalmente o oráculo semanal com energia terminal por ativo não inferior à energia real deixada pelo MPC. Esta informação foi usada apenas pelo oráculo após o MPC acabar, nunca como previsão ou restrição do controlador online. Os limites foram confirmados no replay real. O certificado do solver continua a referir-se à aproximação linear, não ao ótimo global do Simulator.

## Resultados e interpretação

**O MILP de 24 horas é a referência de menor custo observada nestas duas semanas: 222,29 €, menos 17,70% que o BAU.** O oráculo semanal dá 222,97 €, menos 17,44%. As políticas BAU/Basic/Smart/Community reproduziram exatamente os custos anteriormente fechados.

| Controlador | Custo (€) | Poupança / BAU | Pico máximo (kW) | Redução do pico | Média dos picos diários (kW) |
|---|---:|---:|---:|---:|---:|
| BAU | 270,08 | 0,00% | 59,59 | 0,00% | 32,14 |
| Basic | 261,30 | 3,25% | 58,50 | 1,81% | 28,77 |
| Smart | 255,11 | 5,54% | 55,37 | 7,07% | 28,38 |
| Community | 250,99 | 7,07% | 55,37 | 7,07% | 28,12 |
| MILP | 222,29 | 17,70% | 61,79 | -3,69% | 33,12 |
| Oráculo offline | 222,97 | 17,44% | 57,05 | 4,25% | 33,65 |

Percentagens = `100 × (BAU − método) / BAU`. Valores negativos indicam aumento. O MILP aumenta o pico máximo em 3,69% e a média dos picos diários em 3,06%; Community conserva a melhor média diária nesta comparação.

| Método | Janeiro (€) | Julho (€) |
|---|---:|---:|
| BAU | 194,74 | 75,34 |
| Basic | 189,85 | 71,45 |
| Smart | 186,58 | 68,54 |
| Community | 184,78 | 66,20 |
| MILP | 174,48 | 47,81 |
| Oráculo offline | 174,87 | 48,10 |

Todos os seis métodos obtiveram serviço mínimo viável de 100%, 170 saídas EV, 153 ciclos concluídos, zero ciclos perdidos e zero violações elétricas. A precisão EV foi 88,51% em janeiro e 100% em julho; cobertura completa. Os EVs ainda ligados no fim não são saídas observadas e não se contam como serviço de partida concluído.

O pico MILP ocorre às 13h45 UTC de 3 de janeiro: preço 0,061 €/kWh, três EVs a somar 31,40 kW, carga não adiável 34,18 kW, PV −5,74 kW e baterias domésticas a zero. Os restantes consumos somam 1,95 kW. A simultaneidade de carga barata é coerente com custo mínimo; não existe objetivo de pico na formulação. A decomposição usa fluxos válidos, excluindo o sentinela −1 dos exports direcionais dos carregadores.

| Controlador | Média (s) | Mediana (s) | P95 (s) | Máximo (s) | Soma das 1344 decisões (s) |
|---|---:|---:|---:|---:|---:|
| BAU | 0,116 | 0,119 | 0,123 | 0,167 | 156,54 |
| Basic | 0,115 | 0,117 | 0,121 | 0,137 | 154,66 |
| Smart | 0,118 | 0,120 | 0,124 | 0,218 | 158,58 |
| Community | 0,119 | 0,122 | 0,125 | 0,135 | 160,41 |
| MILP | 1,178 | 0,826 | 2,398 | 11,375 | 1583,88 |

O MILP demora cerca de 10 vezes o percurso comum dos RBCs neste host. As fases de serviço foram ótimas nas 1344 decisões. As fases de custo terminaram 1322 vezes dentro da tolerância de ótimo configurada e 22 vezes por limite de tempo, sempre com plano viável; maior gap relativo residual 0,4564%. Não há certificado de ótimo semanal derivado destes gaps por janela.

O oráculo semanal demorou **73,69 s em janeiro e 18,36 s em julho** a construir/resolver o modelo dentro da chamada medida, com uma thread e limite de 180 s por fase. A extração do dataset precede essa medição. Ambos terminaram ótimos dentro da tolerância, com gaps de 0,04545% e 0,04596%. O replay tem cerca de 0,117 s por passo de percurso comum, mas já recebe o plano pronto: esse tempo não equivale a uma resolução online. A soma das decisões MPC é 1583,88 s; as duas resoluções semanais somam 92,05 s, com um regime de informação e frequência diferente.

### Porque o oráculo semanal não é um limite perfeito do Simulator

Mesmo com previsão semanal, o plano fixo custa 0,68 € mais no replay que o MPC. A formulação usa eficiências/potências conservadoras e aproximações lineares. O MPC volta a observar energia real a cada passo; o oráculo executa ações fixas de um modelo aproximado. Também não são idênticas todas as reservas e condições intermédias dos problemas móveis. Portanto, maior horizonte não estabelece uma ordenação garantida dos custos **reais** destes dois controladores.

Os limites duais semanais, 175,01376 € e 48,07443 €, só limitam os respetivos problemas lineares mistos, com as condições terminais declaradas. Em janeiro o custo real do MPC, 174,47672 €, é até inferior ao limite dual daquele modelo: prova direta de que esse limite não pode ser anunciado como limite global do Simulator. A expressão de referência correta é **MILP de 24 h: melhor poupança observada; oráculo semanal: ótimo aproximado do modelo offline**. BAU é o ponto 0% de poupança, não um limite matemático inferior de desempenho.

### Energia final e utilização das baterias

| Método | BESS final janeiro / julho (kWh) | EVs ainda ligados janeiro / julho (kWh) | Trânsito absoluto BESS, 14 dias (kWh) |
|---|---:|---:|---:|
| BAU | 10,59 / 19,77 | 461,06 / 472,17 | 405,33 |
| Basic | 14,05 / 19,77 | 419,27 / 472,76 | 398,39 |
| Smart | 23,04 / 20,76 | 441,24 / 472,82 | 333,82 |
| Community | 22,22 / 20,75 | 441,24 / 473,31 | 449,19 |
| MILP | 12,81 / 12,83 | 461,98 / 473,01 | 1125,26 |
| Oráculo offline | 13,40 / 13,43 | 462,77 / 473,55 | 1107,44 |

Os valores por ativo estão nas [métricas versionadas](validation/core30_milp_mpc_comparison.json). O oráculo passou a verificação real de energia final por bateria e EV ligado ≥ MPC −1e−4 kWh. Os baselines mantiveram as suas regras terminais; não foram modificados para igualar estes estados. O MPC usa mais a bateria (1125,26 kWh de trânsito absoluto, contra 405,33 kWh no BAU); não se inclui preço de desgaste. Há também previsão perfeita de 24 horas e diferentes limites estratégicos/lineares. Logo, a poupança é operacional sob estes pressupostos, não uma garantia de rentabilidade total ou de superioridade universal.

As janelas de 2024 já tinham sido observadas em campanhas anteriores. A preparação incluiu dois dias de 2023 e testes de contrato; não houve ajuste de política para melhorar os números finais de 2024. Esta é uma validação temporal e de execução, não um teste cego nem uma avaliação anual.

## Exportação e reprodução

Template principal: configs/templates/baselines/core30/milp.yaml. Oráculo offline: milp_community.yaml, com plano produzido previamente. O novo agente requer interface entity estática, janela explícita e zero episódios de treino. O bundle é de investigação: descrição da política, configuração e registo das decisões; requer estados, previsões e solver para nova execução. Não é uma política ONNX nem possui paridade de serving certificada.

Artefactos brutos: `runs/milp-mpc-core30-20261103/`. Foram preservados o smoke de 16 passos, o ensaio de dois dias em 2023, a tentativa de timing interrompida localmente e os diagnósticos de orçamento/presolve. O ensaio sem presolve não melhorou o caso estudado; não foi promovido. A tabela principal usa os oito baselines congelados em validated/ (97c9ccd) e os dois novos ensaios MILP em comparison/ (8403423). A implementação dos baselines não mudou entre esses commits. A tentativa MILP de validated/ conserva janeiro completo e julho interrompido por validação incorreta da reserva; nenhum entra na tabela final.

## Verificação e rastreabilidade

- Fonte numérica MPC/oráculo: `84034231f38e7a6fb6308e9008b540c40c862577`; baselines: `97c9ccd4d43c8c183e58da9bdf2e2681a1b82103`. A implementação dos baselines é idêntica nos dois commits. Metadados/documentação posteriores não alteram as decisões.
- 33 testes focados de estado, horizonte, solver e extração; 2056 testes gerais passados e 23 saltados em CPU; 72 testes de jobs completos passados. Ruff, formatação e mypy passaram.
- Uma execução acidental com GPU visível expôs dois testes existentes que misturam tensores CPU/CUDA e um teste MADDPG de reprodução/resume. Não foram ocultados nem alterados nesta tarefa; repetiram-se as suites em CPU, como na CI, com sucesso. Esta entrega não certifica esses caminhos GPU.
- [Histórico da campanha](experiment_history/20261103_milp-mpc-core30-20261103.md) e [métricas completas](validation/core30_milp_mpc_comparison.json). Dados brutos, configurações, hashes, logs, solver e exports oficiais permanecem na pasta `runs/` indicada.

Instalar as dependências opcionais `training,solver,offline` para executar o agente. O template principal mantém a configuração de exemplo de dois dias; para as semanas comparadas usam-se as janelas exatas e os 673 estados/672 decisões registados no protocolo. Configuração de produção deve apontar ao schema Core30 disponível no destino. A etapa 5, a integração especializada de outras famílias e a integração de inferência continuam fora desta entrega.
