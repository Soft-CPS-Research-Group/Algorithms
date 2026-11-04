# Referência física MILP no Core30 — 03/11/2026

Esta revisão substitui a aproximação conservadora do [primeiro ensaio MPC](milp_mpc_core30_2026_11_03.md) por uma formulação compatível com a física declarada no Core30 Normal v3.3 e no `softcpsrecsimulator==2.0.0`. O relatório anterior conserva os seus resultados históricos; os respetivos limites duais não eram limites válidos para a física usada pelo Simulator.

**Resultado:** o ótimo do modelo para as duas semanas arredonda para **221,84 €**. O oráculo executado no Simulator custa **221,83896 €** e o MILP de 24 horas custa **222,17472 €**, a cerca de **0,15%** do ótimo. A decisão online demorou **2,584 s** em média.

## Problema que a referência resolve

A prioridade é cumprir os objetivos de carga dos EVs e os ciclos obrigatórios. Depois minimiza-se a compra líquida de energia à rede pela comunidade. Não se inclui uma penalização de pico ou desgaste. Por isso, «ótimo» nesta página significa ótimo de custo sob estas condições de serviço, informação e armazenamento; não significa melhor pico, menor desgaste ou melhor política sem conhecimento do futuro.

As janelas são 1–7 de janeiro e 1–7 de julho de 2024, com 672 decisões de 15 minutos cada, 30 membros, três baterias estacionárias e 15 carregadores. V2G está desativado no dataset. Não é um resultado anual nem uma avaliação num conjunto nunca antes observado.

O semanal conhece apenas a janela completa avaliada. O controlador `MILP` resolve novamente em cada passo e recebe, no máximo, as próximas 24 horas. Ambos partem do mesmo reset e usam os mesmos dados, preços, equipamentos e limite de serviço. As reservas terminais são definidas antes da resolução: 50% da capacidade das baterias estacionárias e o alvo anunciado dos EVs que continuam ligados no fim. Não se usa o resultado semanal para orientar o MPC.

Os EVs cuja partida ocorre depois do fim da janela têm um alvo terminal de melhor esforço, e não uma partida inventada. Em janeiro, o EV do Member_022 chega quatro horas antes do fim com 22,31071 kWh; o alvo é 45,66655 kWh. O carregador de 3,68 kW, mesmo sempre ligado, só permite chegar a 36,01217 kWh nesse intervalo. O défice terminal inevitável é 9,65438 kWh. A fase de serviço encontra esse mínimo e a fase de custo conserva-o. Não há défice planeado nas partidas que efetivamente ocorrem dentro da janela. Em julho, o défice mínimo é zero, incluindo os alvos terminais.

O mercado mantém os pressupostos anteriores: compra externa ao preço do dataset; transações internas a 80% desse preço; venda externa sem receita; autoconsumo solar sem custo de aquisição. Como os pagamentos e recebimentos internos se cancelam, o custo agregado corresponde ao preço multiplicado pela importação líquida positiva.

## Correções de física

- A eficiência direcional das baterias usa a raiz quadrada da eficiência técnica constante. Foi retirado o desconto artificial de 1%.
- A capacidade é a capacidade física, sem a redução conservadora para 99%.
- O coeficiente de perdas configurado é horário. A perda por passo é esse coeficiente multiplicado pela duração do passo em horas.
- A observação entity apresenta o último SOC concluído; o MPC aplica a perda de repouso antes da ação seguinte.
- A reserva de descarga não cria energia quando as perdas deixam a bateria abaixo dessa reserva.
- A potência dos EVs respeita a curva de redução de potência em função do SOC e a eficiência do carregador na fronteira AC.
- O mínimo de potência é um mínimo do comando. A energia efetivamente entregue pode ser menor quando a bateria limita a potência ou fica cheia. Essas situações têm ramos explícitos no MILP e uma conversão de comando correspondente.
- Dados de carga, PV, preço e SOC de chegada são convertidos com a precisão usada pelo simulador.

O perfil `simulator_linear` exige eficiências constantes, degradação desativada e curvas de potência côncavas. Atualmente exige EVs sem V2G, sem perdas de repouso e com reserva zero. Recusa equipamentos fora desse domínio em vez de lhes atribuir silenciosamente uma aproximação «exata». O perfil `conservative` permanece explícito para experiências antigas e não recebe esta qualificação.

O MPC reserva uma margem numérica de 0,1 Wh nos planos de vários passos e admite 0,01 Wh de tolerância terminal no último passo. Os alvos de serviço incluem uma margem de 0,1 Wh quando há capacidade disponível. Estas margens tratam a representação float32; não acrescentam uma potência fictícia nem arredondam o estado medido para cima.

A fase de custo do perfil alinhado conserva o mínimo de défice EV sem folga adicional. O ensaio inicial ainda herdava uma folga de 1 Wh por resolução: a reotimização podia gastar essa folga repetidamente, acumulando cerca de 8 Wh no EV terminal de janeiro. Esse ensaio fica como diagnóstico; a comparação final usa a folga zero tanto no MPC como no semanal.

Ao converter a solução em comandos, os limites são verificados em **kWh por intervalo**, a unidade da matriz do solver, antes da divisão pela duração do passo. O sentido de carga/descarga selecionado é respeitado; um equipamento sem V2G não recebe potência de descarga. Só se remove resíduo compatível com 10⁻⁶ kWh e com o resíduo medido do indicador inteiro (máximo 10⁻⁶, multiplicado pela potência admissível vezes a duração). Um erro material interrompe o job. A correção máxima de cada plano fica registada. A limpeza afeta a representação da ação; não altera o objetivo, o limite dual nem os dados físicos.

## Significado do limite e da validação

Com eficiência constante, a conservação de energia é linear. Uma curva côncava de potência/SOC é representada exatamente pela interseção dos seus limites afins. Variáveis inteiras representam os sentidos de carga/descarga, os mínimos dos carregadores, a saturação e os instantes de arranque dos ciclos. A primeira fase minimiza défices de serviço; a segunda fixa esse serviço e minimiza o custo.

O modelo impõe as condições físicas necessárias às trajetórias admissíveis. O alocador de segurança do Simulator pode ainda reduzir comandos, sobretudo quando um comando mínimo é maior que a potência efetivamente entregue. Por isso, um plano calculado não é automaticamente uma solução executável: é sempre reproduzido no runner normal e comparado passo a passo com o plano.

O limite dual é um limite inferior do problema físico formulado. Uma reprodução que cumpra serviço, eletricidade e reservas fornece uma solução executável. A distância reportada compara o custo dessa reprodução com o limite inferior, e não apenas a solução interna com o seu próprio solver. São reportados separadamente os resíduos de energia, fluxo e custo.

O certificado numérico refere-se às equações reais declaradas, dentro das tolerâncias do solver. A reprodução independente valida a sua correspondência com o Simulator float32. Um erro observado pequeno não é apresentado como prova formal universal de todas as operações de ponto flutuante. Também não se chama ótimo exato a um resultado que terminou por tempo com intervalo residual aberto.

## Referência semanal e distância ao ótimo

O solver encontra uma solução executável e um limite inferior independente dessa solução. Nos dois problemas semanais, o intervalo do custo ótimo do modelo é:

**221,83526080 € ≤ custo ótimo ≤ 221,83894839 €.**

A largura é **0,00368759 € (0,00166231%)**. Ambos os extremos arredondam para **221,84 €**. Assim, a referência fica fechada à precisão monetária de um cêntimo para estas duas semanas e para o objetivo declarado. Os dois solves terminaram pelo limite de tempo, não com gap zero: mantém-se esse estado nos artefactos e não se afirma um ótimo matemático exato do Simulator float32.

| Janela | Limite inferior (€) | Solução do modelo (€) | Execução no Simulator (€) | Excesso da execução sobre o limite (€) | Tempo real de resolução (s) |
|---|---:|---:|---:|---:|---:|
| 1–7 janeiro | 174,26451398 | 174,26591436 | 174,26591854 | 0,00140456 | 932,347 |
| 1–7 julho | 47,57074682 | 47,57303403 | 47,57303761 | 0,00229079 | 927,775 |
| Total | 221,83526080 | 221,83894839 | 221,83895615 | 0,00369535 | 1860,122 |

A diferença total entre o modelo e a execução é **0,00000776 €**. O maior resíduo de estado foi **0,00001955 kWh (0,01955 Wh)**; o maior resíduo de consumo líquido por edifício foi **0,00000648 kWh**. As verificações abrangem todas as baterias, todas as sessões EV e os 30 edifícios em todos os passos, não apenas o custo agregado.

Os dois replays passaram os critérios de aceitação, com 170 partidas EV observadas, 153 ciclos concluídos e zero violações elétricas. O serviço mínimo e o alvo EV foram cumpridos em 100% das partidas segundo a auditoria do Simulator. A precisão em torno do alvo foi 88,51% em janeiro e 100% em julho; chegar acima do alvo já na entrada conta de forma diferente de falhar a carga necessária. Os alvos terminais dos EVs ainda ligados são auditados separadamente.

O certificado é condicionado ao serviço e às reservas descritos acima. Um controlador que deixe EVs por carregar, ou esvazie mais as baterias no fim, pode apresentar um custo menor sem refutar este limite. Os baselines mantêm os seus próprios limiares estratégicos; a comparação com eles mede o resultado das políticas completas, não uma garantia de que pertencem exatamente ao mesmo conjunto de restrições terminais.

## Comparação com o MILP de 24 horas e os baselines

Custos oficiais do Simulator, após os critérios de aceitação. O pico é a maior importação líquida da comunidade num intervalo de 15 minutos; não é uma componente do objetivo. As poupanças usam o BAU com bateria e limite EV do utilizador como denominador. Os oito jobs BAU/RBC da campanha anterior foram reutilizados: as políticas não mudaram e os manifests confirmam os mesmos 87 ficheiros de entrada, mercado, seed, duração, versão e janela por par.

| Método | Janeiro (€) | Julho (€) | Total (€) | Poupança / BAU | Pico máximo (kW) | Média dos picos diários UTC (kW) |
|---|---:|---:|---:|---:|---:|---:|
| BAU | 194,7389 | 75,3443 | 270,0832 | 0,00% | 59,59 | 32,14 |
| Basic | 189,8507 | 71,4534 | 261,3041 | 3,25% | 58,50 | 28,77 |
| Smart | 186,5761 | 68,5375 | 255,1137 | 5,54% | 55,37 | 28,38 |
| Community | 184,7832 | 66,2045 | 250,9878 | 7,07% | 55,37 | 28,12 |
| MILP 24 h | 174,5036 | 47,6711 | 222,1747 | 17,74% | 56,20 | 34,01 |
| Oráculo semanal | 174,2659 | 47,5730 | 221,8390 | 17,86% | 56,20 | 34,86 |

O MPC custou **0,33577 €** mais que o plano semanal executado. Subtraindo o intervalo de ótimo do modelo ao custo observado do MPC, o excesso fica entre **0,33577 € e 0,33946 €**, ou **0,1514–0,1530%**. Esta comparação respeita as tolerâncias de reprodução declaradas: maior falta de energia terminal do MPC (bateria ou EV, incluindo melhor esforço) 0,01083 Wh; maior falta face ao alvo nas partidas observadas 0,00000 Wh.

Todos os seis métodos cumpriram o serviço mínimo nas 170 partidas observadas, concluíram 153 ciclos, sem ciclos perdidos nem eventos de violação elétrica. Cobertura completa; a precisão mínima semanal foi 88,51%. A precisão mede proximidade ao alvo, enquanto o serviço mínimo mede entrega suficiente. Os dois zeros de custo da energia (PV próprio e exportação à rede) têm sentidos diferentes: não se paga pela produção própria consumida e não se recebe pela exportação externa. Os preços observados foram 0,00173–0,11326 €/kWh em janeiro e 0–0,11041 €/kWh em julho.

### Tempos medidos

Intel Core i9-13980HX; um núcleo físico por job e uma thread HiGHS por solve, sem GPU. Jobs independentes correram em paralelo; trata-se de tempo observado numa máquina partilhada. Python 3.10.22, SciPy 1.15.1, HiGHS 1.8.0. A tabela usa `Runtime/predict_seconds` do runner em todas as 1 344 decisões de cada controlador. Inclui preparação da decisão, solver quando aplicável e conversão da ação; não inclui o passo físico nem o export final.

| Controlador | Média (s) | Mediana (s) | P95 (s) | Máximo (s) | Soma (s) |
|---|---:|---:|---:|---:|---:|
| BAU | 0,116 | 0,119 | 0,123 | 0,167 | 156,545 |
| Basic | 0,115 | 0,117 | 0,121 | 0,137 | 154,658 |
| Smart | 0,118 | 0,120 | 0,124 | 0,218 | 158,585 |
| Community | 0,119 | 0,122 | 0,125 | 0,135 | 160,407 |
| MILP 24 h | 2,584 | 1,979 | 7,732 | 11,168 | 3.472,498 |

MPC: 1 344/1 344 primeiras fases de serviço ótimas; 1 290 resoluções económicas ótimas dentro da tolerância do solver e 54 limitadas por tempo com solução viável. O maior gap de uma decisão isolada foi 2,0705%; refere-se à sua janela móvel, não à distância económica dos 14 dias ao oráculo. O orçamento por tempo pode devolver incumbentes diferentes em execuções distintas. Os valores publicados são os de duas execuções completas da versão final, sem decisões pré-gravadas nem substituição por heurísticas.

O tempo médio por decisão do MPC foi 1,885 s em janeiro e 3,283 s em julho. A soma das 1 344 decisões foi de 57 min 52 s (não inclui os passos do simulador). O semanal resolveu uma vez por semana: 15 min 32 s em janeiro e 15 min 28 s em julho. Estes são os tempos para obter os certificados reportados com orçamento de 900 s por fase, não o tempo mínimo para encontrar qualquer solução viável. O replay posterior só executa um plano pronto. O MPC voltou a resolver a cada 15 minutos; mesmo a sua decisão mais lenta (11,168 s) coube no intervalo de controlo nesta máquina.

## Reprodução

O template `configs/templates/baselines/core30/milp.yaml` usa `physics: simulator_linear`. O mesmo perfil é o valor predefinido do agente `MILP`; datasets incompatíveis precisam de uma escolha explícita de outro modelo.

Para repetir as medições, aponta-se `simulator.dataset_path` ao `evaluation/core_30_normal.json` do Core30 Normal v3.3. Os pares `simulation_start_time_step`/`simulation_end_time_step` são `35040`/`35712` e `52512`/`53184`; `episode_time_steps: 673` dá 672 transições por episódio. Usa-se avaliação única, zero episódios de treino, seed base 22 (avaliação 6763), export final sem BAU nativo, MLflow desligado, `runtime_profiling_enabled: true` e `runtime_profiling_interval: 1`. O MPC usa o runner público normalmente, com 10 s por fase e tolerância relativa de 0,1%; o semanal usa a ferramenta abaixo com tolerância mais apertada e desempate de throughput a zero para que o limite seja de custo puro.

```bash
python scripts/validate_core30_milp_reference.py \
  --config <config-de-avaliacao.yaml> \
  --output-dir <nova-pasta> \
  --time-limit-seconds 900 \
  --mip-relative-gap 0.000001

python scripts/score_milp_reference.py \
  --reference-dir <nova-pasta> \
  --job-dir <nova-pasta>/replay/jobs/<nome-da-pasta>

python run_experiment.py --config <config-de-avaliacao.yaml> \
  --job_id <nome-do-milp-24h> --base-dir <pasta-dos-jobs>
```

Cada fase do solver tem o orçamento indicado. O tempo real de resolução inclui a construção das matrizes da formulação e as fases de serviço/custo; exclui a leitura inicial dos dados e o replay, e pode exceder um único orçamento de fase. O tempo do replay de um plano pronto não é tempo de resolução online. As latências do MPC são medidas em cada decisão pelo runner comum.

Os critérios de aceitação incluem serviço mínimo de EVs ≥99,9%, precisão ≥80%, zero eventos elétricos e energia de violação ≤10⁻⁶ kWh, ciclos obrigatórios completos, reservas e limites físicos. Para a correspondência entre plano e reprodução, o limite explícito é 10⁻⁴ kWh nos estados, fluxos e alvos terminais.

Os artefactos completos desta campanha ficam em `runs/milp-physics-core30-20261103/`. O ensaio curto corrigido já reproduziu o custo com diferença de 3,6×10⁻⁸ € e os estados das três baterias com erro máximo inferior a 7,5×10⁻⁷ kWh. A [tabela de resultados e proveniência](validation/core30_milp_physics_reference.json) e o [registo da campanha](experiment_history/20261103_milp-physics-core30-20261103.md) conservam os resultados finais, critérios, hashes e diagnósticos. A construção atual reproduz byte a byte os dois problemas semanais arquivados; a formulação que gera as matrizes e os limites não mudou durante as correções da conversão de comandos.
