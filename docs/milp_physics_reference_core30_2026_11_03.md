# Referência física MILP no Core30 — 03/11/2026

Esta revisão substitui a aproximação conservadora do [primeiro ensaio MPC](milp_mpc_core30_2026_11_03.md) por uma formulação compatível com a física declarada no Core30 Normal v3.3 e no `softcpsrecsimulator==2.0.0`. O relatório anterior conserva os seus resultados históricos; os respetivos limites duais não eram limites válidos para a física usada pelo Simulator.

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

## Significado do limite e da validação

Com eficiência constante, a conservação de energia é linear. Uma curva côncava de potência/SOC é representada exatamente pela interseção dos seus limites afins. Variáveis inteiras representam os sentidos de carga/descarga, os mínimos dos carregadores, a saturação e os instantes de arranque dos ciclos. A primeira fase minimiza défices de serviço; a segunda fixa esse serviço e minimiza o custo.

O modelo impõe as condições físicas necessárias às trajetórias admissíveis. O alocador de segurança do Simulator pode ainda reduzir comandos, sobretudo quando um comando mínimo é maior que a potência efetivamente entregue. Por isso, um plano calculado não é automaticamente uma solução executável: é sempre reproduzido no runner normal e comparado passo a passo com o plano.

O limite dual é um limite inferior do problema físico formulado. Uma reprodução que cumpra serviço, eletricidade e reservas fornece uma solução executável. A distância reportada compara o custo dessa reprodução com o limite inferior, e não apenas a solução interna com o seu próprio solver. São reportados separadamente os resíduos de energia, fluxo e custo.

O certificado numérico refere-se às equações reais declaradas, dentro das tolerâncias do solver. A reprodução independente valida a sua correspondência com o Simulator float32. Um erro observado pequeno não é apresentado como prova formal universal de todas as operações de ponto flutuante. Também não se chama ótimo exato a um resultado que terminou por tempo com intervalo residual aberto.

## Reprodução

O template `configs/templates/baselines/core30/milp.yaml` usa `physics: simulator_linear`. O mesmo perfil é o valor predefinido do agente `MILP`; datasets incompatíveis precisam de uma escolha explícita de outro modelo.

```bash
python scripts/validate_core30_milp_reference.py \
  --config <config-de-avaliacao.yaml> \
  --output-dir <nova-pasta> \
  --time-limit-seconds 900 \
  --mip-relative-gap 0.000001

python scripts/score_milp_reference.py \
  --reference-dir <nova-pasta> \
  --job-dir <nova-pasta>/replay/jobs/<nome-da-pasta>
```

Cada fase do solver tem o orçamento indicado. O tempo real inclui a construção da formulação e as fases de serviço/custo; pode exceder um único orçamento de fase. O tempo do replay de um plano pronto não é tempo de resolução online. As latências do MPC são medidas em cada decisão pelo runner comum.

Os critérios de aceitação incluem serviço mínimo de EVs ≥99,9%, precisão ≥80%, zero eventos elétricos e energia de violação ≤10⁻⁶ kWh, ciclos obrigatórios completos, reservas e limites físicos. Para a correspondência entre plano e reprodução, o limite explícito é 10⁻⁴ kWh nos estados, fluxos e alvos terminais.

Os artefactos completos desta campanha ficam em `runs/milp-physics-core30-20261103/`. O ensaio curto corrigido já reproduziu o custo com diferença de 3,6×10⁻⁸ € e os estados das três baterias com erro máximo inferior a 7,5×10⁻⁷ kWh. A comparação principal será registada apenas depois de concluídas e auditadas as duas semanas.
