# CC causal price control V4

- Dataset: `citylearn_three_phase_electrical_service_demo_15min_parquet`
- Horizonte de evidência: ano completo, passos `0:35039`
- Mercado comunitário: ativo, preço local `0,8` do preço grid
- Objetivo primário: custo comunitário settled
- Estado: campanha anual V4 concluída; sete probes fixos e três treinos
  CC-SMART anuais recolhidos e auditados

## Motivo da correção CC-PPO

O primeiro CC-PPO anual custou EUR 21 075,18, contra EUR 20 850,00 do PPO
neutro (`+EUR 225,18`). A auditoria do caminho do sinal encontrou uma causa
arquitetural concreta: `local_price_conditioning_enabled` alterava a
observação codificada entregue ao ator PPO congelado, embora esse ator só
tivesse sido treinado com o preço nominal. Isso era inferência fora da
distribuição. Ao mesmo tempo, o `RBCSmartLocalPolicy` usado como base residual
não recebia o contexto do CC.

V4 inverte esse contrato:

1. o ator PPO recebe exatamente a observação original;
2. o multiplicador é entregue apenas ao novo
   `SignalAwareRBCSmartLocal`, que mantém a restrição building-local;
3. a `1,0`, essa base é exatamente igual a `RBCSmartLocalPolicy`;
4. qualquer configuração que tente ativar este canal sem a base local correta
   falha imediatamente.

A base SMART residual original desativa carregamento da bateria motivado por
preço. V4 acrescenta `signal_price_charge_rate=0,6`, ativo apenas para
multiplicadores abaixo de `1,0`. O controlo neutro continua exatamente igual;
um desconto passa a poder carregar a bateria, enquanto sinais acima de `1,0`
continuam a usar a lógica de descarga sensível ao preço. Isto dá autoridade
bidirecional ao canal sem alterar o ator nem fornecer informação comunitária à
folha.

Antes de treinar outro CC, a grelha anual fixa
`0,90/0,95/1,00/1,05/1,10/1,20/1,30` mede a autoridade causal deste canal.
`1,00` é o controlo neutro pré-registado. Só se treina um CC residual em torno
de um multiplicador que supere esse controlo. Se nenhum o superar, o próximo
passo é treinar uma nova versão do PPO local com randomização do preço efetivo,
sem lhe fornecer observações da comunidade, e repetir o mesmo controlo.

## Nova frente CC-SMART

O melhor CC-SMART de desenvolvimento anterior atingiu EUR 21 936,93, uma
melhoria de apenas EUR 27,74 contra o SMART settled. V4 tenta aumentar essa
margem sem confundir mudanças de recompensa e frequência de decisão:

| Receita | Intervalo CC | Horizonte PPO | Recompensa |
|---|---:|---:|---|
| `settled_cost_hourly` | 4 passos / 1 h | 168 decisões / 7 dias | custo settled |
| `settled_cost_15min` | 1 passo / 15 min | 672 decisões / 7 dias | custo settled |
| `settled_cost_peak_15min` | 1 passo / 15 min | 672 decisões / 7 dias | custo + pico 0,05 + ramp 0,02 |

As três receitas usam dez episódios anuais, BC com 8 760 decisões, 4 000
updates supervisionados, referência `1,3`, gama efetiva `0,9--1,3` e
regularização reduzida (`w_factor=0,01`, `w_smoothness=0,005`). Assim o
primeiro par isola a autoridade temporal e o terceiro mede se há ganhos
físicos visíveis sem perder o foco económico.

## Evidência local antes do build

- Testes focados: `65 passed` antes do smoke.
- Suíte completa final: `920 passed`, 29 warnings esperados.
- PPO corrigido `1,0`: smoke real, 17 checkpoints carregados, exit 0,
  artefactos e 53 ficheiros de simulação exportados.
- Paridade causal: num replay de 384 passos, os 53 ficheiros exportados pelo
  PPO original e pelo pipeline corrigido a `1,0` foram byte-a-byte idênticos;
  `exported_kpis.csv` teve SHA-256
  `0c9c1a07e234b0185c6f53afaade22dc2e3b61a007823115b685fe5e71a5d3ec`.
- PPO corrigido `1,3`: smoke real, exit 0, mesmos contratos de exportação.
- Autoridade do desconto: no replay de 384 passos, `0,9` alterou a bateria do
  Building 15, os dados desse edifício, a série comunitária e os KPIs; o hash
  de `exported_kpis.csv` mudou para
  `69aac1f705f371e7721fece1bbfaa5f703b7cfb2198027cba29885ebaae32676`.
  Isto prova atuação causal, mas não é evidência de melhoria económica.
- CC-SMART cost-only 15 min: smoke com três episódios e 672 decisões por
  episódio; BC executada, uma atualização PPO real
  (`pg=-0,0471`, `v=86,8374`, `kl_stop=False`), avaliação final determinística,
  checkpoint e séries exportados, exit 0.

Os smokes são apenas evidência funcional. Resultados de custo, picos e
fairness só serão aceites no ano completo. A promoção de CC+PPO exige:

1. custo settled inferior ao PPO neutro emparelhado;
2. hard gates de EV, deferrables, rede, SoC e outage;
3. scorecard completo de importação, picos, ramping, load factor, emissões,
   autoconsumo, throughput, V2G e fairness;
4. confirmação posterior em três seeds ou superfície temporal held-out.

## Atualização de 2026-08-05: campanha anual concluída

A campanha foi lançada com a imagem do commit `8b73465`. Todos os jobs usam o
mesmo dataset, passos `0:35039`, settlement comunitário ativo e PPO local
congelado seed 789. Os sete probes fixos têm resultados anuais válidos:

| Multiplicador | Custo settled | Delta vs 1,00 | Delta contrafactual | Delta poupança settlement | Pico diário vs BAU | Ramping vs BAU | EV mínimo viável | Decisão parcial |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0,90 | EUR 20 857,60 | EUR +7,60 | EUR +100,62 | EUR +93,02 | 1,0754 | 1,5045 | 99,673% | `REJECT_COST` |
| **0,95** | **EUR 20 845,01** | **EUR -4,99 (-0,024%)** | EUR +70,89 | EUR +75,88 | 1,0772 | 1,4829 | 99,709% | `MARGINAL_CANDIDATE` |
| 1,00 | EUR 20 850,00 | referência | referência | referência | 1,0807 | 1,3620 | 99,927% | `REFERENCE` |
| 1,05 | EUR 20 850,00 | EUR -0,00 | EUR -0,00 | EUR -0,00 | 1,0807 | 1,3620 | 99,927% | `NO_EFFECT` |
| 1,10 | EUR 20 850,00 | EUR +0,00 | EUR +0,00 | EUR +0,00 | 1,0807 | 1,3620 | 99,927% | `NO_EFFECT` |
| 1,20 | EUR 20 850,00 | EUR +0,00 | EUR +0,00 | EUR +0,00 | 1,0807 | 1,3620 | 99,927% | `NO_EFFECT` |
| 1,30 | EUR 20 850,00 | EUR -0,00 | EUR -0,00 | EUR -0,00 | 1,0807 | 1,3620 | 99,927% | `NO_EFFECT` |

O replay `0,95` passa o perfil completo
`phase10_w6_adapted_local_v1`: EV mínimo viável pelo menos 0,99, precisão EV
pelo menos 0,40, zero violações elétricas, zero serviço deferrable perdido,
SOC estacionário em `[0,1]` e zero outage unserved energy. O perfil Phase 6
mais estrito, que exige EV mínimo viável de 0,999, sinaliza contudo `0,95` e
`0,90`; esta diferença de perfil fica explícita e impede apresentar `0,95`
como promoção já confirmada.

Face ao PPO neutro, `0,95`:

- reduz o custo settled em EUR 4,99 e melhora o custo settled de 16/17
  edifícios, mas o Building 15 piora EUR 23,04;
- piora o custo físico contrafactual em EUR 70,89 e aumenta importação em
  92,17 kWh;
- aumenta a poupança do mercado local em EUR 75,88, reduz exportação em
  160,87 kWh e aumenta autoconsumo solar em 0,111 pontos percentuais;
- reduz o pico diário em 0,32% relativo, sem alterar materialmente o pico
  absoluto;
- aumenta ramping em 8,88%, emissões em 0,528% e throughput da bateria em
  7,20%;
- reduz EV mínimo e precisão em 0,218 pontos percentuais, mantendo os gates
  do perfil Phase 10, e conserva zero violações de rede, deferrables, SOC e
  outage.

Assim, o sweep já prova que o novo canal altera causalmente o PPO e identifica
uma região útil perto de `0,95`, mas a melhoria económica é mínima e depende
do settlement para compensar uma regressão física concentrada no Building 15.
Isto justifica testar um CC temporal estreito em torno de `0,95--1,00`, com
penalização de ramping/emissões e atenção explícita ao Building 15; não
justifica ainda afirmar que CC-PPO supera robustamente PPO.

Os probes originais `1,10` e `1,20` falharam apenas na recolha remota: a ação
Union terminou com sucesso, mas o signer não disponibilizou o artefacto após
seis tentativas. Os logs foram preservados, os dois jobs terminais foram
apagados após autorização explícita e as mesmas configurações científicas
foram relançadas como `4b1ff744-b55c-4f44-86b8-a2730946bed8` no servidor e
`9a64e88e-f6ca-46b1-8f1c-5bf60022f302` no Deucalion CPU.

As reposições terminaram e confirmaram a zona morta acima de `1,0`: `1,10` e
`1,20` reproduzem o custo neutro até à precisão numérica. Portanto, não há
justificação para mais probes fixos acima de `1,0`; a única região com sinal
económico nesta arquitetura continua perto de `0,95`.

### Resultado CC-SMART anual

Os três CC-SMART passaram todos os hard gates. Comparados com o SMART settled
neutro emparelhado (EUR 21 964,67), os resultados completos são:

| Receita | Custo settled | Delta custo | Importação | Delta importação | Pico diário vs BAU | Ramping vs BAU | Emissões kgCO2 | Autoconsumo solar | Decisão |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SMART neutro | EUR 21 964,67 | referência | 132 780,13 kWh | referência | 1,07108 | 2,40308 | 22 415,74 | 69,737% | `REFERENCE` |
| CC-SMART hourly | EUR 21 938,27 | EUR -26,40 (-0,120%) | 132 481,51 kWh | -298,62 kWh | 1,06575 | 2,26144 | 22 228,67 | 69,667% | `PASS_CC_SCORECARD` |
| **CC-SMART 15 min** | **EUR 21 921,29** | **EUR -43,38 (-0,198%)** | **132 422,30 kWh** | **-357,83 kWh** | **1,06574** | **2,28716** | **22 315,90** | **69,813%** | **`PASS_CC_SCORECARD`** |
| CC-SMART peak 15 min | EUR 21 928,37 | EUR -36,29 (-0,165%) | 132 473,00 kWh | -307,13 kWh | 1,06581 | 2,31327 | 22 319,57 | 69,781% | `PASS_CC_SCORECARD` |

O `CC-SMART 15 min` é o vencedor económico. A melhoria é pequena, mas não é
apenas redistribuição contabilística do settlement: reduz também importação,
pico diário, ramping e emissões, aumenta ligeiramente o autoconsumo solar e
reduz throughput da bateria em 1 295,61 kWh. O pico absoluto fica inalterado.
O termo adicional de pico/ramping não superou a reward cost-only, pelo que a
receita `peak 15 min` não deve substituir a vencedora.

A fairness ainda é fraca: só 5/17 edifícios melhoram o custo local face ao
SMART emparelhado e não se cumpre `all_buildings_no_worse_than_baseline`.
Assim, a run é um `PASS_CC_SCORECARD` agregado e uma demonstração causal de
melhoria física, mas não uma solução de distribuição justa entre membros.

Evidência bruta e scorecard completo:
`runs/remote_results/cc_causal_price_control_v4_annual_20260805/`.

## Preparação V5 enquanto a campanha anual termina

O protocolo seguinte foi separado da V4 para não alterar nem reinterpretar os
jobs que usam a imagem imutável `8b73465`. A V5 responde primeiro a duas
questões causais antes de autorizar qualquer novo treino do PPO:

1. o PPO congelado reage melhor quando o multiplicador `0,95` altera apenas o
   preço atual ou quando altera também os três forecasts?;
2. o desconto marginalmente útil pode ser aplicado apenas em horas escolhidas,
   evitando a regressão física do sinal constante?

Foram geradas duas configurações anuais auditáveis em
`configs/experiments/cc_ppo_controllability_v5/`. Ambas mantêm settlement,
checkpoint seed 789 e a base `SignalAwareRBCSmartLocal`; diferem apenas entre
`real_unmodified` e `persist_current`. Como o ator atual foi treinado apenas no
preço nominal, estão marcadas `explicit_ood_diagnostic` e não podem ser
promovidas diretamente.

Foi também criado `scripts/build_cc_ppo_schedule_probes.py`. A partir dos
replays anuais emparelhados `1,00` e `0,95`, o script produz schedules horários
com hashes das fontes e cinco hipóteses:

- preço naturalmente barato;
- exportação comunitária;
- união e interseção das duas condições;
- seleção retrospetiva do melhor multiplicador por bloco horário.

O último schedule escolhe `0,95` em 4 430/8 760 horas. A mistura independente
dos dois traces estima EUR 223,33 de margem, mas esse valor **não é evidência**:
combina estados de bateria incompatíveis. Só um replay anual contínuo pode
medir o ganho real. Esse replay foi iniciado localmente como
`cc-ppo-v5-temporal-retrospective-cost-annual-local-r2`, mas o serviço
transiente foi parado externamente aos 31 616/35 040 passos (90,23%). Não há
exceção do algoritmo nos logs e `result.json` ficou `pending`; esta tentativa
não é evidência e só deve ser repetida remotamente mediante nova submissão.

Depois da publicação da imagem imutável
`add-cc-ppo-controllability-v5-protocol-bfecf60`, as duas ablações anuais do
ator foram submetidas ao Union e confirmadas em execução:

- `338cee85-a639-49a2-aedb-0c392b9966fa`: multiplicador aplicado ao preço
  atual do ator, com forecasts reais inalterados;
- `eb25701d-2444-415e-a103-3d4239c43ca4`: multiplicador aplicado ao preço
  atual e ao caminho de forecasts (`persist_current`).

Para aproveitar a janela overnight sem abrir uma nova grelha de parâmetros,
foram ainda submetidos os quatro schedules causais pré-registados e uma
referência neutra na mesma imagem:

- `03c4ee06-bc05-4519-b5a7-b263b465cf6c`: referência neutra no servidor;
- `47bc80bc-376b-4cc4-aae3-dee155cde5a1`: `native_cheap` no Deucalion CPU;
- `5342355a-f15f-4009-8348-d0d071c433e2`: `community_export` no Union;
- `f2306821-e641-46a6-b4d9-ffe6425073ea`: `cheap_or_export` no Union;
- `e5391bb9-d3fe-40e2-86d8-432c8200dd1b`: `cheap_and_export` no Union.

As duas ablações do ator, os cinco replays temporais e a referência neutra
formam o diagnóstico V5. As ablações do ator continuam fora da distribuição e
os schedules derivados são diagnósticos in-sample, não candidatos de promoção.
Servem para decidir se ainda existe margem causal com o PPO nominal congelado
ou se o próximo passo obrigatório é treinar uma folha PPO local explicitamente
condicionada pelo preço efetivo. Não foi aberta uma grelha `0,975` antes de
observar estes resultados, evitando procura oportunista de hiperparâmetros.

### Resultado anual V5

Os sete jobs remotos terminaram com sucesso e exportaram KPIs anuais. A
referência neutra na imagem V5 reproduziu EUR 20 850,00. O diagnóstico completo
é:

| Receita | Custo settled | Delta vs neutro | Violação elétrica | Decisão |
|---|---:|---:|---:|---|
| `actor_current_only` 0,95 | EUR 20 901,29 | EUR +51,28 | 0,01022 kWh | `REJECT_HARD_GATES` |
| `actor_current_and_forecasts` 0,95 | EUR 20 867,38 | EUR +17,38 | 0 | `REJECT_COST` |
| PPO neutro V5 | EUR 20 850,00 | referência | 0 | `REFERENCE` |
| `native_cheap` | EUR 20 838,76 | EUR -11,24 | 0,00830 kWh | `REJECT_HARD_GATES` |
| **`community_export`** | **EUR 20 816,31** | **EUR -33,69 (-0,162%)** | **0** | **`PASS_COST_WITH_TRADEOFFS`** |
| `cheap_or_export` | EUR 20 835,04 | EUR -14,97 | 0,00877 kWh | `REJECT_HARD_GATES` |
| **`cheap_and_export`** | **EUR 20 818,15** | **EUR -31,85 (-0,153%)** | **0** | **`PASS_COST_WITH_TRADEOFFS`** |

Os dois candidatos válidos passam o perfil
`phase10_w6_adapted_local_v1` e a projeção
`phase10_w6_executed_safety_projection_v1`: EV mínimo viável acima de 0,99,
precisão EV acima de 0,40, zero violações elétricas executadas, zero ciclos
deferrable perdidos, serviço deferrable 1,0, zero violações de SoC e zero
outage unserved energy.

Face ao PPO neutro V5:

| Métrica | `community_export` | `cheap_and_export` |
|---|---:|---:|
| Custo settled | EUR -33,69 | EUR -31,85 |
| Importação | -199,42 kWh | -187,81 kWh |
| Pico diário vs BAU | -0,001199 | -0,001199 |
| Pico absoluto vs BAU | sem alteração material | sem alteração material |
| Ramping vs BAU | +1,46% relativo | +1,37% relativo |
| Emissões | +32,62 kgCO2 (+0,15%) | +25,32 kgCO2 (+0,11%) |
| Autoconsumo solar | +0,205 p.p. | +0,181 p.p. |
| Throughput da bateria | +1 242,71 kWh (+2,49%) | +984,70 kWh (+1,97%) |
| Edifícios com custo melhor | 16/17 | **17/17** |
| Pior delta local | EUR +3,81 | **EUR -0,17** |

`community_export` é o vencedor no objetivo primário, mas piora ligeiramente um
edifício. `cheap_and_export` perde apenas EUR 1,84 de poupança agregada e é a
solução mais equilibrada: todos os 17 edifícios poupam e usa menos bateria. Os
dois ficam `PASS_COST_WITH_TRADEOFFS`, não `PASS_CC_SCORECARD`, porque o ramping
piora mais de 1%, a tolerância explícita do scorecard.

Estes schedules foram derivados dos traces anuais e continuam a ser evidência
in-sample, não um CC aprendido ou diretamente destacável. Contudo, demonstram
que o PPO congelado pode ser melhorado por coordenação temporal e identificam
um alvo concreto para o próximo CC: aprender online quando a comunidade está a
exportar, com a interseção preço-barato/exportação como opção de maior fairness
e menor desgaste.

Evidência bruta:
`runs/remote_results/cc_ppo_controllability_v5_annual_20260806/`.
Scorecard completo:
`runs/remote_results/cc_ppo_controllability_v5_annual_20260806/scorecards/v5_valid_temporal_vs_neutral/`.

Validação funcional antes do replay anual:

- 117 testes focados passaram no conjunto V4/V5, adaptador de preço, registry e
  schema;
- smoke real `persist_current`, 384 transições, 17 checkpoints carregados,
  resultado e manifesto exportados, exit 0;
- smoke real do schedule conservador `cheap_and_export`, 384 transições,
  resultado e manifesto exportados, exit 0.

O re-treino price-responsive continua deliberadamente bloqueado por contrato.
Randomizar apenas o preço observado mantendo a reward na tarifa real ensinaria
o PPO a ignorar o CC. Antes desse treino, o mesmo contexto tem de chegar à
observação atual/seguinte, à base residual e a uma reward económica estritamente
local ou a um oracle local condicionado pelo preço. O leaf continuará sem
observações comunitárias.
