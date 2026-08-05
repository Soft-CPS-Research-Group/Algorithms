# CC causal price control V4

- Dataset: `citylearn_three_phase_electrical_service_demo_15min_parquet`
- Horizonte de evidência: ano completo, passos `0:35039`
- Mercado comunitário: ativo, preço local `0,8` do preço grid
- Objetivo primário: custo comunitário settled
- Estado: campanha anual em curso; cinco probes fixos concluídos, dois em
  reposição e três treinos CC-SMART ainda ativos

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

## Atualização de 2026-08-05: campanha anual parcial

A campanha foi lançada com a imagem do commit `8b73465`. Todos os jobs usam o
mesmo dataset, passos `0:35039`, settlement comunitário ativo e PPO local
congelado seed 789. Cinco dos sete probes fixos já têm resultados anuais
válidos:

| Multiplicador | Custo settled | Delta vs 1,00 | Delta contrafactual | Delta poupança settlement | Pico diário vs BAU | Ramping vs BAU | EV mínimo viável | Decisão parcial |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0,90 | EUR 20 857,60 | EUR +7,60 | EUR +100,62 | EUR +93,02 | 1,0754 | 1,5045 | 99,673% | `REJECT_COST` |
| **0,95** | **EUR 20 845,01** | **EUR -4,99 (-0,024%)** | EUR +70,89 | EUR +75,88 | 1,0772 | 1,4829 | 99,709% | `MARGINAL_CANDIDATE` |
| 1,00 | EUR 20 850,00 | referência | referência | referência | 1,0807 | 1,3620 | 99,927% | `REFERENCE` |
| 1,05 | EUR 20 850,00 | EUR -0,00 | EUR -0,00 | EUR -0,00 | 1,0807 | 1,3620 | 99,927% | `NO_EFFECT` |
| 1,10 | pendente | pendente | pendente | pendente | pendente | pendente | pendente | reposição no servidor |
| 1,20 | pendente | pendente | pendente | pendente | pendente | pendente | pendente | reposição no Deucalion |
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
`cc-ppo-v5-temporal-retrospective-cost-annual-local`; resultado ainda pendente.

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
