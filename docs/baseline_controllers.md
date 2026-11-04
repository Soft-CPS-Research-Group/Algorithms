# BAU e controladores por regras

Contrato atualizado em 03/11/2026, Algorithms 0.4, SoftCPS Simulator 2.0.0.
Esta revisão define comportamento e verifica execução e exportação. Não é uma
garantia universal de desempenho nem uma calibração empírica de hábitos humanos.

A [comparação atual do Core30](baseline_comparison_core30.md) usa o BAU com
limite como denominador. Os relatórios históricos mantêm a referência original
e identificam quando foram substituídos.

## Controladores principais

| Controlador | EV | Máquina de lavar ou outro ciclo adiável | Bateria doméstica |
|---|---|---|---|
| `NormalPolicy` — BAU 100% (variante) | Pede carga imediatamente após a ligação, sem esperar por preço ou sol. Por omissão, alvo de 100%. | Inicia no primeiro instante em que existe um pedido e o simulador permite o arranque. | Autoconsumo: carrega com excedente fotovoltaico e descarrega para consumo local. |
| `NormalTargetPolicy` — BAU (referência principal) | Pede carga imediatamente até ao SOC pedido para a sessão. Limita a última ordem para não ultrapassar esse alvo. | Inicia logo que o pedido pode arrancar. | Autoconsumo local, sem arbitragem. |
| `NormalNoBatteryPolicy` — BAU 100% sem controlo da bateria | Igual ao BAU 100%. | Igual ao BAU 100%. | Ação zero. O equipamento continua a existir no simulador, incluindo eventuais perdas passivas. |
| `RBCBasicPolicy` — Basic | Usa preço atual e limiar fixo, sem previsões de preço; mantém serviço até à saída. A urgência pode exigir carga a qualquer hora. | Começa com preço abaixo do limiar, urgência ou proximidade do último arranque possível. | Mesmo autoconsumo reativo do BAU: guarda excedente local e serve consumo local, sem arbitragem de rede. |
| `Smart` | Mantém as proteções de serviço; compara preços previstos e antecipa carga com excedente local. V2G é opcional. | Usa preço relativo, urgência e prazo; acrescenta arranque com excedente local. | Coordena captura solar, arbitragem de preço, reservas e limitação da carga da bateria ao pico local configurado. |
| `RBCCommunityPolicy` — Community | Acrescenta excedente comunitário às decisões de preço/serviço do Smart. | Pode aproveitar excedente comunitário, além do local e dos prazos. | Com mercado ativo, usa excedente partilhável e ajuda nos picos comunitários caros, com gatilho agregado de 15 kW. Sem mercado, usa a política local. |

O **oráculo semanal comunitário** é a referência de otimização escolhida para comparar com MARL. Planeia em conjunto as ações das baterias, dos EVs e dos ciclos adiáveis, dando prioridade ao serviço e depois ao custo comunitário. O controlador `MILP` com horizonte móvel de 24 horas fica como ferramenta de investigação. A formulação, as condições terminais e os resultados exploratórios constam do [estudo do oráculo](milp_physics_reference_core30_2026_11_03.md).

O BAU é uma hipótese operacional explícita: ligar e carregar; pedir uma lavagem
e iniciar. Não pretende otimizar custo. **A referência principal é agora
`NormalTargetPolicy`, com bateria e limite EV definido pelo utilizador.**
`NormalPolicy` mantém-se como variante BAU 100%. A decisão sucede ao ensaio inicial
da variante com limite, em 03/11/2026. O template principal é
`configs/templates/baselines/core30/bau.yaml`; `bau_target.yaml` é uma cópia
compatível. `bau_full_charge.yaml` seleciona explicitamente o BAU 100%.

O BAU com limite lê o SOC pedido de cada EV ligado, sem preços, previsões,
adiamento ou V2G. A última ordem de carga fica limitada à energia que falta até
ao alvo, usando capacidade e SOC observados. Não presume eficiências não
observadas. Se essa ordem ficar abaixo da potência mínima do carregador, para
ligeiramente abaixo do alvo; nunca a arredonda para cima, nem na saída. Com
passos de 15 minutos, esta discretização deve ser medida nos resultados. Se o
EV já chegar acima do alvo, não carrega nem descarrega para o corrigir. Sem SOC,
capacidade ou alvo válidos, não inventa um pedido de carga. Mantém a mesma
repartição física de potência entre carregadores que o BAU original.

O parâmetro histórico `NormalPolicy.ev_normal_target_soc: 0` só muda o ponto de
paragem entre intervalos e pode ultrapassar o pedido no último intervalo. Não
equivale ao novo limite. `NormalTargetPolicy` exige esse parâmetro a zero e
exporta uma identidade própria no bundle executável e no manifesto.

`NormalPolicy` usa por omissão taxas máximas normalizadas de carga/descarga da
bateria de 1, limiares de 0,05 kW e limites estratégicos de SOC de 20–90%.
Os limites físicos observados continuam a prevalecer. Os valores são configuráveis;
a configuração resolvida fica guardada no job. Não há arbitragem de preço ou V2G
no BAU. As capacidades `control_evs`, `control_storage` e `control_deferrables`
permitem experiências explícitas; desativá-las muda o comportamento de referência.

## Horários e preços

Nenhum dos três RBCs está limitado a carregar exclusivamente de dia ou de noite.
O Basic usa exclusivamente o preço atual: barato até **0,0765 €/kWh**, caro a partir
de **0,1078 €/kWh**, neutro entre os dois. São os tercis da tarifa Core30 de 2023,
arredondados a quatro casas decimais; não foram escolhidos com resultados de 2024.
Os parâmetros `basic_cheap_price_eur_kwh` e `basic_expensive_price_eur_kwh` são
fixos durante o job e ficam no export. EVs e eletrodomésticos usam o limiar barato
e a urgência. A bateria atual usa `basic_storage_policy: self_consumption`:
não carrega da rede por preço barato nem espera pelo limiar caro para servir
consumo local. O limiar caro continua disponível para reproduzir a bateria
anterior com `basic_storage_policy: price_only`.
Se faltar o preço, fica neutro e mantém
as decisões de serviço. Uma tarifa constante pode ser barata/cara segundo estes
limiares absolutos. Não há um fallback horário no Basic atual.

Smart e Community mantêm a classificação relativa às previsões auditada anteriormente.
Com tarifa constante ficam neutros. Sem preço atual usam o fallback histórico
(0–7 e 22–23 favoráveis, 17–21 desfavoráveis); sem previsões e com preço atual
ficam neutros. `basic_price_policy: forecast_relative` permite reproduzir a
semântica histórica do Basic; sem escolha explícita da bateria, essa opção
preserva `price_only`. A referência atual é `fixed_thresholds` com
`self_consumption`. Esta última usa os defaults de bateria do BAU (taxas de 1,
limiares de 0,05 kW e limites estratégicos de SOC de 20–90%). Selecionar
explicitamente `price_only` conserva as taxas anteriores de 0,35 e o limiar
solar de 0,25 kW, salvo configuração explícita.

Smart e Community usam por omissão `price_charge_rate: 0.60`; o Basic atual
não faz arbitragem de rede com a bateria. A versão anterior dos dois Smart
desativava essa capacidade, adiava os EV mesmo em
horas baratas e descarregava a bateria para qualquer consumo antes de avaliar
a regra de preço. Isso explicava parte da inversão de custos.

Os três usam a mesma base de proteção de serviço EV e de prazo. Smart
acrescenta carga EV e arranque de ciclos com excedente solar atual. O Basic já
aproveita o excedente na bateria; o Smart coordena também previsões e preço.
Na bateria, dá prioridade ao solar,
alivia picos atuais e períodos caros, reserva espaço quando prevê excedente e
limita a carga comprada à margem até ao pico local configurado. Uma previsão
de pico futuro, isoladamente, não manda gastar já a bateria.

O Community recebe a configuração efetiva do mercado do simulador. Só usa energia
comunitária com desconto se esse mercado estiver ativo. Sem mercado, filtra
os sinais comunitários e reproduz a política local: exportar bateria para outro
membro não pode ser tratado como autoconsumo gratuito. Com mercado, acrescenta
um orçamento de excedente partilhado e alívio de picos comunitários durante
períodos caros. O limiar agregado por omissão é **15 kW**, independentemente do
número de membros. `community_import_threshold_kw` é um gatilho de apoio das
baterias, não um teto garantido de importação. Valores explícitos continuam a
ser respeitados para reprodução de experiências.

V2G continua opcional e protegido por reservas e condições de serviço. As variantes
especializadas com sinais hierárquicos conservam as regras anteriores em classes
de compatibilidade; a sua revisão permanece adiada. Os novos defaults aplicam-se
aos dois controladores operacionais, não aos professores históricos.

Esta é uma hierarquia de capacidades. Nenhuma heurística pode garantir menor
custo e menor pico em todos os datasets, preços, previsões e estados iniciais.
A comparação deve apresentar custo, energia, picos e serviço em conjunto, incluindo
excesso de SOC e casos fisicamente inviáveis.

## Pedidos de lavagem e serviço

Um pedido pendente não pode começar antes da sua janela válida nem reiniciar um
ciclo em curso. `must_run` significa que o serviço é obrigatório; não significa
que o arranque tenha de acontecer já. Os RBCs usam `must_start_now`, o último
arranque possível, a folga e o prazo, com margem de segurança configurável.
O BAU não aproveita essa flexibilidade: inicia logo que `can_start` o permite.

O simulador executa o perfil do ciclo depois do arranque. O controlador não inventa
pedidos, não muda a duração e não deve enviar arranques repetidos. Se o dataset
só publicar uma janela de disponibilidade e não o instante real do pedido humano,
o primeiro instante dessa janela é a hipótese usada pelo BAU.

## Contabilidade e limites

Dentro de cada edifício, os controladores calculam EVs, novos ciclos adiáveis e
bateria por esta ordem, mantendo a ordem original do vetor de ações à saída.
O excedente consumido por um dispositivo deixa de estar disponível para o seguinte.
Nos RBCs, as decisões respeitam a margem elétrica observada; urgência não autoriza
pedir potência acima dessa margem. Os BAU pedem a potência máxima simultaneamente
e deixam a repartição entre carregadores ao Simulator. O simulador mantém a
decisão final sobre a física em todos os casos.

O Community mantém um orçamento comum entre os edifícios da mesma instância.
Energia reservada num edifício não pode voltar a ser oferecida como excedente no
edifício fornecedor. A configuração exige `count: 1`: essa instância recebe os
vários edifícios. Instâncias independentes não têm uma reserva comunitária comum.
A ordem dos edifícios é determinística; esta regra não é um otimizador global
nem garante uma repartição ótima ou equitativa entre participantes.

As grandezas históricas de energia por passo são convertidas em potência com o
passo real. As observações em kW não são reconvertidas. Os carregadores são ligados
às ações pela identidade, não pela ordem alfabética, e a potência observada em
topologias dinâmicas prevalece sobre a configuração inicial.
Também a potência, capacidade e ligação de fase de uma bateria instalada ou
substituída durante o episódio são lidas das observações atuais. O schema inicial
é apenas fallback quando esses sinais não estão disponíveis.

A decisão só conhece os sinais publicados no dataset. Para descontar o novo
ciclo adiável antes da bateria, deve estar disponível `start_power_kw`,
`start_energy_kwh_step` ou, como aproximação conservadora, `cycle_peak_power_kw`.
Sem esses sinais, o arranque continua a funcionar, mas a antecipação da potência
desse ciclo pela bateria não é certificada. As ações auditadas são EV, bateria
elétrica e ciclos adiáveis; controlo direto de HVAC e armazenamento térmico não
está certificado por esta revisão. A presença dessas ações ativas é rejeitada
com uma mensagem explícita, em vez de produzir zeros silenciosamente.

## Serviço EV e potência mínima

O BAU 100% pede carga até 100%, sem usar a tolerância do alvo de saída para parar
antecipadamente. Os RBCs procuram o SOC pedido, com tolerância por omissão de
0,001 (0,1 ponto percentual), em vez dos anteriores 0,04 (4 pontos). Um valor
explícito numa configuração histórica continua a ser respeitado.

Um pedido inferior à potência mínima do carregador produz zero energia. O
último pedido de serviço pode usar o mínimo viável e ultrapassar ligeiramente
o SOC pedido; não deve ficar permanentemente abaixo do alvo. Este ajuste não
aumenta um orçamento solar ou elétrico insuficiente. O Simulator 1.9.2 também corrige a disponibilidade perto de 100% e o arredondamento
float32 na fronteira da potência mínima.
No último intervalo disponível, havendo carga por servir, o RBC dá prioridade
ao pedido urgente dentro da potência disponível. Não o limita ao equivalente
sem perdas do défice de SOC: as perdas do carregador e da bateria deixariam
parte do pedido por cumprir. Esta proteção pode ultrapassar o SOC pedido.

## Definição do BAU nativo nos resultados

O simulador calcula opcionalmente um contrafactual próprio,
`citylearn.agents.baseline.BusinessAsUsualAgent`. Até à versão 1.9.1, esse agente
distribuía a carga dos EVs pelo tempo até à saída. **Não é `NormalPolicy`.**
Os KPIs com referência ao BAU interno não devem ser apresentados como uma
comparação contra o BAU principal com limite do utilizador. Os templates Core30
atuais desligam esse contrafactual. `results/kpis.json` declara a referência
principal em `primary_comparison_reference`, sem fingir que a executou. Os
campos `baseline_reference` e os rácios nativos continuam a identificar o
diagnóstico do Simulator, quando explicitamente solicitado. As reduções
principais exigem um job `NormalTargetPolicy` com inputs equivalentes.

`results/kpis.json` identifica agora essa implementação e a versão do simulador
em `baseline_reference`. Uma comparação com o BAU definido neste documento exige
dois jobs com o mesmo dataset, janela, seed, ativos, mercado e serviço configurado.
O Simulator 1.9.2 muda essa referência para carga imediata até
100%, identificada como `immediate-full-charge-v1`. O Algorithms deteta a
definição no pacote carregado e regista-a em `baseline_reference`; com 1.9.1
preserva a descrição histórica. Mesmo com o objetivo alinhado, as implementações
e os limites configuráveis não são declarados numericamente equivalentes.

## Nomes retirados e compatibilidade

- `RuleBasedPolicy` foi retirado do catálogo e dos templates públicos. Um job
  antigo com esse nome falha com alternativas explícitas; não há substituição
  silenciosa por um algoritmo com outro comportamento.
- A antiga lógica EV permanece privada em `LegacyEVPolicy`, porque coletores e
  experiências offline históricas ainda dependem dela. Não é uma opção para
  novos jobs nem um controlador principal suportado.
- `RBCSmartPolicy` mantém-se como compatibilidade para professores de aprendizagem,
  residuais, configurações e bundles existentes. Pode recorrer a sinais comunitários
  como fallback, pelo que não deve sustentar afirmações de informação estritamente
  local. Para novas comparações, usar `Smart` ou Community.
- `Smart` é o nome público da implementação local já auditada. O identificador
  `RBCSmartLocalPolicy` mantém-se como alias de compatibilidade, com decisões iguais.
  A mudança de nome não altera o comportamento dos professores históricos.
- Os templates `rbc_smart_local.yaml` históricos usam “local” no sentido de execução
  local no computador. Os novos `rbc_smart_building_local*.yaml` selecionam
  explicitamente `Smart` e eliminam essa ambiguidade.
- Há 39 nomes registados, incluindo `NormalTargetPolicy` e o alias público `Smart`. As regras exportam JSON executável e manifesto comum;
  não são redes neuronais a converter para ONNX. A verificação reproduz as decisões
  da avaliação final. Bundles anteriores conservam a sua versão do Algorithms.

## Validação e limites da evidência

`tests/test_baseline_behavior_contract.py` verifica os cinco comportamentos,
preço constante, horários de fallback, instantes de arranque, partilha de energia,
potência/energia a 15 s, 15 min e 1 h, identidade e potência dos carregadores,
limites elétricos e paridade entre construção direta e configuração validada.
Os testes anteriores de baselines e variantes com sinal continuam a ser executados.

`tests/e2e/test_baseline_semantics.py` verifica no simulador real o início único,
conclusão e consumo de uma lavagem; testa também os cinco agentes no dataset
português atual Core-4, com exportação final e replay do bundle. Os testes comuns
de exportação cobrem os cinco controladores e o Smart de compatibilidade.

A antiga campanha anual `run_annual_rec_baseline_audit.py` é um protocolo histórico
do dataset `2023-q15-v1.9`, que já não vem no Simulator 1.9.x. Os cinco testes dos
seus números de catálogo só correm quando esses dados históricos estão presentes.
Não se aplicam esses números aos novos dados de 2023–2024. A migração e nova campanha
anual ficam no trabalho de avaliação posterior, sem reabrir agora a etapa 5.

O Simulator 1.9.0 publicado omitia `citylearn.utils` e falhava a exportação do
manifesto dos KPIs. A correção 1.9.1 inclui o módulo e testa o pacote instalado fora
do código-fonte antes de publicar. A revisão inicial usou 1.9.2; a versão suportada atual é 2.0.0.
A imagem Jetson antiga continua suspensa até à sua migração de plataforma.

Validação local em checkout limpo: 1 882 testes unitários/integração passaram,
23 foram ignorados com motivos explícitos, e 65 testes de jobs completos passaram.
Ruff, formatação, mypy, dependências e instalação mínima fora do checkout passaram.

## Revisão da hierarquia em 29/10/2026

`tests/test_economic_baseline_hierarchy.py` verifica oportunidades de carga barata,
captura solar, limites locais, autoridade da configuração real do mercado,
reserva comunitária sem duplicação e limiares agregados. O Basic mantém as suas
regras nessa campanha histórica. A série histórica e a campanha revista usam as mesmas tarifas e dados;
a variante com mercado ligado é identificada separadamente como cenário
configurado, sem alterar os ficheiros do dataset. Os resultados finais da
campanha ficam no [relatório desta revisão](baseline_hierarchy_2026_10_29.md).

Reprodução pelo ciclo comum de jobs:

```console
python scripts/experiments/run_baseline_hierarchy_audit.py \
  --dataset-path /caminho/Simulator/data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json \
  --output-dir /caminho/resultados --label mercado-ativo --market \
  RBCBasicPolicy RBCSmartLocalPolicy RBCCommunityPolicy
```

Sem `--market`, respeita o mercado do schema original. Os templates antigos
com taxas, V2G ou limites explícitos são experiências com parametrizações próprias;
não representam automaticamente a campanha com os defaults auditados.


## Atualização para Simulator 2.0.0 e Core30 v3.3

As duas variantes históricas `NormalPolicy` e `NormalNoBatteryPolicy` pedem
carga máxima simultaneamente nos carregadores ligados, até 100%. A potência entregue é limitada proporcionalmente pelo Simulator; a
ordem dos carregadores não reserva capacidade para o primeiro. O BAU 100% não reduz
o pedido pela estimativa de potência aceite perto de 100%. Essa estimativa é,
porém, usada para calcular a procura a servir pela bateria da casa: o pedido
nominal de um carro quase cheio não constitui consumo efetivo. A bateria doméstica
mantém autoconsumo e os respetivos limites estratégicos. Testes com dois carros
iguais confirmam 2 kW para cada um numa ligação limitada a 4 kW.

Os datasets de referência são os de `softcpsrecsimulator-v2.0.0`, com
`CURRENT_EVALUATION.json` v3.3. O tag upstream `v2.0.0` é outro produto.
A comparação histórica acima continua a descrever o ensaio de outubro; não
certifica a ordenação dos controladores no Core30 atual.

O Basic limita agora a energia comprada em cada passo ao espaço até ao teto de
SOC configurado. Verificar apenas o SOC antes da carga permitia ultrapassar 90%
num único passo. Nos três RBCs, os limiares de preço continuam iguais aos anteriores;
esta correção não reduz a capacidade económica do Basic para favorecer os outros.
A recolha remota aceita os KPIs finais oficiais `results/kpis.json` quando o
orquestrador ainda só lista CSV e o Simulator exporta Parquet. O JSON original é
preservado e a tabela CSV derivada mantém os mesmos valores e células inválidas.

### Referências MILP

Para novas comparações usar o [protocolo do oráculo semanal](milp_physics_reference_core30_2026_11_03.md#reprodução), que fixa as reservas terminais antes da resolução. A [primeira auditoria MILP](milp_core30_audit_2026_11_03.md) documenta os ensaios históricos com planos individuais e comunitários.

- `FixedServiceOracleReplayPolicy`: otimiza apenas a bateria, mantendo o serviço
  EV/eletrodomésticos de um RBC explícito. É um limite condicional, não um
  otimizador integral da casa.
- `TotalHomeOracleReplayPolicy`: plano com conhecimento do futuro para uma casa,
  incluindo bateria, EVs e ciclos. Minimiza importação individual com exportação
  sem remuneração; não representa, por si, o custo liquidado em comunidade.
- `TotalOracleReplayPolicy`: executa planos conjuntos produzidos pelo MILP de
  energia total, com liquidação individual ou agregada. As variantes decompostas
  individuais só são equivalentes quando não existe acoplamento comunitário.
- O MILP de scorecard procura picos/rampas sob restrições de custo; não tem o
  mesmo objetivo que minimizar o custo exclusivamente.

A preparação do MILP integral preserva fases não ligadas com limite zero e
resolve equipamentos sem fase explícita pela ligação predefinida da casa.
Topologia dinâmica, perturbações de robustez e outages são rejeitados pelos
adaptadores estáticos: não podem produzir um certificado omitindo esses efeitos.
Equipamentos configurados com ações desativadas também são rejeitados, porque
o otimizador não pode impor um plano a um dispositivo que não pode controlar.
Os planos incluem explicitamente os equipamentos sem pedidos durante a janela,
com ações zero; o replay continua a rejeitar ações em falta. A preparação do MILP
de bateria aceita os exports CSV históricos e os Parquet atuais, incluindo partes
ordenadas, e os identificadores `Member_001` do Core30.
Normal e Safety podem ser preparados; a existência de solução depende dos limites
físicos e do serviço. Limites de solver, hipóteses de fronteira, estado inicial,
perdas, energia terminal e diferenças da física têm de acompanhar os resultados.
Só a execução do plano no Simulator permite classificá-lo como referência
fisicamente validada. Estes planos não são políticas online exportáveis para
inferência sobre cenários novos.

O consumo e a geração observados são medições do último intervalo concluído.
Por isso, autoconsumo é a regra de decisão, não uma garantia de exportação nula
em cada intervalo: mudanças de carga/sol, limites e perdas podem causar desvios.
O BAU nativo tem acesso direto ao estado interno do Simulator; os controladores
do Algorithms e os seus bundles usam o contrato de observações. Os dois não
devem ser declarados numericamente equivalentes.

A auditoria atual, incluindo as janelas Core30, correções MILP e limitações da
comparação económica, está no [relatório de novembro](baselines_core30_audit_2026_11_02.md).
A verificação final passou 1 949 testes gerais (23 ignorados por condições
explicitadas) e 68 testes de execução completa/exportação.


### Configurações atuais reproduzíveis

Os templates em [`configs/templates/baselines/core30/`](../configs/templates/baselines/core30/)
usam os defaults auditados, Core30 Normal v3.3, 48 h de janeiro de 2023, seed 22,
uma avaliação determinística, exportação final, MLflow desligado e verificação
obrigatória do bundle. Colocar o dataset da release do Simulator em
`datasets/rec_pt_2023_core_30_v2` ou ajustar `simulator.dataset_path` ao destino.

```console
python run_experiment.py --config configs/templates/baselines/core30/bau.yaml --validate
python run_experiment.py --config configs/templates/baselines/core30/bau.yaml
```

Escolher `bau` (principal), `bau_full_charge` (100%), `bau_no_battery`
(100% sem controlo da bateria), `basic`, `smart` ou `community`.
`bau_target.yaml` permanece como caminho compatível para o BAU principal.
`smart_local.yaml` fica como caminho de compatibilidade para a mesma referência Smart.
Os templates históricos 15 s/15 min/2022 acima mantêm dados e parâmetros próprios;
não reproduzem automaticamente esta referência Core30. Para repetir uma janela
remota, usar a configuração guardada no respetivo job e a imagem registada no
[histórico da auditoria](baselines_core30_audit_2026_11_02.md).

### Calibração do pico comunitário — 03/11/2026

A [campanha de calibração e validação](community_peak_calibration_2026_11_03.md)
executou 28 jobs em quatro semanas de 2023 e 16 jobs nas mesmas épocas de 2024.
O protocolo selecionou 15 kW em 2023. Em 2024, esse valor reduziu o custo de
energia em 2,12% face ao Community original, mas deixou o pico máximo igual
e reduziu a média dos picos diários em apenas 0,055%, com mais 57,53% de energia
movimentada nas baterias e menor reserva final. Todos os critérios de serviço,
limites físicos e exportação passaram.

A decisão inicial da campanha preservou 210 kW por não existir melhoria de pico
validada. **Depois de analisar esses limites, Tiago escolheu a configuração de
15 kW como Community oficial**, sem um nome de variante. O seu ganho de custo e
as limitações de pico permanecem documentados; não se transforma esta decisão
numa alegação de melhoria do pico. A coordenação adicional sugerida no
[diagnóstico](community_peak_diagnosis_2026_11_03.md) ficou adiada.

O mesmo fecho renomeou Smart Local para Smart e simplificou o Basic para preço
atual e limiares fixos. Os BAUs e a política do Smart foram preservados. O
[registo de fecho](baseline_reference_release_2026_11_03.md) descreve a seleção
dos limiares, compatibilidade, validação e âmbito dos resultados.
