# BAU e controladores por regras

Contrato revisto em 29/10/2026, Algorithms 0.4, SoftCPS Simulator 1.9.1 e revisão BAU seguinte.
Esta revisão define comportamento e verifica execução e exportação. Não é uma
classificação anual de desempenho nem uma calibração empírica de hábitos humanos.

## Controladores principais

| Controlador | EV | Máquina de lavar ou outro ciclo adiável | Bateria doméstica |
|---|---|---|---|
| `NormalPolicy` — BAU com bateria | Pede carga imediatamente após a ligação, sem esperar por preço ou sol. Por omissão, alvo de 100%. | Inicia no primeiro instante em que existe um pedido e o simulador permite o arranque. | Autoconsumo: carrega com excedente fotovoltaico e descarrega para consumo local. |
| `NormalNoBatteryPolicy` — BAU sem controlo da bateria | Igual ao BAU. | Igual ao BAU. | Ação zero. O equipamento continua a existir no simulador, incluindo eventuais perdas passivas. |
| `RBCBasicPolicy` | Usa preço relativo e necessidade de serviço até à saída. A urgência pode exigir carga a qualquer hora. | Começa com preço favorável, urgência ou proximidade do último arranque possível. | Carrega a preço favorável e descarrega a preço elevado para consumo útil. |
| `RBCSmartLocalPolicy` | Aproveita excedente local e previsões; adia se houver margem, mas atende a necessidade de saída. V2G é opcional. | Aproveita excedente local ou arranca por urgência/prazo. | Favorece autoconsumo e redução de importações/picos locais; evita disputar energia com serviço urgente. |
| `RBCCommunityPolicy` | Acrescenta excedente e pressão da comunidade, com as condições económicas configuradas. | Pode aproveitar excedente comunitário, além do local e dos prazos. | Usa também excedente comunitário e alívio das importações da comunidade. |

O BAU é uma hipótese operacional explícita: ligar e carregar; pedir uma lavagem
e iniciar. Não pretende otimizar custo. Para comparar apenas o escalonamento
mantendo o mesmo serviço de mobilidade, pode definir-se `ev_normal_target_soc: 0`
para seguir o SOC pedido para a saída. Essa experiência deve declarar a alteração:
não equivale ao BAU de carga até 100%.

`NormalPolicy` usa por omissão taxas máximas normalizadas de carga/descarga da
bateria de 1, limiares de 0,05 kW e limites estratégicos de SOC de 20–90%.
Os limites físicos observados continuam a prevalecer. Os valores são configuráveis;
a configuração resolvida fica guardada no job. Não há arbitragem de preço ou V2G
no BAU. As capacidades `control_evs`, `control_storage` e `control_deferrables`
permitem experiências explícitas; desativá-las muda o comportamento de referência.

## Horários e preços

Nenhum dos três RBCs está limitado a carregar exclusivamente de dia ou de noite.
O Basic compara o preço atual com os preços previstos. Com tarifa constante não
inventa um período barato ou caro. Se faltar o preço atual, usa o fallback
documentado: horas 0–7 e 22–23 favoráveis; 17–21 desfavoráveis; restantes neutras.
Sem preço e sem hora, fica neutro. Com preço atual mas sem previsões, a
classificação também é neutra: não infere uma tarifa horária.

Smart Local e Community favorecem energia solar. Por omissão,
`price_charge_rate: 0` desativa a carga da bateria motivada apenas por preço;
esse valor já era a intenção do construtor e agora também é respeitado pela
validação das configurações. Não desativa a carga necessária dos EVs.
Os templates históricos que definem outro valor continuam a ser explícitos.
V2G exige `allow_v2g: true` e as respetivas reservas e condições de serviço.

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
As decisões respeitam a margem elétrica observada; urgência não autoriza pedir
potência acima dessa margem. O simulador mantém a decisão final sobre a física.

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

A decisão só conhece os sinais publicados no dataset. Para descontar o novo
ciclo adiável antes da bateria, deve estar disponível `start_power_kw`,
`start_energy_kwh_step` ou, como aproximação conservadora, `cycle_peak_power_kw`.
Sem esses sinais, o arranque continua a funcionar, mas a antecipação da potência
desse ciclo pela bateria não é certificada. As ações auditadas são EV, bateria
elétrica e ciclos adiáveis; controlo direto de HVAC e armazenamento térmico não
está certificado por esta revisão. A presença dessas ações ativas é rejeitada
com uma mensagem explícita, em vez de produzir zeros silenciosamente.

## Serviço EV e potência mínima

O BAU pede carga até 100%, sem usar a tolerância do alvo de saída para parar
antecipadamente. Os RBCs procuram o SOC pedido, com tolerância por omissão de
0,001 (0,1 ponto percentual), em vez dos anteriores 0,04 (4 pontos). Um valor
explícito numa configuração histórica continua a ser respeitado.

Um pedido inferior à potência mínima do carregador produz zero energia. O
último pedido de serviço pode usar o mínimo viável e ultrapassar ligeiramente
o SOC pedido; não deve ficar permanentemente abaixo do alvo. Este ajuste não
aumenta um orçamento solar ou elétrico insuficiente. A revisão seguinte do
Simulator também corrige a disponibilidade perto de 100% e o arredondamento
float32 na fronteira da potência mínima.

## Definição do BAU nativo nos resultados

O simulador calcula opcionalmente um contrafactual próprio,
`citylearn.agents.baseline.BusinessAsUsualAgent`. Na versão 1.9.1, esse agente
distribui a carga dos EVs pelo tempo até à saída. **Não é `NormalPolicy`.**
Os KPIs com referência ao BAU interno não devem ser apresentados como uma
comparação contra o nosso BAU imediato.

`results/kpis.json` identifica agora essa implementação e a versão do simulador
em `baseline_reference`. Uma comparação com o BAU definido neste documento exige
dois jobs com o mesmo dataset, janela, seed, ativos, mercado e serviço configurado.
A revisão seguinte do Simulator muda essa referência para carga imediata até
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
  local. Para novas comparações, usar Smart Local ou Community.
- Os templates `rbc_smart_local.yaml` históricos usam “local” no sentido de execução
  local no computador. Os novos `rbc_smart_building_local*.yaml` selecionam
  explicitamente `RBCSmartLocalPolicy` e eliminam essa ambiguidade.
- Há 37 nomes registados. As regras exportam JSON executável e manifesto comum;
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
do código-fonte antes de publicar. Os ambientes suportados ficam fixados na 1.9.1;
a imagem Jetson antiga continua suspensa até à sua migração de plataforma.

Validação local em checkout limpo: 1 882 testes unitários/integração passaram,
23 foram ignorados com motivos explícitos, e 65 testes de jobs completos passaram.
Ruff, formatação, mypy, dependências e instalação mínima fora do checkout passaram.
