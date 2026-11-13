# Referências para comparar MARL no Core30

## Base atual

O runtime usa **SoftCPS REC Simulator 2.1.0**. Os dados Portugal da mesma release
incluem `portugal-realism-4` e `portugal-ev-mobility-1`. A designação geral `v3.3`
não basta para identificar esta revisão: registar também os hashes do schema,
de `CURRENT_EVALUATION.json` e dos ficheiros referenciados. A fonte está fixada no
commit `5f01523e8726a4ff4523bafa18911e991da5fe71` do Simulator. Para Core30 Normal,
o SHA-256 do schema é
`e4a7642eab79b09e40002a74bfb67c53480af6d2841f1d4cb5981f006adf3bca`.

As comparações anteriores usavam outra física e outros inputs. Não sustentam
poupanças, rankings ou calibração nesta revisão. Os templates conservam as
políticas como ponto de partida; não representam um novo benchmark concluído.

## Controladores

Executar os cinco controladores nos mesmos membros, ficheiros, janelas, seeds e condições de mercado do candidato. Os dois BAU iniciam as máquinas no primeiro instante admissível e usam a bateria doméstica para guardar excedente solar e servir consumo local.

| Referência | Política | Configuração |
|---|---|---|
| **BAU com limite do utilizador** | Carrega o EV assim que chega, até ao SOC pedido. Limita a última ordem à carga em falta. Com potência mínima do carregador, o EV pode ficar ligeiramente abaixo do alvo. | [bau.yaml](../configs/templates/baselines/core30/bau.yaml) |
| **BAU sem limite do utilizador** | Carrega imediatamente até 100%. A diferença para o primeiro BAU inclui a energia adicional entregue ao EV. | [bau_full_charge.yaml](../configs/templates/baselines/core30/bau_full_charge.yaml) |
| **Basic** | Usa preço atual e urgência para EVs e ciclos adiáveis. A bateria faz autoconsumo local. | [basic.yaml](../configs/templates/baselines/core30/basic.yaml) |
| **Smart** | Usa previsões de preço e excedente solar local para agendar cargas. A bateria combina autoconsumo, arbitragem e reserva. | [smart.yaml](../configs/templates/baselines/core30/smart.yaml) |
| **Community** | Acrescenta a energia partilhável entre membros. As baterias podem ajudar em períodos caros quando a importação comunitária atinge o gatilho de 15 kW. Este valor desencadeia uma ação e o pico observado pode ultrapassá-lo. | [community.yaml](../configs/templates/baselines/core30/community.yaml) |
| **Oráculo semanal comunitário** | Conhece os dados da semana e planeia de uma vez as 672 ações de quarto de hora. Cada semana começa no estado observado do mesmo episódio. A última pode ser parcial. | [Configuração](../configs/templates/baselines/core30/weekly_oracle.yaml) · [Execução e verificação](oracles/total_energy_milp.md) |

**Denominador.** Calcular a poupança por `100 × (custo_BAU_com_limite − custo_candidato) / custo_BAU_com_limite` quando o custo de referência é positivo. Somar os custos das janelas emparelhadas antes de calcular a percentagem agregada. Usar o custo do job explícito de BAU com limite. O contrafactual nativo do simulador tem uma definição própria.

**Mercado.** Verificar no manifesto os pressupostos do ensaio. Na comparação Core30, a compra externa usa o preço do dataset, a partilha interna usa 80% desse preço, a venda à rede tem receita zero e o autoconsumo solar tem custo de aquisição zero. Usar o custo liquidado pelo simulador.

**Leitura.** Verificar serviço EV, ciclos concluídos, violações elétricas e energia terminal antes do custo. Reportar também o pico. Para controlo online, medir o tempo por decisão. Para o oráculo, medir o tempo de resolução por semana e registar o limite do solver. O controlador móvel `MILP` está disponível para estudos de horizonte.

## Retomar a avaliação

1. **Verificar a integração.** Instalar a versão fixada, validar as configurações e
   correr ensaios curtos pelo programa de execução comum. Confirmar KPIs,
   timestamps, inventário dos exports e ausência de alterações dos inputs.
   Os testes da atualização cobrem este funcionamento; não certificam qualidade anual.
2. **Auditar serviço e energia.** Usar primeiro 14 dias de janeiro de 2023 no
   Core30 Normal. Comparar os dois BAU, Basic, Smart, Community e o oráculo por sessão EV e
   ciclo de aparelho: chegada, alvo, partida, energia AC, energia na bateria,
   perdas, excesso de carga e serviço em falta. Distinguir atingir o alvo exato,
   ficar dentro da tolerância e cumprir o serviço mínimo. Verificar potências,
   limites elétricos e SOC inicial/final das baterias. Explicar diferenças antes
   de alterar regras.
3. **Verificar o encadeamento do oráculo no mesmo ensaio.** Correr duas semanas
   encadeadas em janeiro de 2023 e verificar
   planos contra trajetórias executadas, incluindo uma fronteira com EV ligado
   e outra com aparelho em execução/pedido pendente. Registar limites primal e
   dual, serviço, energia terminal e tempo real por resolução. Uma passagem
   curta sem fronteira semanal não fecha esta validação. O oráculo é uma
   referência offline de informação perfeita para o modelo suportado, não uma
   prova do ótimo anual nem um controlador com latência online comparável.
4. **Executar o ano comum antes de rever políticas.** Após a validação técnica,
   de serviço e energia, avaliar os seis com as políticas atuais em 2024
   inteiro: `[35040, 70176)`, 35 136 transições de 15 minutos, 35 137 observações
   e `terminal_observation_padding: true`. Usar um episódio contínuo por
   controlador, zero episódios de treino, seed do simulador 6763 e seed do
   runtime 22. Exportar KPIs e séries temporais só no final; MLflow desligado.
   A cada semana, o oráculo recebe o estado vivo do mesmo episódio. Comparar
   custo liquidado, pico de importação em kW, serviço, energia dos EVs e estado
   terminal, com o BAU com limite como denominador. Este ensaio anual serve
   para conhecer o comportamento das referências nos dados novos; não ajustar
   os limiares do Basic ou do Community antes dele. Corrigir eventuais falhas
   de implementação ou de dados antes da comparação e registar a nova versão.
5. **Revalidar a política sazonal depois do ano comum.** Manter as quatro
   quinzenas de validação de 2023: 1–14 de fevereiro, maio, agosto e novembro,
   dias completos em UTC. Qualquer calibração usa apenas o restante 2023.
   Os limiares atuais do Basic e o gatilho comunitário de 15 kW ficam para esta
   fase. Não degradar um controlador para impor uma classificação: custo, pico
   e serviço podem ter ordenações diferentes. Se as políticas mudarem, repetir
   a comparação anual com as versões identificadas; não misturar resultados.
6. **Comparar MARL.** Só depois de estabilizar as referências, treinar em 2023
   fora das janelas reservadas, selecionar naquelas quatro quinzenas e congelar
   o candidato antes de o avaliar em 2024. Usar várias seeds de treino e a mesma
   auditoria de serviço/energia. As explorações anteriores de 2024 devem ser
   declaradas; este ano não é um conjunto nunca observado.

O SOC à chegada continua a ser um input destas sessões. O gerador contém um
balanço de mobilidade de referência; não assumir que a energia adicional que
um controlador entrega reaparece na próxima chegada. A versão 2.1.0 corrige a
continuidade em mudanças de carregador/topologia e torna explícita a energia
das chegadas no registo contabilístico. Esta distinção faz parte da leitura dos
dois BAU.

Antes de submeter, disponibilizar uma cópia imutável dos dados atuais e verificar
os hashes no destino. Não reutilizar a antiga pasta `sim200_a0adc12`. Fixar também
o digest de uma imagem com este código e Simulator 2.1.0. Preferir o Server para
os baselines e o solver em CPU; reservar o Union para trabalhos que beneficiem
da GPU, confirmando autenticação recente, capacidade e imagem compatível.
Os estados dos workers devem ser consultados no momento da submissão.
