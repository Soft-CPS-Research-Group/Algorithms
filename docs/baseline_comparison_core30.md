# Referências para comparar MARL no Core30

Executar os cinco controladores nos mesmos membros, ficheiros, janelas, seeds e condições de mercado do candidato. Os dois BAU iniciam as máquinas no primeiro instante admissível e usam a bateria doméstica para guardar excedente solar e servir consumo local.

| Referência | Política | Configuração |
|---|---|---|
| **BAU com limite do utilizador** | Carrega o EV assim que chega, até ao SOC pedido. Limita a última ordem à carga em falta. Com potência mínima do carregador, o EV pode ficar ligeiramente abaixo do alvo. | [bau.yaml](../configs/templates/baselines/core30/bau.yaml) |
| **BAU sem limite do utilizador** | Carrega imediatamente até 100%. A diferença para o primeiro BAU inclui a energia adicional entregue ao EV. | [bau_full_charge.yaml](../configs/templates/baselines/core30/bau_full_charge.yaml) |
| **Basic** | Usa preço atual e urgência para EVs e ciclos adiáveis. A bateria faz autoconsumo local. | [basic.yaml](../configs/templates/baselines/core30/basic.yaml) |
| **Smart** | Usa previsões de preço e excedente solar local para agendar cargas. A bateria combina autoconsumo, arbitragem e reserva. | [smart.yaml](../configs/templates/baselines/core30/smart.yaml) |
| **Community** | Acrescenta a energia partilhável entre membros. As baterias podem ajudar em períodos caros quando a importação comunitária atinge o gatilho de 15 kW. Este valor desencadeia uma ação e o pico observado pode ultrapassá-lo. | [community.yaml](../configs/templates/baselines/core30/community.yaml) |
| **Oráculo semanal comunitário** | Conhece os dados da semana e planeia de uma vez as 672 ações de quarto de hora. É a referência de custo sujeita a serviço, restrições físicas e reservas terminais. | [Procedimento do oráculo](oracles/total_energy_milp.md) |

**Denominador.** Calcular a poupança por `100 × (custo_BAU_com_limite − custo_candidato) / custo_BAU_com_limite` quando o custo de referência é positivo. Somar os custos das janelas emparelhadas antes de calcular a percentagem agregada. Usar o custo do job explícito de BAU com limite. O contrafactual nativo do simulador tem uma definição própria.

**Mercado.** Verificar no manifesto os pressupostos do ensaio. Na comparação Core30, a compra externa usa o preço do dataset, a partilha interna usa 80% desse preço, a venda à rede tem receita zero e o autoconsumo solar tem custo de aquisição zero. Usar o custo liquidado pelo simulador.

**Leitura.** Verificar serviço EV, ciclos concluídos, violações elétricas e energia terminal antes do custo. Reportar também o pico. Para controlo online, medir o tempo por decisão. Para o oráculo, medir o tempo de resolução por semana e registar o limite do solver. O controlador móvel `MILP` está disponível para estudos de horizonte.
