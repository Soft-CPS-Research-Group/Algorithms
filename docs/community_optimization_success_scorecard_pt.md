# Avaliar um controlador comunitário

Comparar jobs concluídos com a mesma versão do simulador, ficheiros de entrada, janela, membros, estado inicial, mercado, seed e política de exportação. Registar o conjunto de informação disponível a cada decisor, especialmente quando há previsões perfeitas. Usar os KPIs válidos da última avaliação determinística.

1. **Serviço.** Contar partidas EV observadas e viáveis, cobertura da auditoria e serviço mínimo atingido. Medir separadamente a precisão em torno do SOC pedido. Um carro pode chegar já acima do alvo. Contar ciclos adiáveis concluídos e perdidos.
2. **Limites físicos.** Verificar eventos e energia de violação por fase, limites de SOC e energia final dos ativos. Um custo menor obtido à custa de serviço ou de energia deixada em dívida no fim tem uma interpretação diferente.
3. **Custo.** Com o mercado ativo, usar `district_cost_community_market_settled_total_eur`. Sem mercado, usar `district_cost_total_control_eur`. Calcular a poupança com um job emparelhado do [BAU com limite do utilizador](baseline_comparison_core30.md).
4. **Outros objetivos.** Mostrar pico máximo de importação, picos diários e, quando ajudam a explicar diferenças, importação externa, autoconsumo, throughput da bateria e V2G. Definir a duração dos intervalos e o fuso horário dos dias.
5. **Tempo.** Para políticas online, medir `Runtime/predict_seconds` em todos os passos e apresentar média, P95 e máximo. Registar hardware e threads. Para o oráculo offline registar tempo por janela e intervalo de otimalidade.

Fixar os critérios antes do ensaio. O perfil Core30 utilizado para validar as referências pede serviço mínimo EV viável ≥99,9%, precisão ≥80%, cobertura completa, zero ciclos perdidos, zero eventos elétricos e energia de violação ≤10⁻⁶ kWh. Reportar valores em falta como desconhecidos. Quando há treino, separar seleção e avaliação e repetir as seeds relevantes com a mesma configuração.
