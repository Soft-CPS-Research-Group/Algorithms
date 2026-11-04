# O que aprendemos nas runs longas

Estas observações vêm de avaliações anteriores, em datasets e versões diferentes do Core30 atual. Servem para escolher experiências e interpretar falhas. Para comparar desempenho entre famílias, voltar a executar os candidatos e as referências nas mesmas condições.

## TI-MARL

Na avaliação anual emparelhada de V20 e V21 (17 membros, 35 040 intervalos), V21 melhorou o serviço EV, mas aumentou a fatura comunitária. O carregamento que poupava nas horas baratas coincidia frequentemente com excedente solar que seria exportado a zero. Parte dessa energia foi reposta mais tarde com importação cara. Uma soma de energia líquida assinada vezes o preço chegou a indicar o sentido oposto ao custo liquidado. Para avaliar uma alteração ao planeamento EV, comparar a importação marginal ao preço real e igualar a reserva terminal. Separar a intenção do ator da ação após proteção e da potência aplicada pelo simulador.

O [guia de TI-MARL](../algorithms/ti_marl/README.md) descreve os vínculos por entidade, os sinais de saúde e o contrato do crítico. Campanha de origem `ti_marl_v21_annual_confirmation`.

## MADDPG e MATD3 com observações de dimensão fixa

Nos dois jobs de 8 760 passos da campanha `phase10_maddpg_matd3_latest_20260606`, ambos os agentes reduziram o custo face ao BAU usado na altura e registaram zero energia de violação elétrica. Ao mesmo tempo, a taxa de partidas viáveis com serviço mínimo ficou abaixo de 98% em ambos. Comparar a energia entregue aos EVs antes de atribuir a redução de custo à aprendizagem. Estes agentes esperam uma disposição estável das observações e ações.

A destilação de um professor de otimização deu um candidato MATD3 mais barato que o Smart emparelhado na campanha anual `matd3_global_v5_annual_tiago_d1001db`. A simulação do candidato registou violações elétricas, enquanto o Smart cumpriu esse critério. A imitação é um ponto de partida para o treino. A validação da política precisa das potências efetivamente aplicadas, por fase.

## Transformer MATD3

Na campanha H7 com mudanças de topologia, a combinação útil foi treinar o crítico sobre as ações executadas, distribuir a carga EV necessária pelo tempo restante e manter alguma imitação dirigida aos EVs durante o treino. A verificação em três seeds, sobre a janela de 11 000 passos, mostrou maior variação nos picos do que no custo. Nos ensaios de uma política nova, guardar as propostas do ator e as correções da proteção permite identificar qual dos dois está a assegurar o serviço. Receita e parâmetros no [guia do agente](transformer_matd3.md). Campanha de origem `tmatd3_dynamic15min_passo5_validate_optimize_20260913`.

## Coordenação hierárquica

Na avaliação anual de CC-L2 sobre 17 folhas PPO, descongelar as folhas desde a primeira transição, alterar a regra de base e atribuir a mesma recompensa comunitária a todas coincidiu com mais importação, menos autoconsumo e maior custo. A existência de um sinal de preço variável não garantiu que a comunidade respondesse bem a esse sinal.

Para isolar o efeito do coordenador, reproduzir primeiro a política local completa, incluindo a regra de base, o ator residual e a proteção. Ligar depois o adaptador de preço com multiplicador 1 e verificar paridade. Manter as folhas determinísticas e congeladas durante a comparação entre coordenadores. Medir a fatura à tarifa real, mesmo quando a tarifa apresentada à folha muda. Campanha de origem `cc_level2_ppo_joint_v1`. Ver o [contrato do coordenador](community_coordinator_design.md).
