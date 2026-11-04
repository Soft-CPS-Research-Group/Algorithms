# Guias do Algorithms

| Quero… | Por onde começar |
|---|---|
| Instalar e correr o primeiro job | [Exemplo executável](../README.md#run-a-first-job) e [instalação por plataforma](runtime/installation.md) |
| Configurar treino, avaliação, seeds e retoma | [Configuração](runtime/configuration.md) e [checkpoints e retoma](runtime/reliability.md) |
| Acrescentar um algoritmo | [Catálogo](agent_catalogue.md), [contrato de um agente](runtime/agents.md) e [ciclo comum do job](runtime/README.md) |
| Comparar MARL com as referências | [BAU, Basic, Smart, Community e oráculo semanal](baseline_comparison_core30.md) e [critérios de avaliação](community_optimization_success_scorecard_pt.md) |
| Resolver uma semana com conhecimento perfeito | [Preparação, resolução e replay do oráculo](oracles/total_energy_milp.md) |
| Entregar modelos para inferência | [Escolha dos outputs](runtime/exports.md) e [manifesto e formatos de inferência](inference_bundle.md) |
| Preparar um job remoto | [Operação da plataforma](platform_guide.md) |
| Escolher uma abordagem de aprendizagem | [Conclusões das runs longas](learning-experience.md) |

Para preparar os dados de entrada, consultar a [seleção de datasets Portugal](portugal_evaluation_protocol_v2.md) e os [perfis de observações entity](entity_encoding_profiles_v1_pt.md).

## Implementações especializadas

- [TI-MARL](../algorithms/ti_marl/README.md) descreve entidades, ações, mudanças de população e persistência.
- [Transformer MATD3](transformer_matd3.md) descreve atores por edifício, crítico central e replay. As [decisões de arquitetura](adr/README.md) explicam as alternativas consideradas.
- [Transformer PPO](transformer_ppo_spec.md) e [contrato entity partilhado](transformer_entity_controller.md) descrevem tokenização e topologia dinâmica.
- [Coordenação hierárquica](community_coordinator_design.md) explica o sinal do coordenador e a sua aplicação pelas políticas locais.
- [Aprendizagem offline](offline_rl/README.md) cobre a preparação dos dados, o treino e a avaliação.

A documentação versionada descreve procedimentos reproduzíveis, contratos e decisões que influenciam trabalhos futuros. Configurações resolvidas, logs, checkpoints e tabelas de uma execução pertencem aos artefactos desse job.
