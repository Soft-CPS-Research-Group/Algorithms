# Plano de Implementação: Offline Reinforcement Learning para Otimização de Carregamento de EVs

## Problema

Desenvolver um pipeline completo de Offline Reinforcement Learning (ORL) para otimização de carregamento de veículos elétricos em comunidades de energia, partindo de dados gerados por políticas existentes (RBC, MADDPG), com foco em:

1. Treino com dados históricos (sem interação online)
2. Generalização entre diferentes tipos de participantes e comunidades
3. Aumento de dataset via data augmentation

## Abordagem

Extensão progressiva do repositório existente com:

- Nova infraestrutura de datasets offline (formato HDF5, compatível com d3rlpy)
- Integração da biblioteca **d3rlpy** para algoritmos de ORL
- Pipeline de feature engineering e seleção de features
- Framework de avaliação cross-domain
- Módulo de data augmentation para enriquecimento do dataset

---

## Tarefas

### Fase 1: Fundação e Baseline

#### T1.1 - Infraestrutura de Coleta de Dados Offline

**Relevância**: O ORL requer datasets de transições (s, a, r, s', done) pré-coletadas. Sem esta infraestrutura, não há como treinar modelos offline.

**Contexto**: O repo atual executa treino online. Precisamos de um collector que guarde as transições durante episódios de simulação com políticas existentes.

**Key Results**:

- Collector funcional integrado no `Wrapper_CityLearn`
- Formato HDF5 compatível com d3rlpy
- Scripts de validação de datasets

**Entregáveis**:

- `utils/offline_data_collector.py` - Módulo de coleta de transições
- `scripts/collect_offline_data.py` - Script CLI para coleta massiva
- Schema de dataset documentado em `docs/offline_dataset_format.md`

---

#### T1.2 - Modelo Baseline de Offline RL (Behavior Cloning)

**Relevância**: Antes de algoritmos sofisticados, um baseline simples (BC) permite validar o pipeline e estabelecer uma referência de comparação.

**Contexto**: Behavior Cloning imita diretamente a política do dataset sem correções de distribuição. É o ponto de partida mais simples.

**Key Results**:

- Agente BC funcional registado no registry
- Treino concluído com dados do RBC
- Métricas baseline documentadas

**Entregáveis**:

- `algorithms/agents/bc_agent.py` - Agente de Behavior Cloning
- Registo em `algorithms/registry.py`
- Config template: `configs/templates/bc_local.yaml`
- Relatório de métricas baseline

---

#### T1.3 - Coleta de Dataset com RBC

**Relevância**: Criar o primeiro dataset offline para treino e validação do pipeline.

**Contexto**: O `RuleBasedPolicy` existente é uma heurística de carregamento baseada em PV. Os dados coletados serão sub-ótimos mas permitem testar o pipeline.

**Key Results**:

- Dataset HDF5 com ~100k transições (múltiplos episódios)
- Validação de integridade do dataset
- Estatísticas descritivas documentadas

**Entregáveis**:

- `datasets/offline/rbc_baseline_v1.hdf5`
- Script de validação executado
- Relatório de estatísticas do dataset

---

#### T1.4 - Avaliação do Modelo Baseline

**Relevância**: Estabelecer métricas de referência e identificar oportunidades de melhoria.

**Contexto**: Avaliar o BC treinado com dados RBC contra o próprio RBC e contra métricas de eficiência energética (KPIs do CityLearn).

**Key Results**:

- Framework de avaliação offline (OPE - Off-Policy Evaluation)
- Comparação BC vs RBC em termos de reward cumulativo
- Identificação de gaps de performance

**Entregáveis**:

- `utils/evaluation/offline_evaluator.py` - Framework OPE
- Resultados comparativos documentados
- Lista de pontos de melhoria identificados

---

### Fase 2: Feature Engineering e Dataset Otimizado

#### T2.1 - Análise Exploratória de Features

**Relevância**: Identificar features cruciais permite reduzir dimensionalidade e melhorar generalização.

**Contexto**: O CityLearn fornece muitas observações (SoC, preços, geração PV, hora, etc.). Nem todas são igualmente relevantes para a decisão de carregamento.

**Key Results**:

- Correlação features-reward analisada
- Features redundantes identificadas
- Importância de features quantificada

**Entregáveis**:

- Notebook de análise: `docs/notebooks/feature_analysis.ipynb`
- Relatório de features selecionadas
- Justificação técnica das escolhas

---

#### T2.2 - Seleção e Engenharia de Features

**Relevância**: Definir o formato final do observation space para treino de modelos futuros.

**Contexto**: Com base na análise, criar um encoder/preprocessor otimizado que transforme observações raw em features relevantes.

**Key Results**:

- Novo encoder definido em `configs/encoders/`
- Testes de validação do encoder
- Documentação do schema de features

**Entregáveis**:

- `configs/encoders/optimized_features.json`
- Atualização de `utils/wrapper_citylearn.py` se necessário
- Documentação em `docs/feature_engineering.md`

---

#### T2.3 - Coleta de Dataset com Políticas Sólidas

**Relevância**: Dados de políticas de maior qualidade melhoram a performance do ORL.

**Contexto**: Usar o MADDPG pré-treinado (ou ensemble de políticas) para gerar um dataset mais diverso e de maior qualidade que o RBC.

**Key Results**:

- Dataset diverso com múltiplas políticas
- Maior variância de ações (melhor cobertura do espaço)
- Rotulagem de proveniência dos dados

**Entregáveis**:

- `datasets/offline/mixed_policies_v1.hdf5`
- Script de coleta multi-política
- Estatísticas comparativas com dataset RBC

---

### Fase 3: Algoritmos Avançados de Offline RL

#### T3.1 - Integração da Biblioteca d3rlpy

**Relevância**: d3rlpy fornece implementações robustas de IQL, CQL, TD3+BC e BCQ, evitando reimplementação.

**Contexto**: Criar adaptadores que permitam usar algoritmos d3rlpy dentro da arquitetura existente (BaseAgent, registry, export).

**Key Results**:

- Wrapper d3rlpy -> BaseAgent funcional
- Compatibilidade com pipeline de export (ONNX)
- Testes de integração passando

**Entregáveis**:

- `algorithms/agents/d3rlpy_wrapper.py` - Adaptador genérico
- Atualização de `requirements.txt` com d3rlpy
- Testes de integração em `tests/`

---

#### T3.2 - Implementação e Treino de IQL

**Relevância**: IQL é particularmente eficaz com dados sub-ótimos, evitando overestimation via expectile regression.

**Contexto**: Treinar IQL com o dataset optimizado (T2.3) e comparar com baseline.

**Key Results**:

- Modelo IQL treinado
- Melhoria mensurável sobre BC baseline
- Hiperparâmetros otimizados documentados

**Entregáveis**:

- Config: `configs/templates/iql_local.yaml`
- Modelo treinado e checkpoints
- Relatório de resultados

---

#### T3.3 - Implementação e Treino de CQL

**Relevância**: CQL penaliza Q-values de ações fora da distribuição, sendo conservativo e estável.

**Contexto**: Alternativa ao IQL para comparação de abordagens conservativas.

**Key Results**:

- Modelo CQL treinado
- Comparação CQL vs IQL
- Trade-offs documentados

**Entregáveis**:

- Config: `configs/templates/cql_local.yaml`
- Modelo treinado
- Análise comparativa

---

#### T3.4 - Implementação e Treino de TD3+BC

**Relevância**: Abordagem mais simples que adiciona termo de BC ao TD3, eficaz em muitos domínios.

**Contexto**: Benchmark adicional para validar se complexidade de IQL/CQL é necessária.

**Key Results**:

- Modelo TD3+BC treinado
- Comparação com IQL e CQL
- Análise de trade-off complexidade vs performance

**Entregáveis**:

- Config: `configs/templates/td3bc_local.yaml`
- Modelo treinado
- Análise comparativa final

---

#### T3.5 - Seleção do Melhor Algoritmo

**Relevância**: Consolidar resultados e selecionar o algoritmo a usar nas fases seguintes.

**Contexto**: Comparar IQL, CQL, TD3+BC em múltiplas métricas (reward, estabilidade, tempo de treino).

**Key Results**:

- Ranking de algoritmos por métrica
- Algoritmo selecionado com justificação
- Configuração final optimizada

**Entregáveis**:

- Relatório comparativo final
- Modelo selecionado pronto para avaliação cross-domain

---

### Fase 4: Avaliação Cross-Domain

#### T4.1 - Framework de Avaliação Multi-Cenário

**Relevância**: Medir generalização é crucial para validar utilidade prática do modelo.

**Contexto**: Testar o modelo em cenários não vistos durante treino:

- Mesmo tipo de edifício, mesma comunidade
- Tipo diferente, mesma comunidade
- Mesmo tipo, comunidade diferente

**Key Results**:

- Pipeline de avaliação automatizado
- Métricas por cenário documentadas
- Visualizações comparativas

**Entregáveis**:

- `utils/evaluation/cross_domain_evaluator.py`
- Script: `scripts/evaluate_cross_domain.py`
- Dashboard ou relatório de resultados

---

#### T4.2 - Avaliação Intra-Comunidade (Mesmo Tipo)

**Relevância**: Validar performance em condições próximas do treino.

**Contexto**: Testar com edifícios do mesmo tipo dentro da mesma comunidade de energia.

**Key Results**:

- Métricas de performance documentadas
- Comparação com RBC e MADDPG online
- Análise de variância entre edifícios

**Entregáveis**:

- Resultados tabulados
- Gráficos comparativos
- Observações qualitativas

---

#### T4.3 - Avaliação Intra-Comunidade (Tipo Diferente)

**Relevância**: Testar transferência entre perfis de consumo distintos.

**Contexto**: Avaliar se um modelo treinado em apartamentos funciona bem em escritórios (ou vice-versa).

**Key Results**:

- Gap de performance quantificado
- Identificação de features sensíveis ao tipo
- Recomendações de adaptação

**Entregáveis**:

- Resultados de transfer learning
- Análise de falhas
- Sugestões de melhoria

---

#### T4.4 - Avaliação Inter-Comunidade (Mesmo Tipo)

**Relevância**: Testar generalização geográfica/temporal.

**Contexto**: Avaliar modelo em comunidade de energia diferente mas com mesmo tipo de edifícios.

**Key Results**:

- Performance em nova comunidade medida
- Fatores de degradação identificados
- Viabilidade de deployment cross-community

**Entregáveis**:

- Resultados comparativos
- Análise de domain shift
- Conclusões de generalização

---

### Fase 5: Data Augmentation

#### T5.1 - Investigação de Técnicas de Data Augmentation

**Relevância**: Aumentar diversidade do dataset pode melhorar robustez e generalização.

**Contexto**: Técnicas simples como noise injection, temporal shifting, feature perturbation.

**Key Results**:

- 3-5 técnicas identificadas e documentadas
- Análise de aplicabilidade ao domínio EV
- Técnica(s) selecionada(s) para implementação

**Entregáveis**:

- Documento de revisão de técnicas
- Justificação da seleção
- Plano de implementação

---

#### T5.2 - Implementação de Data Augmentation

**Relevância**: Aplicar técnicas selecionadas para expandir o dataset.

**Contexto**: Criar módulo de augmentation que pode ser aplicado durante coleta ou pré-processamento.

**Key Results**:

- Módulo de augmentation funcional
- Dataset aumentado gerado
- Validação de que augmentation não introduz bias

**Entregáveis**:

- `utils/data_augmentation.py`
- `datasets/offline/augmented_v1.hdf5`
- Testes de validação

---

#### T5.3 - Treino com Dataset Aumentado

**Relevância**: Validar se data augmentation melhora performance.

**Contexto**: Re-treinar o melhor algoritmo (da Fase 3) com dataset aumentado.

**Key Results**:

- Modelo treinado com dados aumentados
- Comparação com modelo original
- Quantificação do benefício

**Entregáveis**:

- Modelo treinado
- Relatório comparativo
- Conclusões sobre eficácia

---

### Fase 6: Avaliação Final e Conclusões

#### T6.1 - Avaliação Completa do Modelo Aumentado

**Relevância**: Validar se augmentation melhora generalização cross-domain.

**Contexto**: Repetir avaliação cross-domain (T4.2-T4.4) com o modelo treinado em dados aumentados.

**Key Results**:

- Métricas comparativas (antes vs depois de augmentation)
- Melhoria de generalização quantificada
- Identificação de cenários onde augmentation ajuda/prejudica

**Entregáveis**:

- Resultados de avaliação completos
- Gráficos comparativos
- Análise detalhada

---

#### T6.2 - Consolidação de Resultados e Conclusões

**Relevância**: Sintetizar aprendizagens para a tese.

**Contexto**: Reunir todas as observações, métricas e insights num documento coeso.

**Key Results**:

- Sumário executivo de resultados
- Conclusões principais para a tese
- Trabalho futuro identificado

**Entregáveis**:

- Documento de conclusões (`docs/conclusions.md`)
- Tabelas e figuras para a tese
- Recomendações finais

---

## Dependências entre Tarefas

```
T1.1 ─┬─► T1.2 ─► T1.4
      │
      └─► T1.3 ─► T1.4
              │
              └─► T2.1 ─► T2.2 ─► T2.3
                                   │
                                   └─► T3.1 ─┬─► T3.2 ─┐
                                             ├─► T3.3 ─┼─► T3.5 ─► T4.1 ─┬─► T4.2
                                             └─► T3.4 ─┘              ├─► T4.3
                                                                      └─► T4.4
                                                                           │
                                   T5.1 ─► T5.2 ─► T5.3 ◄──────────────────┘
                                                    │
                                                    └─► T6.1 ─► T6.2
```

---

## Notas Técnicas

### Bibliotecas a Adicionar

```
d3rlpy>=2.0.0      # Algoritmos de Offline RL
h5py>=3.0.0        # Formato HDF5 para datasets
scikit-learn       # Feature selection (se não existir)
shap               # Interpretabilidade de features (opcional)
```

### Configuração de Ambiente

O dataset HDF5 deve seguir a estrutura d3rlpy:

```python
{
    "observations": np.array,      # (N, obs_dim)
    "actions": np.array,           # (N, act_dim)
    "rewards": np.array,           # (N,)
    "next_observations": np.array, # (N, obs_dim)
    "terminals": np.array,         # (N,) bool
}
```

### Métricas de Avaliação

- **Reward Cumulativo**: Soma de rewards por episódio
- **KPIs CityLearn**: Custo energético, emissões CO2, conforto
- **Off-Policy Evaluation**: FQE (Fitted Q-Evaluation) para estimativa offline

---

## Estimativa de Complexidade

| Fase | Tarefas | Complexidade |
|------|---------|--------------|
| 1 - Fundação | T1.1-T1.4 | Média |
| 2 - Features | T2.1-T2.3 | Média |
| 3 - Algoritmos | T3.1-T3.5 | Alta |
| 4 - Avaliação | T4.1-T4.4 | Média |
| 5 - Augmentation | T5.1-T5.3 | Média |
| 6 - Conclusões | T6.1-T6.2 | Baixa |

---

## Riscos e Mitigações

| Risco | Mitigação |
|-------|-----------|
| Dataset RBC muito sub-ótimo | Coletar dados com MADDPG cedo (T2.3) |
| d3rlpy incompatível com export ONNX | Implementar export custom no wrapper |
| Generalização cross-domain fraca | Focar em data augmentation e mixed datasets |
| Hiperparâmetros sensíveis | Grid search sistemático, usar defaults d3rlpy |

---

## Referências

- [d3rlpy Documentation](https://d3rlpy.readthedocs.io/)
- [IQL Paper](https://arxiv.org/abs/2110.06169) - Implicit Q-Learning
- [CQL Paper](https://arxiv.org/abs/2006.04779) - Conservative Q-Learning
- [TD3+BC Paper](https://arxiv.org/abs/2106.06860) - A Minimalist Approach to Offline RL
