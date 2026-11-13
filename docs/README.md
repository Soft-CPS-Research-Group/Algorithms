# Algorithms

**Português** · [English](../README.md)

O Algorithms treina e avalia controladores para comunidades de energia. Para correr um job, escolhes um dataset, uma janela de simulação e uma política. O mesmo programa permite avaliar uma regra de carregamento ou treinar uma política de aprendizagem por reforço com vários agentes. Há também um oráculo offline para estimar o menor custo de eletricidade que a comunidade pode atingir numa semana.

## Como funciona o tronco comum

1. **Configurar a experiência.** O ficheiro YAML define a janela, a seed, os episódios de treino e avaliação e os resultados a exportar. O campo `simulator.dataset_path` aponta para o schema do simulador. O schema descreve os edifícios e equipamentos e identifica as séries de consumo, produção solar e preços.
2. **Correr a simulação.** O programa de execução abre o dataset com o SoftCPS REC Simulator (`softcpsrecsimulator==2.1.0`). Em cada intervalo, o agente recebe observações e devolve ações, como a potência de carregamento que pede para um veículo elétrico. O simulador aplica os limites físicos e devolve as observações seguintes e as recompensas.
3. **Treinar e avaliar.** Durante o treino, o programa passa as transições ao agente com a frequência configurada. Cada agente implementa a sua regra de aprendizagem. A avaliação determinística final produz os KPIs usados nas comparações. O programa guarda os resultados e, conforme a configuração, um checkpoint ou um pacote com o modelo.

**Trocar de controlador.** Em `pipeline`, cada etapa indica um `algorithm` registado e os seus parâmetros. Por exemplo, os templates Basic e Smart do Core30 usam o mesmo simulador e dataset. Para comparar as políticas, usa a etapa do template correspondente e mantém a mesma janela, seed e configuração de avaliação. O comando `--dry-run` verifica se o agente é compatível com a interface e a topologia escolhidas. [Basic](../configs/templates/baselines/core30/basic.yaml) · [Smart](../configs/templates/baselines/core30/smart.yaml)

**Acrescentar um agente.** Cria uma subclasse de `BaseAgent` com os métodos de decisão, aprendizagem e exportação. Regista a implementação, o esquema dos parâmetros e as capacidades suportadas num `AgentSpec` em `algorithms/catalogue.py`. A partir daí, o agente usa o mesmo ciclo de simulação e a mesma estrutura de resultados dos restantes. [Contrato do agente (EN)](runtime/agents.md)

## Correr o primeiro job

Na raiz de um clone, com Python 3.10 em Linux x86-64, executa os comandos abaixo. O exemplo usa dois edifícios de um dataset de demonstração incluído no repositório. A instalação inclui as dependências para CPU e para exportação.

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements/lock-cpu-py310-linux-x86_64.txt
python -m pip install '.[export]'

config=configs/templates/experimental/transformer_maddpg_smoke.yaml
algorithms-job --config "$config" --validate
algorithms-job --config "$config" --dry-run
algorithms-job --config "$config" --profile smoke --job_id first-job
```

O primeiro comando `algorithms-job` valida o YAML. O segundo carrega e inicializa o simulador. O último corre um episódio curto de treino e outro de avaliação determinística. Abre `runs/jobs/first-job/results/result.json` e procura `"status": "completed"`. Na mesma pasta do job, `config.resolved.yaml` guarda a configuração efetiva, `results/kpis.json` contém as métricas da avaliação e `bundle/artifact_manifest.json` descreve o modelo exportado. Para correr no Core30, obtém o dataset em separado e indica o caminho do respetivo schema no YAML. [Instalação noutras plataformas (EN)](runtime/installation.md) · [Seleção dos datasets Portugal](portugal_evaluation_protocol_v2.md)

## O que existe e o que falta

- **Controladores.** O catálogo inclui MADDPG, MATD3, agentes Transformer, TI-MARL, coordenadores hierárquicos e agentes com aprendizagem offline. `algorithms-job --list-agents` mostra os nomes registados e as capacidades de cada um. [Catálogo (EN)](agent_catalogue.md)
- **Comparação.** As referências Core30 são os dois BAU (carregar até ao alvo do utilizador ou até 100%), Basic, Smart, Community e o oráculo semanal de custo. A poupança é calculada face ao BAU com limite do utilizador. Falta a comparação de longa duração dos candidatos MARL com este conjunto. [Referências Core30](baseline_comparison_core30.md) · [Critérios de avaliação](community_optimization_success_scorecard_pt.md)
- **Inferência.** O job pode exportar a política avaliada. As políticas neuronais suportadas usam ONNX e as políticas por regras suportadas usam JSON executável. Confirma a elegibilidade para inferência no manifesto do pacote. O TI-MARL e os pipelines completos de coordenação ainda precisam de adaptadores especializados. [Formatos e suporte (EN)](inference_bundle.md#delivery-formats)

## Guias por tarefa

| Quero… | Onde ir |
|---|---|
| Configurar episódios, seeds ou retomar treino | [Configuração](runtime/configuration.md) e [checkpoints e retoma](runtime/reliability.md) |
| Resolver e reproduzir o plano semanal | [Procedimento do oráculo](oracles/total_energy_milp.md) |
| Entregar um modelo ou preparar um job remoto | [Contrato de exportação](runtime/exports.md) e [plataforma](platform_guide.md) |
| Escolher uma abordagem de aprendizagem | [Conclusões das runs longas](learning-experience.md) |
| Trabalhar com Transformers ou TI-MARL | [Transformer PPO](transformer_ppo_spec.md), [Transformer MATD3](transformer_matd3.md) e [TI-MARL](../algorithms/ti_marl/README.md) |
| Trabalhar com outras famílias | [Coordenação hierárquica](community_coordinator_design.md) e [aprendizagem offline](offline_rl/README.md) |
