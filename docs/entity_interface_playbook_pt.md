# Entity Interface Playbook (PT)

Guia rápido para a equipa sobre a interface nova `entity` e como ela entra no
pipeline de treino neste repositório.

## 1) Mensagem Pronta para o Grupo (Gonçalo e Pedro)

Texto pronto para enviar:

> Pessoal, temos agora duas interfaces no simulador: `flat` (legada) e
> `entity` (nova).  
>  
> Regra importante: se o cenário for `topology_mode=dynamic`, o simulador exige
> `interface=entity`.  
>  
> No `Algorithms`, os agentes **não** falam diretamente com `tables/edges`. O
> wrapper (`utils/wrapper_citylearn.py`) recebe payload entity, converte para
> vetores por agente, chama o algoritmo, e depois converte as ações do algoritmo
> de volta para payload entity antes do `env.step(...)`.  
>  
> Quando a topologia muda durante a simulação, o wrapper deteta
> `meta.topology_version`, reconstrói o layout e reanexa metadata para o agente
> (incluindo `entity_specs`).  
>  
> Hoje: `MADDPG` funciona com `entity` em topologia estática; em dinâmica, existe
> guardrail e é preciso usar um agente preparado para topologia variável
> (ex.: `RuleBasedPolicy`).

## 2) Onde é Chamado no Código

### Entrada de configuração e criação de env

- `run_experiment.py` lê:
  - `simulator.interface`
  - `simulator.topology_mode`
- Cria `CityLearnEnv` com esses parâmetros.
- Referência: `run_experiment.py` (bloco de `env_kwargs`).

### Adaptação entity no wrapper

- `Wrapper_CityLearn.__init__`:
  - deteta modo `entity` (`_entity_interface_mode`)
  - inicializa `EntityContractAdapter` se necessário
- `_initialize_entity_agent_state(...)`:
  - faz reset inicial e constrói primeiro layout entity
- `_apply_entity_layout(...)`:
  - converte `tables/edges/meta` -> vetores por agente
  - reconstrói layout quando muda `topology_version`
- `_to_env_actions(...)`:
  - converte ações do agente -> `{"tables": {"building": ..., "charger": ...}}`

### Metadados para o agente

- `_attach_model_environment_metadata(...)` envia ao agente:
  - `interface`
  - `topology_mode`
  - `entity_specs` (quando `interface=entity`)

## 3) Como o Agente Deve Lidar

Para os algoritmos em `algorithms/agents/*`:

- Continuação da API do agente:
  - `predict(observations, deterministic)`
  - `update(observations, actions, rewards, next_observations, ...)`
- Em `entity`, o agente recebe vetores por agente já preparados pelo wrapper.
- Se precisares de nomes/features/ids do contrato entity:
  - usa `attach_environment(..., metadata=...)`
  - ler `metadata["entity_specs"]`

## 4) Quando Usar `flat` vs `entity`

- Usa `flat` quando:
  - cenário estático
  - baseline/compatibilidade legada
- Usa `entity` quando:
  - queres features ricas por entidade (district/building/charger/storage/pv/ev)
  - queres preparar modelos para grafos/hierarquia
  - precisas de topologia dinâmica

## 5) Template Recomendado para Dynamic

Template pronto:

- `configs/templates/rule_based_entity_dynamic_local.yaml`

Exemplo de execução:

```bash
python run_experiment.py \
  --config configs/templates/rule_based_entity_dynamic_local.yaml \
  --job_id entity-dynamic-smoke
```

## 6) Checklist de Implementação para Alunos

1. Confirmar no YAML:
   - `simulator.interface: entity`
   - `simulator.topology_mode: dynamic` (se cenário dinâmico)
2. No agente, usar `attach_environment(...)` para ler `metadata["entity_specs"]`.
3. Não assumir dimensão fixa para sempre em topologia dinâmica.
4. Em troubleshooting, validar:
   - `topology_version` a mudar
   - `action_dimension` e `observation_dimension` no wrapper.
