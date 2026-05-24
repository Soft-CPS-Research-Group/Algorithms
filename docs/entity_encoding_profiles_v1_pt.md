# Perfis de encoding entity para Simulator 1.0.2

Snapshot: 2026-05-24

O Simulator 1.0.2 expõe todos os bundles entity ativos nos datasets principais:

```text
entity_core_electrical
entity_community_operational
entity_forecasts_existing
entity_forecasts_derived
entity_temporal_derived
entity_action_feedback
```

No lado dos algoritmos ficam agora quatro leituras úteis:

```text
minmax_space:
  superfície completa normalizada pelo espaço do simulador.
  Bom para inspeção, RBCs e debugging. Pode ser demasiado larga para RL.

maddpg_v1:
  transforma a superfície completa com tempo cíclico, SOC normalizado,
  deadlines relativos e escalas físicas.
  Bom para compatibilidade e auditoria.

maddpg_v2_compact:
  perfil compacto anterior. Mantido para reproduzir experiências antigas.
  Não é o melhor default para Simulator 1.0.2 porque filtra várias features
  novas de deadline, capacidade factível e feedback.

maddpg_v3_operational:
  novo default para treino. Mantém o core compacto da v2 e adiciona as features
  1.0.2 mais úteis:
    EV min_required_action_normalized
    EV departure_feasibility_ratio / departure_energy_margin_kwh
    EV/BESS can_charge, can_discharge e available_*_action_normalized
    feedback da última ação aplicada/projetada
    must_start_now e estado operacional dos deferrables
    forecasts derivados do simulador

maddpg_v3_realtime:
  variante para cenários mais próximos de deploy. Mantém estado atual,
  capacidade factível e feedback, mas remove campos forecast_* derivados do
  futuro perfeito do simulador. Usa-se quando queremos evitar ensinar o agente
  com forecasts que não existiriam diretamente no mundo real.
```

Regra prática:

```text
RBC / auditoria:
  minmax_space ou raw observations.

Treino RL atual:
  maddpg_v3_operational.

Comparação real-world-safe:
  maddpg_v3_realtime.

Reprodução antiga:
  maddpg_v2_compact.
```

Nota de performance:

```text
O perfil `maddpg_v3_operational` agora usa dois planos compilados:
  1. source plan do entity layout, agrupado por tabela/linha/coluna;
  2. plano de encoding MADDPG, com bloco minmax vetorizado.

No smoke local 2022 all-plus-EVs, 17 agentes:
  model_observation_encoding caiu de ~79 ms/step para ~6.4 ms/step.
  entity_layout caiu de ~10.5 ms/step para ~1.1 ms/step.

Quando o agente nao precisa de raw observations, o wrapper usa a via direta
`entity payload -> maddpg_v3`; nesse modo:
  model_observation_encoding no update fica ~0.0 ms;
  a fase entity-model-observation fica ~7.2 ms;
  a mediana perfilada total fica ~20.7 ms/step.
```

Nota sobre `time_step`:

```text
O raw time_step deve ficar no meta e não como feature principal do agente.
Para controlo real, preferimos calendar features, tempo cíclico, horas até
deadline e forecasts. Um índice absoluto de episódio deixa o agente memorizar
o dataset em vez de aprender estado operacional transferível.
```
