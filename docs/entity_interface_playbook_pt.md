# Usar a interface entity

Escolher `simulator.interface: entity` quando o dataset contém edifícios, carregadores ou equipamentos que entram e saem durante a simulação. Para esses cenários, usar `simulator.topology_mode: dynamic` e um agente que declare suporte a mudanças de topologia no [catálogo](agent_catalogue.md).

O simulador entrega `tables`, `edges` e `meta`. O [adaptador](../utils/entity_adapter.py) constrói as observações por edifício e converte as ações do agente de volta para as tabelas do equipamento correspondente. O [controlador de simulação](../algorithms/runtime/controller.py) observa `meta.topology_version`, reconstrói o layout e fornece `entity_specs` ao agente através de `attach_environment`.

Ao implementar um agente, guardar os identificadores dos membros e equipamentos e verificar como uma mudança de layout afeta ações pendentes, replay e estado temporal. Com MADDPG de dimensão fixa, usar topologia estática. Para testar uma ligação dinâmica a partir da raiz do repositório:

```bash
algorithms-job --config configs/templates/dynamic/rbc_basic_entity_dynamic_local.yaml --dry-run
```

Um dry run valida a ligação inicial. Para verificar reconfiguração, escolher uma janela que inclua um evento de topologia e confirmar o destino das ações antes e depois desse evento. Os [perfis de encoding](entity_encoding_profiles_v1_pt.md) definem a informação visível ao controlador.
