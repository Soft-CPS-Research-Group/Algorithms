# Escolher o encoding de observações entity

Definir `simulator.entity_encoding.normalization` de acordo com as observações que o agente terá durante a utilização. Confirmar as colunas presentes no dataset e manter a mesma ordem, escala e informação no treino e na inferência.

| Perfil | Critério de escolha |
|---|---|
| `minmax_space` | Normaliza pelos limites declarados do espaço. Útil para inspecionar a superfície completa. Agentes que pedem observações brutas seguem o seu contrato próprio. |
| `maddpg_v1` | Preserva a representação com tempo cíclico, SOC, prazos relativos e escalas físicas usada nos modelos desta família. |
| `maddpg_v2_compact` | Reproduz o perfil compacto de políticas existentes com essa dimensão de entrada. |
| `maddpg_v3_operational` | Inclui disponibilidade e margem de serviço dos equipamentos, feedback da ação aplicada e previsões operacionais disponíveis. |
| `maddpg_v3_realtime` | Retira os campos de previsão derivados do futuro perfeito. Usar quando o controlador só terá acesso a estado e medições locais. |

Para comparar duas políticas, declarar as previsões que cada uma recebe. Usar calendário e tempo até ao prazo para representar a tarefa. O índice absoluto do episódio pode permitir ao modelo memorizar o dataset. Ver o [contrato da interface](entity_interface_playbook_pt.md).
