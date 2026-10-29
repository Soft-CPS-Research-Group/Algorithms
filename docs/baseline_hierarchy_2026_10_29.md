# Revisão Basic → Smart Local → Community — 29 outubro 2026

A inversão de custos tinha causas concretas: o Smart desativava por omissão a
carga da bateria por preço, adiava oportunidades baratas de carga dos EV e
esvaziava a bateria para consumos locais antes de avaliar o preço. O Community
podia acrescentar exportação da bateria para outros membros mesmo com o mercado
desligado. O limiar comunitário por omissão tinha a escala de um edifício.

O Basic conserva as suas regras. Smart Local passa a manter a mesma base de
serviço/preço e acrescenta captura solar, reserva e alívio de picos locais.
Community acrescenta partilha com o mercado efetivamente ativo e um limiar
agregado proporcional ao número de membros, salvo configuração explícita.
Sem mercado, reproduz a política local. Professores especializados com sinais
mantêm a implementação histórica; a sua revisão continua adiada.

## Comparação de todo o período

Dataset inalterado `citylearn_challenge_2022_phase_all_plus_evs`: 17 edifícios,
8 EV, 1 aparelho, 8 760 registos horários e 8 759 transições. Não há preços
negativos. Seeds 22/2022; uma avaliação determinística, exportação final, V2G
desligado e defaults dos três agentes. Não são os novos datasets portugueses.
O custo abaixo conserva a unidade reportada pelo simulador; a tarifa histórica
não foi recalibrada para uma moeda real.

| Mercado | Controlador | Custo reportado | Importação (kWh) | Pico agregado (kW) | Excesso médio de SOC (p.p.) |
|---|---|---:|---:|---:|---:|
| Desligado, como no schema original | Basic | 28 622,17 | 159 562,20 | 140,44 | 2,674 |
| Desligado | Smart Local | 27 177,97 | 141 543,53 | 89,83 | 2,728 |
| Desligado | Community | 27 177,97 | 141 543,53 | 89,83 | 2,728 |
| Ligado, preço interno 80% | Basic | 25 254,18 | 159 562,20 | 140,44 | 2,674 |
| Ligado, preço interno 80% | Smart Local | 23 668,54 | 141 543,53 | 89,83 | 2,728 |
| Ligado, preço interno 80% | Community | 22 690,84 | 135 605,47 | 89,83 | 2,682 |

Sem mercado, Smart Local reduz custo em 5,05% e pico em 36,03% face ao Basic.
Com mercado, a redução de custo é 6,28%; Community reduz mais 4,13% face ao
Smart Local, ou 10,15% face ao Basic. O Community não melhora adicionalmente o
pico neste dataset. A ativação do mercado é um cenário explícito aplicado aos
três agentes; não se alteraram preços ou ficheiros do dataset.

Em todos os seis jobs: 365 ciclos do aparelho concluídos, zero ciclos perdidos,
zero violações elétricas e 2 894 partidas EV. Nas 2 858 partidas em que o BAU
conseguiu cumprir o pedido, nenhum RBC ficou mais de 0,1 p.p. abaixo do alvo.
As outras 36 partidas continuam incluídas nas métricas globais. A referência
BAU observada não é uma prova de viabilidade conjunta de escalonamento.
A carga mínima e a proteção do último intervalo podem ultrapassar o alvo;
por isso o serviço não é numericamente idêntico entre políticas.

Os seis bundles passaram replay de 32 amostras. A suite geral passou 1 910 testes
(23 omissões explícitas); os contratos e jobs completos passaram 139 verificações
(11 testes lentos não selecionados). Ruff, formatação e mypy passaram.

## Rastreabilidade e reprodução

[Evidência estruturada](validation/baseline-hierarchy-2026-10-29.json) regista
fontes, mercado, serviço, KPIs, commits e versões originais dos manifestos.
A campanha foi lançada no código Algorithms `c5af041`; os manifestos finais
capturaram `1a08ec7` após documentação, tratamento de limiar explicitamente zero
(não usado nesta campanha) e atualização da dependência.

O código Simulator usado foi `053d57e3`, candidato BAU que ainda declarava 1.9.1.
A [comparação dos 57 módulos](validation/simulator-1.9.2-code-equivalence.json)
confirma equivalência com o pacote oficial 1.9.2, salvo a própria versão.
Depois da publicação, três jobs semanais com o pacote oficial instalado
reproduziram todos os 6 116 indicadores de cada execução equivalente com código
fonte, incluindo os bundles verificados. Os manifestos históricos não foram
reescritos para lhes atribuir retroativamente outra versão.

Usar `scripts/experiments/run_baseline_hierarchy_audit.py` e o
[contrato dos controladores](baseline_controllers.md). A campanha permite
comparar estes defaults neste dataset. Não garante dominância universal de
custo e pico, nem certifica as parametrizações dos templates históricos.
A etapa 5 e a inferência especializada TI-MARL/hierárquica permanecem adiadas.
