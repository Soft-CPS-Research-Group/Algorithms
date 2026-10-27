# Stage 3 delivery: reliable common execution

The 29 stage-3 actions from the master plan are implemented with the supported
scope and limitations in [reliability.md](reliability.md). Branch consolidation and
common interfaces remain the foundation; agents retain their numerical methods.

Local acceptance: **1,799 unit/integration tests passed; 18 explicit skips**
(4 unavailable CUDA and 14 historical external assets). The complete-job suite
contains **27 passing real-job tests**, including seven representative families,
recovered/hierarchical and offline jobs, four independent-process seed/repeat/resume
comparisons, cancellation/retries/export failure, and three dynamic-topology jobs.
Lint, formatting, syntax and five typed boundary modules pass. A built wheel is
validated outside the checkout in configuration-only and CPU training environments;
MADDPG and NormalPolicy finish with Simulator 1.8.3 and no installed/imported MLflow.
Both isolated environments pass `pip check`.

The benchmark retains MLflow as optional code, normally off. CPU episode resume
is supported for four audited families; other serializers explicitly remain
initializers. Jetson r35 publication is paused to preserve its previous working
image until Python/runtime migration is validated. GPU/ARM equivalence, scientific
performance, full manifest/inference delivery and dependency-security remediation
remain separate planned gates.

## Action-to-evidence map

| Action | Implemented requirement | Evidence |
|---|---|---|
| A02.01 | Criar matriz Python/PyTorch/CUDA/Simulator para CPU e hardware escolhido; decidir suporte Jetson/Union sem trocar versões às cegas. | [Inputs, actions and platform decisions](reliability.md#data-and-actions); `tests/test_runtime_reliability.py`, wrapper rollback tests and dynamic real jobs |
| A02.03 | Verificar schema, nomes e ordem, unidades, bounds, datas, timezone, resolução, valores não finitos, lacunas e identidades dos dados. | [Inputs, actions and platform decisions](reliability.md#data-and-actions); `tests/test_runtime_reliability.py`, wrapper rollback tests and dynamic real jobs |
| A02.04 | Inventariar e validar modelos auxiliares, ficheiros externos e respetivos hashes; não substituir dinâmica ausente por um modelo inventado. | [Inputs, actions and platform decisions](reliability.md#data-and-actions); `tests/test_runtime_reliability.py`, wrapper rollback tests and dynamic real jobs |
| A02.05 | Gravar integridade dos ficheiros realmente consumidos, schema/overlays resolvidos, janela e mapping; reutilizar validadores já existentes. | [Inputs, actions and platform decisions](reliability.md#data-and-actions); `tests/test_runtime_reliability.py`, wrapper rollback tests and dynamic real jobs |
| A02.06 | Testar topologia estática e dinâmica, adição/remoção de entidades e rollback, apenas nas combinações que declaram suporte. | [Inputs, actions and platform decisions](reliability.md#data-and-actions); `tests/test_runtime_reliability.py`, wrapper rollback tests and dynamic real jobs |
| A02.07 | Aplicar política explícita para ações inválidas: recusar NaN/dimensões erradas no perfil normal; fallback/clipping autorizado fica registado com ações propostas e aplicadas. | [Inputs, actions and platform decisions](reliability.md#data-and-actions); `tests/test_runtime_reliability.py`, wrapper rollback tests and dynamic real jobs |
| A03.01 | Substituir a ambiguidade de episodes/deterministic_finish por N episódios de treino e avaliação adicional; migrar o legado sem perder treino. | [Training/evaluation and seeds](reliability.md#training-evaluation-and-randomness); four-family independent processes, evaluation-state and scheduling regressions |
| A03.02 | Centralizar seeds Python/NumPy/Torch antes de construir agentes; separar seeds do simulador, exploração, replay e subagentes. | [Training/evaluation and seeds](reliability.md#training-evaluation-and-randomness); four-family independent processes, evaluation-state and scheduling regressions |
| A03.03 | Corrigir e testar a inicialização TPPO/TMATD3 que não aplica training.seed no percurso auditado; registar determinismo e tolerâncias por runtime. | [Training/evaluation and seeds](reliability.md#training-evaluation-and-randomness); four-family independent processes, evaluation-state and scheduling regressions |
| A03.04 | Garantir avaliação sem aprendizagem, atualização de normalizadores aprendidos, exploração ou contaminação de buffers; permitir apenas estado operacional necessário. | [Training/evaluation and seeds](reliability.md#training-evaluation-and-randomness); four-family independent processes, evaluation-state and scheduling regressions |
| A03.05 | Definir política avaliada: final por defeito ou best validado; guardar checkpoint/hash, janela, seed e topologia associados aos KPIs. | [Training/evaluation and seeds](reliability.md#training-evaluation-and-randomness); four-family independent processes, evaluation-state and scheduling regressions |
| A03.06 | Verificar scheduling on-policy/off-policy, warm-up, contadores, terminal/truncation, reset de estado temporal e interação com topologia em cada família. | [Training/evaluation and seeds](reliability.md#training-evaluation-and-randomness); four-family independent processes, evaluation-state and scheduling regressions |
| A03.07 | Testar reproducibilidade em processos independentes e retoma; distinguir repetibilidade no mesmo runtime de tolerância numérica entre CPU/GPU. | [Training/evaluation and seeds](reliability.md#training-evaluation-and-randomness); four-family independent processes, evaluation-state and scheduling regressions |
| A05.01 | Criar índice uniforme de checkpoints com caminho relativo, versão, agente/stage, step, hash e capacidade de retoma, sem impor um único binário. | [Checkpoint/lifecycle contract](reliability.md#checkpoints-and-retries), [state inventory](checkpoint_inventory.md); manager/hash/partial/disk tests, SIGTERM/retry/export failures and exact boundary resume |
| A05.02 | Corrigir descoberta TPPO e separar save_final da cadência periódica; testar cadência desligada com final ligado. | [Checkpoint/lifecycle contract](reliability.md#checkpoints-and-retries), [state inventory](checkpoint_inventory.md); manager/hash/partial/disk tests, SIGTERM/retry/export failures and exact boundary resume |
| A05.03 | Tornar explícita a política para dois pedidos de checkpoint no mesmo step; preservar escrita atómica e marcador de completude existentes. | [Checkpoint/lifecycle contract](reliability.md#checkpoints-and-retries), [state inventory](checkpoint_inventory.md); manager/hash/partial/disk tests, SIGTERM/retry/export failures and exact boundary resume |
| A05.04 | Inventariar por família pesos, otimizadores, RNG, normalizadores, exploração, buffers/replay, scheduler e topologia; testar round-trip. | [Checkpoint/lifecycle contract](reliability.md#checkpoints-and-retries), [state inventory](checkpoint_inventory.md); manager/hash/partial/disk tests, SIGTERM/retry/export failures and exact boundary resume |
| A05.05 | Declarar retoma por fronteira de episódio; suportar meio de episódio só quando estado do ambiente e RNG possam ser restaurados e testados. | [Checkpoint/lifecycle contract](reliability.md#checkpoints-and-retries), [state inventory](checkpoint_inventory.md); manager/hash/partial/disk tests, SIGTERM/retry/export failures and exact boundary resume |
| A05.06 | Cobrir erro de configuração, exceção, cancelamento/SIGTERM, export falhado, disco cheio e limpeza de recursos; publicar completed apenas após outputs pedidos validados. | [Checkpoint/lifecycle contract](reliability.md#checkpoints-and-retries), [state inventory](checkpoint_inventory.md); manager/hash/partial/disk tests, SIGTERM/retry/export failures and exact boundary resume |
| A05.07 | Tornar resultado/manifesto finais atómicos; indexar latest/best sem eliminar versões, monitorizar espaço e isolar tentativas/artefactos parciais. Conservar resultados, modelos, checkpoints e logs, conforme SOF-215. | [Checkpoint/lifecycle contract](reliability.md#checkpoints-and-retries), [state inventory](checkpoint_inventory.md); manager/hash/partial/disk tests, SIGTERM/retry/export failures and exact boundary resume |
| A06.01 | Desligar MLflow nos defaults coerentemente e garantir métricas locais, resultados e checkpoints sem ligação ao serviço. | [Tracking decision and measurements](reliability.md#tracking-and-measured-cost); [raw benchmark](tracking_benchmark.json), installed no-MLflow jobs and tracking-failure tests |
| A06.02 | Isolar dependências/imports MLflow como extra opcional e provar instalação/execução mínima sem o pacote. | [Tracking decision and measurements](reliability.md#tracking-and-measured-cost); [raw benchmark](tracking_benchmark.json), installed no-MLflow jobs and tracking-failure tests |
| A06.03 | Canalizar métricas dos agentes/coordenadores/exporters para uma interface comum de tracking, sem chamadas diretas dispersas. | [Tracking decision and measurements](reliability.md#tracking-and-measured-cost); [raw benchmark](tracking_benchmark.json), installed no-MLflow jobs and tracking-failure tests |
| A06.04 | Aplicar amostragem, minimal/curated e uploads de modelos/checkpoints globalmente; definir falha de rede com fallback local ou abort explícito, sem expiração automática de artefactos. | [Tracking decision and measurements](reliability.md#tracking-and-measured-cost); [raw benchmark](tracking_benchmark.json), installed no-MLflow jobs and tracking-failure tests |
| A06.05 | Comparar off/local/MLflow mínimo/completo em ambiente isolado, quando útil, com condições iguais; medir tempo, memória, CPU/GPU, bytes e chamadas sem reinstalar o serviço removido em produção. | [Tracking decision and measurements](reliability.md#tracking-and-measured-cost); [raw benchmark](tracking_benchmark.json), installed no-MLflow jobs and tracking-failure tests |
| A06.06 | Decidir manter só a integração de código opcional ou retirá-la com base no benefício medido; preservar a remoção do serviço em SOF-222 e documentar eventual migração. | [Tracking decision and measurements](reliability.md#tracking-and-measured-cost); [raw benchmark](tracking_benchmark.json), installed no-MLflow jobs and tracking-failure tests |
| A06.07 | Medir restantes custos: simulação, encoding, predict/update, traces, exports e checkpoints; ajustar limites por perfil com evidência. | [Tracking decision and measurements](reliability.md#tracking-and-measured-cost); [raw benchmark](tracking_benchmark.json), installed no-MLflow jobs and tracking-failure tests |
| A10.01 | Transformar falhas reproduzidas da auditoria em regressões de ponta a ponta pequenas: null exports, config ignorada, seeds, checkpoints, JSON e ações inválidas. | Common job profiles plus `tests/e2e/test_runtime_reliability.py`; full regression suite and installed reference/training jobs |
| A10.03 | Executar um treino curto real e uma política de referência com Simulator fixado, validando ficheiros e sem depender de MLflow. | Common job profiles plus `tests/e2e/test_runtime_reliability.py`; full regression suite and installed reference/training jobs |
