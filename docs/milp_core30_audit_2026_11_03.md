# Auditoria MILP no Core30 — 03/11/2026

Seguimento: o nome `MILP` identifica agora o [controlador de 24 horas com resolução em cada passo](milp_mpc_core30_2026_11_03.md). Esta página conserva a auditoria histórica de planos offline e os respetivos números.

Esta revisão parte dos baselines fechados em `7d76d90`: BAU com limite do
utilizador, Basic com bateria em autoconsumo, Smart e Community. Não altera
essas políticas, o Simulator, os datasets ou os pressupostos do mercado.

## O que existe

| Implementação | O que decide | Objetivo e utilização |
|---|---|---|
| `total_energy_milp.py`, `settlement=community` | Baterias, EVs e ciclos adiáveis das 30 casas | Minimizar custo conjunto com conhecimento de toda a janela. Referência principal de otimização comunitária. |
| O mesmo modelo, `settlement=individual` | Os mesmos equipamentos, resolvidos por casa ou conjuntamente | Minimizar compras isoladas por casa. O plano pode ser executado com o mercado ativo, mas o objetivo individual e a fatura comunitária são grandezas diferentes. |
| `perfect_foresight_milp.py` / `individual_building_milp.py` | Só baterias; restante consumo vem de uma execução anterior | Diagnóstico condicionado ao serviço de origem. Não é um ótimo de todos os consumos. |
| `scorecard_battery_milp.py` | Só baterias | Picos/rampas sob um teto de custo. Objetivo diferente do MILP de custo. |
| `total_home_milp.py` | Uma casa, com EVs e ciclos | Implementação anterior e separada; conservar para reprodução histórica, preferir o modelo total-energy para novas referências Core30. |

Não há treino de pesos. O processo lê as séries futuras, constrói um problema
linear com variáveis inteiras, resolve-o com SciPy/HiGHS e produz um plano.
`TotalOracleReplayPolicy` executa esse plano pelo ciclo comum do Algorithms,
sem professor RBC nem correção de segurança do agente. O Simulator aplica a
física efetiva. O export contém o plano JSON e o manifesto comum de investigação;
não é uma política online para uma janela nova nem um modelo ONNX.

O percurso atual ainda tem três comandos: construir o problema, resolver e
executar o replay. Não existe uma otimização automática em `prepare_training`
para estas classes. A proposta anual de horizonte deslizante nos documentos de
julho é um plano de trabalho, não uma capacidade certificada por esta auditoria.

## Formulação e limites

O modelo total-energy tem balanços de energia, limites de carga/descarga e SOC,
perdas passivas e eficiências lineares, exclusão de carga e descarga simultâneas,
potência mínima dos carregadores, alvos EV, escolhas de arranque de ciclos e
limites elétricos totais e por fase. Primeiro minimiza défices de serviço EV;
só depois minimiza custo, preservando esse resultado. Um limite de tempo não
é uma prova de ótimo. O resultado guarda o estado de cada fase e os limites do
solver. A relaxação contínua fornece um limite do modelo declarado, não uma
prova de ótimo global do Simulator.

A extração usa aproximações conservadoras de capacidade/eficiência e uma reserva
elétrica de 0,1 kW. Os limites físicos da bateria do modelo não são os limiares
estratégicos de 20–90% dos RBCs. Há diferenças de curvas de potência, perdas,
precisão numérica e serviço nas fronteiras; por isso, só o custo e o serviço
observados no replay entram na tabela de controladores.

Core30 Normal v3.3 tem preço comum não negativo, compra externa ao preço integral,
transações internas a 80%, exportação externa a zero e PV próprio sem custo
operacional. Os pagamentos internos cancelam no custo agregado; o objetivo
comunitário representa a importação líquida externa. Os 15 carregadores têm
potência de descarga zero: nenhum ensaio MILP beneficia de V2G.
Noutros datasets, a capacidade V2G tem de ser definida no problema matemático;
o parâmetro `allow_v2g` dos RBCs não reformula um plano MILP já calculado.

## Problemas encontrados e corrigidos

1. O leitor total-energy ignorava `loss_coefficient`, apesar de o modelo já
   suportar perdas. Passa a transmitir as perdas das baterias domésticas e EVs.
2. A exclusão de ciclos adiáveis só impedia dois arranques no mesmo passo. Passa
   a impedir sobreposição durante toda a duração do ciclo, mesmo em passos
   intermédios de potência zero. Aparelhos diferentes continuam independentes.
3. A decomposição individual copiava o inventário de todas as casas para cada
   subproblema. A junção falhava no Core30 com ações de outros proprietários.
   O inventário passa a ser filtrado por casa, e a junção conserva também os
   equipamentos sem pedidos. Dois primeiros ensaios falhados foram preservados.
4. O manifesto do plano assinalava V2G sempre que existissem sessões EV, mesmo
   com descarga impossível. Agora depende da potência de descarga disponível.

5. O modelo completo usava diretamente o wrapper público `scipy.optimize.milp`.
   Na versão instalada 1.15.1, depois de o HiGHS terminar, esse wrapper percorre
   colunas e converte marginais da base que o próprio `milp` descarta. O modelo
   completo passa a reutilizar `_milp_primal_only`, já usado nos oráculos de
   bateria: mesmo HiGHS, solução primal, objetivo e limites preservados, sem
   essa conversão. Em builds sem os bindings privados, conserva o fallback
   público. A dependência SciPy continua fixada.

Os testes de regressão cobrem perdas na extração, sobreposição de ciclos e
recomposição individual com equipamentos inativos na janela. Os testes dos
modelos e certificados passaram também com o adaptador reutilizado.

O mesmo problema comunitário de janeiro, confirmado por SHA-256, passou de
454,86 s para 17,17 s de resolução. O objetivo matemático permaneceu exatamente
173,88143455455233 €. Os replays antigo e novo deram o mesmo custo, pico e serviço.
A tentativa intermédia sem presolve não eliminou o custo de conversão; não se
atribui a melhoria a essa opção. O CLI passa a expô-la para diagnóstico.

## Comparação e fronteiras

Janelas `[35040,35712)` e `[52512,53184)`: 1–7 janeiro e 1–7 julho de 2024,
672 intervalos de 15 minutos cada, com resets separados. Simulator 2.0.0,
seed configurada 22 e seed de avaliação 6763. Reutilizam-se os exports dos
baselines apenas depois de conferir hashes, mercado, versão, janela e seed.

Há nove sessões EV já em curso no início e nove ainda abertas no fim de cada
semana. A extração não detetou SOC inicial de sessões posteriores sem origem
reproduzível, nem omitiu ciclos adiáveis. Os resets e as sessões truncadas ficam
explicitamente identificados; não se chama a estas janelas uma trajetória anual.

O protocolo original deixa sessões abertas sem alvo artificial e exige energia
terminal das baterias não inferior à inicial. Para isolar a vantagem de adiar
carga para depois da janela, acrescentou-se uma comparação com limites terminais:
para cada bateria e cada EV ainda ligado, energia final pelo menos igual à do
Community correspondente. Esses limites usam os exports já existentes do
Community e foram definidos antes de observar os novos custos MILP. A execução
real tem de confirmar os limites; a restrição linear, isoladamente, não basta.

Serviço mínimo viável ≥99,9%, precisão EV ≥80%, cobertura 100%, zero desconhecidos,
zero ciclos perdidos, zero eventos elétricos e energia de violação ≤1e−6 kWh,
SOC físico em [0,1] ±1e−6. Um custo baixo não compensa falhar estes critérios.

## Resultados finais: 14 dias em 2024

Valores oficiais depois de executar os planos. Redução de custo calculada como
`100 × (BAU − controlador) / BAU`, com `NormalTargetPolicy` como BAU.

| Controlador | Custo (€) | Redução face ao BAU | Pico máximo (kW) | Média dos picos diários (kW) |
|---|---:|---:|---:|---:|
| BAU | 270,08 | 0,00% | 59,59 | 32,14 |
| Basic | 261,30 | 3,25% | 58,50 | 28,77 |
| Smart | 255,11 | 5,54% | 55,37 | 28,38 |
| Community | 250,99 | 7,07% | 55,37 | 28,12 |
| MILP individual | 240,69 | 10,88% | 57,05 | 32,06 |
| MILP comunitário | 222,51 | 17,61% | 58,08 | 32,24 |

O MILP comunitário reduz o custo em **11,35% face ao Community**, e o individual
em **4,10%**. O pico máximo comunitário do MILP é **4,89% superior** ao Community,
e a média dos picos diários é **14,63% superior**. O objetivo otimiza custo,
não pico. A vantagem económica não implica dominância em todos os KPIs.

| Plano final | Custo no Simulator (€) | Pico (kW) | Resolver (s) | Solver: gap MIP |
|---|---:|---:|---:|---:|
| MILP individual — janeiro | 179,61 | 57,05 | 8,55 | 0,0065% |
| MILP comunitário — janeiro | 173,66 | 58,08 | 17,17 | 0,0263% |
| MILP individual — julho | 61,08 | 41,10 | 5,43 | 0,0058% |
| MILP comunitário — julho | 48,85 | 23,65 | 21,53 | 0,0918% |

Os quatro planos atingiram o estado `optimal` dentro da tolerância relativa
configurada de 0,1%, com défice EV linear zero. Cada controlador completou
**170 saídas EV e 153 ciclos**, com serviço mínimo viável 100%, zero violações
elétricas, zero ciclos perdidos e precisão EV de 94,12% no agregado. Todos os
limites terminais por equipamento foram confirmados no Simulator, sem usar
apenas a soma agregada como substituto dessa verificação.

A formulação individual custa 279,82 € no seu objetivo de compras isoladas,
mas 240,69 € depois da liquidação comunitária efetiva no replay. A formulação
comunitária soma 222,74 € no modelo linear e 222,51 € no Simulator. Esta última
diferença confirma que os limites do modelo não são uma prova de ótimo exato
da física não linear. Não há preços de desgaste e não se força energia final
igual a todos os baselines; o controlo terminal usa explicitamente o Community.

Os planos não foram afinados para inverter a ordenação dos controladores. Usam
os defaults físicos conservadores existentes, depois das correções descritas,
e os limites terminais definidos a partir dos exports do Community. Todas as
semanas e todos os baselines continuam com os mesmos preços e mercado.

## Execução, exportação e reprodução

O código dos quatro pares finais solve/replay é
`c639417e11fe9886dc48bb645188b21f614e4243`. Foram executados localmente, em CPU,
com dois processos em paralelo. Limite HiGHS de 180 s **por solve/fase**, gap
relativo de 0,001, presolve desligado no protocolo final e limite externo de
480 s para cada comando. Um limite HiGHS não limita o tempo total de conversão
Python nem o conjunto de todos os subproblemas individuais.

A exportação final e os manifestos foram produzidos pelo ciclo comum. A
verificação de inferência permanece `unverified: requires_specialised_runtime`,
com artefactos declarados de investigação e `require_deployable: false`.
Não se apresenta isso como paridade de inferência aprovada. O replay no Simulator
e a verificação de serviço/limites constituem a evidência operacional desta revisão.

Os novos templates `milp_community.yaml` e `milp_individual.yaml` usam o mesmo
contrato de jobs. São exemplos de **48 horas de janeiro de 2023**, como os restantes
templates Core30; não reproduzem automaticamente a tabela de 2024. Exemplo:

```bash
python scripts/build_citylearn_total_energy_problem.py \
  --schema datasets/rec_pt_2023_core_30_v2/evaluation/core_30_normal.json \
  --start-time-step 0 --end-time-step 192 --settlement community \
  --output-dir runs/milp-core30/community
python scripts/solve_total_energy_problem.py \
  --problem runs/milp-core30/community/problem.json \
  --output-dir runs/milp-core30/community --mode schedule \
  --time-limit-seconds 180 --mip-relative-gap 0.001 --no-presolve
python run_experiment.py \
  --config configs/templates/baselines/core30/milp_community.yaml
```

Para o individual, mudar `community` para `individual` nas três etapas e juntar
`--decompose-individual` ao solver. Validar sempre que o plano e o job usam a
mesma janela, dataset, passo e seed. Os templates desligam o BAU nativo do
Simulator; as reduções usam um job do BAU principal correspondente.

A reprodução exata da comparação de 2024 requer os limites terminais por ativo
do Community. Os problemas serializados, configurações completas e scripts estão
em `runs/milp-core30-20261103/final/`; os limites constam de
`problem.json -> metadata.terminal_energy_reference`. O JSON versionado
[core30_milp_comparison.json](validation/core30_milp_comparison.json) guarda
métricas, critérios, hashes dos problemas, versões, tempos e limites do solver.
O [registo da campanha](experiment_history/20261103_milp-core30-20261103.md)
preserva também os ensaios falhados e intermédios.

Verificação: **2046 testes gerais passaram**, 23 omissões justificadas;
**71 testes de jobs/exportação passaram**. Ruff, formatação, mypy e validação
estrutural dos novos templates passaram. Os quatro jobs finais constituem a
validação efetiva dos planos semanais no Core30.

## Limitações das outras implementações

O replay de serviço fixado volta a chamar o RBC e pode aplicar uma projeção de
segurança; portanto, serviço fixo no problema matemático não prova trajetória
idêntica no Simulator. Os modelos históricos só de bateria e total-home não
representam todas as perdas/fases como o total-energy atual. O total-home tem
extração de SOC nas fronteiras e regras de fases diferentes; não deve ser
promovido com base nos resultados do total-energy.

O replay estrito valida unidades, identidades, passo temporal e disponibilidade
do plano, mas não confirma automaticamente toda a identidade de dataset/janela
na inicialização. A campanha faz essa verificação explicitamente. Consolidar os
adaptadores antigos e acrescentar esse controlo automático são ações separadas;
não são capacidades declaradas como concluídas nesta revisão.
