# Portugal — construção dos datasets

Catálogo corrente: `portugal-native-2023-2024-v3.1`, para o Simulator original
`978be68b0e0740deab15ecf536805f323b86fa57`. A tarefa atual é construir os dados;
avaliações de utilização, referências de controlo e experiências estão suspensas.

São nove configurações sobre quatro bases de 2023–2024, com intervalos de 15 minutos:
Core-4; Core-15; Core-30 Normal, Topologia, Safety, Robustness e Combinado;
Premium-100 Normal e All-in. DR só no Premium All-in. As janelas reservadas no
manifesto não acrescentam datasets. O inventário mantém-se. Duas visitas ao Café 028 foram prolongadas em 15 minutos para tornar viáveis os SOC pedidos; a viabilidade foi comprovada em janelas curtas no motor original, sem avaliações de algoritmos.

Todos usam a mesma série de compra à rede, `shared/pricing_btn_bi.parquet`,
referência BTN bi-horária indexada a OMIE 2023–2024. As séries guardam o preço completo.
A exportação à rede vale zero. O mercado está desligado nos schemas de dados;
para uma experiência, ativar `community_market.enabled` e configurar
`community_market.local_price_ratio_to_grid_import: 0.8` no schema respetivo.
A liquidação é a do simulador original, sem reequilíbrio externo ou alterações ao motor.

O manifesto em `algorithms/ti_marl/experiments/portugal_evaluation_v2_manifest.json`
espelha `Simulator/results/portugal_rec_native/manifest.json`. Os schemas estão
em `evaluation/` e os índices de cada família em `CURRENT_EVALUATION.json`.
A documentação corrente, as fontes, o atlas e a decisão tarifária encontram-se
em `Simulator/docs/portugal_rec_native/README.md`. A verificação desta revisão é
estática e não certifica desempenho de algoritmos ou integração do pipeline.

As limitações nativas continuam documentadas: SOC por sessão de EV, fases de
máquinas por instalação, backup agregado, possível visibilidade dos eventos de
health e DR com média da hora anterior. Quem sai deixa o cenário com os seus ativos.

O protocolo anterior e os resultados com tarifas individuais estão no arquivo
`/home/tiago/dev/Simulator-recovery-archive/2026-09-27-common-tariff/individual-v1`. Não representam a configuração corrente.
A implementação Algorithms, os seletores e as regras executáveis não foram
alterados nesta revisão. A integração e a avaliação serão tratadas quando
retomarmos explicitamente as experiências.

2024 é continuação cronológica: 70176 registos ao todo, sem reinício de membros/equipamentos nem repetição da topologia. Os inputs foram verificados estaticamente; não se executaram avaliações de controladores. Janelas isoladas de 2024 devem respeitar o estado acumulado. O fecho de 2023 foi preservado no Simulator, commit `393cd200`.

As histórias são geradas dos eventos por configuração. As previsões de preço no fim de 2024 usam um dia externo de apoio, mantendo o período simulado. A variabilidade foi quantificada estaticamente em `Simulator/docs/portugal_rec_native/Previsoes_e_historias.md`. A persistência diária não modela uma degradação progressiva da qualidade com o horizonte. O recorte nativo das previsões derivadas de consumo/PV na fronteira do episódio permanece; não houve alterações ao motor ou ao pipeline Algorithms.
