# Ponte do PPO local congelado para o Community Coordinator

- Campanha: `ppo_frozen_cc_bridge_validation_20260804`
- Arquivo: `2026-08-04T06:51:36+01:00`
- Dataset: `citylearn_three_phase_electrical_service_demo_15min_parquet`
- Janelas: `0:511`, `0:1023` e `0:2735`
- Código final da infraestrutura: `2134667`
- Leaf local: PPO residual-battery anual seed `789`, congelado
- Mercado comunitário: desligado nesta fase de isolamento do sinal

## Decisão

A integração `CC -> preço efetivo -> PPO local congelado` fica validada. O
controlo neutro com multiplicador `1.0` reproduziu exatamente a execução PPO
standalone: os 52 ficheiros de timeseries e o `exported_kpis.csv` foram
idênticos byte a byte na janela `0:1023`.

O PPO anual e o bundle entregue ao CC mantêm a decisão `ACCEPTED` documentada
em `2026-08-04_ppo_annual_three_seed_freeze.md`. Nenhum treino desta campanha
alterou os 17 atores locais: apenas o checkpoint do CC foi atualizado.

As receitas aprendidas atuais do `CCLevel1` ficam `REJECTED`. Todas mantiveram
17/17 gates locais tolerantes, mas nenhuma bateu o controlo neutro na janela de
quatro semanas. Não se justifica ainda escalar o treino do CC para o ano
inteiro.

## Superfície económica de quatro semanas

O comparador correspondente é o mesmo PPO seed `789` com multiplicador neutro
`1.0`, cujo custo foi EUR 2 007,9620.

### Resposta a multiplicadores constantes

| Multiplicador | Custo | Delta vs. neutro | Casas abaixo do neutro | Gates tolerantes |
|---:|---:|---:|---:|---:|
| 0,850 | EUR 2 004,2403 | EUR -3,7217 | 10/17 | 17/17 |
| 0,900 | **EUR 2 000,8431** | **EUR -7,1188** | 12/17 | 17/17 |
| 0,925 | EUR 2 000,9646 | EUR -6,9974 | 12/17 | 17/17 |
| 0,950 | EUR 2 002,6178 | EUR -5,3441 | 14/17 | 17/17 |
| 0,975 | EUR 2 004,6967 | EUR -3,2653 | 13/17 | 17/17 |
| 1,000 | EUR 2 007,9620 | EUR 0,0000 | controlo | 17/17 |
| 1,025 | EUR 2 010,6382 | EUR +2,6763 | 5/17 tolerantes | 17/17 |
| 1,050 | EUR 2 014,2457 | EUR +6,2838 | 4/17 tolerantes | 17/17 |

O sweep prova duas coisas. Primeiro, o adaptador de preço não é inerte: o PPO
responde de forma material ao sinal do CC. Segundo, existe nesta janela uma
zona útil perto de `0.90--0.925`; baixar indefinidamente não é a solução, pois
`0.85` já volta a piorar.

O ponto `0.90` é apenas um controlo diagnóstico. Melhora o custo comunitário,
mas piora cinco casas e não constitui uma política comunitária adaptativa.

### Receitas aprendidas

| Receita CC | Custo | Delta vs. neutro | Decisão |
|---|---:|---:|---|
| BC físico + PPO | EUR 2 036,2116 | EUR +28,2496 | REJECT |
| PPO cost-first, gama 0,95--1,05 | EUR 2 034,2041 | EUR +26,2422 | REJECT |
| PPO cost-only, gama 0,85--0,95 | EUR 2 026,2074 | EUR +18,2454 | REJECT |

Na última receita, a política determinística terminou com multiplicador médio
`0.899779`, mínimo `0.895064`, máximo `0.904206` e correlação preço-multiplicador
`+0.4519`. Apesar da média coincidir com o melhor controlo fixo, a modulação
temporal piorou o custo em EUR 25,3643 relativamente ao `0.90` constante. O
problema deixou de ser escolher a escala média: o padrão temporal aprendido é
prejudicial e as respostas dos 17 edifícios são heterogéneas.

Houve ainda um smoke treinável de dois episódios em `0:511`, também rejeitado:
EUR 353,5209 contra EUR 350,8228 do neutro correspondente.

## Contrato confirmado para o handoff

O leaf entregue continua a ser o composto:

`RBCSmartLocalPolicy + PPO residual battery + safety projector + price adapter`

- cada PPO recebe apenas o perfil `building_local_v1`;
- o CC recebe apenas `cc_level1`;
- o PPO não observa comunidade, outros edifícios ou a decisão interna do CC;
- o preço efetivo altera a observação do ator, mas não o preço real de
  settlement;
- um escalar é difundido pelos 17 atores e o pipeline já aceita também um vetor
  com um contexto por membro, permitindo preços individualizados no passo
  seguinte;
- carregar o checkpoint do PPO é independente de iniciar o CC de raiz;
- checkpoints de stages congelados não são duplicados no bundle do CC.

## Próximo gate técnico

Não repetir simplesmente mais épocas do mesmo `CCLevel1`. A evidência aponta
para duas alterações a avaliar numa janela curta antes de qualquer ano inteiro:

1. vetor de 17 multiplicadores ou outra parametrização que trate a resposta
   heterogénea das casas, mantendo os PPOs locais cegos à comunidade;
2. reward/advantage centrado num controlo contemporâneo, para isolar o efeito
   marginal da bateria do consumo exógeno e evitar que o gradiente aprenda uma
   calendarização errada.

Qualquer candidato novo deve primeiro superar EUR 2 007,9620, manter 17/17
gates tolerantes e depois ser comparado com EUR 2 000,8431 do controlo fixo
`0.90`. Só depois se escala para múltiplas seeds e ano inteiro.

## Evidência local ignorada pelo Git

- Auditoria consolidada:
  `runs/analysis/ppo_cc_bridge_four_week_consolidated_audit_20260804`
- Paridade neutra:
  `runs/jobs/ppo-seed789-fixed-neutral-cc-smoke-0-1023-c03cf52-r2` e
  `runs/jobs/ppo-seed789-standalone-parity-smoke-0-1023-c03cf52`
- Controlo neutro de quatro semanas:
  `runs/jobs/fixed-neutral-cc-frozen-ppo-seed789-control-0-2735-470d86d`
- Sweep fixo:
  `runs/jobs/fixed-price-*-frozen-ppo-seed789-0-2735-2134667`
- CC BC+PPO:
  `runs/jobs/cclevel1-bc-ppo-frozen-ppo-seed789-0-2735-s789-2134667`
- CC cost-first:
  `runs/jobs/cclevel1-cost-first-frozen-ppo-seed789-0-2735-s789-2134667`
- CC cost-only na gama medida:
  `runs/jobs/cclevel1-cost-only-measured-range-frozen-ppo-seed789-0-2735-s789-2134667`
