# CC causal price control V4

- Dataset: `citylearn_three_phase_electrical_service_demo_15min_parquet`
- Horizonte de evidência: ano completo, passos `0:35039`
- Mercado comunitário: ativo, preço local `0,8` do preço grid
- Objetivo primário: custo comunitário settled
- Estado: implementação e smokes concluídos; campanha anual aguarda imagem remota

## Motivo da correção CC-PPO

O primeiro CC-PPO anual custou EUR 21 075,18, contra EUR 20 850,00 do PPO
neutro (`+EUR 225,18`). A auditoria do caminho do sinal encontrou uma causa
arquitetural concreta: `local_price_conditioning_enabled` alterava a
observação codificada entregue ao ator PPO congelado, embora esse ator só
tivesse sido treinado com o preço nominal. Isso era inferência fora da
distribuição. Ao mesmo tempo, o `RBCSmartLocalPolicy` usado como base residual
não recebia o contexto do CC.

V4 inverte esse contrato:

1. o ator PPO recebe exatamente a observação original;
2. o multiplicador é entregue apenas ao novo
   `SignalAwareRBCSmartLocal`, que mantém a restrição building-local;
3. a `1,0`, essa base é exatamente igual a `RBCSmartLocalPolicy`;
4. qualquer configuração que tente ativar este canal sem a base local correta
   falha imediatamente.

A base SMART residual original desativa carregamento da bateria motivado por
preço. V4 acrescenta `signal_price_charge_rate=0,6`, ativo apenas para
multiplicadores abaixo de `1,0`. O controlo neutro continua exatamente igual;
um desconto passa a poder carregar a bateria, enquanto sinais acima de `1,0`
continuam a usar a lógica de descarga sensível ao preço. Isto dá autoridade
bidirecional ao canal sem alterar o ator nem fornecer informação comunitária à
folha.

Antes de treinar outro CC, a grelha anual fixa
`0,90/0,95/1,00/1,05/1,10/1,20/1,30` mede a autoridade causal deste canal.
`1,00` é o controlo neutro pré-registado. Só se treina um CC residual em torno
de um multiplicador que supere esse controlo. Se nenhum o superar, o próximo
passo é treinar uma nova versão do PPO local com randomização do preço efetivo,
sem lhe fornecer observações da comunidade, e repetir o mesmo controlo.

## Nova frente CC-SMART

O melhor CC-SMART de desenvolvimento anterior atingiu EUR 21 936,93, uma
melhoria de apenas EUR 27,74 contra o SMART settled. V4 tenta aumentar essa
margem sem confundir mudanças de recompensa e frequência de decisão:

| Receita | Intervalo CC | Horizonte PPO | Recompensa |
|---|---:|---:|---|
| `settled_cost_hourly` | 4 passos / 1 h | 168 decisões / 7 dias | custo settled |
| `settled_cost_15min` | 1 passo / 15 min | 672 decisões / 7 dias | custo settled |
| `settled_cost_peak_15min` | 1 passo / 15 min | 672 decisões / 7 dias | custo + pico 0,05 + ramp 0,02 |

As três receitas usam dez episódios anuais, BC com 8 760 decisões, 4 000
updates supervisionados, referência `1,3`, gama efetiva `0,9--1,3` e
regularização reduzida (`w_factor=0,01`, `w_smoothness=0,005`). Assim o
primeiro par isola a autoridade temporal e o terceiro mede se há ganhos
físicos visíveis sem perder o foco económico.

## Evidência local antes do build

- Testes focados: `65 passed` antes do smoke.
- Suíte completa final: `920 passed`, 29 warnings esperados.
- PPO corrigido `1,0`: smoke real, 17 checkpoints carregados, exit 0,
  artefactos e 53 ficheiros de simulação exportados.
- Paridade causal: num replay de 384 passos, os 53 ficheiros exportados pelo
  PPO original e pelo pipeline corrigido a `1,0` foram byte-a-byte idênticos;
  `exported_kpis.csv` teve SHA-256
  `0c9c1a07e234b0185c6f53afaade22dc2e3b61a007823115b685fe5e71a5d3ec`.
- PPO corrigido `1,3`: smoke real, exit 0, mesmos contratos de exportação.
- Autoridade do desconto: no replay de 384 passos, `0,9` alterou a bateria do
  Building 15, os dados desse edifício, a série comunitária e os KPIs; o hash
  de `exported_kpis.csv` mudou para
  `69aac1f705f371e7721fece1bbfaa5f703b7cfb2198027cba29885ebaae32676`.
  Isto prova atuação causal, mas não é evidência de melhoria económica.
- CC-SMART cost-only 15 min: smoke com três episódios e 672 decisões por
  episódio; BC executada, uma atualização PPO real
  (`pg=-0,0471`, `v=86,8374`, `kl_stop=False`), avaliação final determinística,
  checkpoint e séries exportados, exit 0.

Os smokes são apenas evidência funcional. Resultados de custo, picos e
fairness só serão aceites no ano completo. A promoção de CC+PPO exige:

1. custo settled inferior ao PPO neutro emparelhado;
2. hard gates de EV, deferrables, rede, SoC e outage;
3. scorecard completo de importação, picos, ramping, load factor, emissões,
   autoconsumo, throughput, V2G e fairness;
4. confirmação posterior em três seeds ou superfície temporal held-out.
