# Contrato dos oráculos MILP locais e comunitários

Data: 2026-07-29

Atualizado em 2026-07-31 após implementação, correção do SOC inicial de EVs,
novo solve e replay CityLearn dos dois níveis total-energy.

## Separação obrigatória

Há dois problemas de benchmark distintos e nenhum resultado deve atravessar a
fronteira entre eles:

1. **Local/retalho:** cada `Building_i` paga a sua própria importação à rede.
   Exportação de outra casa não pode compensar esta importação. O comparador
   heurístico é `RBCSmartLocal`, o RL recebe apenas observações/reward locais e o
   oráculo é resolvido 17 vezes, uma por casa.
2. **Comunidade/mercado:** o objetivo é o custo conjunto depois de coordenação e
   netting entre membros. O comparador principal é `RBCCommunity`; `RBCSmart`
   pode aparecer apenas como baseline local secundário. O MILP é um problema
   conjunto com as 17 casas.

Para qualquer configuração em que `community_market.enabled: true`, o custo
settled de um membro já depende das restantes casas. Esse caso não pode ser
rotulado como benchmark local independente.

## Quatro níveis de oráculo

| Nível | Estado | Variáveis controladas | Claim permitido |
|---|---|---|---|
| Local fixed-service | implementado e replayado | bateria estacionária por casa; EV e deferrable fixos ao RBCSmart | limite condicional por casa |
| Comunidade fixed-service | implementado e replayado | baterias estacionárias em conjunto; EV e deferrable fixos | limite condicional comunitário |
| Local total-energy/full-house | implementado, resolvido e replayado | bateria, EV/V2G e início de deferrables de uma casa | ótimo do modelo linear local e upper feasible CityLearn após replay |
| Comunidade total-energy/full | implementado, resolvido e replayado | todos os ativos flexíveis das 17 casas e settlement conjunto | ótimo do modelo linear comunitário e upper feasible CityLearn após replay |

Os quatro níveis estão implementados. Os modelos fixed-service continuam como
referências condicionais históricas. O
resultado conservador é uma solução do modelo; só passa a ser upper bound
CityLearn depois do replay no simulador. O relaxamento otimista é lower bound
do modelo fornecido, não uma prova automática sobre toda a dinâmica não linear
do CityLearn.

## Formulação full-house implementada

Para cada casa `i`, passo `t` e ativo aplicável:

- bateria estacionária: potência de carga/descarga, SOC, eficiências PWL,
  complementaridade e SOC terminal;
- EV por sessão: carga/V2G, SOC, disponibilidade entre chegada e partida,
  potência mínima/máxima, eficiência, serviço mínimo e precisão à partida;
- deferrable: binária de início e convolução com o perfil do ciclo, janela de
  início, deadline, exclusão de sobreposição e conclusão integral;
- importação local `g[i,t] >= net[i,t]` e `g[i,t] >= 0`;
- limites totais e por fase, incluindo a ligação de cada charger/ativo à fase;
- objetivo lexicográfico: primeiro maximizar serviço fisicamente alcançável,
  depois cumprir gates e finalmente minimizar `sum(price[t] * g[i,t] * dt)`.

O oráculo local é a soma de 17 solves independentes. Além do custo total, deve
publicar custo, regret, gates e certificado para cada `Building_i`.

## Formulação comunitária implementada

O modelo reutiliza todas as restrições full-house e acrescenta a importação
distrital `G[t]`:

```text
G[t] >= sum_i net[i,t]
G[t] >= 0
min sum_t price[t] * G[t] * dt
```

No dataset atual, o preço interno simétrico de 0,8 cancela no total da
comunidade. Variáveis explícitas de trade só são necessárias para reproduzir
faturas individuais, pesos de settlement ou objetivos de fairness; não são
necessárias para o custo distrital mínimo.

## Testes de consistência e promoção

1. Com mercado e constraints partilhadas desligados, o MILP comunitário deve
   decompor exatamente na soma dos 17 MILPs locais.
2. Toda solução conservadora deve ser replayada no mesmo dataset/janela.
3. Gates de EV, precisão, rede total/fases, deferrables, SOC e outage são
   avaliados antes do custo.
4. O scorecard local publica número de casas melhores, mediana, pior casa,
   regret ao oráculo e fração do gap RBCSmart-oráculo fechada.
5. `PPO > RBCSmartLocal` é uma meta de promoção, não uma garantia matemática. Casas
   sem flexibilidade útil podem empatar; uma melhoria agregada dominada por uma
   só casa não é suficiente para promoção robusta.

## Evidência fixed-service atual

No cenário local sem mercado, janela `0:1023`, seed 123:

- `RBCSmart`: EUR 673,4518, 17/17 casas passam os gates;
- replay dos 17 MILPs independentes: EUR 629,3619, 17/17 passam e 17/17 têm
  custo inferior ao RBCSmart;
- melhoria condicional observada: EUR 44,0898 (6,55%).

Estes valores demonstram oportunidade de controlo local, mas ainda não são o
ótimo full-house porque EV e deferrables permanecem fixos ao RBCSmart.

## Evidência total-energy corrigida

Na janela matching `[0,672)`, após alinhar o fallback de SOC inicial dos EVs
com o gerador determinístico do CityLearn:

- MILP individual: 17/17 solves `optimal`, EUR 424,3984 no modelo; replay
  EUR 424,0119 contra EUR 477,0224 do RBC Smart local (-11,11%), 17/17 gates e
  17/17 casas abaixo do RBC;
- MILP comunitário: `optimal`, objetivo EUR 347,7026176791, dual bound
  EUR 347,6787858945 e gap 0,0068547%; replay EUR 347,7026162444 contra
  EUR 440,0609948344 do RBCCommunity (-20,99%), 17/17 gates e 17/17 casas
  abaixo do RBC.

As duas claims são “ótimo do modelo linear fornecido + replay CityLearn
feasible”, não “ótimo global do simulador”. A janela tem oito sessões EV
truncadas à direita e mantém natureza week-one diagnóstica.
