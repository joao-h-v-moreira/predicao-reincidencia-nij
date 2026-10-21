# Inventário das variáveis do Full Dataset (NIJ Recidivism Forecasting Challenge)

Este documento registra a revisão de todas as 53 variáveis do codebook oficial
do NIJ (Appendix 2), a decisão tomada para cada uma e a justificativa, em
resposta à exigência de que todo o trabalho utilize apenas dados reais e de
que a mitigação de viés racial seja feita de forma criteriosa — e não por
simples exclusão de uma única coluna.

Fonte do codebook: [NIJ Recidivism Forecasting Challenge — Appendix 2: Codebook](https://nij.ojp.gov/funding/recidivism-forecasting-challenge-appendix-2-codebook.pdf)

## Princípio adotado

Em vez de excluir variáveis manualmente "a olho" (abordagem pouco defensável
e difícil de justificar numa banca), o pipeline usa um critério estatístico
sistemático e documentado (RF05, UC04): toda variável remanescente após a
remoção da raça é testada quanto à sua associação com a raça original —
V de Cramér (≥ 0,20) para variáveis categóricas, correlação ponto-bisserial
(|r| ≥ 0,30) para variáveis numéricas — e sinalizada como proxy quando
ultrapassa o limiar. A raça em si nunca é um atributo de entrada do modelo;
é retida separadamente, exclusivamente para a avaliação de equidade (UC08).

## Variáveis excluídas do conjunto de atributos preditivos

| Variável | Motivo |
|---|---|
| `ID` | Identificador de linha, sem valor preditivo |
| `Recidivism_Arrest_Year1` | Vazamento de dados: componente que forma o próprio alvo `Recidivism_Within_3years` |
| `Recidivism_Arrest_Year2` | Idem |
| `Recidivism_Arrest_Year3` | Idem |
| `Race` | Removida do conjunto de atributos; retida separadamente apenas para avaliação de equidade (nunca como entrada do modelo) |
| `Training_Sample` | **Não pertence ao codebook oficial (Apêndice 2) do NIJ.** É a 54ª coluna presente no CSV real, ausente das 53 variáveis documentadas — um artefato administrativo do desafio original (indicador de qual lote treino/teste o NIJ usou na competição), não um atributo real da pessoa avaliada. Associação empírica com raça é desprezível (V de Cramér ≈ 0,008), portanto não é um proxy; é excluída por não ser uma variável válida sobre o indivíduo, e não por motivo de viés |

## Variável-alvo

| Variável | Observação |
|---|---|
| `Recidivism_Within_3years` | Booleana (`Yes`/`No` no CSV real); convertida para 0/1 pelo pipeline |

## Correção de tipagem: variáveis categóricas que o pandas lê como número

| Variável | Problema identificado | Correção |
|---|---|---|
| `Residence_PUMA` | É um código de área censitária (Public Use Microdata Area) — categórico nominal, sem ordem ou magnitude real — mas o pandas infere como número inteiro a partir do CSV. Se não corrigido, (i) o modelo recebe uma "escala" que não existe entre códigos de área e (ii) o teste de proxy usaria correlação ponto-bisserial em vez de V de Cramér, mascarando um proxy geográfico clássico de raça (efeito histórico de segregação residencial / "redlining", amplamente documentado na literatura de justiça criminal) | Forçada para tipo categórico (`string`) na importação, antes de qualquer outra etapa |

## Colunas sem cabeçalho legível no CSV real (`_v1` a `_v4`)

No arquivo real baixado pelo usuário, 4 colunas chegam sem nome legível —
o pandas as rotula automaticamente `_v1`, `_v2`, `_v3`, `_v4` (provável
efeito colateral de um problema de codificação/exportação a montante, antes
da distribuição pelo NIJ; não é algo introduzido pelo pipeline). A posição
exata dessas colunas no arquivo foi comparada com o codebook oficial do
NIJ e corresponde, de forma inequívoca, a 4 variáveis documentadas que
"faltavam" nos nomes visíveis:

| Nome no CSV | Posição observada | Variável real (codebook oficial) |
|---|---|---|
| `_v1` | Entre `Prior_Arrest_Episodes_Drug` e `Prior_Arrest_Episodes_DVCharges` | `Prior_Arrest_Episodes_PPViolationCharges` |
| `_v2` | Logo após `Prior_Conviction_Episodes_Drug` | `Prior_Conviction_Episodes_PPViolationCharges` |
| `_v3` | Em seguida a `_v2` | `Prior_Conviction_Episodes_DomesticViolenceCharges` |
| `_v4` | Em seguida a `_v3`, antes de `Prior_Revocations_Parole` | `Prior_Conviction_Episodes_GunCharges` |

O grupo `Prior_Arrest_Episodes_*` tem 8 subtipos no codebook (Felony, Misd,
Violent, Property, Drug, PPViolationCharges, DVCharges, GunCharges), mas só
7 chegam nomeados no CSV real — a posição da lacuna bate exatamente com
`_v1`. O mesmo vale para `Prior_Conviction_Episodes_*`: o codebook lista
`PPViolationCharges`, `DomesticViolenceCharges` e `GunCharges` nas posições
26-28, imediatamente após `Prior_Conviction_Episodes_Drug` (posição 25) —
exatamente onde `_v2`, `_v3` e `_v4` aparecem no CSV.

O pipeline renomeia essas 4 colunas para seus nomes reais automaticamente
na importação (`MAPA_RENOMEACAO_COLUNAS_SEM_CABECALHO` em `src/dataset.py`),
de forma condicional — só renomeia se encontrar literalmente `_v1`..`_v4`
entre as colunas, então não quebra caso uma versão futura do CSV já venha
com os nomes corretos. Nenhuma das quatro variáveis excede o limiar de
associação com raça nos testes realizados (valores entre 0,005 e 0,072,
muito abaixo do limiar de 0,20) — não são proxies.

## Variáveis do período de supervisão: fora da análise principal

A pergunta do trabalho é: **no momento da soltura**, qual o risco de a
pessoa ser presa novamente em até 3 anos? Dezesseis variáveis do Full
Dataset são medidas **durante** a liberdade condicional, ou seja, depois
desse momento e dentro do mesmo período em que a reincidência é contada.
O relatório oficial do NIJ registra que elas não foram fornecidas para a
previsão do primeiro ano do desafio, justamente porque esse dado não existe
quando a pessoa é solta (WHITE et al., 2022). Usá-las para prever a
reincidência em 3 anos introduziria vazamento temporal: para quem é preso
de novo cedo, parte desses valores é consequência da própria nova prisão
(por exemplo, `Percent_Days_Employed` cai porque a pessoa foi presa).

| Grupo | Variáveis |
|---|---|
| Violações das condições | `Violations_ElectronicMonitoring`, `Violations_Instruction`, `Violations_FailToReport`, `Violations_MoveWithoutPermission` |
| Relatórios e programas | `Delinquency_Reports`, `Program_Attendances`, `Program_UnexcusedAbsences`, `Residence_Changes` |
| Testes de droga | `Avg_Days_per_DrugTest`, `DrugTests_THC_Positive`, `DrugTests_Cocaine_Positive`, `DrugTests_Meth_Positive`, `DrugTests_Other_Positive` |
| Emprego | `Percent_Days_Employed`, `Jobs_Per_Year`, `Employment_Exempt` |

Decisão: ficam **fora** do conjunto de atributos da análise principal
(conjunto "soltura"). Entram apenas na análise de sensibilidade (conjunto
"completo"), que mede quanto do desempenho e da disparidade entre grupos
vinha dessa informação posterior à soltura. Lista no código:
`COLUNAS_PERIODO_SUPERVISAO` em `src/dataset.py`.

## Variáveis disponíveis na soltura, sujeitas ao teste estatístico de proxy

As demais variáveis (sexo, idade em faixas, escolaridade, dependentes,
afiliação a gangue, área de residência, score e nível de supervisão
atribuídos no primeiro contato, crime e tempo da pena cumprida, prisões e
condenações anteriores por tipo, revogações anteriores e condições da
condicional) compõem o conjunto "soltura". Todas passam pelo teste de V de
Cramér / correlação ponto-bisserial contra a raça, calculado **somente no
conjunto de treino** de cada divisão, antes da decisão de remoção. Duas
merecem atenção por serem apontadas na literatura como canais conhecidos
de viés racial em sistemas de justiça criminal, embora não sejam excluídas
automaticamente:

- **`Supervision_Risk_Score_First`**: um score calculado por outro
  instrumento de avaliação de risco, que pode carregar viés embutido no
  próprio cálculo, de forma análoga à crítica ao COMPAS.
- **`Gang_Affiliated`**: bancos de dados de afiliação a gangues são
  documentados como desproporcionalmente aplicados a grupos raciais
  específicos.

Ambas seguem o mesmo critério estatístico das demais. A importância de cada
variável na reconstrução multivariada da raça fica registrada em
`results/importancia_proxy_multivariado.csv`.

## `Gender` (sexo)

Mantida como atributo preditivo normal. O escopo do trabalho é mitigação de
viés **racial**; o sexo é tratado como um segundo atributo protegido para
fins de avaliação de equidade (UC08), não como uma variável a ser removida
do conjunto de atributos.
