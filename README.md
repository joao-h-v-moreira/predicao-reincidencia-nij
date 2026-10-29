# Predição de Reincidência Criminal com Mitigação de Viés Racial

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23713062.svg)](https://doi.org/10.5281/zenodo.23713062)

Trabalho de Conclusão de Curso (TCC) — Universidade São Judas Tadeu (USJT)
Curso de Bacharelado em Engenharia de Computação

**Equipe:** Douglas Sanches Ferreira, João Henrique Vasques Moreira, João
Victor Alves Navarro, Lucas Castilho Bueno, Verônica de Souza Menezes Costa
**Orientador:** Prof. Nelson Aguiar

## Sobre o projeto

Pipeline experimental para estimar, **no momento da soltura**, o risco de
uma pessoa em liberdade condicional ser presa novamente em até 3 anos, a
partir do **Full Dataset** do [NIJ Recidivism Forecasting
Challenge](https://nij.ojp.gov/funding/recidivism-forecasting-challenge)
(pessoas liberadas para condicional na Geórgia, EUA, entre 2013 e 2015).

O ponto central é a **exclusão da variável racial e de variáveis proxy** do
conjunto de atributos preditivos. A raça é retida à parte, apenas para medir
a equidade do modelo entre grupos raciais (e de sexo).

### Decisões metodológicas principais

| Decisão | O que o código faz |
|---|---|
| Atributos disponíveis na soltura | A análise principal usa só o que se sabe quando a pessoa é solta (histórico criminal, pena, dados demográficos, score/nível de supervisão inicial, condições da condicional). As 16 variáveis medidas **durante** a condicional (`COLUNAS_PERIODO_SUPERVISAO` em `src/dataset.py`) ficam de fora, porque se sobrepõem ao período em que a reincidência é contada. Elas entram só numa análise de sensibilidade (conjunto "completo"). |
| Sem vazamento no pré-processamento | Imputação, one-hot e normalização ficam dentro de um `sklearn.pipeline.Pipeline` e são ajustadas só no treino (e em cada dobra interna da busca de hiperparâmetros). |
| Proxies identificados no treino | V de Cramér (≥ 0,20) e correlação ponto-bisserial (\|r\| ≥ 0,30), calculados só no conjunto de treino de cada divisão. |
| Validação aninhada | Laço externo: 10 divisões treino/teste 70/30 estratificadas (sementes 42 a 51). Laço interno: k-fold (k = 3) no treino para a busca de hiperparâmetros. A divisão de semente 42 é a de referência (figuras e análises detalhadas). |
| Incerteza | IC de 95% por bootstrap (1000 reamostragens) para métricas de desempenho, métricas por grupo e diferenças entre grupos; bootstrap pareado para a diferença de AUC entre modelos; média ± desvio-padrão nas 10 repetições. |
| Comparações entre grupos com direção fixa | Grupo A × grupo B de referência: BLACK × WHITE (raça) e M × F (sexo) (`ORDEM_GRUPOS` em `src/fairness.py`). As diferenças `dif_fpr`, `dif_fnr`, `dif_tpr`, `dif_ppv` são **A − B, com sinal**, e o IC pode conter zero. Uma diferença do tipo "maior − menor" nunca é negativa, e o IC por bootstrap dela nunca contém zero, mesmo sem disparidade alguma; por isso ela não é usada. |
| Convenção do disparate impact | Taxa de predição **favorável** (baixo risco) do grupo A / a do grupo B. Abaixo de 0,80 é alerta contra o grupo A (regra dos quatro quintos); acima de 1, a direção se inverte (acima de 1,25, alerta contra o grupo B). |
| Pós-processamento (limiar por grupo) | Limiares ajustados **só no treino**, sobre escores fora da dobra (validação cruzada de 5 dobras), e avaliados no conjunto de teste **inteiro** de cada uma das 10 divisões, com IC para acurácia, recall e diferenças entre grupos. |
| Robustez a `Gang_Affiliated` | A variável está ausente para todas as mulheres, então a categoria "Ausente" codifica o sexo. A análise principal é repetida sem ela, e sem ela e sem `Gender`, nas 10 divisões (mesmos hiperparâmetros de cada divisão, para isolar o efeito da remoção). |
| Sensibilidade aos limiares de proxy | Além do critério principal (V ≥ 0,20; \|r\| ≥ 0,30), um rigoroso (0,10; 0,10) e um permissivo (0,30; 0,50), nas 10 divisões (`LIMIARES_PROXY` em `run_pipeline.py`). |
| Interseccionalidade | Métricas por raça × sexo (quatro subgrupos) e diferenças raciais dentro de cada sexo, com IC. |
| Comparação com o desafio | Desenho do primeiro ano do desafio do NIJ (divisão oficial `Training_Sample`, alvo `Recidivism_Arrest_Year1`): erro de Brier por sexo e índice de equidade, ao lado dos resultados oficiais publicados pelo NIJ. |
| Decomposição da disparidade de FPR | Identidade de Chouldechova (2017): quanto da razão FPR_A / FPR_B vem da taxa-base, da precisão (PPV) e da sensibilidade (FNR). As três parcelas somam 1. Uma parcela negativa é um fator que reduz a disparidade; uma parcela acima de 1 é um fator compensado em parte pelos outros. Com razão de FPR próxima de 1, as parcelas ficam instáveis; nesse caso, leia as razões dos termos. |

## Estrutura do repositório

```
predicao-reincidencia-nij/
├── README.md
├── run_pipeline.py          # script principal (orquestra UC01-UC10)
├── requirements.txt
├── docs/
│   └── variaveis_dataset.md  # inventário das 53 variáveis e decisões
├── src/
│   ├── dataset.py            # UC01 - importar dataset; conjuntos de atributos
│   ├── eda.py                # UC02 - análise exploratória (EDA)
│   ├── preprocess.py         # UC03 - pré-processamento (transformador sklearn)
│   ├── bias_removal.py       # UC04 - remover raça; identificar proxies
│   ├── split.py              # divisão treino/teste e validação aninhada
│   ├── models.py             # UC05/UC06 - algoritmos, busca de hiperparâmetros, baseline
│   ├── evaluation.py         # UC07 - métricas de desempenho e ICs
│   ├── fairness.py           # UC08 - métricas de equidade, ICs, decomposição, índice NIJ
│   ├── ablation.py           # estudo de ablação (RF11)
│   ├── comparison.py         # UC09 - comparar modelos
│   ├── proxy_multivariado.py # diagnóstico de proxy combinatório
│   ├── postprocessing.py     # pós-processamento por limiar por grupo
│   ├── analises_complementares.py # interseccionalidade e comparação com o desafio do NIJ
│   ├── visualizacao.py       # curvas ROC e de calibração
│   ├── report.py             # UC10 - exportar relatório
│   ├── events.py             # registrador de eventos (RNF11)
│   └── synthetic_data.py     # dados SINTÉTICOS para o smoke test
├── data/                     # coloque aqui o CSV real (não versionado)
└── results/                  # saídas do pipeline
```

## Como obter o dataset real

1. Acesse `https://nij.ojp.gov/funding/recidivism-forecasting-challenge`.
2. Baixe o **Full Dataset** (CSV).
3. Salve em `data/nij_full_dataset.csv` (ou informe outro caminho com `--dataset`).

O dataset não é redistribuído neste repositório.

## Como executar

Localmente:

```bash
pip install -r requirements.txt
python run_pipeline.py --dataset data/nij_full_dataset.csv
```

No Google Colab:

```python
!git clone https://github.com/joao-h-v-moreira/predicao-reincidencia-nij.git
%cd predicao-reincidencia-nij
!pip install -q -r requirements.txt
# envie o CSV para data/nij_full_dataset.csv pelo painel de arquivos
!python run_pipeline.py --dataset data/nij_full_dataset.csv --resultados results
```

Opções:

- `--repeticoes N` — número de divisões do laço externo (padrão 10). A busca
  de hiperparâmetros é refeita em cada divisão, e as análises complementares
  (robustez, limiares de proxy, pós-processamento) também percorrem as 10
  divisões, por isso a execução completa leva bem mais que a versão anterior
  (o tempo varia com o hardware). Para um teste rápido, use `--repeticoes 2`.
- `--sem-complementares` — roda só a análise principal (sem robustez,
  limiares de proxy, interseccionalidade e comparação com o desafio).
- `--sintetico` — smoke test com dados artificiais (ver abaixo).

O terminal mostra o horário de cada etapa, para indicar que o processo não travou.

**Reprodutibilidade:** cada execução grava `results/ambiente_execucao.json`
com as versões de Python e das bibliotecas usadas. As versões exatas da
execução publicada estão em `requirements-lock.txt` (`pip install -r
requirements-lock.txt` para reproduzir os mesmos números).

## Saídas geradas (pasta `results/`)

A pasta `results/` deste repositório contém a execução com os dados reais cujos números
são apresentados no Capítulo 5 do TCC (ver `results/resumo_resultados.txt`). Uma nova
execução com as mesmas versões de bibliotecas (`results/ambiente_execucao.json`)
reproduz os mesmos valores.

Resumo:
- `resumo_resultados.txt` — resumo em texto corrido dos principais resultados
  (desempenho, equidade, ablação, pós-processamento e sensibilidade), com ICs,
  legível sem consultar o código (RNF05).

Desempenho:
- `comparacao_modelos.csv` — divisão de referência: métricas de cada modelo
  com IC de 95% (`*_ic_inferior`, `*_ic_superior`), disparate impact e
  diferenças de FPR e FNR (A − B) por raça e sexo, também com IC, índice
  oficial do NIJ e hiperparâmetros.
- `resumo_repeticoes.csv` / `repeticoes_por_semente.csv` — média,
  desvio-padrão, mínimo e máximo das métricas nas 10 divisões / valores de
  cada divisão.
- `diferencas_auc_pareadas.csv` — diferença de AUC-ROC entre pares de
  modelos, com IC de 95% por bootstrap pareado.
- `importancia_atributos_modelo.csv` — importância por permutação de cada
  variável para o Gradient Boosting (queda de AUC no teste).
- `curvas_roc.png`, `curvas_calibracao.png`.

Equidade:
- `equidade_detalhada.csv` — formato longo: por modelo, atributo protegido
  e grupo, cada métrica (taxa-base, taxa de alto risco, FPR, FNR, TPR, PPV,
  calibração no agregado, ECE, Brier, Brier skill) e cada diferença entre
  grupos (linha `BLACK_vs_WHITE` ou `M_vs_F`), com IC de 95%.
- `decomposicao_disparidade_fpr.csv` — decomposição da razão de FPR entre
  grupos em taxa-base, precisão (PPV) e sensibilidade (FNR).
- `calibracao_por_grupo_raca.png` — curva de calibração do Gradient
  Boosting separada por grupo racial.
- `taxas_base_reincidencia.csv` — taxa de reincidência observada por grupo.

Proxies e mitigação:
- `diagnostico_associacao_raca.csv` — associação de cada variável com a
  raça (calculada no treino da divisão de referência).
- `diagnostico_proxy_multivariado.csv` — AUC de um classificador auxiliar
  que tenta reconstruir a raça a partir dos atributos finais, com o
  tamanho de efeito segundo Rice e Harris (2005), para os conjuntos
  "soltura" e "completo".
- `importancia_proxy_multivariado.csv` — quais variáveis mais contribuem
  para essa reconstrução.
- `estudo_ablacao.csv` — quatro variantes (com/sem raça × com/sem proxies),
  todas com os mesmos hiperparâmetros do Gradient Boosting de referência.
  Para cada variante: AUC, disparate impact e `dif_fpr`/`dif_fnr` por raça,
  com IC de 95%. As colunas `delta_*` trazem a diferença pareada em relação
  à variante "com raça, com proxies", com IC por bootstrap pareado (na
  própria variante de referência, os `delta_*` são zero).
- `pos_processamento_equidade.csv` — divisão de referência: antes e depois
  do ajuste de limiar por grupo (HARDT; PRICE; SREBRO, 2016), com ICs para
  acurácia, recall, FPR/TPR/PPV por grupo e diferenças entre grupos. Os
  limiares são ajustados só no treino (escores fora da dobra) e avaliados no
  teste inteiro. A linha `diferenca_depois_menos_antes` traz a diferença
  pareada, com IC. Atenção: esse ajuste usa a raça no momento da decisão
  (ver docstring de `src/postprocessing.py`).
- `pos_processamento_repeticoes.csv` / `pos_processamento_resumo.csv` — o
  mesmo procedimento em cada uma das 10 divisões (com IC em cada uma) e o
  resumo (média ± desvio-padrão e quantas divisões têm IC contendo zero,
  diferença negativa significativa etc.); figura
  `pos_processamento_repeticoes.png`.

Robustez e análises complementares:
- `robustez_gang_affiliated.csv` — divisão de referência, Gradient Boosting:
  modelo principal × sem `Gang_Affiliated` × sem `Gang_Affiliated` e
  `Gender`, com IC e diferenças pareadas em relação ao principal (por raça e
  por sexo); `robustez_gang_affiliated_repeticoes.csv` e `_resumo.csv` — as
  três variantes, para os três algoritmos, nas 10 divisões.
- `equidade_detalhada_variantes.csv` — métricas por grupo (com IC) de cada
  variante de robustez na divisão de referência, incluindo a calibração por
  grupo (calibração no agregado, ECE e Brier skill): mostra, por exemplo, se o
  modelo sem `Gang_Affiliated` e `Gender` passa a superestimar o risco das
  mulheres. As repetições trazem a calibração no agregado por grupo
  (`sexo_calib_M`, `sexo_calib_F`, `raca_calib_BLACK`, `raca_calib_WHITE`).
- `sensibilidade_limiares_proxy.csv` (+ `_repeticoes.csv`, `_resumo.csv`) —
  variáveis removidas, desempenho, disparidade e reconstrução da raça com os
  limiares rigoroso, principal e permissivo.
- `robustez_dif_fpr_racial.png` — diferença de FPR racial (com IC) em todas
  as análises de robustez, lado a lado.
- `equidade_interseccional.csv` — métricas por raça × sexo e diferenças
  raciais dentro de cada sexo (e entre sexos dentro de cada grupo racial),
  com IC. As repetições (`repeticoes_por_semente.csv`) trazem também
  `raca_dif_fpr_homens` e `raca_dif_fpr_mulheres`.
- `comparacao_desafio_nij.csv` — desenho do primeiro ano do desafio
  (divisão oficial, alvo de 1 ano): Brier de homens, de mulheres e médio,
  índice de equidade por sexo e AUC dos modelos deste trabalho, de um
  modelo ingênuo (taxa-base por sexo) e os resultados oficiais dos
  vencedores publicados pelo NIJ.

Sensibilidade e auditoria:
- `analise_sensibilidade_atributos.csv` — conjunto "soltura" (principal) ×
  "completo" (com variáveis do período de supervisão).
- `valores_ausentes.csv`, `eda_distribuicoes.png`.
- `valores_ausentes_por_sexo.csv` — proporção de ausentes de cada variável
  entre homens e mulheres. Se a ausência se concentra num sexo, a categoria
  "Ausente" da imputação passa a codificar o sexo.
- `ambiente_execucao.json`, `eventos.jsonl` (log de auditoria, RNF11, só
  com metadados de execução).

## Versão e como citar

A versão usada no artigo e no TCC é marcada com uma tag (`v2.0.0`) e
arquivada no Zenodo, que atribui um DOI à versão (ver `CITATION.cff`):
[10.5281/zenodo.23713062](https://doi.org/10.5281/zenodo.23713062).
Cite o código pelo DOI da versão, e não apenas pelo endereço do GitHub, para
que o leitor encontre exatamente o código que gerou os números publicados.

## Licença

Código distribuído sob a licença MIT (arquivo `LICENSE`). O Full Dataset do NIJ não
faz parte do repositório e segue os termos de uso do Office of Justice Programs.

## Observação sobre o smoke test

`--sintetico` gera um dataset artificial com a mesma estrutura e os mesmos
formatos do Full Dataset do NIJ (faixas como "3 or more", booleanos,
ausentes). Serve **apenas** para verificar que o pipeline roda de ponta a
ponta. **Nenhuma métrica produzida nesse modo é resultado do trabalho.**

## Referências citadas no código

- CHOULDECHOVA, A. Fair prediction with disparate impact. *Big Data*, 2017.
- COHEN, J. *Statistical power analysis for the behavioral sciences*. 1988.
- DRESSEL, J.; FARID, H. The accuracy, fairness, and limits of predicting recidivism. *Science Advances*, 2018.
- HARDT, M.; PRICE, E.; SREBRO, N. Equality of opportunity in supervised learning. *NeurIPS*, 2016.
- KLEINBERG, J.; MULLAINATHAN, S.; RAGHAVAN, M. Inherent trade-offs in the fair determination of risk scores. *ITCS*, 2017.
- LIPTON, Z. C.; CHOULDECHOVA, A.; MCAULEY, J. Does mitigating ML's impact disparity require treatment disparity? *NeurIPS*, 2018.
- PEDREGOSA, F. et al. Scikit-learn: machine learning in Python. *JMLR*, 2011.
- PEDRESCHI, D.; RUGGIERI, S.; TURINI, F. Discrimination-aware data mining. *ACM SIGKDD*, 2008.
- REA, L. M.; PARKER, R. A. *Designing and conducting survey research*. 1992.
- RICE, M. E.; HARRIS, G. T. Comparing effect sizes in follow-up studies: ROC area, Cohen's d, and r. *Law and Human Behavior*, 2005.
- WHITE, V. et al. *NIJ Recidivism Forecasting Challenge: contextualizing the results*. NIJ, 2022.
- NATIONAL INSTITUTE OF JUSTICE. *Recidivism Forecasting Challenge Results*. https://nij.ojp.gov/funding/recidivism-forecasting-challenge-results.
