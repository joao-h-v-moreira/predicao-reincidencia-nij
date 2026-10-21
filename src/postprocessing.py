"""
Pós-processamento de equidade por ajuste de limiar de decisão por grupo
(equalized odds), seguindo Hardt, Price e Srebro (2016), implementado
com fairlearn.postprocessing.ThresholdOptimizer.

O que a técnica faz: parte do classificador JÁ TREINADO (sem reabrir o
conjunto de atributos nem retreinar) e escolhe um limiar de decisão
diferente para cada grupo racial, de modo a aproximar as taxas de falso
positivo e de verdadeiro positivo entre os grupos. O limiar ótimo pode
ser aleatorizado (para um mesmo escore, uma fração das pessoas recebe
"alto risco"), por isso a predição usa uma semente fixa.

O que a técnica EXIGE, e que precisa ficar explícito no texto: para
aplicar o limiar do grupo de cada pessoa, a raça dessa pessoa precisa
ser conhecida NO MOMENTO DA DECISÃO. Ou seja, a raça deixa de ser usada
como atributo do modelo, mas passa a ser usada na regra de decisão --
o que entra em tensão com a premissa central do trabalho e configura
tratamento explicitamente diferenciado por raça (LIPTON; CHOULDECHOVA;
MCAULEY, 2018). Por isso o resultado é reportado como um limite
do que seria alcançável com esse tipo de intervenção, e não como uma
recomendação de uso.

Cuidado metodológico: o otimizador é ajustado numa metade do conjunto
de teste original (calibração) e avaliado na outra metade (avaliação);
ajustar e avaliar o limiar nos mesmos dados inflaria o ganho aparente.
"""
from __future__ import annotations

import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.model_selection import train_test_split


def dividir_calibracao_avaliacao(x_teste, y_teste, raca_teste, proporcao_calibracao: float = 0.5, semente_aleatoria: int = 42):
    idx_calib, idx_aval = train_test_split(
        x_teste.index, test_size=1 - proporcao_calibracao, random_state=semente_aleatoria, stratify=y_teste
    )
    return (
        x_teste.loc[idx_calib], y_teste.loc[idx_calib], raca_teste.loc[idx_calib],
        x_teste.loc[idx_aval], y_teste.loc[idx_aval], raca_teste.loc[idx_aval],
    )


def ajustar_limiar_por_grupo(pipeline_treinado, x_calib, y_calib, raca_calib, x_aval, raca_aval, semente_aleatoria: int = 42):
    """
    `pipeline_treinado` é o Pipeline (pré-processamento + classificador)
    já ajustado no treino. Para o ThresholdOptimizer, os atributos são
    transformados pelo pré-processamento já ajustado e o classificador é
    passado sozinho (prefit=True) -- nada é reajustado.
    Retorna (predições ajustadas na metade de avaliação, limiares por grupo).
    """
    prep = pipeline_treinado.named_steps["prep"]
    clf = pipeline_treinado.named_steps["clf"]
    # matrizes numéricas já transformadas pelo pré-processamento ajustado
    # no treino (o classificador foi ajustado sobre matrizes, sem nomes de coluna)
    xc = prep.transform(x_calib)
    xa = prep.transform(x_aval)

    otimizador = ThresholdOptimizer(
        estimator=clf, constraints="equalized_odds", predict_method="predict_proba", prefit=True
    )
    otimizador.fit(xc, y_calib, sensitive_features=raca_calib)
    y_ajustado = otimizador.predict(xa, sensitive_features=raca_aval, random_state=semente_aleatoria)
    return y_ajustado, otimizador
