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
MCAULEY, 2018). Por isso o resultado é reportado como uma estimativa do
que seria alcançável com esse tipo de intervenção, e não como uma
recomendação de uso.

Desenho (revisão do orientador):
- os limiares são ajustados SOMENTE COM O TREINO, sobre escores fora da
  dobra (out-of-fold): o pipeline (pré-processamento + classificador,
  com os hiperparâmetros já escolhidos) é reajustado em k dobras do
  treino, e cada registro do treino recebe o escore de um modelo que não
  o viu. Isso evita ajustar os limiares sobre escores de treino
  otimistas, sem consumir nenhuma parte do conjunto de teste;
- os limiares assim obtidos são aplicados aos escores do modelo final
  (ajustado no treino inteiro) sobre o conjunto de teste INTEIRO, e o
  procedimento é repetido nas dez divisões do laço externo, de modo que
  o efeito é reportado com IC por bootstrap em cada divisão e com média
  e desvio-padrão entre as divisões.
"""
from __future__ import annotations

import numpy as np
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

K_DOBRAS_LIMIAR = 5


class EscorePronto(ClassifierMixin, BaseEstimator):
    """
    "Classificador" que devolve, como probabilidade, o próprio escore
    recebido na única coluna de X. Permite ajustar o ThresholdOptimizer
    sobre escores já calculados (fora da dobra no treino; do modelo final
    no teste) sem retreinar nada.
    """

    def fit(self, x, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, x):
        s = np.asarray(x, dtype=float).reshape(-1)
        return np.column_stack([1 - s, s])

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


def escores_fora_da_dobra(pipeline_treinado, x_treino, y_treino, k: int = K_DOBRAS_LIMIAR,
                          semente_aleatoria: int = 42) -> np.ndarray:
    """
    Escores out-of-fold no treino: clona o Pipeline (mesmos
    hiperparâmetros, ainda não ajustado) e o reajusta em k dobras
    estratificadas do próprio treino.
    """
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=semente_aleatoria)
    return cross_val_predict(clone(pipeline_treinado), x_treino, y_treino, cv=cv,
                             method="predict_proba", n_jobs=-1)[:, 1]


def ajustar_limiares(escores_treino, y_treino, raca_treino) -> ThresholdOptimizer:
    """Ajusta os limiares por grupo (equalized odds) sobre escores fora da dobra do treino."""
    estimador = EscorePronto().fit(None)
    otimizador = ThresholdOptimizer(estimator=estimador, constraints="equalized_odds",
                                    predict_method="predict_proba", prefit=True)
    otimizador.fit(np.asarray(escores_treino).reshape(-1, 1), np.asarray(y_treino),
                   sensitive_features=np.asarray(raca_treino, dtype=object))
    return otimizador


def aplicar_limiares(otimizador: ThresholdOptimizer, escores_teste, raca_teste, semente_aleatoria: int = 42) -> np.ndarray:
    """Aplica os limiares por grupo aos escores do modelo final no conjunto de teste."""
    return np.asarray(otimizador.predict(np.asarray(escores_teste).reshape(-1, 1),
                                         sensitive_features=np.asarray(raca_teste, dtype=object),
                                         random_state=semente_aleatoria)).astype(int)


def pos_processar(pipeline_treinado, x_treino, y_treino, raca_treino, x_teste, raca_teste,
                  semente_aleatoria: int = 42) -> tuple[np.ndarray, np.ndarray, ThresholdOptimizer]:
    """
    Fluxo completo: escores fora da dobra no treino -> limiares por grupo
    -> decisões no teste. Retorna (decisões antes, decisões depois,
    otimizador). "Antes" é a decisão do modelo no limiar único de 0,5.
    """
    oof = escores_fora_da_dobra(pipeline_treinado, x_treino, y_treino, semente_aleatoria=semente_aleatoria)
    otimizador = ajustar_limiares(oof, y_treino, raca_treino)
    escores_teste = pipeline_treinado.predict_proba(x_teste)[:, 1]
    antes = (escores_teste >= 0.5).astype(int)
    depois = aplicar_limiares(otimizador, escores_teste, raca_teste, semente_aleatoria)
    return antes, depois, otimizador
