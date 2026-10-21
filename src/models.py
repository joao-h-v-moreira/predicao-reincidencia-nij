"""
ConfiguradorAlgoritmo e hierarquia de ModeloPreditivo (UC05, UC06).

Inclui os três modelos propostos (Regressão Logística, Random Forest e
Gradient Boosting) e o modelo de referência de dois atributos (idade e
total de condenações anteriores) adaptado de Dressel e Farid (2018).

Cada ModeloPreditivo é um `sklearn.pipeline.Pipeline` com duas etapas:
"prep" (o transformador do Preprocessador: imputação, one-hot e
normalização) e "clf" (o classificador). Treinar o modelo = ajustar o
pipeline inteiro SOMENTE no conjunto de treino, de modo que nenhuma
estatística do pré-processamento venha do conjunto de teste.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from preprocess import Preprocessador

SEMENTE_MODELOS = 42


def _classificador_base(algoritmo: str):
    return {
        "regressao_logistica": SkLogisticRegression(max_iter=2000, random_state=SEMENTE_MODELOS),
        "random_forest": RandomForestClassifier(random_state=SEMENTE_MODELOS, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=SEMENTE_MODELOS),
        "baseline_dressel_farid": SkLogisticRegression(max_iter=2000, random_state=SEMENTE_MODELOS),
    }[algoritmo]


class ModeloPreditivo(ABC):
    nome: str = "modelo_base"

    def __init__(self, hiperparametros: dict | None = None):
        self.hiperparametros = hiperparametros or {}
        self._modelo: Pipeline | None = None

    @abstractmethod
    def _construir(self):
        """Devolve o classificador scikit-learn (sem pré-processamento)."""

    def treinar(self, x_treino: pd.DataFrame, y_treino: pd.Series) -> None:
        transformador = Preprocessador().construir_transformador(x_treino)
        self._modelo = Pipeline([("prep", transformador), ("clf", self._construir())])
        self._modelo.fit(x_treino, y_treino)

    def prever(self, x_teste: pd.DataFrame) -> np.ndarray:
        return self._modelo.predict(x_teste)

    def prever_probabilidade(self, x_teste: pd.DataFrame) -> np.ndarray:
        return self._modelo.predict_proba(x_teste)[:, 1]

    @property
    def pipeline(self) -> Pipeline:
        return self._modelo


class RegressaoLogistica(ModeloPreditivo):
    nome = "regressao_logistica"

    def _construir(self):
        return _classificador_base(self.nome).set_params(**self.hiperparametros)


class RandomForest(ModeloPreditivo):
    nome = "random_forest"

    def _construir(self):
        return _classificador_base(self.nome).set_params(**self.hiperparametros)


class GradientBoosting(ModeloPreditivo):
    nome = "gradient_boosting"

    def _construir(self):
        return _classificador_base(self.nome).set_params(**self.hiperparametros)


def _contagem_inicial(valor) -> float:
    """
    Converte as faixas de contagem do codebook do NIJ ("0", "1", "3 or
    more", ...) no número inicial da faixa. Valores booleanos viram 1/0.
    A faixa superior é truncada (ex.: "3 or more" -> 3), o que é uma
    limitação registrada no texto.
    """
    if pd.isna(valor):
        return np.nan
    texto = str(valor).strip().lower()
    if texto in ("true", "yes", "y"):
        return 1.0
    if texto in ("false", "no", "n"):
        return 0.0
    achado = re.match(r"^(\d+)", texto)
    return float(achado.group(1)) if achado else np.nan


class ModeloBaselineDresselFarid(ModeloPreditivo):
    """
    Modelo de referência de dois atributos, adaptado de Dressel e Farid
    (2018): idade e total de condenações anteriores, numa regressão
    logística.

    Adaptação ao codebook do NIJ: a idade é fornecida em faixas
    (Age_at_Release) e o total de condenações anteriores é aproximado
    pela soma de Prior_Conviction_Episodes_Felony e
    Prior_Conviction_Episodes_Misd, que particionam as condenações
    anteriores por gravidade (as demais colunas Prior_Conviction_* dividem
    as mesmas condenações por tipo de crime e, portanto, se sobrepõem a
    essas duas). As faixas superiores são truncadas no codebook (ex.: "3
    or more"), então a soma é um limite inferior do total real.
    """

    nome = "baseline_dressel_farid"
    COLUNA_IDADE = "Age_at_Release"
    COLUNAS_CONDENACOES = ["Prior_Conviction_Episodes_Felony", "Prior_Conviction_Episodes_Misd"]

    def _construir(self):
        return _classificador_base(self.nome)

    def preparar_atributos(self, df: pd.DataFrame) -> pd.DataFrame:
        faltando = [c for c in [self.COLUNA_IDADE, *self.COLUNAS_CONDENACOES] if c not in df.columns]
        if faltando:
            raise ValueError(
                f"Colunas esperadas pelo baseline de Dressel e Farid não encontradas: {faltando}. "
                "Verifique os nomes de coluna do arquivo CSV."
            )
        total = sum(df[c].map(_contagem_inicial) for c in self.COLUNAS_CONDENACOES)
        idade = df[self.COLUNA_IDADE].astype(object).map(lambda v: str(v) if pd.notna(v) else np.nan)
        return pd.DataFrame(
            {"Age_at_Release": idade, "Total_Condenacoes_Anteriores": total.astype(float)},
            index=df.index,
        )


# Grades de busca de hiperparâmetros por algoritmo (UC05). Mantidas
# deliberadamente modestas (poucos valores por hiperparâmetro) para que a
# busca termine em tempo viável em hardware comum; essa limitação é
# registrada no protocolo experimental (Seção 2.5).
GRADES_HIPERPARAMETROS = {
    "regressao_logistica": {
        "C": [0.01, 0.1, 1.0, 10.0],
    },
    "random_forest": {
        "n_estimators": [100, 300],
        "max_depth": [5, 10, None],
        "min_samples_leaf": [1, 5],
    },
    "gradient_boosting": {
        "n_estimators": [100, 200],
        "max_depth": [2, 3],
        "learning_rate": [0.05, 0.1],
    },
}


class ConfiguradorAlgoritmo:
    def __init__(self, algoritmo: str, hiperparametros: dict | None = None):
        self.algoritmo = algoritmo
        self.hiperparametros = hiperparametros or {}

    def definir_hiperparametros(self, parametros: dict) -> None:
        """Define hiperparâmetros fixos, validando-os contra o algoritmo (UC05, passo 3 e E01)."""
        validos = set(_classificador_base(self.algoritmo).get_params())
        invalidos = sorted(set(parametros) - validos)
        if invalidos:
            raise ValueError(f"Hiperparâmetro(s) incompatível(is) com '{self.algoritmo}': {invalidos}. "
                             f"Parâmetros aceitos: {sorted(validos)}")
        self.hiperparametros = dict(parametros)

    def buscar_hiperparametros(self, x_treino: pd.DataFrame, y_treino: pd.Series, grade: dict, cv_interno) -> dict:
        """
        Busca em grade com validação cruzada interna, SOMENTE no conjunto
        de treino. O pipeline inteiro (pré-processamento + classificador)
        é reajustado em cada dobra interna, então nem a busca usa
        estatísticas de dados de validação no pré-processamento.
        """
        pipeline = Pipeline([
            ("prep", Preprocessador().construir_transformador(x_treino)),
            ("clf", _classificador_base(self.algoritmo)),
        ])
        grade_pipeline = {f"clf__{k}": v for k, v in grade.items()}
        busca = GridSearchCV(pipeline, grade_pipeline, cv=cv_interno, scoring="roc_auc", n_jobs=-1)
        busca.fit(x_treino, y_treino)
        self.hiperparametros = {k.removeprefix("clf__"): v for k, v in busca.best_params_.items()}
        return dict(self.hiperparametros)

    def instanciar(self) -> ModeloPreditivo:
        classes = {
            "regressao_logistica": RegressaoLogistica,
            "random_forest": RandomForest,
            "gradient_boosting": GradientBoosting,
            "baseline_dressel_farid": ModeloBaselineDresselFarid,
        }
        return classes[self.algoritmo](self.hiperparametros)
