"""
Classe Preprocessador (UC03 - Pré-processar dados).

Separa a variável-alvo, a raça (retida apenas para avaliação) e os
atributos preditivos, e constrói o transformador de pré-processamento
(tratamento de ausentes -> codificação -> normalização).

Diferença importante em relação às versões anteriores: o transformador
NÃO é ajustado aqui, sobre a base inteira. Ele é devolvido "vazio" e
encaixado como primeira etapa de um `sklearn.pipeline.Pipeline` junto
com o classificador (ver models.py). Assim, a mediana usada na
imputação, as categorias do one-hot encoding e a média/desvio da
normalização são aprendidas SOMENTE no conjunto de treino (e, durante a
busca de hiperparâmetros, somente em cada dobra interna de treino),
nunca com dados do conjunto de teste -- evitando o vazamento de
informação do teste para o pré-processamento.

Tratamento de valores ausentes:
- variáveis numéricas: imputação pela mediana do treino + indicador de
  ausência (uma coluna 0/1 por variável com ausentes), porque no dataset
  do NIJ a ausência costuma ser informativa (ex.: score de risco não
  calculado);
- variáveis categóricas: a ausência vira uma categoria explícita
  ("Ausente"), em vez de ser substituída pela moda, pelo mesmo motivo.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bias_removal import RemovedorDeVies
from dataset import COLUNAS_EXCLUIDAS_DOS_ATRIBUTOS, selecionar_conjunto_atributos

MAPA_ALVO_BINARIO = {
    "yes": 1, "no": 0,
    "true": 1, "false": 0,
    "y": 1, "n": 0,
    "1": 1, "0": 0,
    "1.0": 1, "0.0": 0,
}

ROTULO_AUSENTE = "Ausente"


def converter_alvo_binario(coluna: pd.Series) -> pd.Series:
    """
    Converte a coluna-alvo para inteiro binário (0/1), aceitando as
    diversas codificações usadas pelo Full Dataset real do NIJ
    (ex.: "Yes"/"No") além de valores já numéricos (0/1, True/False).
    """
    if pd.api.types.is_bool_dtype(coluna) or pd.api.types.is_numeric_dtype(coluna):
        return coluna.astype(int)

    normalizado = coluna.astype(str).str.strip().str.lower()
    convertido = normalizado.map(MAPA_ALVO_BINARIO)

    if convertido.isna().any():
        valores_nao_reconhecidos = sorted(normalizado[convertido.isna()].unique().tolist())
        raise ValueError(
            "A coluna-alvo contém valores que o pipeline não sabe "
            f"converter para 0/1: {valores_nao_reconhecidos}. Atualize "
            "MAPA_ALVO_BINARIO em src/preprocess.py para incluir esses "
            "valores."
        )
    return convertido.astype(int)


def _padronizar_tipos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Conversão de tipos SEM estado (não aprende nada dos dados, então
    pode ser aplicada antes da divisão treino/teste sem vazamento):
    toda coluna não numérica (texto, booleana, categórica) vira texto,
    preservando os ausentes como NaN.
    """
    df = df.copy()
    for col in df.columns:
        serie = df[col]
        if pd.api.types.is_bool_dtype(serie) or not pd.api.types.is_numeric_dtype(serie):
            df[col] = serie.astype(object).map(lambda v: str(v) if pd.notna(v) else np.nan)
    return df


class Preprocessador:
    def __init__(self, estrategia_imputacao: str = "mediana_com_indicador"):
        self.estrategia_imputacao = estrategia_imputacao

    # ---- etapas do pré-processamento (cada uma devolve um transformador
    # scikit-learn ainda não ajustado; o ajuste acontece só no treino) ----
    def tratar_valores_ausentes(self, tipo: str) -> SimpleImputer:
        if tipo == "numerica":
            return SimpleImputer(strategy="median", add_indicator=True)
        return SimpleImputer(strategy="constant", fill_value=ROTULO_AUSENTE)

    def codificar_variaveis(self) -> OneHotEncoder:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def normalizar(self) -> StandardScaler:
        return StandardScaler()

    def construir_transformador(self, x: pd.DataFrame) -> ColumnTransformer:
        """Monta o ColumnTransformer para as colunas de `x` (não ajusta)."""
        numericas = x.select_dtypes("number").columns.tolist()
        categoricas = [c for c in x.columns if c not in numericas]
        etapas = []
        if numericas:
            etapas.append((
                "num",
                Pipeline([("imputar", self.tratar_valores_ausentes("numerica")),
                          ("normalizar", self.normalizar())]),
                numericas,
            ))
        if categoricas:
            etapas.append((
                "cat",
                Pipeline([("imputar", self.tratar_valores_ausentes("categorica")),
                          ("codificar", self.codificar_variaveis())]),
                categoricas,
            ))
        return ColumnTransformer(etapas, remainder="drop", verbose_feature_names_out=False)

    def preparar_atributos(
        self,
        df: pd.DataFrame,
        coluna_alvo: str,
        coluna_raca: str,
        conjunto: str = "soltura",
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Separa (atributos brutos, alvo, raça retida) sem ajustar nada aos
        dados. Os atributos devolvidos já estão sem a raça, sem as
        colunas excluídas por construção (ID, alvos parciais por ano,
        Training_Sample) e restritos ao conjunto de atributos pedido
        ("soltura" na análise principal; "completo" na sensibilidade).
        A remoção de proxies acontece depois, com o diagnóstico calculado
        apenas no conjunto de treino (ver run_pipeline.py).
        """
        sem_alvo = df[coluna_alvo].isna()
        if sem_alvo.any():
            # UC03-E01: registro sem variável-alvo não serve ao treino supervisionado
            warnings.warn(f"{int(sem_alvo.sum())} registro(s) sem a variável-alvo foram excluídos.")
            df = df.loc[~sem_alvo]
        y = converter_alvo_binario(df[coluna_alvo])
        colunas_a_remover = [coluna_alvo] + [c for c in COLUNAS_EXCLUIDAS_DOS_ATRIBUTOS if c in df.columns]
        atributos = df.drop(columns=colunas_a_remover)
        atributos = selecionar_conjunto_atributos(atributos, conjunto)
        atributos, raca_retida = RemovedorDeVies().remover_variavel_racial(atributos, coluna_raca)
        raca_retida = raca_retida.astype(object).map(lambda v: str(v) if pd.notna(v) else np.nan)
        return _padronizar_tipos(atributos), y, raca_retida
